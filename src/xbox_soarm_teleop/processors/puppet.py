"""Puppet mode processor — crane geometry with Joy-Con IMU wrist orientation.

Extends crane mode by reading Joy-Con IMU (accelerometer + gyroscope) to drive
the wrist joints directly from the user's physical hand orientation:

- Left stick X   → shoulder_pan (base rotation, direct joint)
- Left stick Y   → reach (extend/retract, 2-DOF planar IK)
- D-pad Y        → height (raise/lower, 2-DOF planar IK); in JoyConController
                   this is populated by the height_up/height_down buttons (SR/B)
- IMU pitch      → wrist_flex (J4): tilt hand fwd/back
- IMU roll       → wrist_roll (J5): tilt hand left/right
- Right trigger  → gripper (direct position)
- LB             → deadman switch (SL on Joy-Con)
- A button       → return to home

When the IIO device is not found (IMU unavailable) the wrist joints hold their
current position — the processor degrades gracefully to crane-without-IMU-wrist.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    JOINT_LIMITS_DEG,
    JOINT_NAMES_WITH_GRIPPER,
)
from xbox_soarm_teleop.diagnostics.xbox_joint_drive import map_trigger_to_gripper_deg
from xbox_soarm_teleop.processors.joint_direct import JointCommand
from xbox_soarm_teleop.teleoperators.xbox import XboxState

# Velocity limits for the direct joints driven by stick / buttons
_PAN_VEL_DEG_S: float = 60.0
_HEIGHT_VEL_M_S: float = 0.06
_REACH_VEL_M_S: float = 0.06
_IK_MAX_VEL_DEG_S: float = 90.0

# Workspace bounds
_REACH_MIN: float = 0.06
_REACH_MAX: float = 0.30
_HEIGHT_MIN: float = 0.04
_HEIGHT_MAX: float = 0.40

# How many degrees of wrist deflection per degree of IMU tilt from neutral
_IMU_WRIST_SCALE: float = 1.0

# Complementary filter coefficient (0 = accel only, 1 = gyro only)
_COMP_ALPHA: float = 0.95


# ---------------------------------------------------------------------------
# JoyConIMU — reads pitch/roll from Linux IIO sysfs (hid-nintendo driver)
# ---------------------------------------------------------------------------

class JoyConIMU:
    """Read Joy-Con pitch and roll from the Linux IIO subsystem.

    The ``hid-nintendo`` kernel driver exposes accelerometer and gyroscope
    data at ``/sys/bus/iio/devices/iio:deviceN/`` as raw sysfs attributes.
    A complementary filter fuses both sensors to give stable orientation.

    Args:
        device_index: IIO device index to try first.  Pass ``None`` to
            auto-scan for a Joy-Con IMU device.
        alpha: Complementary filter coefficient (gyro weight, 0–1).
    """

    _ACCEL_ATTRS = ("in_accel_x_raw", "in_accel_y_raw", "in_accel_z_raw")
    _GYRO_ATTRS = ("in_anglvel_x_raw", "in_anglvel_y_raw", "in_anglvel_z_raw")
    _NAME_KEYWORDS = ("joycon", "joy-con", "nintendo")

    def __init__(self, device_index: int | None = None, alpha: float = _COMP_ALPHA) -> None:
        self.alpha = alpha
        self._iio_path: Path | None = self._find_iio_device(device_index)
        self._accel_scale: float = self._read_scale("in_accel_scale", 1.0)
        self._gyro_scale: float = self._read_scale("in_anglvel_scale", 1.0)
        self._pitch: float = 0.0   # rad — updated by filter
        self._roll: float = 0.0    # rad
        self._last_t: float = time.monotonic()
        self._calibrated = False
        if self._iio_path:
            print(f"JoyConIMU: found IIO device at {self._iio_path}", flush=True)
        else:
            print(
                "JoyConIMU: no IIO device found — wrist orientation from IMU unavailable. "
                "Ensure hid-nintendo driver is loaded and Joy-Con is connected.",
                flush=True,
            )

    @property
    def available(self) -> bool:
        return self._iio_path is not None

    def read(self) -> tuple[float, float]:
        """Return current ``(pitch_rad, roll_rad)`` from the IMU.

        Returns ``(0.0, 0.0)`` if the IIO device is not available.
        """
        if self._iio_path is None:
            return 0.0, 0.0

        try:
            ax, ay, az = self._read_accel()
            gx, gy, gz = self._read_gyro()
        except OSError:
            return self._pitch, self._roll

        now = time.monotonic()
        dt = min(now - self._last_t, 0.1)   # cap stale dt on first call
        self._last_t = now

        # Accelerometer-based pitch/roll (gravity reference, noisy)
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 0.01:
            accel_pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
            accel_roll = math.atan2(ay, az)
        else:
            accel_pitch = self._pitch
            accel_roll = self._roll

        # Gyro integration (drift-prone but smooth)
        gyro_pitch = self._pitch + gy * dt
        gyro_roll = self._roll + gx * dt

        # Complementary filter
        self._pitch = self.alpha * gyro_pitch + (1.0 - self.alpha) * accel_pitch
        self._roll = self.alpha * gyro_roll + (1.0 - self.alpha) * accel_roll

        return self._pitch, self._roll

    def calibrate(self) -> None:
        """Reset the filter state from current accelerometer reading."""
        if self._iio_path is None:
            return
        try:
            ax, ay, az = self._read_accel()
            norm = math.sqrt(ax * ax + ay * ay + az * az)
            if norm > 0.01:
                self._pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
                self._roll = math.atan2(ay, az)
        except OSError:
            pass
        self._last_t = time.monotonic()
        self._calibrated = True

    # ── Private helpers ────────────────────────────────────────────────────

    def _find_iio_device(self, index: int | None) -> Path | None:
        iio_root = Path("/sys/bus/iio/devices")
        if not iio_root.exists():
            return None

        candidates: list[Path] = []
        for dev in sorted(iio_root.iterdir()):
            name_file = dev / "name"
            if not name_file.exists():
                continue
            try:
                name = name_file.read_text().strip().lower()
            except OSError:
                continue
            if any(kw in name for kw in self._NAME_KEYWORDS):
                # Require at least accel attributes
                if (dev / "in_accel_x_raw").exists():
                    candidates.append(dev)

        if not candidates:
            return None
        if index is not None and index < len(candidates):
            return candidates[index]
        return candidates[0]

    def _read_scale(self, attr: str, default: float) -> float:
        if self._iio_path is None:
            return default
        f = self._iio_path / attr
        if not f.exists():
            return default
        try:
            return float(f.read_text().strip())
        except (OSError, ValueError):
            return default

    def _read_accel(self) -> tuple[float, float, float]:
        p = self._iio_path
        ax = float((p / "in_accel_x_raw").read_text()) * self._accel_scale  # type: ignore[operator]
        ay = float((p / "in_accel_y_raw").read_text()) * self._accel_scale  # type: ignore[operator]
        az = float((p / "in_accel_z_raw").read_text()) * self._accel_scale  # type: ignore[operator]
        return ax, ay, az

    def _read_gyro(self) -> tuple[float, float, float]:
        p = self._iio_path
        gx = float((p / "in_anglvel_x_raw").read_text()) * self._gyro_scale  # type: ignore[operator]
        gy = float((p / "in_anglvel_y_raw").read_text()) * self._gyro_scale  # type: ignore[operator]
        gz = float((p / "in_anglvel_z_raw").read_text()) * self._gyro_scale  # type: ignore[operator]
        return gx, gy, gz


# ---------------------------------------------------------------------------
# PuppetProcessor
# ---------------------------------------------------------------------------

class PuppetProcessor:
    """Crane geometry with Joy-Con IMU wrist orientation.

    Wrist joints are driven by the IMU's absolute pitch/roll relative to
    a calibrated neutral pose (set on ``reset()``).  All other joints use
    crane-style decoupled control.

    Args:
        urdf_path: Path to robot URDF (required for 2-DOF planar IK).
        pan_vel_deg_s: Max shoulder_pan velocity at full stick deflection.
        height_vel_m_s: Max height velocity at full button press.
        reach_vel_m_s: Max reach velocity at full stick deflection.
        loop_dt: Control loop period in seconds.
        imu_wrist_scale: Degrees of wrist motion per degree of IMU tilt.
        imu_device_index: IIO device index for JoyConIMU.
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        pan_vel_deg_s: float = _PAN_VEL_DEG_S,
        height_vel_m_s: float = _HEIGHT_VEL_M_S,
        reach_vel_m_s: float = _REACH_VEL_M_S,
        loop_dt: float = 1.0 / 30.0,
        imu_wrist_scale: float = _IMU_WRIST_SCALE,
        imu_device_index: int | None = None,
    ) -> None:
        self.pan_vel_deg_s = pan_vel_deg_s
        self.height_vel_m_s = height_vel_m_s
        self.reach_vel_m_s = reach_vel_m_s
        self.loop_dt = loop_dt
        self.imu_wrist_scale = imu_wrist_scale

        # IMU
        self._imu = JoyConIMU(device_index=imu_device_index)

        # Direct joint targets
        self._pan_deg = float(HOME_POSITION_DEG["shoulder_pan"])
        self._wrist_flex_deg = float(HOME_POSITION_DEG["wrist_flex"])
        self._wrist_roll_deg = float(HOME_POSITION_DEG["wrist_roll"])
        self._gripper_deg = float(HOME_POSITION_DEG["gripper"])

        # IK joint targets
        self._sl_deg = float(HOME_POSITION_DEG["shoulder_lift"])
        self._ef_deg = float(HOME_POSITION_DEG["elbow_flex"])

        # Cylindrical targets (reach, height)
        self._reach_m: float = 0.15
        self._height_m: float = 0.15

        # Neutral IMU orientation captured at reset
        self._neutral_pitch: float = 0.0
        self._neutral_roll: float = 0.0

        # Placo 2-DOF IK
        self._planar_ik = None
        if urdf_path is not None:
            self._init_kinematics(urdf_path)

    def _init_kinematics(self, urdf_path: str) -> None:
        try:
            from lerobot.model.kinematics import RobotKinematics

            self._planar_ik = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_frame_link",
                joint_names=["shoulder_lift", "elbow_flex"],
            )
            full_ik = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_frame_link",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"],
            )
            home_4j = np.array(
                [
                    HOME_POSITION_DEG["shoulder_pan"],
                    HOME_POSITION_DEG["shoulder_lift"],
                    HOME_POSITION_DEG["elbow_flex"],
                    HOME_POSITION_DEG["wrist_flex"],
                ],
                dtype=float,
            )
            ee_home = full_ik.forward_kinematics(home_4j)[:3, 3]
            self._reach_m = float(np.clip(
                np.sqrt(float(ee_home[0]) ** 2 + float(ee_home[1]) ** 2),
                _REACH_MIN, _REACH_MAX,
            ))
            self._height_m = float(np.clip(float(ee_home[2]), _HEIGHT_MIN, _HEIGHT_MAX))
        except Exception as exc:
            print(f"PuppetProcessor: kinematics init failed ({exc}), using defaults.", flush=True)
            self._planar_ik = None

    def __call__(self, state: XboxState) -> JointCommand:
        """Compute joint targets from controller state + IMU.

        Args:
            state: Current controller state.

        Returns:
            JointCommand with updated goal positions.
        """
        # Gripper always active
        g_lo, g_hi = JOINT_LIMITS_DEG["gripper"]
        self._gripper_deg = map_trigger_to_gripper_deg(state.right_trigger, g_lo, g_hi)

        # A button: go home and recalibrate IMU neutral
        if state.a_button_pressed:
            self.reset()
            return JointCommand(
                goals_deg={n: HOME_POSITION_DEG[n] for n in JOINT_NAMES_WITH_GRIPPER},
                selected_joint="",
            )

        # Deadman
        if not state.left_bumper:
            return self._current_command()

        dt = self.loop_dt

        # --- Pan ---
        pan_lo, pan_hi = JOINT_LIMITS_DEG["shoulder_pan"]
        self._pan_deg = float(np.clip(
            self._pan_deg + state.left_stick_x * self.pan_vel_deg_s * dt,
            pan_lo, pan_hi,
        ))

        # --- Reach (left stick Y) ---
        self._reach_m = float(np.clip(
            self._reach_m + (-state.left_stick_y) * self.reach_vel_m_s * dt,
            _REACH_MIN, _REACH_MAX,
        ))

        # --- Height (dpad_y populated by JoyCon height buttons) ---
        # dpad_y convention: +1.0 = up button pressed, -1.0 = down button pressed
        self._height_m = float(np.clip(
            self._height_m + state.dpad_y * self.height_vel_m_s * dt,
            _HEIGHT_MIN, _HEIGHT_MAX,
        ))

        # --- 2-DOF planar IK ---
        if self._planar_ik is not None:
            target = np.eye(4)
            target[0, 3] = self._reach_m
            target[1, 3] = 0.0
            target[2, 3] = self._height_m
            ik_result = self._planar_ik.inverse_kinematics(
                np.array([self._sl_deg, self._ef_deg], dtype=float),
                target,
                position_weight=1.0,
                orientation_weight=0.0,
            )
            if ik_result is not None:
                sl_new, ef_new = float(ik_result[0]), float(ik_result[1])
                sl_lo, sl_hi = JOINT_LIMITS_DEG["shoulder_lift"]
                ef_lo, ef_hi = JOINT_LIMITS_DEG["elbow_flex"]
                # Velocity-cap IK output
                sl_delta = float(np.clip(sl_new - self._sl_deg, -_IK_MAX_VEL_DEG_S * dt, _IK_MAX_VEL_DEG_S * dt))
                ef_delta = float(np.clip(ef_new - self._ef_deg, -_IK_MAX_VEL_DEG_S * dt, _IK_MAX_VEL_DEG_S * dt))
                self._sl_deg = float(np.clip(self._sl_deg + sl_delta, sl_lo, sl_hi))
                self._ef_deg = float(np.clip(self._ef_deg + ef_delta, ef_lo, ef_hi))

        # --- Wrist from IMU (absolute orientation relative to neutral) ---
        if self._imu.available:
            pitch_rad, roll_rad = self._imu.read()
            delta_pitch_deg = math.degrees(pitch_rad - self._neutral_pitch)
            delta_roll_deg = math.degrees(roll_rad - self._neutral_roll)

            wf_lo, wf_hi = JOINT_LIMITS_DEG["wrist_flex"]
            wr_lo, wr_hi = JOINT_LIMITS_DEG["wrist_roll"]
            wf_home = float(HOME_POSITION_DEG["wrist_flex"])
            wr_home = float(HOME_POSITION_DEG["wrist_roll"])

            self._wrist_flex_deg = float(np.clip(
                wf_home + delta_pitch_deg * self.imu_wrist_scale,
                wf_lo, wf_hi,
            ))
            self._wrist_roll_deg = float(np.clip(
                wr_home + delta_roll_deg * self.imu_wrist_scale,
                wr_lo, wr_hi,
            ))
        # else: wrist holds current position (no IMU available)

        return self._current_command()

    def reset(self) -> None:
        """Return to home position and recalibrate IMU neutral."""
        self._pan_deg = float(HOME_POSITION_DEG["shoulder_pan"])
        self._wrist_flex_deg = float(HOME_POSITION_DEG["wrist_flex"])
        self._wrist_roll_deg = float(HOME_POSITION_DEG["wrist_roll"])
        self._gripper_deg = float(HOME_POSITION_DEG["gripper"])
        self._sl_deg = float(HOME_POSITION_DEG["shoulder_lift"])
        self._ef_deg = float(HOME_POSITION_DEG["elbow_flex"])
        self._reach_m = 0.15
        self._height_m = 0.15
        # Capture current IMU orientation as new neutral reference
        if self._imu.available:
            self._imu.calibrate()
            self._neutral_pitch, self._neutral_roll = self._imu.read()

    def _current_command(self) -> JointCommand:
        return JointCommand(
            goals_deg={
                "shoulder_pan": self._pan_deg,
                "shoulder_lift": self._sl_deg,
                "elbow_flex": self._ef_deg,
                "wrist_flex": self._wrist_flex_deg,
                "wrist_roll": self._wrist_roll_deg,
                "gripper": self._gripper_deg,
            },
            selected_joint="",
        )
