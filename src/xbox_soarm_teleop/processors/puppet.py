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

import numpy as np

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    JOINT_LIMITS_DEG,
    JOINT_NAMES_WITH_GRIPPER,
)
from xbox_soarm_teleop.diagnostics.xbox_joint_drive import map_trigger_to_gripper_deg
from xbox_soarm_teleop.processors.joint_direct import JointCommand
from xbox_soarm_teleop.teleoperators.joycon_imu import JoyConIMU
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
            from xbox_soarm_teleop.kinematics.cylindrical import CylindricalKinematics

            self._planar_ik = CylindricalKinematics(urdf_path)
            self._sync_cylindrical_target()
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
        # Gate every commanded joint, including homing and gripper.
        if not state.left_bumper:
            return self._current_command()

        # Position-controlled gripper while enabled
        g_lo, g_hi = JOINT_LIMITS_DEG["gripper"]
        self._gripper_deg = map_trigger_to_gripper_deg(state.right_trigger, g_lo, g_hi)

        # A button: go home and recalibrate IMU neutral
        if state.a_button_pressed:
            self.reset()
            return JointCommand(
                goals_deg={n: HOME_POSITION_DEG[n] for n in JOINT_NAMES_WITH_GRIPPER},
                selected_joint="",
            )

        previous = (self._pan_deg, self._sl_deg, self._ef_deg,
                    self._wrist_flex_deg, self._wrist_roll_deg, self._reach_m, self._height_m)
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


        if self._planar_ik is not None:
            solved = self._planar_ik.inverse_cylindrical(
                np.array([self._sl_deg, self._ef_deg]), self._reach_m, self._height_m,
                pan_deg=self._pan_deg, wrist_flex_deg=self._wrist_flex_deg,
                wrist_roll_deg=self._wrist_roll_deg,
            )
            status = getattr(self._planar_ik, "last_ik_result", None)
            if solved is None or (status is not None and not status.accepted):
                (self._pan_deg, self._sl_deg, self._ef_deg, self._wrist_flex_deg,
                 self._wrist_roll_deg, self._reach_m, self._height_m) = previous
            else:
                before = np.asarray(previous[:5])
                proposed = np.array([self._pan_deg, *solved, self._wrist_flex_deg, self._wrist_roll_deg])
                change = proposed - before
                peak = float(np.max(np.abs(change)))
                scale = min(1.0, _IK_MAX_VEL_DEG_S * dt / peak) if peak > 0 else 1.0
                (self._pan_deg, self._sl_deg, self._ef_deg, self._wrist_flex_deg,
                 self._wrist_roll_deg) = map(float, before + scale * change)

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
        if self._planar_ik is not None:
            self._sync_cylindrical_target()
        # Capture current IMU orientation as new neutral reference
        if self._imu.available:
            self._imu.calibrate()
            self._neutral_pitch, self._neutral_roll = self._imu.read()

    def _sync_cylindrical_target(self) -> None:
        """Initialize reach and height from the full current model pose."""
        for name, angle in (("shoulder_pan", self._pan_deg),
                            ("wrist_flex", self._wrist_flex_deg),
                            ("wrist_roll", self._wrist_roll_deg)):
            self._planar_ik.robot.set_joint(name, float(np.deg2rad(angle)))
        pose = self._planar_ik.forward_kinematics(np.array([self._sl_deg, self._ef_deg]))
        self._reach_m = float(np.hypot(*pose[:2, 3]))
        self._height_m = float(pose[2, 3])

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
