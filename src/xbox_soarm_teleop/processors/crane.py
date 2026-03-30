"""Crane mode processor — cylindrical, decoupled control for SO-ARM101.

In crane mode the arm operates in a natural cylindrical geometry:

- Left stick X  → shoulder_pan (base rotation, direct joint)
- Left stick Y  → reach (extend/retract, 2-DOF planar IK with elbow)
- Right stick Y → height (raise/lower, 2-DOF planar IK with elbow)
- Right stick X → wrist_roll (direct joint)
- D-pad Y       → wrist_flex (direct joint)
- Right trigger → gripper (direct position)
- LB            → deadman switch

shoulder_lift and elbow_flex are solved by a 2-DOF planar IK that maps
(reach, height) to joint angles. The solver uses Placo via a 2-joint
RobotKinematics instance so all joint-limit handling is inherited.
"""

from __future__ import annotations

import numpy as np

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    JOINT_LIMITS_DEG,
)
from xbox_soarm_teleop.control.home import scalar_reached, step_scalar_toward
from xbox_soarm_teleop.diagnostics.xbox_joint_drive import map_trigger_to_gripper_deg
from xbox_soarm_teleop.processors.joint_direct import JointCommand
from xbox_soarm_teleop.teleoperators.xbox import XboxState

# Velocity limits shared by the direct joints
_PAN_VEL_DEG_S: float = 60.0
_WRIST_VEL_DEG_S: float = 60.0
_REACH_VEL_M_S: float = 0.06
_HEIGHT_VEL_M_S: float = 0.06

# Per-step joint velocity cap applied to the IK output (deg/s)
_IK_MAX_VEL_DEG_S: float = 90.0

# Workspace bounds for (reach, height) targets
_REACH_MIN: float = 0.06
_REACH_MAX: float = 0.30
_HEIGHT_MIN: float = 0.04
_HEIGHT_MAX: float = 0.40

# Neutral crane home used for teleoperation. This is intentionally different
# from the folded parked pose in HOME_POSITION_DEG.
_CRANE_HOME_REACH_M: float = 0.16
_CRANE_HOME_HEIGHT_M: float = 0.12
_CRANE_HOME_WRIST_FLEX_DEG: float = 13.0


class CraneProcessor:
    """Crane-style decoupled cylindrical control for SO-ARM101.

    shoulder_pan, wrist_flex, and wrist_roll are driven directly from
    sticks/D-pad.  shoulder_lift and elbow_flex are solved by a 2-DOF
    planar IK given the target (reach, height).

    Args:
        urdf_path: Path to the robot URDF file.  Required for the 2-DOF IK
            solver; if *None* the processor falls back to the stubbed zero-
            motion behaviour.
        pan_vel_deg_s: Max shoulder_pan velocity (deg/s) at full stick.
        wrist_vel_deg_s: Max wrist velocity (deg/s) at full stick / D-pad.
        reach_vel_m_s: Max reach velocity (m/s) at full stick.
        height_vel_m_s: Max height velocity (m/s) at full stick.
        loop_dt: Control loop period in seconds.
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        pan_vel_deg_s: float = _PAN_VEL_DEG_S,
        wrist_vel_deg_s: float = _WRIST_VEL_DEG_S,
        reach_vel_m_s: float = _REACH_VEL_M_S,
        height_vel_m_s: float = _HEIGHT_VEL_M_S,
        loop_dt: float = 1.0 / 30.0,
    ) -> None:
        self.pan_vel_deg_s = pan_vel_deg_s
        self.wrist_vel_deg_s = wrist_vel_deg_s
        self.reach_vel_m_s = reach_vel_m_s
        self.height_vel_m_s = height_vel_m_s
        self.loop_dt = loop_dt

        # Direct-joint state
        self._pan_deg = float(HOME_POSITION_DEG["shoulder_pan"])
        self._wrist_flex_deg = float(_CRANE_HOME_WRIST_FLEX_DEG)
        self._wrist_roll_deg = float(HOME_POSITION_DEG["wrist_roll"])
        self._gripper_deg = float(HOME_POSITION_DEG["gripper"])

        # IK joint state (warm-started from parked home until crane home is solved)
        self._sl_deg = float(HOME_POSITION_DEG["shoulder_lift"])
        self._ef_deg = float(HOME_POSITION_DEG["elbow_flex"])

        # Planar IK solver (shoulder_lift + elbow_flex only)
        self._planar_ik = None
        self._reach_m: float = _CRANE_HOME_REACH_M
        self._height_m: float = _CRANE_HOME_HEIGHT_M
        self._homing_active: bool = False

        if urdf_path is not None:
            self._init_kinematics(urdf_path)
        self.reset()

    def _init_kinematics(self, urdf_path: str) -> None:
        """Load the 2-joint planar IK solver and initialise reach/height."""
        try:
            from lerobot.model.kinematics import RobotKinematics

            # 2-joint solver: shoulder_pan and wrist_flex are fixed at 0
            self._planar_ik = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_frame_link",
                joint_names=["shoulder_lift", "elbow_flex"],
            )

        except Exception as exc:
            print(f"CraneProcessor: kinematics init failed ({exc}), using defaults.", flush=True)
            self._planar_ik = None

    def __call__(self, state: XboxState) -> JointCommand:
        """Compute joint position targets from controller state.

        Args:
            state: Current Xbox controller state.

        Returns:
            JointCommand with updated goal positions.
        """
        # A button: start a smooth return to the neutral crane pose.
        if state.a_button_pressed:
            self._homing_active = True

        if self._homing_active:
            self._step_home()
            return self._current_command()

        # Gripper is always active (position control via trigger)
        g_lower, g_upper = JOINT_LIMITS_DEG["gripper"]
        self._gripper_deg = map_trigger_to_gripper_deg(state.right_trigger, g_lower, g_upper)

        # Deadman — hold current position if LB is released
        if not state.left_bumper:
            return self._current_command()

        dt = self.loop_dt

        # --- Direct joints ---
        pan_lo, pan_hi = JOINT_LIMITS_DEG["shoulder_pan"]
        self._pan_deg = float(np.clip(
            self._pan_deg + state.left_stick_x * self.pan_vel_deg_s * dt,
            pan_lo, pan_hi,
        ))

        # D-pad Y: -1 = up → tilt wrist up
        wf_lo, wf_hi = JOINT_LIMITS_DEG["wrist_flex"]
        self._wrist_flex_deg = float(np.clip(
            self._wrist_flex_deg + state.dpad_y * self.wrist_vel_deg_s * dt,
            wf_lo, wf_hi,
        ))

        wr_lo, wr_hi = JOINT_LIMITS_DEG["wrist_roll"]
        self._wrist_roll_deg = float(np.clip(
            self._wrist_roll_deg + state.right_stick_x * self.wrist_vel_deg_s * dt,
            wr_lo, wr_hi,
        ))

        # --- Cylindrical targets ---
        # Left stick Y forward (negative raw) → extend reach
        self._reach_m = float(np.clip(
            self._reach_m + (-state.left_stick_y) * self.reach_vel_m_s * dt,
            _REACH_MIN, _REACH_MAX,
        ))
        # Right stick Y up (negative raw) → raise height
        self._height_m = float(np.clip(
            self._height_m + (-state.right_stick_y) * self.height_vel_m_s * dt,
            _HEIGHT_MIN, _HEIGHT_MAX,
        ))

        # --- 2-DOF planar IK ---
        if self._planar_ik is not None:
            target = np.eye(4)
            target[0, 3] = self._reach_m   # x = reach in arm plane (pan=0 frame)
            target[1, 3] = 0.0              # y = 0 (arm stays in xz-plane)
            target[2, 3] = self._height_m   # z = height

            ik_result = self._planar_ik.inverse_kinematics(
                np.array([self._sl_deg, self._ef_deg], dtype=float),
                target,
                position_weight=1.0,
                orientation_weight=0.0,
            )
            if ik_result is not None:
                # Velocity-limit the IK output to prevent sudden jumps
                max_step = _IK_MAX_VEL_DEG_S * dt
                sl_raw = float(ik_result[0])
                ef_raw = float(ik_result[1])
                sl_new = float(np.clip(sl_raw, self._sl_deg - max_step, self._sl_deg + max_step))
                ef_new = float(np.clip(ef_raw, self._ef_deg - max_step, self._ef_deg + max_step))

                # Apply joint limits
                sl_lo, sl_hi = JOINT_LIMITS_DEG["shoulder_lift"]
                ef_lo, ef_hi = JOINT_LIMITS_DEG["elbow_flex"]
                self._sl_deg = float(np.clip(sl_new, sl_lo, sl_hi))
                self._ef_deg = float(np.clip(ef_new, ef_lo, ef_hi))

        return self._current_command()

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

    def _step_home(self) -> None:
        """Move crane state toward its neutral teleoperation pose."""
        self._pan_deg = step_scalar_toward(
            self._pan_deg,
            HOME_POSITION_DEG["shoulder_pan"],
            self.pan_vel_deg_s * self.loop_dt,
        )
        self._wrist_flex_deg = step_scalar_toward(
            self._wrist_flex_deg,
            _CRANE_HOME_WRIST_FLEX_DEG,
            self.wrist_vel_deg_s * self.loop_dt,
        )
        self._wrist_roll_deg = step_scalar_toward(
            self._wrist_roll_deg,
            HOME_POSITION_DEG["wrist_roll"],
            self.wrist_vel_deg_s * self.loop_dt,
        )
        self._gripper_deg = step_scalar_toward(
            self._gripper_deg,
            HOME_POSITION_DEG["gripper"],
            _IK_MAX_VEL_DEG_S * self.loop_dt,
        )
        self._reach_m = step_scalar_toward(
            self._reach_m,
            _CRANE_HOME_REACH_M,
            self.reach_vel_m_s * self.loop_dt,
        )
        self._height_m = step_scalar_toward(
            self._height_m,
            _CRANE_HOME_HEIGHT_M,
            self.height_vel_m_s * self.loop_dt,
        )

        solved = self._solve_planar_target(self._reach_m, self._height_m)
        if solved is not None:
            self._sl_deg = step_scalar_toward(
                self._sl_deg,
                solved[0],
                _IK_MAX_VEL_DEG_S * self.loop_dt,
            )
            self._ef_deg = step_scalar_toward(
                self._ef_deg,
                solved[1],
                _IK_MAX_VEL_DEG_S * self.loop_dt,
            )

        done = (
            scalar_reached(self._pan_deg, HOME_POSITION_DEG["shoulder_pan"])
            and scalar_reached(self._wrist_flex_deg, _CRANE_HOME_WRIST_FLEX_DEG)
            and scalar_reached(self._wrist_roll_deg, HOME_POSITION_DEG["wrist_roll"])
            and scalar_reached(self._gripper_deg, HOME_POSITION_DEG["gripper"])
            and scalar_reached(self._reach_m, _CRANE_HOME_REACH_M)
            and scalar_reached(self._height_m, _CRANE_HOME_HEIGHT_M)
        )
        if solved is not None:
            done = (
                done
                and scalar_reached(self._sl_deg, solved[0])
                and scalar_reached(self._ef_deg, solved[1])
            )
        self._homing_active = not done

    def _solve_planar_target(self, reach_m: float, height_m: float) -> tuple[float, float] | None:
        """Solve the 2-DOF planar IK for a cylindrical target."""
        if self._planar_ik is None:
            return None

        target = np.eye(4)
        target[0, 3] = float(np.clip(reach_m, _REACH_MIN, _REACH_MAX))
        target[2, 3] = float(np.clip(height_m, _HEIGHT_MIN, _HEIGHT_MAX))

        ik_result = self._planar_ik.inverse_kinematics(
            np.array([self._sl_deg, self._ef_deg], dtype=float),
            target,
            position_weight=1.0,
            orientation_weight=0.0,
        )
        if ik_result is None:
            return None

        sl_lo, sl_hi = JOINT_LIMITS_DEG["shoulder_lift"]
        ef_lo, ef_hi = JOINT_LIMITS_DEG["elbow_flex"]
        return (
            float(np.clip(float(ik_result[0]), sl_lo, sl_hi)),
            float(np.clip(float(ik_result[1]), ef_lo, ef_hi)),
        )

    def reset(self) -> None:
        """Reset crane mode to a neutral teleoperation pose."""
        self._pan_deg = float(HOME_POSITION_DEG["shoulder_pan"])
        self._wrist_flex_deg = float(
            np.clip(
                _CRANE_HOME_WRIST_FLEX_DEG,
                JOINT_LIMITS_DEG["wrist_flex"][0],
                JOINT_LIMITS_DEG["wrist_flex"][1],
            )
        )
        self._wrist_roll_deg = float(HOME_POSITION_DEG["wrist_roll"])
        self._gripper_deg = float(HOME_POSITION_DEG["gripper"])
        self._reach_m = _CRANE_HOME_REACH_M
        self._height_m = _CRANE_HOME_HEIGHT_M
        self._homing_active = False

        solved = self._solve_planar_target(self._reach_m, self._height_m)
        if solved is None:
            self._sl_deg = float(HOME_POSITION_DEG["shoulder_lift"])
            self._ef_deg = float(HOME_POSITION_DEG["elbow_flex"])
        else:
            self._sl_deg, self._ef_deg = solved
