"""Processor for direct joint velocity control (bypasses IK)."""

from __future__ import annotations

from dataclasses import dataclass

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    JOINT_LIMITS_DEG,
    JOINT_NAMES_WITH_GRIPPER,
)
from xbox_soarm_teleop.control.home import scalar_reached, step_scalar_toward
from xbox_soarm_teleop.diagnostics.xbox_joint_drive import (
    advance_goal,
    dpad_edge,
    map_trigger_to_gripper_deg,
)
from xbox_soarm_teleop.teleoperators.xbox import XboxState

# ---------------------------------------------------------------------------
# Per-joint axis mapping for multi-joint keyboard mode
# Each entry: (joint_name, XboxState attribute, sign)
# Sign convention: positive state value → positive joint velocity.
# ---------------------------------------------------------------------------
_MULTI_JOINT_AXES: list[tuple[str, str, float]] = [
    ("shoulder_pan",  "left_stick_x",  +1.0),  # A=neg / D=pos
    ("shoulder_lift", "right_stick_y", -1.0),  # W→state neg → lift pos
    ("elbow_flex",    "left_stick_y",  -1.0),  # R→state neg → flex pos
    ("wrist_flex",    "right_stick_x", +1.0),  # Q=neg / E=pos
    ("wrist_roll",    "dpad_y",        -1.0),  # ↑→HAT -1 → roll pos
]


@dataclass
class JointCommand:
    """Direct joint position targets and metadata.

    Attributes:
        goals_deg: Mapping of joint name to target position in degrees.
        selected_joint: The joint currently being commanded by the left stick.
        cmd_vel_deg_s: Commanded velocity for the selected joint (deg/s), for logging.
    """

    goals_deg: dict[str, float]
    selected_joint: str
    cmd_vel_deg_s: float = 0.0


class JointDirectProcessor:
    """Maps Xbox controller state to direct joint position targets.

    Maintains mutable state: current goal positions per joint and the
    currently-selected joint index. Each call integrates the stick
    command and returns updated target positions.

    Two modes of operation:
    - **Single-joint** (``multi_joint=False``, default): left stick X drives
      the currently-selected joint; D-pad left/right cycles the selection.
      Designed for Xbox / Joy-Con.
    - **Multi-joint** (``multi_joint=True``): five dedicated axes drive all
      arm joints simultaneously. Designed for keyboard control.

      Key mapping (keyboard defaults):
          A / D       → shoulder_pan
          W / S       → shoulder_lift
          R / F       → elbow_flex
          Q / E       → wrist_flex
          ↑ / ↓       → wrist_roll
          Space       → gripper (via right_trigger)

    Common to both modes:
        LB (hold):          Enable joint motion (deadman; always True for keyboard).
        A button (press):   Reset all joints to home position.
        Right trigger:      Gripper position (0=open, 1=closed).

    Args:
        max_vel_deg_s: Max velocity at full deflection (deg/s).
        dt: Control loop period in seconds. Must match the actual loop rate.
        multi_joint: When True, use the per-joint axis mapping described above.
    """

    def __init__(
        self,
        max_vel_deg_s: float = 70.0,
        dt: float = 1.0 / 30.0,
        multi_joint: bool = False,
    ) -> None:
        self.max_vel_deg_s = max_vel_deg_s
        self.dt = dt
        self.multi_joint = multi_joint
        self._selected_idx: int = 0
        self._goals_deg: dict[str, float] = {
            name: HOME_POSITION_DEG[name] for name in JOINT_NAMES_WITH_GRIPPER
        }
        self._prev_dpad_x: float = 0.0
        self._homing_active: bool = False

    @property
    def selected_joint(self) -> str:
        """Name of the joint currently driven by the left stick (single-joint mode)."""
        return JOINT_NAMES_WITH_GRIPPER[self._selected_idx]

    def __call__(self, state: XboxState) -> JointCommand:
        """Compute joint position targets from controller state.

        Args:
            state: Current Xbox controller state.

        Returns:
            JointCommand with updated goal positions and metadata.
        """
        # A button press: start a smooth return to home.
        if state.a_button_pressed:
            self._homing_active = True

        if self._homing_active:
            max_step = self.max_vel_deg_s * self.dt
            for name in JOINT_NAMES_WITH_GRIPPER:
                self._goals_deg[name] = step_scalar_toward(
                    self._goals_deg[name],
                    HOME_POSITION_DEG[name],
                    max_step,
                )
            if all(
                scalar_reached(self._goals_deg[name], HOME_POSITION_DEG[name])
                for name in JOINT_NAMES_WITH_GRIPPER
            ):
                self._homing_active = False
            return JointCommand(
                goals_deg=dict(self._goals_deg),
                selected_joint=self.selected_joint,
                cmd_vel_deg_s=0.0,
            )

        # Gripper: always active via trigger (position control)
        g_lower, g_upper = JOINT_LIMITS_DEG["gripper"]
        self._goals_deg["gripper"] = map_trigger_to_gripper_deg(
            state.right_trigger, g_lower, g_upper
        )

        cmd_vel = 0.0

        if self.multi_joint:
            # Drive all arm joints simultaneously from dedicated axes.
            if state.left_bumper:
                for joint, attr, sign in _MULTI_JOINT_AXES:
                    vel = getattr(state, attr) * sign * self.max_vel_deg_s
                    lower, upper = JOINT_LIMITS_DEG[joint]
                    self._goals_deg[joint] = advance_goal(
                        self._goals_deg[joint], vel, self.dt, lower, upper
                    )
        else:
            # Single-joint mode: D-pad cycles selection, left stick X drives it.
            edge = dpad_edge(state.dpad_x, self._prev_dpad_x)
            self._prev_dpad_x = state.dpad_x
            if edge != 0:
                n = len(JOINT_NAMES_WITH_GRIPPER)
                self._selected_idx = (self._selected_idx + edge) % n

            if state.left_bumper:
                cmd_vel = state.left_stick_x * self.max_vel_deg_s
                joint = self.selected_joint
                lower, upper = JOINT_LIMITS_DEG[joint]
                self._goals_deg[joint] = advance_goal(
                    self._goals_deg[joint], cmd_vel, self.dt, lower, upper
                )

        return JointCommand(
            goals_deg=dict(self._goals_deg),
            selected_joint=self.selected_joint,
            cmd_vel_deg_s=cmd_vel,
        )

    def reset(self) -> None:
        """Reset to home position and first joint selection."""
        self._selected_idx = 0
        self._goals_deg = {name: HOME_POSITION_DEG[name] for name in JOINT_NAMES_WITH_GRIPPER}
        self._prev_dpad_x = 0.0
        self._homing_active = False
