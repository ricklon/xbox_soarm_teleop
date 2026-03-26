"""Processor for direct joint velocity control (bypasses IK)."""

from __future__ import annotations

from dataclasses import dataclass

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    JOINT_LIMITS_DEG,
    JOINT_NAMES_WITH_GRIPPER,
)
from xbox_soarm_teleop.diagnostics.xbox_joint_drive import (
    advance_goal,
    dpad_edge,
    map_trigger_to_gripper_deg,
)
from xbox_soarm_teleop.teleoperators.xbox import XboxState


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

    Control mapping:
        LB (hold):          Enable joint motion (deadman).
        Left stick X:       Selected-joint velocity command.
        D-pad left/right:   Select previous/next joint.
        A button (press):   Reset all joints to home position.
        Right trigger:      Gripper position (0=open, 1=closed).

    Args:
        max_vel_deg_s: Max velocity for the selected joint at full stick deflection.
        dt: Control loop period in seconds. Must match the actual loop rate.
    """

    def __init__(
        self,
        max_vel_deg_s: float = 70.0,
        dt: float = 1.0 / 30.0,
    ) -> None:
        self.max_vel_deg_s = max_vel_deg_s
        self.dt = dt
        self._selected_idx: int = 0
        self._goals_deg: dict[str, float] = {
            name: HOME_POSITION_DEG[name] for name in JOINT_NAMES_WITH_GRIPPER
        }
        self._prev_dpad_x: float = 0.0

    @property
    def selected_joint(self) -> str:
        """Name of the joint currently driven by the left stick."""
        return JOINT_NAMES_WITH_GRIPPER[self._selected_idx]

    def __call__(self, state: XboxState) -> JointCommand:
        """Compute joint position targets from controller state.

        Args:
            state: Current Xbox controller state.

        Returns:
            JointCommand with updated goal positions and metadata.
        """
        # D-pad edge: select joint
        edge = dpad_edge(state.dpad_x, self._prev_dpad_x)
        self._prev_dpad_x = state.dpad_x
        if edge != 0:
            n = len(JOINT_NAMES_WITH_GRIPPER)
            self._selected_idx = (self._selected_idx + edge) % n

        # A button press: go home
        if state.a_button_pressed:
            self._goals_deg = {name: HOME_POSITION_DEG[name] for name in JOINT_NAMES_WITH_GRIPPER}

        # Gripper: always active via trigger (position control)
        g_lower, g_upper = JOINT_LIMITS_DEG["gripper"]
        self._goals_deg["gripper"] = map_trigger_to_gripper_deg(
            state.right_trigger, g_lower, g_upper
        )

        # Deadman-gated joint velocity on left stick X
        cmd_vel = 0.0
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
