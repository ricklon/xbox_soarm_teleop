"""Processor for mapping Xbox controller input to end effector deltas."""

from dataclasses import dataclass

from xbox_soarm_teleop.teleoperators.xbox import XboxState


@dataclass
class EEDelta:
    """End effector delta command.

    Attributes:
        dx: Linear velocity in X direction (m/s) - forward/back in arm plane.
        dy: Linear velocity in Y direction (m/s) - left/right.
        dz: Linear velocity in Z direction (m/s) - up/down.
        droll: Angular velocity around roll axis (rad/s) - wrist roll.
        dpitch: Angular velocity around pitch axis (rad/s) - tilt up/down.
        dyaw: Angular velocity around yaw axis (rad/s) - rotate heading.
        gripper: Gripper position (0=open, 1=closed).
    """

    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0
    dpitch: float = 0.0
    dyaw: float = 0.0
    gripper: float = 0.0

    def as_array(self) -> list[float]:
        """Return delta as a list [dx, dy, dz, droll, dpitch, dyaw, gripper]."""
        return [self.dx, self.dy, self.dz, self.droll, self.dpitch, self.dyaw, self.gripper]

    def is_zero_motion(self) -> bool:
        """Check if all motion commands are zero."""
        return (
            self.dx == 0.0
            and self.dy == 0.0
            and self.dz == 0.0
            and self.droll == 0.0
            and self.dpitch == 0.0
            and self.dyaw == 0.0
        )


class MapXboxToEEDelta:
    """Maps Xbox controller state to end effector velocity commands.

    This processor transforms normalized controller inputs into
    velocity commands for the end effector and base rotation.

    The deadman switch (left bumper) must be held for any motion
    to occur. The gripper is position-controlled and always active.

    Args:
        linear_scale: Maximum linear velocity in m/s at full stick deflection.
        angular_scale: Maximum angular velocity in rad/s at full stick deflection.
        orientation_scale: Maximum angular velocity in rad/s for pitch/yaw (D-pad).
        invert_pitch: Invert pitch direction.
        invert_yaw: Invert yaw direction.
    """

    def __init__(
        self,
        linear_scale: float = 0.1,
        angular_scale: float = 0.5,
        orientation_scale: float = 1.0,
        wrist_roll_threshold: float = 0.25,  # Extra threshold for wrist roll to prevent drift
        invert_pitch: bool = False,
        invert_yaw: bool = False,
    ):
        self.linear_scale = linear_scale
        self.angular_scale = angular_scale
        self.orientation_scale = orientation_scale
        self.wrist_roll_threshold = wrist_roll_threshold
        self.invert_pitch = invert_pitch
        self.invert_yaw = invert_yaw

    def __call__(self, state: XboxState) -> EEDelta:
        """Map controller state to end effector delta.

        Args:
            state: Current Xbox controller state.

        Returns:
            End effector delta command.
        """
        # Gripper is always active (position control)
        gripper = state.right_trigger

        # Deadman switch check - no arm motion unless left bumper held
        if not state.left_bumper:
            return EEDelta(gripper=gripper)

        # Apply extra threshold for wrist roll to prevent drift when using right stick Y
        wrist_roll_input = state.right_stick_x
        if abs(wrist_roll_input) < self.wrist_roll_threshold:
            wrist_roll_input = 0.0

        # D-pad for pitch/yaw
        # D-pad Y: -1 = up, 1 = down (HAT convention)
        # We want up = pitch up (positive), so negate
        pitch_input = -state.dpad_y
        if self.invert_pitch:
            pitch_input = -pitch_input

        # D-pad X: -1 = left, 1 = right
        # We want right = yaw right (positive)
        yaw_input = state.dpad_x
        if self.invert_yaw:
            yaw_input = -yaw_input

        return EEDelta(
            dx=-state.right_stick_y * self.linear_scale,  # Forward/back (right stick Y)
            dy=-state.left_stick_x * self.linear_scale,  # Left/right (left stick X)
            dz=-state.left_stick_y * self.linear_scale,  # Up/down (left stick Y)
            droll=wrist_roll_input * self.angular_scale,  # Wrist roll (right stick X)
            dpitch=pitch_input * self.orientation_scale,  # Pitch (D-pad up/down)
            dyaw=yaw_input * self.orientation_scale,  # Yaw (D-pad left/right)
            gripper=gripper,
        )

    def reset(self) -> None:
        """Reset processor state (no-op for stateless processor)."""
        pass
