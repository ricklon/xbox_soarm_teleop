"""Configuration for Xbox controller input."""

from dataclasses import dataclass, field


@dataclass
class XboxConfig:
    """Configuration for Xbox controller teleoperator.

    Attributes:
        deadzone: Threshold below which analog inputs are treated as zero.
        linear_scale: Maximum linear velocity in m/s at full stick deflection.
        angular_scale: Maximum angular velocity in rad/s at full stick deflection.
        orientation_scale: Maximum angular velocity in rad/s for pitch/yaw (D-pad).
        gripper_rate: Gripper position change per second (0-1 range).
        invert_x: Invert left stick X axis.
        invert_y: Invert left stick Y axis.
        invert_z: Invert right stick Y axis (vertical).
        invert_roll: Invert right stick X axis (roll).
        invert_pitch: Invert D-pad Y axis (pitch).
        invert_yaw: Invert D-pad X axis (yaw).
        device_index: Index of the gamepad device to use.
    """

    deadzone: float = 0.1
    dominant_axis_min: float = 0.55  # minimum stick magnitude before cross-axis suppression
    dominant_axis_ratio: float = 2.5  # dominant axis must exceed minor axis by this ratio
    cross_axis_attenuation: float = 0.2  # scale factor applied to minor axis when suppressed
    linear_scale: float = 0.1  # m/s at full stick
    angular_scale: float = 2.0  # rad/s at full stick (wrist rotation)
    orientation_scale: float = 0.5  # rad/s for pitch/yaw (D-pad) - slower for digital input
    gripper_rate: float = 2.0  # gripper position change per second (0-1 range)
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    invert_roll: bool = False
    invert_pitch: bool = False
    invert_yaw: bool = False
    device_index: int = 0

    # Button mappings (can be overridden for different controller types)
    deadman_button: str = "BTN_TL"  # Left bumper
    home_button: str = "BTN_SOUTH"  # A button
    aux_button: str = "BTN_WEST"  # Y button on Linux Xbox layout, reserved for script-specific actions

    # Axis mappings
    left_stick_x_axis: str = "ABS_X"
    left_stick_y_axis: str = "ABS_Y"
    right_stick_x_axis: str = "ABS_RX"
    right_stick_y_axis: str = "ABS_RY"
    right_trigger_axis: str = "ABS_RZ"  # Some controllers use ABS_Z
    dpad_x_axis: str = "ABS_HAT0X"  # D-pad left/right (-1, 0, 1)
    dpad_y_axis: str = "ABS_HAT0Y"  # D-pad up/down (-1, 0, 1)

    # Axis ranges for normalization (typical Xbox controller values)
    stick_range: tuple[int, int] = field(default_factory=lambda: (-32768, 32767))
    trigger_range: tuple[int, int] = field(default_factory=lambda: (0, 255))
