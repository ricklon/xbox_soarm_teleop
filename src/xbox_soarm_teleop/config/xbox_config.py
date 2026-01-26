"""Configuration for Xbox controller input."""

from dataclasses import dataclass, field


@dataclass
class XboxConfig:
    """Configuration for Xbox controller teleoperator.

    Attributes:
        deadzone: Threshold below which analog inputs are treated as zero.
        linear_scale: Maximum linear velocity in m/s at full stick deflection.
        angular_scale: Maximum angular velocity in rad/s at full stick deflection.
        invert_x: Invert left stick X axis.
        invert_y: Invert left stick Y axis.
        invert_z: Invert right stick Y axis (vertical).
        invert_roll: Invert right stick X axis (roll).
        device_index: Index of the gamepad device to use.
    """

    deadzone: float = 0.1
    linear_scale: float = 0.1  # m/s at full stick
    angular_scale: float = 2.0  # rad/s at full stick (wrist rotation)
    gripper_rate: float = 2.0  # gripper position change per second (0-1 range)
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    invert_roll: bool = False
    device_index: int = 0

    # Button mappings (can be overridden for different controller types)
    deadman_button: str = "BTN_TL"  # Left bumper
    home_button: str = "BTN_SOUTH"  # A button
    frame_toggle_button: str = "BTN_NORTH"  # Y button

    # Axis mappings
    left_stick_x_axis: str = "ABS_X"
    left_stick_y_axis: str = "ABS_Y"
    right_stick_x_axis: str = "ABS_RX"
    right_stick_y_axis: str = "ABS_RY"
    right_trigger_axis: str = "ABS_RZ"  # Some controllers use ABS_Z

    # Axis ranges for normalization (typical Xbox controller values)
    stick_range: tuple[int, int] = field(default_factory=lambda: (-32768, 32767))
    trigger_range: tuple[int, int] = field(default_factory=lambda: (0, 255))
