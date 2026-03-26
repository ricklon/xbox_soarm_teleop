"""Configuration for Nintendo Switch Right Joy-Con input (single horizontal mode)."""

from dataclasses import dataclass, field


@dataclass
class JoyConConfig:
    """Configuration for Right Joy-Con teleoperator (horizontal single-controller mode).

    Button layout (Joy-Con R held sideways, SL on left, SR on right):
        Stick          — ABS_RX / ABS_RY
        A (right face) — BTN_EAST
        B (bottom)     — BTN_SOUTH
        X (top)        — BTN_NORTH
        Y (left face)  — BTN_WEST
        SL             — BTN_TL      (deadman switch)
        SR             — BTN_TR
        ZR             — BTN_TR2     (gripper trigger)
        +              — BTN_START   (home / reset)
        Home           — BTN_MODE
        Stick click    — BTN_THUMBR

    Axis convention (hid-nintendo driver, horizontal mode):
        Stick right → ABS_RX positive
        Stick down  → ABS_RY positive
    """

    deadzone: float = 0.1
    linear_scale: float = 0.1   # m/s at full stick
    angular_scale: float = 2.0  # rad/s at full stick
    orientation_scale: float = 0.5  # rad/s for orientation axes
    gripper_rate: float = 2.0

    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    invert_roll: bool = False
    invert_pitch: bool = False
    invert_yaw: bool = False

    device_index: int = 0

    # Button mappings
    deadman_button: str = "BTN_TL"       # SL — hold to enable motion
    home_button: str = "BTN_START"       # + button — return to home
    frame_toggle_button: str = "BTN_WEST"  # Y face button — toggle frame

    # Puppet mode height control buttons (used to synthesise dpad_y)
    height_up_button: str = "BTN_TR"     # SR — raise end effector
    height_down_button: str = "BTN_SOUTH"  # B face button — lower end effector

    # Axis mappings (Right Joy-Con has only right-stick axes)
    left_stick_x_axis: str = "ABS_RX"
    left_stick_y_axis: str = "ABS_RY"
    right_stick_x_axis: str = "ABS_RX"   # same stick — single Joy-Con
    right_stick_y_axis: str = "ABS_RY"
    right_trigger_axis: str = ""          # ZR is a button (BTN_TR2), not an axis
    dpad_x_axis: str = ""                 # no d-pad on single Joy-Con
    dpad_y_axis: str = ""

    # hid-nintendo axis range
    stick_range: tuple[int, int] = field(default_factory=lambda: (-32767, 32767))
    trigger_range: tuple[int, int] = field(default_factory=lambda: (0, 1))

    # ZR button code (used as digital gripper trigger)
    zr_button: str = "BTN_TR2"

    # Device discovery — joycond virtual device name patterns
    device_name_patterns: list = field(
        default_factory=lambda: ["Nintendo Switch Right Joy-Con", "Joy-Con (R)"]
    )
