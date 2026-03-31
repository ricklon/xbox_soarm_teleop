"""Configuration for Nintendo Switch Joy-Con input devices."""

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
    aux_button: str = "BTN_WEST"  # Y face button — reserved for script-specific actions

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


@dataclass
class DualJoyConConfig:
    """Configuration for split dual Joy-Con teleoperation.

    The intended default posture is standard vertical use:
    - Left Joy-Con drives translation and clutch/deadman.
    - Right Joy-Con drives gripper plus natural wrist orientation from the IMU.
    """

    deadzone: float = 0.1
    linear_scale: float = 0.1
    vertical_scale: float = 0.08
    angular_scale: float = 0.5
    orientation_scale: float = 1.0
    roll_scale: float = 1.0
    pitch_scale: float = 1.0
    yaw_scale: float = 0.75
    gripper_rate: float = 2.0

    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    invert_roll: bool = False
    invert_pitch: bool = False
    invert_yaw: bool = False

    left_device_index: int = 0
    right_device_index: int = 0
    imu_device_index: int | None = None

    left_stick_x_axis: str = "ABS_X"
    left_stick_y_axis: str = "ABS_Y"
    right_stick_x_axis: str = "ABS_RX"
    right_stick_y_axis: str = "ABS_RY"

    deadman_button: str = "BTN_TL2"      # ZL
    home_button: str = "BTN_START"       # +
    aux_button: str = "BTN_MODE"         # Home
    gripper_button: str = "BTN_TR2"      # ZR
    z_up_button: str = "BTN_DPAD_UP"
    z_down_button: str = "BTN_DPAD_DOWN"
    dpad_left_button: str = "BTN_DPAD_LEFT"
    dpad_right_button: str = "BTN_DPAD_RIGHT"

    stick_range: tuple[int, int] = field(default_factory=lambda: (-32767, 32767))

    left_device_name_patterns: list[str] = field(
        default_factory=lambda: ["Nintendo Switch Left Joy-Con", "Joy-Con (L)"]
    )
    right_device_name_patterns: list[str] = field(
        default_factory=lambda: ["Nintendo Switch Right Joy-Con", "Joy-Con (R)"]
    )
    right_imu_name_patterns: list[str] = field(
        default_factory=lambda: ["Nintendo Switch Right Joy-Con IMU", "Joy-Con (R) IMU"]
    )
