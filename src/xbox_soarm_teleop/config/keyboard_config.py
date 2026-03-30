"""Configuration for keyboard teleoperator."""

from dataclasses import dataclass, field


@dataclass
class KeyboardConfig:
    """Configuration for keyboard teleoperator.

    Key names match evdev KEY_ constants (e.g. "KEY_W", "KEY_UP").

    Speed levels (1–5 keys):
        1 = 25%   2 = 50%   3 = 75% (default)   4 = 100%   5 = 150%
    Hold Shift to double current speed.

    Control layout:
        W / S       → forward / back     (X axis, right stick Y)
        A / D       → left / right       (Y axis, left stick X)
        R / F       → up / down          (Z axis, left stick Y)
        Q / E       → wrist roll         (right stick X)
        ↑ / ↓       → pitch              (dpad Y)
        ← / →       → yaw / joint cycle  (dpad X)
        Space       → gripper            (right trigger, hold=closed)
        H           → home position      (a_button_pressed edge)
        Y           → auxiliary action   (y_button_pressed edge)
        1–5         → speed level
        Shift       → 2× speed
        Ctrl+C      → exit
    """

    # Speed levels — fraction of full stick deflection
    speed_levels: tuple = field(default_factory=lambda: (0.25, 0.50, 0.75, 1.00, 1.50))
    default_speed_level: int = 2  # 0-indexed → level 3 = 0.75

    # Shift multiplier applied on top of current speed level
    shift_multiplier: float = 2.0

    # Movement keys
    key_forward: str = "KEY_W"
    key_back: str = "KEY_S"
    key_left: str = "KEY_A"
    key_right: str = "KEY_D"
    key_up: str = "KEY_R"
    key_down: str = "KEY_F"
    key_roll_left: str = "KEY_Q"
    key_roll_right: str = "KEY_E"

    # Orientation keys (arrow cluster)
    key_pitch_up: str = "KEY_UP"
    key_pitch_down: str = "KEY_DOWN"
    key_yaw_left: str = "KEY_LEFT"
    key_yaw_right: str = "KEY_RIGHT"

    # Function keys
    key_gripper: str = "KEY_SPACE"
    key_home: str = "KEY_H"
    key_aux: str = "KEY_Y"

    # Shift keys (either triggers the multiplier)
    key_shift_left: str = "KEY_LEFTSHIFT"
    key_shift_right: str = "KEY_RIGHTSHIFT"

    # Speed level keys
    key_speed_1: str = "KEY_1"
    key_speed_2: str = "KEY_2"
    key_speed_3: str = "KEY_3"
    key_speed_4: str = "KEY_4"
    key_speed_5: str = "KEY_5"

    # Device selection — None = auto-detect first keyboard with letter keys
    device_path: str | None = None

    # Exclusive device access — when True, device.grab() is called on connect so
    # keypresses are not forwarded to other applications (e.g. the X11 desktop).
    # Prevents accidental arm movement when the terminal loses focus.
    # Requires no other process to have the device open.
    grab: bool = False

    # Record/playback ─────────────────────────────────────────────────────────
    # key_record_toggle: press to start/stop recording during a live session.
    key_record_toggle: str = "KEY_TAB"

    # Where to save recordings. None = auto-name as recording_<timestamp>.json
    # in the current working directory.
    record_path: str | None = None

    # Path to a previously saved recording to play back instead of live input.
    # When set, no physical keyboard device is opened.
    playback_path: str | None = None
