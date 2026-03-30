"""Nintendo Switch Right Joy-Con teleoperator for SO-ARM teleoperation.

Requires the hid-nintendo kernel module and joycond daemon.
See scripts/setup_joycon.sh for setup instructions.
"""

import math
import threading
from typing import Any

from xbox_soarm_teleop.config.joycon_config import JoyConConfig
from xbox_soarm_teleop.teleoperators.xbox import XboxState


def _ecodes_name(mapping: dict, code: int) -> str:
    """Return a stable string name for an evdev event code."""
    result = mapping.get(code, str(code))
    return result[0] if isinstance(result, list) else result


class JoyConController:
    """Right Joy-Con teleoperator (horizontal single-controller mode).

    Reads from the joycond virtual uinput device via evdev.  joycond must be
    running and the Joy-Con must have been activated in single-controller mode
    (press SL+SR after connecting).

    Produces XboxState output so it is a drop-in replacement for XboxController
    in any processor or control loop.

    Setup:
        sudo bash scripts/setup_joycon.sh
        bluetoothctl connect <MAC>
        # Press SL+SR on Joy-Con when LEDs alternate

    Args:
        config: Joy-Con configuration. Defaults to JoyConConfig().
    """

    def __init__(self, config: JoyConConfig | None = None):
        self.config = config or JoyConConfig()
        self._state = XboxState()
        self._prev_state = XboxState()
        self._device: Any = None
        self._raw_state: dict[str, int] = {}
        self._raw_state_lock = threading.Lock()
        self._connected = False
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── Public API ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to the Joy-Con via joycond's virtual device.

        Returns:
            True if a matching device was found and opened.
        """
        try:
            import evdev  # noqa: F401
        except ImportError:
            raise ImportError(
                "evdev not installed. Run: uv pip install evdev\n"
                "Note: evdev is Linux-only."
            )

        device = self._find_device()
        if device is None:
            return False

        self._device = device
        self._connected = True
        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._read_events_loop, daemon=True)
        self._reader_thread.start()
        return True

    def disconnect(self) -> None:
        """Disconnect from the Joy-Con device."""
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        self._connected = False
        with self._raw_state_lock:
            self._raw_state.clear()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read(self) -> XboxState:
        """Return current normalized Joy-Con state.

        Non-blocking — events are consumed by a background thread.

        Returns:
            XboxState with Joy-Con inputs mapped to Xbox-compatible fields:
              left_stick_x/y  — physical stick (ABS_RX/RY)
              right_stick_x/y — same physical stick (single Joy-Con)
              right_trigger   — ZR button (0.0 or 1.0)
              left_bumper     — SL button (deadman switch)
              a_button        — A face button (BTN_EAST)
              y_button        — Y face button (BTN_WEST), reserved as an auxiliary button
              dpad_x          — always 0.0 (no d-pad on single Joy-Con)
              dpad_y          — ±1.0 from height_up/height_down buttons (puppet mode)
        """
        if not self._connected:
            return XboxState()

        # Save previous for edge detection
        self._prev_state = XboxState(
            left_stick_x=self._state.left_stick_x,
            left_stick_y=self._state.left_stick_y,
            right_stick_x=self._state.right_stick_x,
            right_stick_y=self._state.right_stick_y,
            right_trigger=self._state.right_trigger,
            left_bumper=self._state.left_bumper,
            a_button=self._state.a_button,
            y_button=self._state.y_button,
            dpad_x=self._state.dpad_x,
            dpad_y=self._state.dpad_y,
        )

        with self._raw_state_lock:
            raw = self._raw_state.copy()

        # Stick (single physical stick → both left and right fields)
        raw_x = self._normalize_stick_raw(
            raw.get(self.config.left_stick_x_axis, 0),
            invert=self.config.invert_x,
        )
        raw_y = self._normalize_stick_raw(
            raw.get(self.config.left_stick_y_axis, 0),
            invert=self.config.invert_y,
        )
        sx, sy = self._apply_radial_deadzone(raw_x, raw_y)
        self._state.left_stick_x = sx
        self._state.left_stick_y = sy
        self._state.right_stick_x = sx
        self._state.right_stick_y = sy

        # ZR is a digital button, not an analog axis
        self._state.right_trigger = 1.0 if raw.get(self.config.zr_button, 0) else 0.0

        # Buttons
        self._state.left_bumper = bool(raw.get(self.config.deadman_button, 0))
        self._state.a_button = bool(raw.get(self.config.home_button, 0))
        self._state.y_button = bool(raw.get(self.config.aux_button, 0))

        # No hardware d-pad on single Joy-Con; synthesise dpad_y from height buttons
        # (used in puppet mode: SR=up → +1.0, B=down → -1.0)
        self._state.dpad_x = 0.0
        height_up = bool(raw.get(self.config.height_up_button, 0))
        height_down = bool(raw.get(self.config.height_down_button, 0))
        if height_up and not height_down:
            self._state.dpad_y = 1.0
        elif height_down and not height_up:
            self._state.dpad_y = -1.0
        else:
            self._state.dpad_y = 0.0

        # Edge detection
        self._state.a_button_pressed = self._state.a_button and not self._prev_state.a_button
        self._state.y_button_pressed = self._state.y_button and not self._prev_state.y_button

        return self._state

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_device(self) -> Any:
        """Scan evdev devices and return the first matching Joy-Con device."""
        import evdev

        candidates = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                if any(pat in dev.name for pat in self.config.device_name_patterns):
                    candidates.append(dev)
                else:
                    dev.close()
            except (OSError, PermissionError):
                continue

        if not candidates:
            return None

        idx = self.config.device_index
        chosen = candidates[idx] if idx < len(candidates) else candidates[0]
        for i, c in enumerate(candidates):
            if c is not chosen:
                c.close()
        return chosen

    def _read_events_loop(self) -> None:
        """Background thread: read evdev events and update _raw_state."""
        import select

        import evdev

        fd = self._device.fd
        while not self._stop_event.is_set():
            try:
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    continue
                for event in self._device.read():
                    if event.type == evdev.ecodes.EV_ABS:
                        name = _ecodes_name(evdev.ecodes.ABS, event.code)
                        with self._raw_state_lock:
                            self._raw_state[name] = event.value
                    elif event.type == evdev.ecodes.EV_KEY:
                        name = _ecodes_name(evdev.ecodes.BTN, event.code)
                        with self._raw_state_lock:
                            self._raw_state[name] = event.value
            except OSError:
                self._connected = False
                break
            except Exception:
                pass

    def _normalize_stick_raw(self, value: int, invert: bool = False) -> float:
        """Normalize raw stick value to [-1, 1]."""
        min_val, max_val = self.config.stick_range
        center = (min_val + max_val) / 2
        half_range = (max_val - min_val) / 2
        normalized = (value - center) / half_range
        if invert:
            normalized = -normalized
        return max(-1.0, min(1.0, normalized))

    def _apply_radial_deadzone(self, x: float, y: float) -> tuple[float, float]:
        """Apply circular deadzone and rescale to full range."""
        magnitude = math.sqrt(x * x + y * y)
        if magnitude < self.config.deadzone:
            return 0.0, 0.0
        scale = min((magnitude - self.config.deadzone) / (1.0 - self.config.deadzone), 1.0)
        return (x / magnitude) * scale, (y / magnitude) * scale
