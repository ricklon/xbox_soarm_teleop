"""Keyboard teleoperator for SO-ARM teleoperation.

Uses evdev for simultaneous key detection (no extra window, no root required
if user is in the 'input' group).  stdin is put into cbreak mode to suppress
key echo while the controller is active.

Requirements:
    evdev >= 1.6   (uv pip install evdev  or  uv pip install -e ".[joycon]")
    User in 'input' group: sudo usermod -aG input $USER  (then re-login)

Usage:
    ctrl = KeyboardController()
    if ctrl.connect():
        state = ctrl.read()   # returns XboxState — same interface as XboxController
    ctrl.disconnect()
"""

from __future__ import annotations

import sys
import termios
import threading
import tty
from typing import Any

from xbox_soarm_teleop.config.keyboard_config import KeyboardConfig
from xbox_soarm_teleop.teleoperators.xbox import XboxState


class KeyboardController:
    """Keyboard teleoperator that produces XboxState output.

    Drop-in replacement for XboxController.  No deadman switch — movement
    occurs whenever movement keys are held.  left_bumper is always True so
    all downstream processors work without modification.

    Simultaneous key detection is handled via evdev (kernel-level events),
    so diagonal movement (e.g. W+D) works correctly.

    Args:
        config: Keyboard configuration.  Defaults to KeyboardConfig().
    """

    def __init__(self, config: KeyboardConfig | None = None) -> None:
        self.config = config or KeyboardConfig()
        self._device: Any = None
        self._held_keys: set[int] = set()
        self._held_keys_lock = threading.Lock()
        self._connected = False
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._speed_level: int = self.config.default_speed_level
        self._key_map: dict[str, int] = {}   # config key name → evdev keycode

        # Edge-detection state (home / frame-toggle)
        self._prev_home: bool = False
        self._prev_frame: bool = False

        # Saved terminal attributes for restore on disconnect
        self._orig_term: list | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Find keyboard device, suppress echo, start event reader.

        Returns:
            True if a keyboard device was found and opened.

        Raises:
            ImportError: if evdev is not installed.
        """
        try:
            import evdev  # noqa: F401
        except ImportError:
            raise ImportError(
                "evdev not installed. Run: uv pip install evdev\n"
                "Note: evdev is Linux-only."
            )

        device = self._find_keyboard()
        if device is None:
            return False

        self._device = device
        self._build_key_map()
        self._enter_cbreak()
        self._connected = True
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._read_events_loop, daemon=True, name="keyboard-reader"
        )
        self._reader_thread.start()
        return True

    def disconnect(self) -> None:
        """Stop reader, restore terminal settings, close device."""
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        self._restore_terminal()
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        self._connected = False
        with self._held_keys_lock:
            self._held_keys.clear()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read(self) -> XboxState:
        """Return current keyboard state as XboxState.

        Non-blocking — key state is maintained by the background reader thread.

        Stick sign conventions match XboxController so all existing processors
        work correctly:
            right_stick_y negative  → arm moves forward (+X)
            left_stick_x  positive  → arm moves right   (+Y direction, post-negation in processor)
            left_stick_y  negative  → arm moves up       (+Z)
            right_stick_x positive  → wrist rolls right
        """
        if not self._connected:
            return XboxState()

        with self._held_keys_lock:
            held = frozenset(self._held_keys)

        cfg = self.config

        # Speed magnitude
        speed = cfg.speed_levels[self._speed_level]
        if self._is_held(held, cfg.key_shift_left) or self._is_held(held, cfg.key_shift_right):
            speed = min(speed * cfg.shift_multiplier, 2.0)

        # ── Stick axes (opposing keys cancel, result clamped to [-1, 1]) ──────

        # Forward/back → right_stick_y (W=forward → negative, matching Xbox up-stick convention)
        right_stick_y = _combine(
            pos=self._is_held(held, cfg.key_back),
            neg=self._is_held(held, cfg.key_forward),
            scale=speed,
        )

        # Left/right → left_stick_x (D=right → positive)
        left_stick_x = _combine(
            pos=self._is_held(held, cfg.key_right),
            neg=self._is_held(held, cfg.key_left),
            scale=speed,
        )

        # Up/down → left_stick_y (R=up → negative, matching Xbox up-stick convention)
        left_stick_y = _combine(
            pos=self._is_held(held, cfg.key_down),
            neg=self._is_held(held, cfg.key_up),
            scale=speed,
        )

        # Wrist roll → right_stick_x (E=roll right → positive)
        right_stick_x = _combine(
            pos=self._is_held(held, cfg.key_roll_right),
            neg=self._is_held(held, cfg.key_roll_left),
            scale=speed,
        )

        # ── D-pad: pitch / yaw / joint selection (discrete ±1) ───────────────

        dpad_y = 0.0
        if self._is_held(held, cfg.key_pitch_up):
            dpad_y = -1.0   # HAT convention: up = -1
        elif self._is_held(held, cfg.key_pitch_down):
            dpad_y = 1.0

        dpad_x = 0.0
        if self._is_held(held, cfg.key_yaw_right):
            dpad_x = 1.0
        elif self._is_held(held, cfg.key_yaw_left):
            dpad_x = -1.0

        # ── Gripper (Space held = closed = 1.0) ───────────────────────────────
        right_trigger = 1.0 if self._is_held(held, cfg.key_gripper) else 0.0

        # ── Buttons (edge detection) ──────────────────────────────────────────
        home_held = self._is_held(held, cfg.key_home)
        frame_held = self._is_held(held, cfg.key_frame_toggle)
        a_pressed = home_held and not self._prev_home
        y_pressed = frame_held and not self._prev_frame
        self._prev_home = home_held
        self._prev_frame = frame_held

        return XboxState(
            left_stick_x=left_stick_x,
            left_stick_y=left_stick_y,
            right_stick_x=right_stick_x,
            right_stick_y=right_stick_y,
            right_trigger=right_trigger,
            left_bumper=True,         # no deadman switch for keyboard
            a_button=home_held,
            y_button=frame_held,
            dpad_x=dpad_x,
            dpad_y=dpad_y,
            a_button_pressed=a_pressed,
            y_button_pressed=y_pressed,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_keyboard(self) -> Any:
        """Return the first evdev device that has letter keys (a keyboard)."""
        import evdev

        if self.config.device_path:
            try:
                return evdev.InputDevice(self.config.device_path)
            except (OSError, PermissionError) as exc:
                raise RuntimeError(
                    f"Cannot open keyboard device {self.config.device_path}: {exc}"
                ) from exc

        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                caps = dev.capabilities()
                if evdev.ecodes.EV_KEY not in caps:
                    dev.close()
                    continue
                # Require letter keys to exclude mice, power buttons, HID media remotes
                if evdev.ecodes.KEY_W not in caps[evdev.ecodes.EV_KEY]:
                    dev.close()
                    continue
                return dev
            except (OSError, PermissionError):
                continue
        return None

    def _build_key_map(self) -> None:
        """Populate _key_map: config key name → evdev keycode integer."""
        try:
            import evdev

            for attr, name in vars(self.config).items():
                if attr.startswith("key_") and isinstance(name, str) and name.startswith("KEY_"):
                    code = getattr(evdev.ecodes, name, None)
                    if code is not None:
                        self._key_map[name] = code
        except ImportError:
            pass

    def _is_held(self, held: frozenset, key_name: str) -> bool:
        """Return True if the key named key_name is currently held."""
        code = self._key_map.get(key_name)
        return code is not None and code in held

    def _enter_cbreak(self) -> None:
        """Put stdin into cbreak mode: no echo, no line-buffering, signals intact."""
        try:
            fd = sys.stdin.fileno()
            self._orig_term = termios.tcgetattr(fd)
            tty.cbreak(fd)
        except (termios.error, AttributeError, OSError):
            self._orig_term = None

    def _restore_terminal(self) -> None:
        """Restore terminal to its state before connect()."""
        if self._orig_term is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._orig_term)
            except (termios.error, AttributeError, OSError):
                pass
            self._orig_term = None

    def _read_events_loop(self) -> None:
        """Background thread: read evdev key events, maintain _held_keys set."""
        import select

        import evdev

        fd = self._device.fd

        # Map evdev keycodes for speed-level keys
        speed_keys: dict[int, int] = {}
        for i, attr in enumerate(
            ["key_speed_1", "key_speed_2", "key_speed_3", "key_speed_4", "key_speed_5"]
        ):
            name = getattr(self.config, attr, None)
            if name:
                code = getattr(evdev.ecodes, name, None)
                if code is not None:
                    speed_keys[code] = i

        while not self._stop_event.is_set():
            try:
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    continue
                for event in self._device.read():
                    if event.type != evdev.ecodes.EV_KEY:
                        continue
                    if event.value == 1:     # key down
                        with self._held_keys_lock:
                            self._held_keys.add(event.code)
                        if event.code in speed_keys:
                            self._speed_level = speed_keys[event.code]
                    elif event.value == 0:   # key up
                        with self._held_keys_lock:
                            self._held_keys.discard(event.code)
                    # value == 2 is key-repeat — ignored (state tracked via held set)
            except OSError:
                self._connected = False
                break
            except Exception:
                pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _combine(pos: bool, neg: bool, scale: float) -> float:
    """Return ±scale or 0.0 from two bool directions, clamped to [-1, 1]."""
    value = (1.0 if pos else 0.0) - (1.0 if neg else 0.0)
    return max(-1.0, min(1.0, value * scale))
