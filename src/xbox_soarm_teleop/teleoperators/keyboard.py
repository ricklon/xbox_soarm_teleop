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

Record/playback:
    # Live session — press Tab to start/stop recording:
    ctrl = KeyboardController(KeyboardConfig(record_path="demo.json"))

    # Playback a saved recording (no physical keyboard needed):
    ctrl = KeyboardController(KeyboardConfig(playback_path="demo.json"))
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from xbox_soarm_teleop.config.keyboard_config import KeyboardConfig
from xbox_soarm_teleop.teleoperators.xbox import XboxState

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows import path
    termios = None
    tty = None


_FALLBACK_ECODES = {
    "KEY_1": 2,
    "KEY_2": 3,
    "KEY_3": 4,
    "KEY_4": 5,
    "KEY_5": 6,
    "KEY_TAB": 15,
    "KEY_Q": 16,
    "KEY_W": 17,
    "KEY_E": 18,
    "KEY_R": 19,
    "KEY_Y": 21,
    "KEY_A": 30,
    "KEY_S": 31,
    "KEY_D": 32,
    "KEY_F": 33,
    "KEY_H": 35,
    "KEY_LEFTSHIFT": 42,
    "KEY_SPACE": 57,
    "KEY_RIGHTSHIFT": 54,
    "KEY_UP": 103,
    "KEY_LEFT": 105,
    "KEY_RIGHT": 106,
    "KEY_DOWN": 108,
}

_PYNPUT_KEY_TOKENS = {
    "KEY_SPACE": "special:space",
    "KEY_TAB": "special:tab",
    "KEY_UP": "special:up",
    "KEY_DOWN": "special:down",
    "KEY_LEFT": "special:left",
    "KEY_RIGHT": "special:right",
    "KEY_LEFTSHIFT": "special:shift_l",
    "KEY_RIGHTSHIFT": "special:shift_r",
}


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
        self._held_keys: set[Any] = set()
        self._held_keys_lock = threading.Lock()
        self._connected = False
        self._reader_thread: threading.Thread | None = None
        self._listener: Any = None
        self._stop_event = threading.Event()
        self._speed_level: int = self.config.default_speed_level
        self._key_map: dict[str, Any] = {}   # config key name → backend token
        self._playback_code_style: str = "evdev"

        self._grabbed: bool = False

        # Edge-detection state (home / auxiliary Y button)
        self._prev_home: bool = False
        self._prev_aux: bool = False

        # Saved terminal attributes for restore on disconnect
        self._orig_term: list | None = None

        # Recording state (all accessed only from the reader thread — no lock needed)
        self._recording: bool = False
        self._record_start: float = 0.0
        self._record_events: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Find keyboard device (or load playback file), start event reader.

        In playback mode (``config.playback_path`` set) no physical device is
        opened; synthetic events are fed from the recording file instead.

        Returns:
            True if the controller connected (or playback file loaded) successfully.

        Raises:
            ImportError: if evdev is not installed (live mode only).
        """
        if self.config.playback_path:
            return self._connect_playback()

        backend = self._resolve_backend()
        if backend == "pynput":
            return self._connect_pynput()
        if backend != "evdev":
            raise RuntimeError(f"Keyboard backend '{backend}' is not implemented yet.")

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
        if self.config.grab:
            try:
                device.grab()
                self._grabbed = True
            except OSError as exc:
                device.close()
                raise RuntimeError(
                    f"Could not grab keyboard device (another process may have it open): {exc}"
                ) from exc
        self._enter_cbreak()
        self._connected = True
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._read_events_loop, daemon=True, name="keyboard-reader"
        )
        self._reader_thread.start()
        return True

    def _resolve_backend(self) -> str:
        """Resolve the live keyboard backend for the current platform."""
        if self.config.playback_path:
            return "playback"
        if self.config.backend != "auto":
            return self.config.backend
        if sys.platform == "linux":
            return "evdev"
        return "pynput"

    def _connect_pynput(self) -> bool:
        """Connect using the desktop pynput listener backend."""
        try:
            from pynput import keyboard as pynput_keyboard
        except ImportError as exc:
            raise ImportError(
                "pynput not installed. Run: uv sync --extra keyboard\n"
                "macOS and Windows keyboard mode use pynput."
            ) from exc

        if self.config.grab:
            print(
                "WARNING: --keyboard-grab is only supported by the Linux evdev backend.",
                flush=True,
            )

        self._build_key_map()
        self._connected = True
        self._stop_event.clear()
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_pynput_press,
            on_release=self._on_pynput_release,
        )
        self._listener.start()

        if not getattr(pynput_keyboard.Listener, "IS_TRUSTED", True):
            self.disconnect()
            raise RuntimeError(
                "Keyboard monitoring is not trusted on this platform. "
                "On macOS, grant Accessibility/Input Monitoring to your terminal or IDE."
            )
        return True

    def _connect_playback(self) -> bool:
        """Load a recording file and start the playback thread."""
        path = Path(self.config.playback_path)  # type: ignore[arg-type]
        if not path.exists():
            print(f"ERROR: playback file not found: {path}", flush=True)
            return False
        try:
            with path.open() as f:
                events = json.load(f)
        except Exception as exc:
            print(f"ERROR: could not load playback file {path}: {exc}", flush=True)
            return False

        if events and isinstance(events[0].get("code"), str):
            self._playback_code_style = "token"
        else:
            self._playback_code_style = "evdev"

        # Build key map from config (needed so read() can resolve key names)
        self._build_key_map()
        self._connected = True
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._playback_loop,
            args=(events,),
            daemon=True,
            name="keyboard-playback",
        )
        self._reader_thread.start()
        print(f"Playback: loaded {len(events)} events from {path}", flush=True)
        return True

    def disconnect(self) -> None:
        """Stop reader, restore terminal settings, close device."""
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        self._restore_terminal()
        if self._device is not None:
            if getattr(self, "_grabbed", False):
                try:
                    self._device.ungrab()
                except Exception:
                    pass
                self._grabbed = False
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
            left_stick_y  negative  → arm moves forward (+X)
            left_stick_x  positive  → arm moves right   (+Y direction, post-negation in processor)
            right_stick_y negative  → arm moves up       (+Z)
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

        # Forward/back → left_stick_y (W=forward → negative, matching Xbox up-stick convention)
        left_stick_y = _combine(
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

        # Up/down → right_stick_y (R=up → negative, matching Xbox up-stick convention)
        right_stick_y = _combine(
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
        aux_held = self._is_held(held, cfg.key_aux)
        a_pressed = home_held and not self._prev_home
        y_pressed = aux_held and not self._prev_aux
        self._prev_home = home_held
        self._prev_aux = aux_held

        return XboxState(
            left_stick_x=left_stick_x,
            left_stick_y=left_stick_y,
            right_stick_x=right_stick_x,
            right_stick_y=right_stick_y,
            right_trigger=right_trigger,
            left_bumper=True,         # no deadman switch for keyboard
            a_button=home_held,
            y_button=aux_held,
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
        """Populate _key_map with backend-specific tokens."""
        backend = self._resolve_backend()
        if backend == "pynput" or (backend == "playback" and self._playback_code_style == "token"):
            self._key_map = {
                name: self._pynput_token_for_name(name)
                for attr, name in vars(self.config).items()
                if attr.startswith("key_") and isinstance(name, str)
            }
            return

        self._key_map = {}
        try:
            import evdev

            for attr, name in vars(self.config).items():
                if attr.startswith("key_") and isinstance(name, str) and name.startswith("KEY_"):
                    code = getattr(evdev.ecodes, name, None)
                    if code is not None:
                        self._key_map[name] = code
        except ImportError:
            for attr, name in vars(self.config).items():
                if attr.startswith("key_") and isinstance(name, str) and name.startswith("KEY_"):
                    code = _FALLBACK_ECODES.get(name)
                    if code is not None:
                        self._key_map[name] = code

    def _is_held(self, held: frozenset, key_name: str) -> bool:
        """Return True if the key named key_name is currently held."""
        code = self._key_map.get(key_name)
        return code is not None and code in held

    def _pynput_token_for_name(self, key_name: str) -> str:
        """Return a canonical token for a configured key name."""
        if key_name in _PYNPUT_KEY_TOKENS:
            return _PYNPUT_KEY_TOKENS[key_name]
        if key_name.startswith("KEY_") and len(key_name) == 5:
            return f"char:{key_name[-1].lower()}"
        raise KeyError(f"Unsupported pynput key mapping: {key_name}")

    def _pynput_token_from_event(self, key: Any) -> str | None:
        """Translate a pynput key event into a canonical token."""
        char = getattr(key, "char", None)
        if isinstance(char, str) and char:
            return f"char:{char.lower()}"

        text = str(key)
        special = {
            "Key.space": "special:space",
            "Key.tab": "special:tab",
            "Key.up": "special:up",
            "Key.down": "special:down",
            "Key.left": "special:left",
            "Key.right": "special:right",
            "Key.shift": "special:shift_l",
            "Key.shift_l": "special:shift_l",
            "Key.shift_r": "special:shift_r",
        }
        return special.get(text)

    def _enter_cbreak(self) -> None:
        """Put stdin into cbreak mode: no echo, no line-buffering, signals intact."""
        if termios is None or tty is None:
            self._orig_term = None
            return
        try:
            fd = sys.stdin.fileno()
            self._orig_term = termios.tcgetattr(fd)
            tty.cbreak(fd)
        except (termios.error, AttributeError, OSError):
            self._orig_term = None

    def _restore_terminal(self) -> None:
        """Restore terminal to its state before connect()."""
        if self._orig_term is not None and termios is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._orig_term)
            except (termios.error, AttributeError, OSError):
                pass
            self._orig_term = None

    def _speed_key_tokens(self) -> dict[Any, int]:
        """Return the configured speed-level tokens for the active backend."""
        speed_keys: dict[Any, int] = {}
        for i, attr in enumerate(
            ["key_speed_1", "key_speed_2", "key_speed_3", "key_speed_4", "key_speed_5"]
        ):
            name = getattr(self.config, attr, None)
            token = self._key_map.get(name)
            if token is not None:
                speed_keys[token] = i
        return speed_keys

    def _handle_key_event(self, token: Any, value: int) -> None:
        """Update held state, speed selection, and recording for one logical key event."""
        speed_keys = self._speed_key_tokens()
        record_toggle = self._key_map.get(self.config.key_record_toggle)

        if value == 1:
            with self._held_keys_lock:
                self._held_keys.add(token)
            if token in speed_keys:
                self._speed_level = speed_keys[token]
            if token == record_toggle:
                self._toggle_recording()
        elif value == 0:
            with self._held_keys_lock:
                self._held_keys.discard(token)
        else:
            return

        if self._recording:
            self._record_events.append(
                {
                    "t": time.monotonic() - self._record_start,
                    "code": token,
                    "value": value,
                }
            )

    def _read_events_loop(self) -> None:
        """Background thread: read evdev key events, maintain _held_keys set."""
        import select

        import evdev

        fd = self._device.fd

        while not self._stop_event.is_set():
            try:
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    continue
                for event in self._device.read():
                    if event.type != evdev.ecodes.EV_KEY:
                        continue
                    self._handle_key_event(event.code, event.value)
            except OSError:
                self._connected = False
                break
            except Exception:
                pass

    def _on_pynput_press(self, key: Any) -> None:
        """Handle a pynput key-down event."""
        token = self._pynput_token_from_event(key)
        if token is not None:
            self._handle_key_event(token, 1)

    def _on_pynput_release(self, key: Any) -> None:
        """Handle a pynput key-up event."""
        token = self._pynput_token_from_event(key)
        if token is not None:
            self._handle_key_event(token, 0)

    def _toggle_recording(self) -> None:
        """Start or stop recording; auto-save on stop."""
        if not self._recording:
            self._recording = True
            self._record_start = time.monotonic()
            self._record_events = []
            print("\n[REC] Recording started — press Tab to stop.", flush=True)
        else:
            self._recording = False
            path = self._save_recording()
            print(
                f"\n[REC] Recording stopped — {len(self._record_events)} events saved to {path}",
                flush=True,
            )

    def _save_recording(self) -> Path:
        """Write _record_events to JSON and return the path used."""
        if self.config.record_path:
            path = Path(self.config.record_path)
        else:
            import datetime

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(f"recording_{ts}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self._record_events, f, indent=2)
        return path

    def _playback_loop(self, events: list[dict]) -> None:
        """Background thread: replay recorded events into _held_keys."""
        t0 = time.monotonic()
        for ev in events:
            if self._stop_event.is_set():
                break
            target = t0 + ev["t"]
            remaining = target - time.monotonic()
            if remaining > 0:
                # Sleep in short chunks so stop_event is checked promptly
                while remaining > 0.02 and not self._stop_event.is_set():
                    time.sleep(0.02)
                    remaining = target - time.monotonic()
                if not self._stop_event.is_set() and remaining > 0:
                    time.sleep(remaining)
            if self._stop_event.is_set():
                break
            code, value = ev["code"], ev["value"]
            if value == 1:
                with self._held_keys_lock:
                    self._held_keys.add(code)
            elif value == 0:
                with self._held_keys_lock:
                    self._held_keys.discard(code)
        # Playback finished — release all held keys and mark disconnected
        with self._held_keys_lock:
            self._held_keys.clear()
        self._connected = False
        print("\nPlayback complete.", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _combine(pos: bool, neg: bool, scale: float) -> float:
    """Return ±scale or 0.0 from two bool directions, clamped to [-1, 1]."""
    value = (1.0 if pos else 0.0) - (1.0 if neg else 0.0)
    return max(-1.0, min(1.0, value * scale))
