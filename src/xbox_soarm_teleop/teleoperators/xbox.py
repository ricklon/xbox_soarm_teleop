"""Xbox controller teleoperator for SO-ARM teleoperation."""

import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from xbox_soarm_teleop.config.xbox_config import XboxConfig

# Subprocess backend: a report older than this releases the deadman.
SUBPROCESS_STALE_S = 0.15
SUBPROCESS_STARTUP_S = 3.0


def _platform_backend() -> str:
    """In-process backend for this platform."""
    return "sdl" if sys.platform in {"darwin", "win32"} else "inputs"


def _plain_python() -> str:
    """The environment's interpreter, never the mjpython launcher."""
    for name in ("python3", "python"):
        candidate = Path(sys.prefix) / "bin" / name
        if candidate.exists():
            return str(candidate)
    return sys.executable


@dataclass
class XboxState:
    """Normalized state of Xbox controller inputs.

    All stick values are normalized to [-1, 1].
    D-pad values are -1, 0, or 1.
    Trigger values are normalized to [0, 1].
    Button values are boolean.
    """

    left_stick_x: float = 0.0
    left_stick_y: float = 0.0
    right_stick_x: float = 0.0
    right_stick_y: float = 0.0
    right_trigger: float = 0.0
    left_bumper: bool = False
    a_button: bool = False
    y_button: bool = False
    b_button: bool = False  # latch the virtual stop
    start_button: bool = False  # release the virtual stop

    # D-pad state (-1, 0, or 1)
    dpad_x: float = 0.0  # -1 = left, 1 = right
    dpad_y: float = 0.0  # -1 = up, 1 = down (raw HAT convention)

    # Edge detection for buttons (True only on press, not hold)
    a_button_pressed: bool = False
    y_button_pressed: bool = False
    b_button_pressed: bool = False
    start_button_pressed: bool = False

    # Optional IMU orientation (radians) for controllers that expose it.
    imu_roll: float = 0.0
    imu_pitch: float = 0.0
    imu_yaw: float = 0.0
    imu_orientation_valid: bool = False


# Level (non-edge) fields, as streamed by the subprocess helper.
_LEVEL_FIELDS = [f.name for f in fields(XboxState) if not f.name.endswith("_pressed")]


class XboxController:
    """Xbox controller teleoperator for SO-ARM teleoperation.

    Reads input from an Xbox controller and provides normalized state.
    Supports deadzone filtering and axis inversion.

    Uses a background thread to read events from the controller,
    allowing non-blocking reads in the main control loop.

    Args:
        config: Controller configuration.
    """

    def __init__(self, config: XboxConfig | None = None):
        self.config = config or XboxConfig()
        self._state = XboxState()
        self._prev_state = XboxState()
        self._backend = self._resolve_backend()
        self._gamepad: Any = None
        self._sdl_controller: Any = None
        self._raw_state: dict[str, int] = {}
        self._raw_state_lock = threading.Lock()
        self._connected = False
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Subprocess backend: latest helper message and when it arrived.
        self._helper: subprocess.Popen | None = None
        self._helper_message: dict[str, Any] | None = None
        self._helper_received_at = 0.0

    def connect(self) -> bool:
        """Connect to the Xbox controller.

        Returns:
            True if connection successful, False otherwise.
        """
        self._backend = self._resolve_backend()
        if self._backend == "inputs":
            return self._connect_inputs()
        if self._backend == "sdl":
            return self._connect_sdl()
        if self._backend == "subprocess":
            return self._connect_subprocess()
        raise RuntimeError(
            f"Unsupported Xbox backend '{self._backend}'. "
            "Use one of: auto, inputs, sdl, subprocess."
        )

    def disconnect(self) -> None:
        """Disconnect from the controller."""
        self._stop_event.set()
        if self._helper is not None:
            self._stop_helper()
        if self._reader_thread is not None:
            try:
                self._reader_thread.join(timeout=1.0)
            except KeyboardInterrupt:
                pass
            self._reader_thread = None
        if self._sdl_controller is not None:
            try:
                self._sdl_controller.quit()
            except Exception:
                pass
            self._sdl_controller = None
        self._gamepad = None
        self._connected = False
        with self._raw_state_lock:
            self._raw_state.clear()

    @property
    def is_connected(self) -> bool:
        """Check if controller is connected."""
        if self._backend == "subprocess":
            return self._fresh_helper_message() is not None
        return self._connected

    def read(self) -> XboxState:
        """Read and return the current controller state.

        Returns normalized state with deadzone filtering applied.
        This method is non-blocking as events are read in a background thread.

        Returns:
            Current normalized controller state.
        """
        if not self._connected:
            return XboxState()

        # Store previous state for edge detection
        self._snapshot_previous_state()
        if self._backend == "sdl":
            self._read_sdl_state()
        elif self._backend == "subprocess":
            self._read_subprocess_state()
        else:
            self._read_inputs_state()
        self._apply_edge_detection()

        return self._state

    def _resolve_backend(self) -> str:
        """Pick the controller backend to use for the current platform."""
        if self.config.backend != "auto":
            return self.config.backend
        if os.environ.get("XBOX_SOARM_MJPYTHON") == "1":
            # mjpython owns the macOS main thread; SDL can't pump events here.
            return "subprocess"
        return _platform_backend()

    def _helper_command(self) -> list[str]:
        """Command line for the helper process that reads the controller."""
        config = asdict(self.config)
        config.pop("backend")
        return [
            _plain_python(),
            "-m",
            "xbox_soarm_teleop.teleoperators.xbox_helper",
            "--backend",
            _platform_backend(),
            "--config",
            json.dumps(config),
        ]

    def _connect_subprocess(self) -> bool:
        """Start the helper process and wait for its first report."""
        env = os.environ.copy()
        env.pop("XBOX_SOARM_MJPYTHON", None)
        self._helper_message = None
        self._helper = subprocess.Popen(
            self._helper_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._read_helper_loop, daemon=True)
        self._reader_thread.start()
        deadline = time.monotonic() + SUBPROCESS_STARTUP_S
        while time.monotonic() < deadline and self._helper.poll() is None:
            if self._helper_message is not None:
                break
            time.sleep(0.01)
        # Stay "connected" to the helper even before a pad is plugged in; the
        # helper keeps retrying and is_connected reports the pad itself.
        self._connected = self._helper_message is not None or self._helper.poll() is None
        return self._connected

    def _read_helper_loop(self) -> None:
        """Keep the newest helper message; anything malformed is ignored."""
        assert self._helper is not None and self._helper.stdout is not None
        for line in self._helper.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict):
                self._helper_message = message
                self._helper_received_at = time.monotonic()
            if self._stop_event.is_set():
                break

    def _fresh_helper_message(self) -> dict[str, Any] | None:
        """Return the latest message if it is recent and reports a controller."""
        message = self._helper_message
        if message is None or not message.get("connected"):
            return None
        if time.monotonic() - self._helper_received_at > SUBPROCESS_STALE_S:
            return None
        if self._helper is None or self._helper.poll() is not None:
            return None
        return message

    def _read_subprocess_state(self) -> None:
        """Apply the helper's state, or neutral (deadman released) if stale."""
        message = self._fresh_helper_message()
        state = message.get("state", {}) if message else {}
        neutral = XboxState()
        for name in _LEVEL_FIELDS:
            value = state.get(name, getattr(neutral, name))
            setattr(self._state, name, type(getattr(neutral, name))(value))

    def _stop_helper(self) -> None:
        helper, self._helper = self._helper, None
        if helper is None:
            return
        try:
            if helper.stdin is not None:
                helper.stdin.close()
            helper.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            helper.kill()
            helper.wait(timeout=1.0)

    def _connect_inputs(self) -> bool:
        """Connect using the legacy inputs backend."""
        try:
            import inputs

            gamepads = inputs.devices.gamepads
            if not gamepads:
                return False

            if self.config.device_index >= len(gamepads):
                return False

            self._gamepad = gamepads[self.config.device_index]
            self._connected = True

            self._stop_event.clear()
            self._reader_thread = threading.Thread(target=self._read_events_loop, daemon=True)
            self._reader_thread.start()
            return True
        except ImportError as exc:
            raise ImportError("inputs library not found. Install with: pip install inputs") from exc
        except Exception:
            return False

    def _connect_sdl(self) -> bool:
        """Connect using SDL game-controller support via pygame."""
        try:
            import pygame
            from pygame._sdl2 import controller as sdl_controller

            pygame.init()
            sdl_controller.init()
            controller_indices = [
                index
                for index in range(sdl_controller.get_count())
                if sdl_controller.is_controller(index)
            ]
            if self.config.device_index >= len(controller_indices):
                return False
            self._sdl_controller = sdl_controller.Controller(
                controller_indices[self.config.device_index]
            )
            self._connected = bool(self._sdl_controller.attached())
            return self._connected
        except ImportError as exc:
            raise ImportError("pygame SDL controller support is not available.") from exc
        except Exception:
            return False

    def _snapshot_previous_state(self) -> None:
        """Store the previous logical state for button edge detection."""
        self._prev_state = XboxState(
            left_stick_x=self._state.left_stick_x,
            left_stick_y=self._state.left_stick_y,
            right_stick_x=self._state.right_stick_x,
            right_stick_y=self._state.right_stick_y,
            right_trigger=self._state.right_trigger,
            left_bumper=self._state.left_bumper,
            a_button=self._state.a_button,
            y_button=self._state.y_button,
            b_button=self._state.b_button,
            start_button=self._state.start_button,
            dpad_x=self._state.dpad_x,
            dpad_y=self._state.dpad_y,
        )

    def _apply_edge_detection(self) -> None:
        """Populate rising-edge button flags after a fresh state read."""
        self._state.a_button_pressed = self._state.a_button and not self._prev_state.a_button
        self._state.y_button_pressed = self._state.y_button and not self._prev_state.y_button
        self._state.b_button_pressed = self._state.b_button and not self._prev_state.b_button
        self._state.start_button_pressed = (
            self._state.start_button and not self._prev_state.start_button
        )

    def _read_inputs_state(self) -> None:
        """Read and normalize state from the legacy inputs backend."""
        with self._raw_state_lock:
            raw_copy = self._raw_state.copy()

        left_x = self._normalize_stick_raw(
            raw_copy.get(self.config.left_stick_x_axis, 0),
            invert=self.config.invert_x,
        )
        left_y = self._normalize_stick_raw(
            raw_copy.get(self.config.left_stick_y_axis, 0),
            invert=self.config.invert_y,
        )
        right_x = self._normalize_stick_raw(
            raw_copy.get(self.config.right_stick_x_axis, 0),
            invert=self.config.invert_roll,
        )
        right_y = self._normalize_stick_raw(
            raw_copy.get(self.config.right_stick_y_axis, 0),
            invert=self.config.invert_z,
        )

        self._state.left_stick_x, self._state.left_stick_y = self._apply_radial_deadzone(
            left_x, left_y
        )
        self._state.right_stick_x, self._state.right_stick_y = self._apply_radial_deadzone(
            right_x, right_y
        )
        self._state.right_trigger = self._normalize_trigger(
            raw_copy.get(self.config.right_trigger_axis, 0)
        )
        self._state.left_bumper = bool(raw_copy.get(self.config.deadman_button, 0))
        self._state.a_button = bool(raw_copy.get(self.config.home_button, 0))
        self._state.y_button = bool(raw_copy.get(self.config.aux_button, 0))
        self._state.b_button = bool(raw_copy.get(self.config.stop_button, 0))
        self._state.start_button = bool(raw_copy.get(self.config.stop_reset_button, 0))
        self._state.dpad_x = float(raw_copy.get(self.config.dpad_x_axis, 0))
        self._state.dpad_y = float(raw_copy.get(self.config.dpad_y_axis, 0))

    def _read_sdl_state(self) -> None:
        """Read and normalize state from the SDL controller backend."""
        import pygame

        if self._sdl_controller is None or not self._sdl_controller.attached():
            self._connected = False
            return

        pygame.event.pump()
        left_x = self._normalize_stick_raw(
            self._sdl_controller.get_axis(pygame.CONTROLLER_AXIS_LEFTX),
            invert=self.config.invert_x,
        )
        left_y = self._normalize_stick_raw(
            self._sdl_controller.get_axis(pygame.CONTROLLER_AXIS_LEFTY),
            invert=self.config.invert_y,
        )
        right_x = self._normalize_stick_raw(
            self._sdl_controller.get_axis(pygame.CONTROLLER_AXIS_RIGHTX),
            invert=self.config.invert_roll,
        )
        right_y = self._normalize_stick_raw(
            self._sdl_controller.get_axis(pygame.CONTROLLER_AXIS_RIGHTY),
            invert=self.config.invert_z,
        )

        self._state.left_stick_x, self._state.left_stick_y = self._apply_radial_deadzone(
            left_x, left_y
        )
        self._state.right_stick_x, self._state.right_stick_y = self._apply_radial_deadzone(
            right_x, right_y
        )
        self._state.right_trigger = self._normalize_sdl_trigger(
            self._sdl_controller.get_axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT)
        )
        self._state.left_bumper = bool(
            self._sdl_controller.get_button(pygame.CONTROLLER_BUTTON_LEFTSHOULDER)
        )
        self._state.a_button = bool(self._sdl_controller.get_button(pygame.CONTROLLER_BUTTON_A))
        self._state.y_button = bool(self._sdl_controller.get_button(pygame.CONTROLLER_BUTTON_Y))
        self._state.b_button = bool(self._sdl_controller.get_button(pygame.CONTROLLER_BUTTON_B))
        self._state.start_button = bool(
            self._sdl_controller.get_button(pygame.CONTROLLER_BUTTON_START)
        )
        self._state.dpad_x = float(
            bool(self._sdl_controller.get_button(pygame.CONTROLLER_BUTTON_DPAD_RIGHT))
            - bool(self._sdl_controller.get_button(pygame.CONTROLLER_BUTTON_DPAD_LEFT))
        )
        self._state.dpad_y = float(
            bool(self._sdl_controller.get_button(pygame.CONTROLLER_BUTTON_DPAD_DOWN))
            - bool(self._sdl_controller.get_button(pygame.CONTROLLER_BUTTON_DPAD_UP))
        )

    def _read_events_loop(self) -> None:
        """Background thread that reads controller events."""
        import inputs

        while not self._stop_event.is_set():
            try:
                # This blocks until an event arrives
                events = self._gamepad.read()
                with self._raw_state_lock:
                    for event in events:
                        if event.ev_type in ("Absolute", "Key"):
                            self._raw_state[event.code] = event.state
            except inputs.UnpluggedError:
                self._connected = False
                break
            except Exception:
                # Ignore other errors, keep trying
                pass

    def _normalize_stick_raw(self, value: int, invert: bool = False) -> float:
        """Normalize stick value to [-1, 1] without deadzone.

        Args:
            value: Raw stick value.
            invert: Whether to invert the axis.

        Returns:
            Normalized value in [-1, 1].
        """
        min_val, max_val = self.config.stick_range
        center = (min_val + max_val) / 2
        half_range = (max_val - min_val) / 2

        # Normalize to [-1, 1]
        normalized = (value - center) / half_range

        # Apply inversion
        if invert:
            normalized = -normalized

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, normalized))

    def _apply_radial_deadzone(self, x: float, y: float) -> tuple[float, float]:
        """Apply radial (circular) deadzone to a stick's X/Y values.

        This treats both axes together, so diagonal movements and slight
        off-axis drift are properly filtered.

        Args:
            x: Normalized X axis value in [-1, 1].
            y: Normalized Y axis value in [-1, 1].

        Returns:
            Tuple of (x, y) with radial deadzone applied.
        """
        magnitude = math.sqrt(x * x + y * y)

        if magnitude < self.config.deadzone:
            return 0.0, 0.0

        # Rescale to maintain full range outside deadzone
        # Map [deadzone, 1] -> [0, 1]
        scale = (magnitude - self.config.deadzone) / (1.0 - self.config.deadzone)

        # Normalize direction and apply scaled magnitude
        # Clamp scale to avoid exceeding 1.0 at corners
        scale = min(scale, 1.0)
        new_x = (x / magnitude) * scale
        new_y = (y / magnitude) * scale

        return self._attenuate_minor_axis(new_x, new_y)

    def _attenuate_minor_axis(self, x: float, y: float) -> tuple[float, float]:
        """Reduce small cross-axis bleed when one stick axis is clearly dominant.

        This preserves intentional diagonals while making strong cardinal motions
        less likely to leak a small amount of the orthogonal axis.
        """
        abs_x = abs(x)
        abs_y = abs(y)
        dominant_min = self.config.dominant_axis_min
        ratio = self.config.dominant_axis_ratio
        attenuation = self.config.cross_axis_attenuation

        if abs_x >= dominant_min and abs_y > 0.0 and abs_x / abs_y >= ratio:
            return x, y * attenuation
        if abs_y >= dominant_min and abs_x > 0.0 and abs_y / abs_x >= ratio:
            return x * attenuation, y
        return x, y

    def _normalize_trigger(self, value: int) -> float:
        """Normalize trigger value to [0, 1].

        Args:
            value: Raw trigger value.

        Returns:
            Normalized value in [0, 1].
        """
        min_val, max_val = self.config.trigger_range
        if max_val == min_val:
            return 0.0
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    def _normalize_sdl_trigger(self, value: int) -> float:
        """Normalize an SDL trigger axis value to [0, 1]."""
        return max(0.0, min(1.0, value / 32767.0))
