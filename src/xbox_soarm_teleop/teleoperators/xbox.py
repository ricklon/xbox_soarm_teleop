"""Xbox controller teleoperator for SO-ARM teleoperation."""

import math
import threading
from dataclasses import dataclass
from typing import Any

from xbox_soarm_teleop.config.xbox_config import XboxConfig


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

    # D-pad state (-1, 0, or 1)
    dpad_x: float = 0.0  # -1 = left, 1 = right
    dpad_y: float = 0.0  # -1 = up, 1 = down (raw HAT convention)

    # Edge detection for buttons (True only on press, not hold)
    a_button_pressed: bool = False
    y_button_pressed: bool = False

    # Optional IMU orientation (radians) for controllers that expose it.
    imu_roll: float = 0.0
    imu_pitch: float = 0.0
    imu_yaw: float = 0.0
    imu_orientation_valid: bool = False


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
        self._gamepad: Any = None
        self._raw_state: dict[str, int] = {}
        self._raw_state_lock = threading.Lock()
        self._connected = False
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def connect(self) -> bool:
        """Connect to the Xbox controller.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            import inputs

            gamepads = inputs.devices.gamepads
            if not gamepads:
                return False

            if self.config.device_index >= len(gamepads):
                return False

            self._gamepad = gamepads[self.config.device_index]
            self._connected = True

            # Start background reader thread
            self._stop_event.clear()
            self._reader_thread = threading.Thread(target=self._read_events_loop, daemon=True)
            self._reader_thread.start()

            return True
        except ImportError:
            raise ImportError("inputs library not found. Install with: pip install inputs")
        except Exception:
            return False

    def disconnect(self) -> None:
        """Disconnect from the controller."""
        self._stop_event.set()
        if self._reader_thread is not None:
            try:
                self._reader_thread.join(timeout=1.0)
            except KeyboardInterrupt:
                pass
            self._reader_thread = None
        self._gamepad = None
        self._connected = False
        with self._raw_state_lock:
            self._raw_state.clear()

    @property
    def is_connected(self) -> bool:
        """Check if controller is connected."""
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

        # Get current raw state (thread-safe)
        with self._raw_state_lock:
            raw_copy = self._raw_state.copy()

        # Normalize stick values (without deadzone - applied radially below)
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

        # Apply radial deadzone to each stick (treats both axes together)
        self._state.left_stick_x, self._state.left_stick_y = self._apply_radial_deadzone(
            left_x, left_y
        )
        self._state.right_stick_x, self._state.right_stick_y = self._apply_radial_deadzone(
            right_x, right_y
        )

        self._state.right_trigger = self._normalize_trigger(
            raw_copy.get(self.config.right_trigger_axis, 0)
        )

        # Buttons
        self._state.left_bumper = bool(raw_copy.get(self.config.deadman_button, 0))
        self._state.a_button = bool(raw_copy.get(self.config.home_button, 0))
        self._state.y_button = bool(raw_copy.get(self.config.aux_button, 0))

        # D-pad (HAT switch) - values are -1, 0, or 1
        self._state.dpad_x = float(raw_copy.get(self.config.dpad_x_axis, 0))
        self._state.dpad_y = float(raw_copy.get(self.config.dpad_y_axis, 0))

        # Edge detection - True only on rising edge
        self._state.a_button_pressed = self._state.a_button and not self._prev_state.a_button
        self._state.y_button_pressed = self._state.y_button and not self._prev_state.y_button

        return self._state

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
