"""Nintendo Switch Joy-Con teleoperators for SO-ARM control."""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

from xbox_soarm_teleop.config.joycon_config import DualJoyConConfig, JoyConConfig
from xbox_soarm_teleop.teleoperators.joycon_imu import JoyConIMU
from xbox_soarm_teleop.teleoperators.xbox import XboxState


def _ecodes_name(mapping: dict, code: int) -> str:
    """Return a stable string name for an evdev event code."""
    result = mapping.get(code, str(code))
    return result[0] if isinstance(result, list) else result


def _is_matching_joycon_device_name(name: str, patterns: list[str]) -> bool:
    """Return True for usable Joy-Con input devices and False for IMU-only nodes."""
    return any(pat in name for pat in patterns) and "IMU" not in name


def _patterns_target_left(patterns: list[str]) -> bool:
    """Return True when the requested Joy-Con patterns target the left unit."""
    lowered = " ".join(patterns).lower()
    return "left" in lowered or "(l)" in lowered


class _PyJoyConDevice:
    """macOS Joy-Con device wrapper using joycon-python over HID."""

    _STICK_RANGE = (0, 4095)

    def __init__(self, joycon_id: tuple[int, int, str], *, track_orientation: bool = False, alpha: float = 0.95):
        from pyjoycon.wrappers import PythonicJoyCon

        self._joycon = PythonicJoyCon(*joycon_id)
        self._track_orientation = track_orientation
        self._alpha = alpha
        self._pitch = 0.0
        self._roll = 0.0
        self._yaw = 0.0
        self._last_t = time.monotonic()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._joycon, name)

    @property
    def stick_range(self) -> tuple[int, int]:
        return self._STICK_RANGE

    @property
    def available(self) -> bool:
        return self._track_orientation

    def disconnect(self) -> None:
        self._joycon.disconnect()

    def calibrate(self) -> None:
        self._pitch = 0.0
        self._roll = 0.0
        self._yaw = 0.0
        self._last_t = time.monotonic()

    def read_orientation(self) -> tuple[float, float, float]:
        if not self._track_orientation:
            return 0.0, 0.0, 0.0

        accel = self._joycon.accel_in_g
        gyro = self._joycon.gyro_in_rad
        if not accel or not gyro:
            return self._pitch, self._roll, self._yaw

        ax = sum(sample[0] for sample in accel) / len(accel)
        ay = sum(sample[1] for sample in accel) / len(accel)
        az = sum(sample[2] for sample in accel) / len(accel)
        gx = sum(sample[0] for sample in gyro) / len(gyro)
        gy = sum(sample[1] for sample in gyro) / len(gyro)
        gz = sum(sample[2] for sample in gyro) / len(gyro)

        now = time.monotonic()
        dt = min(now - self._last_t, 0.1)
        self._last_t = now

        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 0.01:
            accel_pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
            accel_roll = math.atan2(ay, az)
        else:
            accel_pitch = self._pitch
            accel_roll = self._roll

        gyro_pitch = self._pitch + gy * dt
        gyro_roll = self._roll + gx * dt

        self._pitch = self._alpha * gyro_pitch + (1.0 - self._alpha) * accel_pitch
        self._roll = self._alpha * gyro_roll + (1.0 - self._alpha) * accel_roll
        self._yaw += gz * dt
        return self._pitch, self._roll, self._yaw


@dataclass
class _JoyConDeviceReader:
    """Background evdev reader for one Joy-Con input node."""

    device: Any
    raw_state: dict[str, int]
    raw_state_lock: threading.Lock
    stop_event: threading.Event
    connected_ref: dict[str, bool]
    reader_thread: threading.Thread | None = None

    def start(self) -> None:
        self.stop_event.clear()
        self.reader_thread = threading.Thread(target=self._read_events_loop, daemon=True)
        self.reader_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1.0)
            self.reader_thread = None
        try:
            self.device.close()
        except Exception:
            pass

    def _read_events_loop(self) -> None:
        import select

        import evdev

        fd = self.device.fd
        while not self.stop_event.is_set():
            try:
                readable, _, _ = select.select([fd], [], [], 0.05)
                if not readable:
                    continue
                for event in self.device.read():
                    if event.type == evdev.ecodes.EV_ABS:
                        name = _ecodes_name(evdev.ecodes.ABS, event.code)
                        with self.raw_state_lock:
                            self.raw_state[name] = event.value
                    elif event.type == evdev.ecodes.EV_KEY:
                        name = _ecodes_name(evdev.ecodes.BTN, event.code)
                        with self.raw_state_lock:
                            self.raw_state[name] = event.value
            except OSError:
                self.connected_ref["connected"] = False
                break
            except Exception:
                pass


class _JoyConMathMixin:
    """Shared normalization helpers for Joy-Con teleoperators."""

    def _normalize_stick_raw(
        self,
        value: int,
        *,
        stick_range: tuple[int, int] | None = None,
        invert: bool = False,
    ) -> float:
        if stick_range is None:
            stick_range = self.config.stick_range
        min_val, max_val = stick_range
        center = (min_val + max_val) / 2
        half_range = (max_val - min_val) / 2
        normalized = (value - center) / half_range
        if invert:
            normalized = -normalized
        return max(-1.0, min(1.0, normalized))

    def _apply_radial_deadzone(
        self,
        x: float,
        y: float,
        *,
        deadzone: float | None = None,
    ) -> tuple[float, float]:
        if deadzone is None:
            deadzone = self.config.deadzone
        magnitude = math.sqrt(x * x + y * y)
        if magnitude < deadzone:
            return 0.0, 0.0
        scale = min((magnitude - deadzone) / (1.0 - deadzone), 1.0)
        return (x / magnitude) * scale, (y / magnitude) * scale

    def _find_device(self, patterns: list[str], device_index: int) -> Any:
        import evdev

        candidates = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                if _is_matching_joycon_device_name(dev.name, patterns):
                    candidates.append(dev)
                else:
                    dev.close()
            except (OSError, PermissionError):
                continue

        if not candidates:
            return None

        chosen = candidates[device_index] if device_index < len(candidates) else candidates[0]
        for candidate in candidates:
            if candidate is not chosen:
                candidate.close()
        return chosen

    def _find_pyjoycon_id(self, patterns: list[str], device_index: int) -> tuple[int, int, str] | None:
        from pyjoycon.device import get_L_ids, get_R_ids

        candidates = get_L_ids() if _patterns_target_left(patterns) else get_R_ids()
        if not candidates:
            return None
        return candidates[device_index] if device_index < len(candidates) else candidates[0]


class JoyConController(_JoyConMathMixin):
    """Right Joy-Con teleoperator (horizontal single-controller mode)."""

    def __init__(self, config: JoyConConfig | None = None):
        self.config = config or JoyConConfig()
        self._state = XboxState()
        self._prev_state = XboxState()
        self._device_reader: _JoyConDeviceReader | None = None
        self._raw_state: dict[str, int] = {}
        self._raw_state_lock = threading.Lock()
        self._connected = False
        self._connected_ref = {"connected": False}
        self._stop_event = threading.Event()

    def connect(self) -> bool:
        """Connect to the single Joy-Con via evdev."""
        try:
            device = self._find_device(
                self.config.device_name_patterns,
                self.config.device_index,
            )
        except ImportError as exc:
            raise ImportError(
                "evdev not installed. Run: uv pip install evdev\n"
                "Note: evdev is Linux-only."
            ) from exc
        if device is None:
            return False

        self._connected = True
        self._connected_ref["connected"] = True
        self._device_reader = _JoyConDeviceReader(
            device=device,
            raw_state=self._raw_state,
            raw_state_lock=self._raw_state_lock,
            stop_event=self._stop_event,
            connected_ref=self._connected_ref,
        )
        self._device_reader.start()
        return True

    def disconnect(self) -> None:
        """Disconnect from the Joy-Con device."""
        if self._device_reader is not None:
            self._device_reader.stop()
            self._device_reader = None
        self._connected = False
        self._connected_ref["connected"] = False
        with self._raw_state_lock:
            self._raw_state.clear()

    @property
    def is_connected(self) -> bool:
        return self._connected_ref["connected"] or (self._connected and self._device_reader is None)

    def read(self) -> XboxState:
        """Return current normalized Joy-Con state."""
        if not self.is_connected:
            return XboxState()

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

        raw_x = self._normalize_stick_raw(
            raw.get(self.config.left_stick_x_axis, 0),
            stick_range=self.config.stick_range,
            invert=self.config.invert_x,
        )
        raw_y = self._normalize_stick_raw(
            raw.get(self.config.left_stick_y_axis, 0),
            stick_range=self.config.stick_range,
            invert=self.config.invert_y,
        )
        sx, sy = self._apply_radial_deadzone(raw_x, raw_y, deadzone=self.config.deadzone)
        self._state.left_stick_x = sx
        self._state.left_stick_y = sy
        self._state.right_stick_x = sx
        self._state.right_stick_y = sy

        self._state.right_trigger = 1.0 if raw.get(self.config.zr_button, 0) else 0.0
        self._state.left_bumper = bool(raw.get(self.config.deadman_button, 0))
        self._state.a_button = bool(raw.get(self.config.home_button, 0))
        self._state.y_button = bool(raw.get(self.config.aux_button, 0))

        self._state.dpad_x = 0.0
        height_up = bool(raw.get(self.config.height_up_button, 0))
        height_down = bool(raw.get(self.config.height_down_button, 0))
        if height_up and not height_down:
            self._state.dpad_y = 1.0
        elif height_down and not height_up:
            self._state.dpad_y = -1.0
        else:
            self._state.dpad_y = 0.0

        self._state.a_button_pressed = self._state.a_button and not self._prev_state.a_button
        self._state.y_button_pressed = self._state.y_button and not self._prev_state.y_button
        self._state.imu_orientation_valid = False
        self._state.imu_roll = 0.0
        self._state.imu_pitch = 0.0
        self._state.imu_yaw = 0.0
        return self._state


class DualJoyConController(_JoyConMathMixin):
    """Split dual Joy-Con teleoperator with right-hand IMU orientation."""

    def __init__(self, config: DualJoyConConfig | None = None):
        self.config = config or DualJoyConConfig()
        self._state = XboxState()
        self._prev_state = XboxState()
        self._left_reader: _JoyConDeviceReader | None = None
        self._right_reader: _JoyConDeviceReader | None = None
        self._left_pyjoycon: _PyJoyConDevice | None = None
        self._right_pyjoycon: _PyJoyConDevice | None = None
        self._left_raw_state: dict[str, int] = {}
        self._right_raw_state: dict[str, int] = {}
        self._left_lock = threading.Lock()
        self._right_lock = threading.Lock()
        self._left_stop = threading.Event()
        self._right_stop = threading.Event()
        self._connected = False
        self._connected_ref = {"connected": False}
        self._imu = JoyConIMU(
            device_index=self.config.imu_device_index,
            device_name_patterns=self.config.right_imu_name_patterns,
        )

    def connect(self) -> bool:
        """Connect to left and right Joy-Con input devices."""
        if sys.platform == "darwin":
            left_id = self._find_pyjoycon_id(
                self.config.left_device_name_patterns,
                self.config.left_device_index,
            )
            right_id = self._find_pyjoycon_id(
                self.config.right_device_name_patterns,
                self.config.right_device_index,
            )
            if left_id is None or right_id is None:
                return False

            try:
                self._left_pyjoycon = _PyJoyConDevice(left_id)
                self._right_pyjoycon = _PyJoyConDevice(right_id, track_orientation=True)
                self._right_pyjoycon.calibrate()
            except Exception:
                if self._left_pyjoycon is not None:
                    try:
                        self._left_pyjoycon.disconnect()
                    except Exception:
                        pass
                    self._left_pyjoycon = None
                if self._right_pyjoycon is not None:
                    try:
                        self._right_pyjoycon.disconnect()
                    except Exception:
                        pass
                    self._right_pyjoycon = None
                return False

            self._connected = True
            self._connected_ref["connected"] = True
            return True

        try:
            left = self._find_device(
                self.config.left_device_name_patterns,
                self.config.left_device_index,
            )
            right = self._find_device(
                self.config.right_device_name_patterns,
                self.config.right_device_index,
            )
        except ImportError as exc:
            raise ImportError(
                "evdev not installed. Run: uv pip install evdev\n"
                "Note: evdev is Linux-only."
            ) from exc
        if left is None or right is None:
            if left is not None:
                left.close()
            if right is not None:
                right.close()
            return False

        self._connected = True
        self._connected_ref["connected"] = True
        self._left_reader = _JoyConDeviceReader(
            device=left,
            raw_state=self._left_raw_state,
            raw_state_lock=self._left_lock,
            stop_event=self._left_stop,
            connected_ref=self._connected_ref,
        )
        self._right_reader = _JoyConDeviceReader(
            device=right,
            raw_state=self._right_raw_state,
            raw_state_lock=self._right_lock,
            stop_event=self._right_stop,
            connected_ref=self._connected_ref,
        )
        self._left_reader.start()
        self._right_reader.start()
        if self._imu.available:
            self._imu.calibrate()
        return True

    def disconnect(self) -> None:
        """Disconnect both Joy-Con devices."""
        if self._left_pyjoycon is not None:
            try:
                self._left_pyjoycon.disconnect()
            except Exception:
                pass
            self._left_pyjoycon = None
        if self._right_pyjoycon is not None:
            try:
                self._right_pyjoycon.disconnect()
            except Exception:
                pass
            self._right_pyjoycon = None
        if self._left_reader is not None:
            self._left_reader.stop()
            self._left_reader = None
        if self._right_reader is not None:
            self._right_reader.stop()
            self._right_reader = None
        self._connected = False
        self._connected_ref["connected"] = False
        with self._left_lock:
            self._left_raw_state.clear()
        with self._right_lock:
            self._right_raw_state.clear()

    @property
    def is_connected(self) -> bool:
        if sys.platform == "darwin" and (
            self._left_pyjoycon is not None or self._right_pyjoycon is not None
        ):
            return self._connected and self._left_pyjoycon is not None and self._right_pyjoycon is not None
        return self._connected_ref["connected"] or (
            self._connected and self._left_reader is None and self._right_reader is None
        )

    def read(self) -> XboxState:
        """Return merged left/right Joy-Con state plus IMU orientation."""
        if not self.is_connected:
            return XboxState()

        if sys.platform == "darwin" and (
            self._left_pyjoycon is not None or self._right_pyjoycon is not None
        ):
            return self._read_pyjoycon()

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

        with self._left_lock:
            left_raw = self._left_raw_state.copy()
        with self._right_lock:
            right_raw = self._right_raw_state.copy()

        left_x = self._normalize_stick_raw(
            left_raw.get(self.config.left_stick_x_axis, 0),
            stick_range=self.config.stick_range,
            invert=self.config.invert_x,
        )
        left_y = self._normalize_stick_raw(
            left_raw.get(self.config.left_stick_y_axis, 0),
            stick_range=self.config.stick_range,
            invert=self.config.invert_y,
        )
        right_x = self._normalize_stick_raw(
            right_raw.get(self.config.right_stick_x_axis, 0),
            stick_range=self.config.stick_range,
            invert=False,
        )
        right_y = self._normalize_stick_raw(
            right_raw.get(self.config.right_stick_y_axis, 0),
            stick_range=self.config.stick_range,
            invert=False,
        )
        self._state.left_stick_x, self._state.left_stick_y = self._apply_radial_deadzone(
            left_x,
            left_y,
            deadzone=self.config.deadzone,
        )
        self._state.right_stick_x, self._state.right_stick_y = self._apply_radial_deadzone(
            right_x,
            right_y,
            deadzone=self.config.deadzone,
        )

        self._state.left_bumper = bool(left_raw.get(self.config.deadman_button, 0))
        self._state.right_trigger = 1.0 if right_raw.get(self.config.gripper_button, 0) else 0.0
        self._state.a_button = bool(right_raw.get(self.config.home_button, 0))
        self._state.y_button = bool(right_raw.get(self.config.aux_button, 0))
        self._state.a_button_pressed = self._state.a_button and not self._prev_state.a_button
        self._state.y_button_pressed = self._state.y_button and not self._prev_state.y_button

        left_pressed = bool(left_raw.get(self.config.dpad_left_button, 0))
        right_pressed = bool(left_raw.get(self.config.dpad_right_button, 0))
        if left_pressed and not right_pressed:
            self._state.dpad_x = -1.0
        elif right_pressed and not left_pressed:
            self._state.dpad_x = 1.0
        else:
            self._state.dpad_x = 0.0

        up_pressed = bool(left_raw.get(self.config.z_up_button, 0))
        down_pressed = bool(left_raw.get(self.config.z_down_button, 0))
        if up_pressed and not down_pressed:
            self._state.dpad_y = -1.0
        elif down_pressed and not up_pressed:
            self._state.dpad_y = 1.0
        else:
            self._state.dpad_y = 0.0

        if self._imu.available:
            pitch, roll, yaw = self._imu.read_orientation()
            self._state.imu_pitch = pitch
            self._state.imu_roll = roll
            self._state.imu_yaw = yaw
            self._state.imu_orientation_valid = True
        else:
            self._state.imu_pitch = 0.0
            self._state.imu_roll = 0.0
            self._state.imu_yaw = 0.0
            self._state.imu_orientation_valid = False

        return self._state

    def _read_pyjoycon(self) -> XboxState:
        if self._left_pyjoycon is None or self._right_pyjoycon is None:
            return XboxState()

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

        left_stick = self._left_pyjoycon.stick_l
        right_stick = self._right_pyjoycon.stick_r

        left_x = self._normalize_stick_raw(
            int(left_stick[0]),
            stick_range=self._left_pyjoycon.stick_range,
            invert=self.config.invert_x,
        )
        left_y = self._normalize_stick_raw(
            int(left_stick[1]),
            stick_range=self._left_pyjoycon.stick_range,
            invert=self.config.invert_y,
        )
        right_x = self._normalize_stick_raw(
            int(right_stick[0]),
            stick_range=self._right_pyjoycon.stick_range,
            invert=False,
        )
        right_y = self._normalize_stick_raw(
            int(right_stick[1]),
            stick_range=self._right_pyjoycon.stick_range,
            invert=False,
        )
        self._state.left_stick_x, self._state.left_stick_y = self._apply_radial_deadzone(
            left_x,
            left_y,
            deadzone=self.config.deadzone,
        )
        self._state.right_stick_x, self._state.right_stick_y = self._apply_radial_deadzone(
            right_x,
            right_y,
            deadzone=self.config.deadzone,
        )

        self._state.left_bumper = bool(self._left_pyjoycon.zl)
        self._state.right_trigger = 1.0 if self._right_pyjoycon.zr else 0.0
        self._state.a_button = bool(self._right_pyjoycon.plus)
        self._state.y_button = bool(self._right_pyjoycon.home)
        self._state.a_button_pressed = self._state.a_button and not self._prev_state.a_button
        self._state.y_button_pressed = self._state.y_button and not self._prev_state.y_button

        left_pressed = bool(self._left_pyjoycon.left)
        right_pressed = bool(self._left_pyjoycon.right)
        if left_pressed and not right_pressed:
            self._state.dpad_x = -1.0
        elif right_pressed and not left_pressed:
            self._state.dpad_x = 1.0
        else:
            self._state.dpad_x = 0.0

        up_pressed = bool(self._left_pyjoycon.up)
        down_pressed = bool(self._left_pyjoycon.down)
        if up_pressed and not down_pressed:
            self._state.dpad_y = -1.0
        elif down_pressed and not up_pressed:
            self._state.dpad_y = 1.0
        else:
            self._state.dpad_y = 0.0

        pitch, roll, yaw = self._right_pyjoycon.read_orientation()
        self._state.imu_pitch = pitch
        self._state.imu_roll = roll
        self._state.imu_yaw = yaw
        self._state.imu_orientation_valid = True
        return self._state
