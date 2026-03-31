"""Joy-Con IMU access via the Linux IIO subsystem."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Sequence


class JoyConIMU:
    """Read Joy-Con orientation from the Linux IIO subsystem.

    The hid-nintendo driver exposes accelerometer and gyroscope channels under
    ``/sys/bus/iio/devices``. A complementary filter fuses accel + gyro into a
    stable orientation estimate and keeps a yaw term from integrated gyro data.
    """

    _ACCEL_ATTRS = ("in_accel_x_raw", "in_accel_y_raw", "in_accel_z_raw")
    _GYRO_ATTRS = ("in_anglvel_x_raw", "in_anglvel_y_raw", "in_anglvel_z_raw")

    def __init__(
        self,
        *,
        device_index: int | None = None,
        device_name_patterns: Sequence[str] | None = None,
        alpha: float = 0.95,
    ) -> None:
        self.alpha = alpha
        self._device_name_patterns = tuple(device_name_patterns or ())
        self._iio_path = self._find_iio_device(device_index)
        self._accel_scale = self._read_scale("in_accel_scale", 1.0)
        self._gyro_scale = self._read_scale("in_anglvel_scale", 1.0)
        self._pitch = 0.0
        self._roll = 0.0
        self._yaw = 0.0
        self._last_t = time.monotonic()

    @property
    def available(self) -> bool:
        """Return True when the IMU sysfs device is available."""
        return self._iio_path is not None

    def read(self) -> tuple[float, float]:
        """Return ``(pitch_rad, roll_rad)`` for backward compatibility."""
        pitch, roll, _ = self.read_orientation()
        return pitch, roll

    def read_orientation(self) -> tuple[float, float, float]:
        """Return ``(pitch_rad, roll_rad, yaw_rad)`` from the IMU."""
        if self._iio_path is None:
            return 0.0, 0.0, 0.0

        try:
            ax, ay, az = self._read_accel()
            gx, gy, gz = self._read_gyro()
        except OSError:
            return self._pitch, self._roll, self._yaw

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

        self._pitch = self.alpha * gyro_pitch + (1.0 - self.alpha) * accel_pitch
        self._roll = self.alpha * gyro_roll + (1.0 - self.alpha) * accel_roll
        self._yaw += gz * dt
        return self._pitch, self._roll, self._yaw

    def calibrate(self) -> None:
        """Reset the filter from the current sensor reading."""
        if self._iio_path is None:
            return
        try:
            ax, ay, az = self._read_accel()
        except OSError:
            self._last_t = time.monotonic()
            self._yaw = 0.0
            return

        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 0.01:
            self._pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
            self._roll = math.atan2(ay, az)
        self._yaw = 0.0
        self._last_t = time.monotonic()

    def _find_iio_device(self, index: int | None) -> Path | None:
        root = Path("/sys/bus/iio/devices")
        if not root.exists():
            return None

        candidates: list[Path] = []
        for dev in sorted(root.iterdir()):
            name_file = dev / "name"
            if not name_file.exists():
                continue
            try:
                name = name_file.read_text().strip()
            except OSError:
                continue
            lowered = name.lower()
            patterns = tuple(p.lower() for p in self._device_name_patterns)
            if patterns:
                if not any(pattern in lowered for pattern in patterns):
                    continue
            elif not any(keyword in lowered for keyword in ("joy-con", "joycon", "nintendo")):
                continue
            if (dev / "in_accel_x_raw").exists():
                candidates.append(dev)

        if not candidates:
            return None
        if index is not None and 0 <= index < len(candidates):
            return candidates[index]
        return candidates[0]

    def _read_scale(self, attr: str, default: float) -> float:
        if self._iio_path is None:
            return default
        path = self._iio_path / attr
        if not path.exists():
            return default
        try:
            return float(path.read_text().strip())
        except (OSError, ValueError):
            return default

    def _read_accel(self) -> tuple[float, float, float]:
        path = self._iio_path
        ax = float((path / "in_accel_x_raw").read_text()) * self._accel_scale  # type: ignore[operator]
        ay = float((path / "in_accel_y_raw").read_text()) * self._accel_scale  # type: ignore[operator]
        az = float((path / "in_accel_z_raw").read_text()) * self._accel_scale  # type: ignore[operator]
        return ax, ay, az

    def _read_gyro(self) -> tuple[float, float, float]:
        path = self._iio_path
        gx = float((path / "in_anglvel_x_raw").read_text()) * self._gyro_scale  # type: ignore[operator]
        gy = float((path / "in_anglvel_y_raw").read_text()) * self._gyro_scale  # type: ignore[operator]
        gz = float((path / "in_anglvel_z_raw").read_text()) * self._gyro_scale  # type: ignore[operator]
        return gx, gy, gz
