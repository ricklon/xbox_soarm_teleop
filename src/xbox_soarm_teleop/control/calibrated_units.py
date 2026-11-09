"""Explicit model-angle mapping for LeRobot RANGE_M100_100 arm joints.

Encoder coordinates here are the values seen by LeRobot *after* the active
hardware homing offset. Never apply that offset again. A calibration range does
not identify URDF zero: the zero, sign, and scale must come from separate evidence.
This module deliberately does not replace legacy live-control conversions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibratedJointMapping:
    """Map normalized arm positions through an explicitly referenced encoder.

    Args:
        range_min: Lower calibrated encoder endpoint.
        range_max: Upper calibrated encoder endpoint.
        drive_mode: Effective LeRobot inversion (0 or 1, with apply_drive_mode).
        raw_at_model_zero: Encoder reading at zero URDF joint angle, in the same
            homing-offset coordinate system as the range endpoints.
        model_sign: +1 or -1: model-angle direction versus encoder direction.
        ticks_per_turn: Encoder ticks per physical revolution, not maximum code.

    These are arm-joint conversions, not gripper opening percentages. Calibration
    bounds are checked here; workspace and model joint limits are separate.
    """

    range_min: int
    range_max: int
    drive_mode: int
    raw_at_model_zero: float
    model_sign: int
    ticks_per_turn: float

    def __post_init__(self) -> None:
        values = (self.range_min, self.range_max, self.raw_at_model_zero, self.ticks_per_turn)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Calibration and reference values must be finite")
        if self.range_min >= self.range_max:
            raise ValueError("Calibration range_min must be less than range_max")
        if self.drive_mode not in (0, 1) or self.model_sign not in (-1, 1):
            raise ValueError("drive_mode must be 0 or 1; model_sign must be -1 or 1")
        if self.ticks_per_turn <= 0:
            raise ValueError("ticks_per_turn must be positive")

    def normalized_to_degrees(self, normalized: float) -> float:
        """Convert feedback to model degrees without clipping away bad inputs."""
        if not math.isfinite(normalized) or not -100 <= normalized <= 100:
            raise ValueError("Normalized arm position must be finite and within [-100, 100]")
        value = -normalized if self.drive_mode else normalized
        raw = self.range_min + (value + 100) * (self.range_max - self.range_min) / 200
        return self.model_sign * (raw - self.raw_at_model_zero) * 360 / self.ticks_per_turn

    def degrees_to_normalized(self, degrees: float) -> float:
        """Convert a model target, rejecting targets outside calibration bounds."""
        if not math.isfinite(degrees):
            raise ValueError("Model angle must be finite")
        raw = self.raw_at_model_zero + self.model_sign * degrees * self.ticks_per_turn / 360
        # Permit only floating-point roundoff at the calibrated endpoints.
        if raw < self.range_min - 1e-9 or raw > self.range_max + 1e-9:
            raise ValueError("Model target is outside the calibrated encoder range")
        raw = min(self.range_max, max(self.range_min, raw))
        value = (raw - self.range_min) * 200 / (self.range_max - self.range_min) - 100
        return -value if self.drive_mode else value
