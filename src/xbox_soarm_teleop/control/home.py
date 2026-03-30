"""Shared helpers for rate-limited homing motions."""

from __future__ import annotations

import numpy as np


def step_scalar_toward(current: float, target: float, max_step: float) -> float:
    """Move a scalar toward a target by at most ``max_step``."""
    if max_step <= 0.0:
        return float(current)
    delta = float(target) - float(current)
    if abs(delta) <= max_step:
        return float(target)
    return float(current + np.sign(delta) * max_step)


def step_array_toward(
    current: np.ndarray,
    target: np.ndarray,
    max_step: float | np.ndarray,
) -> np.ndarray:
    """Move each array element toward its target by at most ``max_step``."""
    current_arr = np.asarray(current, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    max_step_arr = np.broadcast_to(np.asarray(max_step, dtype=float), current_arr.shape)
    delta = target_arr - current_arr
    return current_arr + np.clip(delta, -max_step_arr, max_step_arr)


def scalar_reached(current: float, target: float, *, atol: float = 1e-3) -> bool:
    """Return True when a scalar is within tolerance of its target."""
    return abs(float(current) - float(target)) <= atol


def array_reached(current: np.ndarray, target: np.ndarray, *, atol: float = 1e-3) -> bool:
    """Return True when every array element is within tolerance of its target."""
    return bool(np.all(np.abs(np.asarray(current, dtype=float) - np.asarray(target, dtype=float)) <= atol))
