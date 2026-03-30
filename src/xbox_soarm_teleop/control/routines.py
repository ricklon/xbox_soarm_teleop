"""Motion routines for demo or testing."""

from __future__ import annotations

import numpy as np


def square_offset(u: float, size: float) -> tuple[float, float]:
    """Parametric square offset in 2D."""
    half = size / 2.0
    s = (u % 1.0) * 4.0
    seg = int(s)
    f = s - seg
    if seg == 0:
        return (-half + f * size, -half)
    if seg == 1:
        return (half, -half + f * size)
    if seg == 2:
        return (half - f * size, half)
    return (-half, half - f * size)


def plane_offset(plane: str, u: float, size: float) -> np.ndarray:
    """Square offset applied to a named plane (xy/xz/yz)."""
    a, b = square_offset(u, size)
    if plane == "xy":
        return np.array([a, b, 0.0])
    if plane == "xz":
        return np.array([a, 0.0, b])
    return np.array([0.0, a, b])
