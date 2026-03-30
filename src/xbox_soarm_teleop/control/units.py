"""Unit conversion helpers."""

from __future__ import annotations

import numpy as np

from xbox_soarm_teleop.config.joints import JOINT_LIMITS_DEG


def deg_to_normalized(deg: float, joint_name: str) -> float:
    """Convert degrees to LeRobot normalized value [-100, 100]."""
    if joint_name == "gripper":
        return deg
    lower, upper = JOINT_LIMITS_DEG[joint_name]
    max_abs = max(abs(lower), abs(upper), 1e-6)
    normalized = deg * (100.0 / max_abs)
    return float(np.clip(normalized, -100.0, 100.0))


def normalized_to_deg(normalized: float, joint_name: str) -> float:
    """Convert LeRobot normalized value [-100, 100] to degrees."""
    if joint_name == "gripper":
        return normalized
    lower, upper = JOINT_LIMITS_DEG[joint_name]
    max_abs = max(abs(lower), abs(upper), 1e-6)
    deg = normalized * (max_abs / 100.0)
    return float(np.clip(deg, lower, upper))
