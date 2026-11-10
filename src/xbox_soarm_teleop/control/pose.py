"""Pose helpers for control loops."""

from __future__ import annotations

import numpy as np


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX euler angles to a rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def rotation_matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """Invert :func:`euler_to_rotation_matrix`: return (roll, pitch, yaw) in radians."""
    r = np.asarray(rotation, dtype=float)
    pitch = float(-np.arcsin(np.clip(r[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(r[2, 1], r[2, 2]))
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    return roll, pitch, yaw
