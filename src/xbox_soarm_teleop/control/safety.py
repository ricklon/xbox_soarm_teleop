"""Safety helpers for cartesian control."""

from __future__ import annotations

from typing import Dict

import numpy as np

from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta


def apply_strict_safety(
    delta: EEDelta,
    *,
    max_linear_speed: float,
    max_angular_speed: float,
    allow_orientation: bool,
) -> tuple[EEDelta, Dict[str, int]]:
    """Apply strict safety limits to an EE delta.

    Returns the adjusted delta and a dict of flags:
    - speed_clip
    - orient_clip
    """
    flags = {"speed_clip": 0, "orient_clip": 0}

    lin = np.array([delta.dx, delta.dy, delta.dz], dtype=float)
    lin_norm = float(np.linalg.norm(lin))
    max_lin = max(0.001, max_linear_speed)
    if lin_norm > max_lin:
        lin *= max_lin / lin_norm
        flags["speed_clip"] = 1

    droll = float(np.clip(delta.droll, -max_angular_speed, max_angular_speed))

    dpitch = float(delta.dpitch)
    dyaw = float(delta.dyaw)
    if not allow_orientation:
        if abs(dpitch) > 1e-6 or abs(dyaw) > 1e-6:
            flags["orient_clip"] = 1
        dpitch = 0.0
        dyaw = 0.0
    else:
        old_dpitch, old_dyaw = dpitch, dyaw
        dpitch = float(np.clip(dpitch, -max_angular_speed, max_angular_speed))
        dyaw = float(np.clip(dyaw, -max_angular_speed, max_angular_speed))
        if old_dpitch != dpitch or old_dyaw != dyaw:
            flags["orient_clip"] = 1

    return (
        EEDelta(
            dx=float(lin[0]),
            dy=float(lin[1]),
            dz=float(lin[2]),
            droll=droll,
            dpitch=dpitch,
            dyaw=dyaw,
            gripper=delta.gripper,
        ),
        flags,
    )


def clip_workspace(
    target_pos: np.ndarray,
    workspace_limits: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, Dict[str, int]]:
    """Clip target position to workspace limits.

    Returns clipped target position and flags:
    - ws_clip_x
    - ws_clip_y
    - ws_clip_z
    """
    clipped = target_pos.copy()
    flags = {"ws_clip_x": 0, "ws_clip_y": 0, "ws_clip_z": 0}
    axes = ("x", "y", "z")
    for idx, axis in enumerate(axes):
        lo, hi = workspace_limits[axis]
        val = float(np.clip(clipped[idx], lo, hi))
        if val != clipped[idx]:
            flags[f"ws_clip_{axis}"] = 1
        clipped[idx] = val
    return clipped, flags
