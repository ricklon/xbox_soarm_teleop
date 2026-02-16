"""Utility functions for direct joint-drive diagnostics."""

from __future__ import annotations


def advance_goal(
    current_goal_deg: float,
    velocity_cmd_deg_s: float,
    dt_s: float,
    lower_deg: float,
    upper_deg: float,
) -> float:
    """Integrate a velocity command and clamp to joint limits."""
    if dt_s <= 0.0:
        return max(lower_deg, min(upper_deg, current_goal_deg))
    next_goal = current_goal_deg + velocity_cmd_deg_s * dt_s
    return max(lower_deg, min(upper_deg, next_goal))


def map_trigger_to_gripper_deg(trigger: float, lower_deg: float, upper_deg: float) -> float:
    """Map trigger [0,1] to gripper angle with 0=open (upper), 1=closed (lower)."""
    trig = max(0.0, min(1.0, trigger))
    return upper_deg - trig * (upper_deg - lower_deg)


def dpad_edge(current: float, previous: float, threshold: float = 0.5) -> int:
    """Return directional edge on D-pad axis.

    Returns -1 or 1 on a new edge crossing the threshold, else 0.
    """
    if current <= -threshold and previous > -threshold:
        return -1
    if current >= threshold and previous < threshold:
        return 1
    return 0
