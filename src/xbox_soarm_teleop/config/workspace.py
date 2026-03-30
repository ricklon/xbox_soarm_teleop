"""Workspace limits configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_POSITION = {
    "x": (0.05, 0.50),
    "y": (-0.30, 0.30),
    "z": (0.05, 0.45),
}

_DEFAULT_STRICT_POSITION = {
    "x": (0.10, 0.32),
    "y": (-0.20, 0.20),
    "z": (0.05, 0.30),
}


def _coerce_axis_bounds(
    raw: dict[str, Any] | None,
    fallback: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    if not raw:
        return dict(fallback)

    result: dict[str, tuple[float, float]] = {}
    for axis in ("x", "y", "z"):
        axis_cfg = raw.get(axis, {}) if isinstance(raw, dict) else {}
        if isinstance(axis_cfg, dict) and "min" in axis_cfg and "max" in axis_cfg:
            result[axis] = (float(axis_cfg["min"]), float(axis_cfg["max"]))
        else:
            result[axis] = fallback[axis]
    return result


def load_workspace_limits(
    path: Path | None = None,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Load workspace limits from YAML.

    Returns:
        Tuple of (position_limits, strict_position_limits). Each is a dict:
        {"x": (min, max), "y": (min, max), "z": (min, max)}.
    """
    if path is None:
        path = Path(__file__).with_name("workspace_limits.yaml")

    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception:
            data = {}

    position = _coerce_axis_bounds(data.get("position"), _DEFAULT_POSITION)
    strict_position = _coerce_axis_bounds(data.get("strict_position"), _DEFAULT_STRICT_POSITION)

    return position, strict_position
