"""Analysis helpers for direct-joint diagnostic CSV logs."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from xbox_soarm_teleop.config.joints import JOINT_LIMITS_DEG, JOINT_NAMES_WITH_GRIPPER


@dataclass
class JointSummary:
    """Computed summary for a single joint diagnostic stream."""

    joint: str
    samples: int
    active_cmd_samples: int
    cmd_abs_mean_deg_s: float
    cmd_abs_peak_deg_s: float
    measured_abs_mean_deg_s: float
    measured_abs_peak_deg_s: float
    tracking_err_mean_deg: float
    tracking_err_p95_deg: float
    observed_min_deg: float
    observed_max_deg: float
    observed_span_deg: float
    limit_span_deg: float
    span_utilization_pct: float
    near_limit_samples: int


@dataclass
class DiagnosticSummary:
    """High-level results for the whole diagnostic log."""

    source: Path
    total_rows: int
    per_joint: list[JointSummary]


def _to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if pct <= 0.0:
        return sorted_values[0]
    if pct >= 100.0:
        return sorted_values[-1]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_values[lo]
    frac = k - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _summarize_joint_rows(
    joint: str,
    rows: list[dict[str, str]],
    cmd_threshold_deg_s: float,
    near_limit_margin_deg: float,
) -> JointSummary:
    positions = [_to_float(r, "selected_pos_deg") for r in rows]
    goals = [_to_float(r, "selected_goal_deg") for r in rows]
    times = [_to_float(r, "t_s") for r in rows]
    cmds = [_to_float(r, "cmd_vel_deg_s") for r in rows]

    abs_cmds = [abs(v) for v in cmds]
    active_cmds = [v for v in abs_cmds if v >= cmd_threshold_deg_s]

    tracking_errors = [abs(g - p) for g, p in zip(goals, positions)]
    sorted_tracking = sorted(tracking_errors)

    measured_abs = []
    for i in range(1, len(rows)):
        dt = times[i] - times[i - 1]
        if dt <= 1e-9:
            continue
        dp = positions[i] - positions[i - 1]
        measured_abs.append(abs(dp / dt))

    lower, upper = JOINT_LIMITS_DEG[joint]
    span = upper - lower
    observed_min = min(positions) if positions else 0.0
    observed_max = max(positions) if positions else 0.0
    observed_span = observed_max - observed_min
    utilization = 0.0 if span <= 0 else (observed_span / span) * 100.0

    near_limit = 0
    for pos in positions:
        if (pos - lower) <= near_limit_margin_deg or (upper - pos) <= near_limit_margin_deg:
            near_limit += 1

    return JointSummary(
        joint=joint,
        samples=len(rows),
        active_cmd_samples=len(active_cmds),
        cmd_abs_mean_deg_s=sum(abs_cmds) / len(abs_cmds) if abs_cmds else 0.0,
        cmd_abs_peak_deg_s=max(abs_cmds) if abs_cmds else 0.0,
        measured_abs_mean_deg_s=(sum(measured_abs) / len(measured_abs) if measured_abs else 0.0),
        measured_abs_peak_deg_s=max(measured_abs) if measured_abs else 0.0,
        tracking_err_mean_deg=(
            sum(tracking_errors) / len(tracking_errors) if tracking_errors else 0.0
        ),
        tracking_err_p95_deg=_percentile(sorted_tracking, 95.0),
        observed_min_deg=observed_min,
        observed_max_deg=observed_max,
        observed_span_deg=observed_span,
        limit_span_deg=span,
        span_utilization_pct=utilization,
        near_limit_samples=near_limit,
    )


def analyze_joint_diagnostic_csv(
    path: str | Path,
    cmd_threshold_deg_s: float = 5.0,
    near_limit_margin_deg: float = 1.0,
) -> DiagnosticSummary:
    """Analyze a direct-joint diagnostic CSV file."""
    csv_path = Path(path)
    rows_by_joint: dict[str, list[dict[str, str]]] = {name: [] for name in JOINT_NAMES_WITH_GRIPPER}
    total_rows = 0

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            joint = row.get("selected_joint", "")
            if joint in rows_by_joint:
                rows_by_joint[joint].append(row)

    per_joint: list[JointSummary] = []
    for joint in JOINT_NAMES_WITH_GRIPPER:
        if not rows_by_joint[joint]:
            continue
        per_joint.append(
            _summarize_joint_rows(
                joint,
                rows_by_joint[joint],
                cmd_threshold_deg_s=max(0.0, cmd_threshold_deg_s),
                near_limit_margin_deg=max(0.0, near_limit_margin_deg),
            )
        )

    return DiagnosticSummary(source=csv_path, total_rows=total_rows, per_joint=per_joint)
