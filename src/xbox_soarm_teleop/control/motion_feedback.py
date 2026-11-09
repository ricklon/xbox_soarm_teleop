"""User-facing motion feedback.

Explains why the arm is not following a command (workspace bound, joint limit,
unreachable target, homing) and summarizes IK tracking error without counting
deliberate joint-space moves such as homing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES, JOINT_LIMITS_DEG
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta

WorkspaceBounds = dict[str, tuple[float | None, float | None]]

_LINEAR_EPS = 1e-6
_BOUND_TOL_M = 1e-6
_JOINT_LIMIT_TOL_DEG = 0.5
_STALLED_PROGRESS = 0.5

# (axis index, axis key, side, label)
_BOUND_LABELS = (
    (2, "z", 0, "at floor"),
    (2, "z", 1, "at ceiling"),
    (0, "x", 1, "at front limit"),
    (0, "x", 0, "at back limit"),
    (1, "y", 0, "at side limit"),
    (1, "y", 1, "at side limit"),
)

_IK_REASON_LABELS = {
    "not_converged": "out of reach",
    "joint_limit": "joint limit",
    "invalid_output": "IK error",
    "solver_error": "IK error",
}


def commanded_progress(
    before_pos: np.ndarray, after_pos: np.ndarray, delta: EEDelta, dt: float
) -> float | None:
    """Fraction of the commanded linear displacement achieved this frame.

    Returns None when no linear motion was commanded.
    """
    commanded = np.array([delta.dx, delta.dy, delta.dz], dtype=float) * dt
    length = float(np.linalg.norm(commanded))
    if length <= _LINEAR_EPS * dt:
        return None
    moved = np.asarray(after_pos, dtype=float) - np.asarray(before_pos, dtype=float)
    return float(np.dot(moved, commanded) / (length * length))


def describe_motion_limit(
    *,
    homing: bool,
    delta: EEDelta,
    target_pos: np.ndarray,
    workspace_bounds: WorkspaceBounds,
    ik_joints_deg: np.ndarray,
    ik_result=None,
    progress: float | None = None,
) -> str:
    """Return a short reason the arm is not moving as commanded, or "".

    Only limits the operator is pushing against are reported: an arm resting at
    the floor with no downward command reports nothing.

    Args:
        homing: True while a home move is in progress.
        delta: This frame's end-effector command.
        target_pos: Commanded tool position after workspace clipping (m).
        workspace_bounds: ``{"x"|"y"|"z": (low, high)}``; ``None`` means unbounded.
        ik_joints_deg: IK-chain joint angles after this frame, ordered as ``IK_JOINT_NAMES``.
        ik_result: The solver's ``last_ik_result`` for this frame, if IK ran.
        progress: Fraction of the commanded displacement the tool achieved this
            frame. A joint resting at its limit is only reported when progress is
            below ``_STALLED_PROGRESS`` (the limit is actually blocking motion).

    Returns:
        A message such as ``"at floor"``, ``"joint limit: wrist_flex"``,
        ``"out of reach"`` or ``"homing"``; empty when motion is unconstrained.
    """
    if homing:
        return "homing"

    linear = np.array([delta.dx, delta.dy, delta.dz], dtype=float)
    if not np.any(np.abs(linear) > _LINEAR_EPS):
        return ""

    pos = np.asarray(target_pos, dtype=float)
    for idx, key, side, label in _BOUND_LABELS:
        bound = workspace_bounds.get(key, (None, None))[side]
        if bound is None:
            continue
        pushing = linear[idx] < -_LINEAR_EPS if side == 0 else linear[idx] > _LINEAR_EPS
        at_bound = pos[idx] <= bound + _BOUND_TOL_M if side == 0 else pos[idx] >= bound - _BOUND_TOL_M
        if pushing and at_bound:
            return label

    stalled = progress is None or progress < _STALLED_PROGRESS
    joints = np.asarray(ik_joints_deg, dtype=float)
    for name, angle in zip(IK_JOINT_NAMES, joints if stalled else []):
        lower, upper = JOINT_LIMITS_DEG[name]
        if angle <= lower + _JOINT_LIMIT_TOL_DEG or angle >= upper - _JOINT_LIMIT_TOL_DEG:
            return f"joint limit: {name}"

    if ik_result is not None and not ik_result.accepted:
        return _IK_REASON_LABELS.get(ik_result.reason, "IK error")
    return ""


@dataclass
class TrackingStats:
    """IK tracking-error statistics with a verdict that excludes homing.

    Homing interpolates in joint space toward a fixed home pose, so its
    "error" is distance-to-home rather than tracking error.

    Args:
        max_err_mm: Fail if the largest tracked error exceeds this.
        mean_err_mm: Fail if the mean tracked error exceeds this.
    """

    max_err_mm: float
    mean_err_mm: float
    _errors: list[float] = field(default_factory=list, repr=False)
    _phases: Counter = field(default_factory=Counter, repr=False)
    limit_counts: dict[str, int] = field(default_factory=dict)

    def add(self, err_m: float, *, phase: str, limit: str = "") -> None:
        """Record one sample.

        Args:
            err_m: Target-to-actual tool distance in meters.
            phase: ``"moving"``, ``"idle"`` or ``"homing"``.
            limit: Message from :func:`describe_motion_limit`, if any.
        """
        self._phases[phase] += 1
        if phase != "homing":
            self._errors.append(float(err_m))
        if limit and limit != "homing":
            self.limit_counts[limit] = self.limit_counts.get(limit, 0) + 1

    def count(self, phase: str) -> int:
        """Number of samples recorded in a phase."""
        return self._phases[phase]

    def max_mm(self) -> float:
        """Largest tracked error in millimeters (0 with no tracked samples)."""
        return max(self._errors, default=0.0) * 1000.0

    def mean_mm(self) -> float:
        """Mean tracked error in millimeters (0 with no tracked samples)."""
        return float(np.mean(self._errors)) * 1000.0 if self._errors else 0.0

    @property
    def passed(self) -> bool:
        """True when tracked error is within both thresholds."""
        return self.max_mm() <= self.max_err_mm and self.mean_mm() <= self.mean_err_mm

    def summary_lines(self) -> list[str]:
        """Human-readable summary ending with ``PASS`` or a ``FAIL`` line."""
        lines = ["IK error summary (kinematics FK)"]
        if not self._errors:
            lines.append("  no tracked samples (nothing to evaluate)")
        else:
            lines += [
                f"  tracked samples: {len(self._errors)} "
                f"(moving {self.count('moving')}, idle {self.count('idle')})",
                f"  max position error: {self.max_mm():.1f} mm",
                f"  mean position error: {self.mean_mm():.1f} mm",
            ]
        if self.count("homing"):
            lines.append(
                f"  homing samples excluded: {self.count('homing')} "
                "(joint-space move; error there is distance to home)"
            )
        if self.limit_counts:
            stops = ", ".join(f"{name} x{n}" for name, n in sorted(self.limit_counts.items()))
            lines.append(f"  limit stops (frames): {stops}")
        lines.append("PASS" if self.passed else "FAIL: IK error exceeded thresholds.")
        return lines
