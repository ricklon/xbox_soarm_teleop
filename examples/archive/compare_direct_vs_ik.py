#!/usr/bin/env python3
"""Compare direct-joint diagnostic capability against IK run behavior.

Usage:
    uv run python examples/archive/compare_direct_vs_ik.py \
      --joint-log joint_diag_20260215_212931.csv \
      --ik-log ik_error.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from xbox_soarm_teleop.diagnostics.joint_diag_analysis import analyze_joint_diagnostic_csv


def read_ik_log(path: Path) -> dict[str, float]:
    rows = 0
    clipped = 0
    errs = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            try:
                errs.append(float(row.get("pos_err_mm", "0") or 0.0))
            except ValueError:
                pass

            clipped_raw = (row.get("clipped", "0") or "0").strip().lower()
            if clipped_raw in {"1", "1.0", "true"}:
                clipped += 1

    mean_err = (sum(errs) / len(errs)) if errs else 0.0
    max_err = max(errs) if errs else 0.0
    clip_rate = (clipped / rows * 100.0) if rows else 0.0

    return {
        "rows": float(rows),
        "clipped_rows": float(clipped),
        "clip_rate_pct": clip_rate,
        "mean_err_mm": mean_err,
        "max_err_mm": max_err,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct-joint and IK logs")
    parser.add_argument("--joint-log", required=True, help="Path to joint_diag_*.csv")
    parser.add_argument("--ik-log", required=True, help="Path to ik_error.csv")
    args = parser.parse_args()

    joint_log = Path(args.joint_log)
    ik_log = Path(args.ik_log)

    if not joint_log.exists():
        raise SystemExit(f"Missing joint log: {joint_log}")
    if not ik_log.exists():
        raise SystemExit(f"Missing IK log: {ik_log}")

    joint_summary = analyze_joint_diagnostic_csv(joint_log)
    ik_summary = read_ik_log(ik_log)

    if not joint_summary.per_joint:
        raise SystemExit("No valid joint rows found in direct-joint log")

    util_values = [j.span_utilization_pct for j in joint_summary.per_joint if j.joint != "gripper"]
    speed_ratios = []
    for j in joint_summary.per_joint:
        if j.cmd_abs_peak_deg_s > 1e-6:
            speed_ratios.append(j.measured_abs_peak_deg_s / j.cmd_abs_peak_deg_s)

    avg_util = sum(util_values) / len(util_values) if util_values else 0.0
    avg_ratio = sum(speed_ratios) / len(speed_ratios) if speed_ratios else 0.0

    print("Direct vs IK Comparison")
    print("=" * 72)
    print(f"Direct-joint log: {joint_log}")
    print(f"IK log:           {ik_log}")
    print()

    print("Direct-joint capability")
    print(f"  samples: {joint_summary.total_rows}")
    print(f"  mean non-gripper ROM utilization: {avg_util:.1f}%")
    print(f"  mean measured/cmd peak-speed ratio: {avg_ratio:.2f}x")

    print("\nIK run behavior")
    print(f"  samples: {int(ik_summary['rows'])}")
    print(f"  clipped samples: {int(ik_summary['clipped_rows'])}")
    print(f"  clip rate: {ik_summary['clip_rate_pct']:.1f}%")
    print(f"  mean position error: {ik_summary['mean_err_mm']:.2f} mm")
    print(f"  max position error: {ik_summary['max_err_mm']:.2f} mm")

    print("\nInterpretation")
    if avg_util > 70.0 and avg_ratio > 0.8:
        print("  Direct mode shows strong hardware motion capability.")
    else:
        print("  Direct mode does not show strong hardware headroom yet.")

    if ik_summary["clip_rate_pct"] > 5.0:
        print("  IK clipping is non-trivial; likely limiting perceived motion speed.")
    else:
        print("  IK clipping is low; perceived slowness is likely from other limits/scales.")

    if ik_summary["max_err_mm"] > 25.0:
        print("  IK occasionally misses target significantly (check workspace/orientation demands).")


if __name__ == "__main__":
    main()
