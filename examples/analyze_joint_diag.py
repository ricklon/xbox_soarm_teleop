#!/usr/bin/env python3
"""Analyze direct-joint diagnostic CSV and print per-joint performance metrics.

Usage:
    uv run python examples/analyze_joint_diag.py --input joint_diag_20260216_120000.csv
"""

from __future__ import annotations

import argparse
import sys

from xbox_soarm_teleop.diagnostics.joint_diag_analysis import analyze_joint_diagnostic_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Xbox direct-joint diagnostic CSV")
    parser.add_argument("--input", required=True, help="Path to joint diagnostic CSV")
    parser.add_argument(
        "--cmd-threshold",
        type=float,
        default=5.0,
        help="Minimum |cmd_vel_deg_s| counted as active command (default: 5)",
    )
    parser.add_argument(
        "--limit-margin",
        type=float,
        default=1.0,
        help="Near-limit margin in degrees (default: 1)",
    )
    args = parser.parse_args()

    summary = analyze_joint_diagnostic_csv(
        args.input,
        cmd_threshold_deg_s=args.cmd_threshold,
        near_limit_margin_deg=args.limit_margin,
    )

    if summary.total_rows == 0:
        print("No rows found in CSV")
        sys.exit(1)

    if not summary.per_joint:
        print("No valid joint rows found in CSV")
        sys.exit(1)

    print(f"Source: {summary.source}")
    print(f"Rows:   {summary.total_rows}")
    print("\nPer-joint summary")
    print(
        "joint           samples active cmd_mean cmd_peak meas_mean meas_peak "
        "err_mean err_p95 span_deg util_% near_lim"
    )
    for joint_summary in summary.per_joint:
        print(
            f"{joint_summary.joint:14s} {joint_summary.samples:7d} {joint_summary.active_cmd_samples:6d} "
            f"{joint_summary.cmd_abs_mean_deg_s:8.2f} {joint_summary.cmd_abs_peak_deg_s:8.2f} "
            f"{joint_summary.measured_abs_mean_deg_s:9.2f} {joint_summary.measured_abs_peak_deg_s:9.2f} "
            f"{joint_summary.tracking_err_mean_deg:8.2f} {joint_summary.tracking_err_p95_deg:7.2f} "
            f"{joint_summary.observed_span_deg:8.2f} {joint_summary.span_utilization_pct:6.1f} "
            f"{joint_summary.near_limit_samples:8d}"
        )

    weakest = min(summary.per_joint, key=lambda j: j.span_utilization_pct)
    print("\nQuick flags")
    if weakest.span_utilization_pct < 25.0:
        print(
            f"- Low range usage: {weakest.joint} used {weakest.span_utilization_pct:.1f}% "
            "of its configured limit span"
        )
    else:
        print("- No obvious low-range joint detected")


if __name__ == "__main__":
    main()
