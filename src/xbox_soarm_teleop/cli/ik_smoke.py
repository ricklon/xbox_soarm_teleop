#!/usr/bin/env python3
"""IK smoke test routine for SO-ARM101."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES

URDF_PATH = Path(__file__).resolve().parents[3] / "assets" / "so101_abs.urdf"
EE_FRAME = "gripper_frame_link"


def build_pattern(
    t: float,
    center: np.ndarray,
    amp: np.ndarray,
    freq: np.ndarray,
) -> np.ndarray:
    """Generate a smooth 3D Lissajous-style pattern around center."""
    return center + amp * np.array(
        [
            np.sin(2 * np.pi * freq[0] * t),
            np.cos(2 * np.pi * freq[1] * t),
            np.sin(2 * np.pi * freq[2] * t + np.pi / 4),
        ]
    )


def run_smoke_test(
    duration_s: float,
    step_hz: float,
    max_pos_err_m: float,
    mean_pos_err_m: float,
    verbose: bool,
) -> int:
    """Run the IK smoke test. Returns process exit code."""
    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found at {URDF_PATH}")
        return 2

    try:
        from lerobot.model.kinematics import RobotKinematics
    except Exception as exc:  # pragma: no cover - import error path
        print(f"ERROR: Failed to import RobotKinematics: {exc}")
        return 2

    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=EE_FRAME,
        joint_names=IK_JOINT_NAMES,
    )

    joint_guess = np.zeros(len(IK_JOINT_NAMES))
    home_pose = kinematics.forward_kinematics(joint_guess)

    center = home_pose[:3, 3].copy()
    amp = np.array([0.03, 0.03, 0.02])  # meters
    freq = np.array([0.20, 0.15, 0.12])  # Hz

    dt = 1.0 / step_hz
    steps = max(int(duration_s * step_hz), 1)

    errors = []
    failed = 0

    start = time.monotonic()
    for i in range(steps):
        t = i * dt
        target_pos = build_pattern(t, center, amp, freq)

        target_pose = home_pose.copy()
        target_pose[:3, 3] = target_pos

        try:
            solved = kinematics.inverse_kinematics(joint_guess, target_pose)
        except Exception:
            failed += 1
            continue

        solved = np.array(solved[: len(joint_guess)])
        if np.any(~np.isfinite(solved)):
            failed += 1
            continue

        fk_pose = kinematics.forward_kinematics(solved)
        pos_err = np.linalg.norm(fk_pose[:3, 3] - target_pos)
        errors.append(pos_err)

        joint_guess = solved

        if verbose and (i % max(int(step_hz), 1) == 0):
            elapsed = time.monotonic() - start
            print(
                f"t={elapsed:5.2f}s pos_err={pos_err*1000:5.1f}mm "
                f"fail={failed}/{i+1}",
                flush=True,
            )

    if not errors:
        print("ERROR: No successful IK solutions.")
        return 3

    errors = np.array(errors)
    max_err = float(np.max(errors))
    mean_err = float(np.mean(errors))

    print("\nIK smoke test results")
    print(f"  steps: {steps}")
    print(f"  failures: {failed}")
    print(f"  max position error: {max_err*1000:.1f} mm")
    print(f"  mean position error: {mean_err*1000:.1f} mm")

    if failed > 0:
        print("FAIL: IK failed to solve some targets.")
        return 1
    if max_err > max_pos_err_m or mean_err > mean_pos_err_m:
        print("FAIL: Position error exceeded thresholds.")
        return 1

    print("PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-ARM101 IK smoke test")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds.")
    parser.add_argument("--hz", type=float, default=50.0, help="Steps per second.")
    parser.add_argument("--max-err-mm", type=float, default=30.0, help="Max error threshold (mm).")
    parser.add_argument("--mean-err-mm", type=float, default=10.0, help="Mean error threshold (mm).")
    parser.add_argument("--verbose", action="store_true", help="Print per-second progress.")
    args = parser.parse_args()

    return run_smoke_test(
        duration_s=args.duration,
        step_hz=args.hz,
        max_pos_err_m=args.max_err_mm / 1000.0,
        mean_pos_err_m=args.mean_err_mm / 1000.0,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
