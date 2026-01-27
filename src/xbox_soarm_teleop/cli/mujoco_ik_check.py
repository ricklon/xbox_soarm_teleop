#!/usr/bin/env python3
"""Run MuJoCo IK check with logging and thresholds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="MuJoCo IK check helper")
    parser.add_argument("--no-controller", action="store_true", help="Run demo mode")
    parser.add_argument("--deadzone", type=float, default=0.15, help="Controller deadzone (0.0-1.0).")
    parser.add_argument("--linear-scale", type=float, default=None, help="Linear velocity scale (m/s).")
    parser.add_argument("--debug-ik", action="store_true", help="Print IK target/achieved error.")
    parser.add_argument("--debug-ik-every", type=int, default=10, help="Print IK debug every N loops.")
    parser.add_argument("--ik-log", type=str, default="ik_error.csv", help="CSV output path.")
    parser.add_argument("--ik-max-err-mm", type=float, default=30.0, help="Fail if max error exceeds (mm).")
    parser.add_argument("--ik-mean-err-mm", type=float, default=10.0, help="Fail if mean error exceeds (mm).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    sim_path = repo_root / "examples" / "simulate_mujoco.py"
    if not sim_path.exists():
        print(f"ERROR: {sim_path} not found")
        return 2

    # Execute the script in-process to preserve exit code.
    argv = [
        str(sim_path),
        "--ik-log",
        args.ik_log,
        "--ik-max-err-mm",
        str(args.ik_max_err_mm),
        "--ik-mean-err-mm",
        str(args.ik_mean_err_mm),
        "--debug-ik-every",
        str(args.debug_ik_every),
    ]
    if args.no_controller:
        argv.append("--no-controller")
    if args.debug_ik:
        argv.append("--debug-ik")
    if args.linear_scale is not None:
        argv.extend(["--linear-scale", str(args.linear_scale)])
    if args.deadzone is not None:
        argv.extend(["--deadzone", str(args.deadzone)])

    globals_dict = {"__file__": str(sim_path), "__name__": "__main__"}
    try:
        code = compile(sim_path.read_text(), str(sim_path), "exec")
        exec(code, globals_dict)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
