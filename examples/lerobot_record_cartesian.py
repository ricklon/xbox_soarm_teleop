#!/usr/bin/env python3
"""Convenience wrapper for cartesian dataset recording.

This keeps a dedicated cartesian-recording command without depending on
LeRobot's teleoperator registry path, which currently trips over an upstream
teleoperator import cycle in standalone use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xbox_soarm_teleop.cli.record_xbox import run_recording


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record cartesian teleop demonstrations with the project recorder."
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repo id (user/name).")
    parser.add_argument("--task", required=True, help="Task description.")
    parser.add_argument("--robot-port", required=True, help="Robot serial port.")
    parser.add_argument(
        "--controller",
        choices=["xbox", "joycon", "dual_joycon", "keyboard"],
        default="xbox",
        help="Input device type.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Recording fps.")
    parser.add_argument("--episodes", type=int, default=10, help="Maximum number of episodes.")
    parser.add_argument(
        "--urdf-path",
        default=str(Path(__file__).parent.parent / "assets" / "so101_abs.urdf"),
        help="URDF path for IK.",
    )
    parser.add_argument("--no-swap-xy", action="store_true", help="Disable XY swap for mapping.")
    parser.add_argument(
        "--no-strict-safety",
        action="store_true",
        help="Disable strict workspace limits.",
    )
    parser.add_argument(
        "--ik-vel-scale",
        type=float,
        default=1.0,
        help="Scale factor for IK joint velocity limits.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Local root directory for the dataset (default: ~/.cache/huggingface/lerobot).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload dataset to HuggingFace Hub after recording.",
    )
    args = parser.parse_args()

    run_recording(
        repo_id=args.repo_id,
        robot_port=args.robot_port,
        task=args.task,
        mode="cartesian",
        controller_type=args.controller,
        urdf_path=args.urdf_path,
        fps=args.fps,
        max_episodes=args.episodes,
        root=args.root,
        push_to_hub=args.push_to_hub,
        swap_xy=not args.no_swap_xy,
        strict_safety=not args.no_strict_safety,
        ik_vel_scale=args.ik_vel_scale,
    )


if __name__ == "__main__":
    main()
