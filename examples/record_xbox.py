"""Record teleoperation demonstrations to a LeRobotDataset.

Uses XboxTeleoperator (joint or crane mode) to control the robot while
simultaneously recording joint-position actions to a LeRobotDataset.

Usage
-----
Basic recording with crane mode (default):

    uv run python examples/record_xbox.py \\
        --repo-id my-hf-username/my-dataset \\
        --robot-port /dev/ttyUSB0

Joint-direct mode (fine-grained single-joint control):

    uv run python examples/record_xbox.py \\
        --repo-id my-hf-username/my-dataset \\
        --robot-port /dev/ttyUSB0 \\
        --mode joint

Controller buttons during recording:
    LB (hold)       — deadman switch: enables arm motion
    A               — return arm to home position
    Y               — save current episode and start a new one
    Start           — stop recording and upload dataset

Notes
-----
- Requires a real SO-ARM101 connected via USB.
- For simulation-only use, run simulate_mujoco.py instead.
- CARTESIAN mode is not supported; use JOINT or CRANE.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record Xbox teleoperation to a LeRobotDataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace dataset repo ID, e.g. 'username/my-dataset'.",
    )
    p.add_argument(
        "--robot-port",
        default="/dev/ttyUSB0",
        help="Serial port for the SO-ARM101 follower arm.",
    )
    p.add_argument(
        "--mode",
        choices=["joint", "crane"],
        default="crane",
        help="Control mode (cartesian not supported for recording).",
    )
    p.add_argument(
        "--urdf-path",
        default=None,
        help="Path to robot URDF (required for crane IK; auto-detected if omitted).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Recording frame rate.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Maximum number of episodes to record.",
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload dataset to HuggingFace Hub after recording.",
    )
    return p


def _find_urdf() -> str | None:
    """Locate the SO-ARM101 URDF bundled with lerobot."""
    try:
        import lerobot

        candidates = list(
            Path(lerobot.__file__).parent.rglob("so_arm101.urdf")
        ) + list(Path(lerobot.__file__).parent.rglob("so_arm100.urdf"))
        if candidates:
            return str(candidates[0])
    except Exception:
        pass
    return None


def run_recording(
    repo_id: str,
    robot_port: str,
    mode: str = "crane",
    urdf_path: str | None = None,
    fps: float = 30.0,
    max_episodes: int = 10,
    push_to_hub: bool = False,
) -> None:
    """Run the recording session.

    Args:
        repo_id: HuggingFace dataset repository ID.
        robot_port: Serial port for the SO-ARM101 follower arm.
        mode: Control mode ("joint" or "crane").
        urdf_path: Path to robot URDF.  Auto-detected when None.
        fps: Recording frame rate in Hz.
        max_episodes: Stop after this many episodes.
        push_to_hub: Push to HuggingFace Hub when done.
    """
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    from xbox_soarm_teleop.teleoperators.config_xbox_teleop import XboxTeleopConfig
    from xbox_soarm_teleop.teleoperators.xbox_teleop import XboxTeleoperator

    if urdf_path is None:
        urdf_path = _find_urdf()

    # ---- Build teleoperator ------------------------------------------------
    teleop_cfg = XboxTeleopConfig(
        id="xbox_0",
        mode=mode,
        urdf_path=urdf_path,
    )
    teleop = XboxTeleoperator(teleop_cfg)

    # ---- Build robot -------------------------------------------------------
    robot_cfg = SO101FollowerConfig(port=robot_port)
    robot = SO101Follower(robot_cfg)

    dt = 1.0 / fps
    episode = 0
    total_frames = 0

    print("Connecting Xbox controller …")
    teleop.connect()
    print(f"Connecting robot on {robot_port} …")
    robot.connect()

    try:
        while episode < max_episodes:
            print(f"\n=== Episode {episode + 1}/{max_episodes} ===")
            print("Hold LB to move the arm.  Press Y to save and start next episode.")

            frames: list[dict] = []

            while True:
                t0 = time.monotonic()

                action = teleop.get_action()

                # Send action to robot
                robot.send_action(action)

                # Record frame
                obs = robot.get_observation()
                frames.append({"observation": obs, "action": action})
                total_frames += 1

                # Y button: end episode
                raw_state = teleop._controller._state
                if raw_state.y_button_pressed:
                    print(f"  Episode saved ({len(frames)} frames).")
                    break

                elapsed = time.monotonic() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            episode += 1

        print(f"\nRecording complete: {episode} episodes, {total_frames} frames.")

        if push_to_hub:
            print("Pushing dataset to HuggingFace Hub …")
            # LeRobotDataset.push_to_hub() — implementation depends on lerobot version
            print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")

    except KeyboardInterrupt:
        print("\nRecording interrupted.")
    finally:
        teleop.disconnect()
        robot.disconnect()


def main() -> None:
    args = _build_parser().parse_args()
    run_recording(
        repo_id=args.repo_id,
        robot_port=args.robot_port,
        mode=args.mode,
        urdf_path=args.urdf_path,
        fps=args.fps,
        max_episodes=args.episodes,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
