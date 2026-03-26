"""Record teleoperation demonstrations to a LeRobotDataset.

Uses XboxTeleoperator (joint or crane mode) to control the robot while
simultaneously recording joint-position observations and actions to a
LeRobotDataset.

Usage
-----
Basic recording with crane mode (default):

    uv run python examples/record_xbox.py \\
        --repo-id my-hf-username/my-dataset \\
        --robot-port /dev/ttyUSB0 \\
        --task "pick up the red block"

Joint-direct mode with keyboard controller:

    uv run python examples/record_xbox.py \\
        --repo-id my-hf-username/my-dataset \\
        --robot-port /dev/ttyUSB0 \\
        --mode joint \\
        --controller keyboard \\
        --task "move joint 1 to 45 degrees"

Controller buttons during recording:
    LB (hold)       — deadman switch: enables arm motion  (Xbox/Joy-Con)
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

import numpy as np

from xbox_soarm_teleop.config.joints import JOINT_NAMES_WITH_GRIPPER

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _obs_to_array(obs: dict) -> np.ndarray:
    """Convert robot observation dict to shape-(6,) float32 array."""
    return np.array(
        [obs[f"{m}.pos"] for m in JOINT_NAMES_WITH_GRIPPER], dtype=np.float32
    )


def _action_to_array(action: dict) -> np.ndarray:
    """Convert action dict to shape-(6,) float32 array."""
    return np.array(
        [action[f"{m}.pos"] for m in JOINT_NAMES_WITH_GRIPPER], dtype=np.float32
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        "--controller",
        choices=["xbox", "joycon", "keyboard"],
        default="xbox",
        help="Input device type.",
    )
    p.add_argument(
        "--task",
        required=True,
        help="Natural-language description of the task being demonstrated.",
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
        "--root",
        default=None,
        help="Local root directory for the dataset (default: ~/.cache/huggingface/lerobot).",
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload dataset to HuggingFace Hub after recording.",
    )
    return p


# ---------------------------------------------------------------------------
# Recording session
# ---------------------------------------------------------------------------

def run_recording(
    repo_id: str,
    robot_port: str,
    task: str,
    mode: str = "crane",
    controller_type: str = "xbox",
    urdf_path: str | None = None,
    fps: float = 30.0,
    max_episodes: int = 10,
    root: str | None = None,
    push_to_hub: bool = False,
) -> None:
    """Run the recording session.

    Args:
        repo_id: HuggingFace dataset repository ID.
        robot_port: Serial port for the SO-ARM101 follower arm.
        task: Natural-language task description stored with every frame.
        mode: Control mode ("joint" or "crane").
        controller_type: Input device ("xbox", "joycon", or "keyboard").
        urdf_path: Path to robot URDF.  Auto-detected when None.
        fps: Recording frame rate in Hz.
        max_episodes: Stop after this many episodes.
        root: Local dataset root directory.
        push_to_hub: Push to HuggingFace Hub when done.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    from xbox_soarm_teleop.teleoperators.config_xbox_teleop import XboxTeleopConfig
    from xbox_soarm_teleop.teleoperators.xbox_teleop import XboxTeleoperator

    if urdf_path is None:
        urdf_path = _find_urdf()

    # ---- Dataset features --------------------------------------------------
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES_WITH_GRIPPER),),
            "names": [f"{m}.pos" for m in JOINT_NAMES_WITH_GRIPPER],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES_WITH_GRIPPER),),
            "names": [f"{m}.pos" for m in JOINT_NAMES_WITH_GRIPPER],
        },
    }

    dataset_kwargs: dict = {"features": features}
    if root is not None:
        dataset_kwargs["root"] = root

    dataset = LeRobotDataset.create(repo_id, fps=fps, **dataset_kwargs)

    # ---- Build teleoperator ------------------------------------------------
    teleop_cfg = XboxTeleopConfig(
        id=f"{controller_type}_0",
        mode=mode,
        urdf_path=urdf_path,
        controller_type=controller_type,
    )
    teleop = XboxTeleoperator(teleop_cfg)

    # ---- Build robot -------------------------------------------------------
    robot_cfg = SOFollowerRobotConfig(port=robot_port)
    robot = SOFollower(robot_cfg)

    dt = 1.0 / fps
    episode = 0
    total_frames = 0

    print(f"Connecting {controller_type} controller …")
    teleop.connect()
    print(f"Connecting robot on {robot_port} …")
    robot.connect()

    try:
        while episode < max_episodes:
            print(f"\n=== Episode {episode + 1}/{max_episodes} ===")
            if controller_type == "keyboard":
                print("Use keyboard to move the arm.  Press Y to save and start next episode.")
            else:
                print("Hold LB to move the arm.  Press Y to save and start next episode.")

            frames: list[dict] = []

            while True:
                t0 = time.monotonic()

                action = teleop.get_action()

                # Send action to robot and record observation
                robot.send_action(action)
                obs = robot.get_observation()

                frame = {
                    "observation.state": _obs_to_array(obs),
                    "action": _action_to_array(action),
                    "task": task,
                }
                dataset.add_frame(frame)
                frames.append(frame)
                total_frames += 1

                # Y button (or Y key): end episode
                raw_state = teleop._controller.read()
                if getattr(raw_state, "y_button", False):
                    dataset.save_episode(task=task)
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
            dataset.push_to_hub()
            print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")

    except KeyboardInterrupt:
        print("\nRecording interrupted.")
        if frames:
            print("Saving partial episode …")
            dataset.save_episode(task=task)
    finally:
        teleop.disconnect()
        robot.disconnect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    run_recording(
        repo_id=args.repo_id,
        robot_port=args.robot_port,
        task=args.task,
        mode=args.mode,
        controller_type=args.controller,
        urdf_path=args.urdf_path,
        fps=args.fps,
        max_episodes=args.episodes,
        root=args.root,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
