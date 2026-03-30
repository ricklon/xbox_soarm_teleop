"""Record teleoperation demonstrations to a LeRobotDataset.

Uses XboxTeleoperator (joint or crane mode) or a cartesian control loop
to control the robot while recording joint-position observations and actions
to a LeRobotDataset.

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
- Cartesian mode uses IK with a 4-joint chain and drives wrist_roll directly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    JOINT_NAMES_WITH_GRIPPER,
)
from xbox_soarm_teleop.control.units import deg_to_normalized
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta
from xbox_soarm_teleop.recording.features import build_dataset_features, build_schema_metadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF = PROJECT_ROOT / "assets" / "so101_abs.urdf"


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


def _pose_to_array(pose: np.ndarray) -> np.ndarray:
    return pose.astype(np.float32).reshape(-1)


def _build_cartesian_controller(controller_type: str):
    from xbox_soarm_teleop.config.joycon_config import JoyConConfig
    from xbox_soarm_teleop.config.keyboard_config import KeyboardConfig
    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta
    from xbox_soarm_teleop.teleoperators.joycon import JoyConController
    from xbox_soarm_teleop.teleoperators.keyboard import KeyboardController
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    if controller_type == "joycon":
        cfg = JoyConConfig()
        controller = JoyConController(cfg)
        proc_cfg = cfg
    elif controller_type == "keyboard":
        cfg = KeyboardConfig()
        controller = KeyboardController(cfg)
        proc_cfg = XboxConfig()
    else:
        cfg = XboxConfig()
        controller = XboxController(cfg)
        proc_cfg = cfg

    mapper = MapXboxToEEDelta(
        linear_scale=proc_cfg.linear_scale,
        angular_scale=proc_cfg.angular_scale,
        orientation_scale=proc_cfg.orientation_scale,
        invert_pitch=proc_cfg.invert_pitch,
        invert_yaw=proc_cfg.invert_yaw,
    )
    return controller, mapper, proc_cfg


def _ee_delta_to_action_dict(delta: EEDelta) -> dict:
    return {
        "dx": delta.dx,
        "dy": delta.dy,
        "dz": delta.dz,
        "droll": delta.droll,
        "dpitch": delta.dpitch,
        "dyaw": delta.dyaw,
        "gripper": delta.gripper,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
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
        choices=["joint", "crane", "cartesian"],
        default="crane",
        help="Control mode for recording.",
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
        "--no-swap-xy",
        action="store_true",
        help="Disable XY swap for Cartesian mapping (cartesian mode only).",
    )
    p.add_argument(
        "--no-strict-safety",
        action="store_true",
        help="Disable strict workspace limits (cartesian mode only).",
    )
    p.add_argument(
        "--ik-vel-scale",
        type=float,
        default=1.0,
        help="Scale factor for IK joint velocity limits (cartesian mode only).",
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
    swap_xy: bool = True,
    strict_safety: bool = True,
    ik_vel_scale: float = 1.0,
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
        swap_xy: Swap Cartesian X/Y axes (cartesian mode only).
        strict_safety: Use strict workspace bounds (cartesian mode only).
        ik_vel_scale: Scale IK joint velocity limits (cartesian mode only).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    from xbox_soarm_teleop.teleoperators.config_xbox_teleop import XboxTeleopConfig
    from xbox_soarm_teleop.teleoperators.xbox_teleop import XboxTeleoperator

    if urdf_path is None:
        urdf_path = _find_urdf()
    if urdf_path is None and DEFAULT_URDF.exists():
        urdf_path = str(DEFAULT_URDF)

    # ---- Dataset features --------------------------------------------------
    features = build_dataset_features(mode=mode, joint_count=len(JOINT_NAMES_WITH_GRIPPER))
    features["observation.state"]["names"] = [f"{m}.pos" for m in JOINT_NAMES_WITH_GRIPPER]
    features["action"]["names"] = [f"{m}.pos" for m in JOINT_NAMES_WITH_GRIPPER]
    if mode == "cartesian":
        features["action.ee_delta"]["names"] = [
            "dx",
            "dy",
            "dz",
            "droll",
            "dpitch",
            "dyaw",
            "gripper",
        ]
        features["safety.flags"]["names"] = [
            "ws_clip_x",
            "ws_clip_y",
            "ws_clip_z",
            "speed_clip",
            "orient_clip",
            "joint_clip",
            "reject",
        ]

    dataset_kwargs: dict = {"features": features}
    if root is not None:
        dataset_kwargs["root"] = root

    dataset = LeRobotDataset.create(repo_id, fps=fps, **dataset_kwargs)
    schema_path = dataset.root / "meta" / "schema.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        json.dumps(
            build_schema_metadata(mode=mode, joint_names=JOINT_NAMES_WITH_GRIPPER),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # ---- Build robot -------------------------------------------------------
    robot_cfg = SOFollowerRobotConfig(port=robot_port)
    robot = SOFollower(robot_cfg)

    dt = 1.0 / fps
    episode = 0
    total_frames = 0

    controller = None
    if mode != "cartesian":
        # ---- Build teleoperator ------------------------------------------------
        teleop_cfg = XboxTeleopConfig(
            id=f"{controller_type}_0",
            mode=mode,
            urdf_path=urdf_path,
            controller_type=controller_type,
        )
        teleop = XboxTeleoperator(teleop_cfg)
        print(f"Connecting {controller_type} controller …")
        teleop.connect()
    else:
        teleop = None
        controller, mapper, proc_cfg = _build_cartesian_controller(controller_type)
        print(f"Connecting {controller_type} controller …")
        if not controller.connect():
            raise RuntimeError(f"Failed to connect {controller_type} controller.")

        from lerobot.processor.converters import (
            robot_action_observation_to_transition,
            transition_to_robot_action,
        )
        from lerobot.processor.pipeline import RobotProcessorPipeline

        from xbox_soarm_teleop.lerobot_steps.cartesian_ik import SoArmCartesianIKProcessor

        if urdf_path is None:
            raise RuntimeError("URDF path is required for cartesian recording.")

        teleop_action_processor = RobotProcessorPipeline(
            steps=[
                SoArmCartesianIKProcessor(
                    urdf_path=urdf_path,
                    dt=dt,
                    swap_xy=swap_xy,
                    strict_safety=strict_safety,
                    ik_vel_scale=ik_vel_scale,
                    gripper_rate=float(getattr(proc_cfg, "gripper_rate", 2.0)),
                )
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
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

                if mode != "cartesian":
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
                else:
                    state = controller.read()
                    if state.a_button_pressed:
                        home_action = {
                            f"{name}.pos": deg_to_normalized(HOME_POSITION_DEG[name], name)
                            for name in JOINT_NAMES_WITH_GRIPPER[:-1]
                        }
                        home_action["gripper.pos"] = 0.0
                        robot.send_action(home_action)
                        teleop_action_processor.reset()
                        continue

                    ee_delta = mapper(state)
                    action_input = _ee_delta_to_action_dict(ee_delta)
                    obs_before = robot.get_observation()
                    action = teleop_action_processor((action_input, obs_before))
                    robot.send_action(action)
                    obs = robot.get_observation()

                    ik_step = teleop_action_processor.steps[0]
                    ee_target_pose = ik_step.last_target_pose
                    ee_obs_pose = ik_step.last_obs_pose
                    if ee_target_pose is None:
                        ee_target_pose = np.eye(4, dtype=np.float32)
                    if ee_obs_pose is None:
                        ee_obs_pose = np.eye(4, dtype=np.float32)
                    flags_order = [
                        "ws_clip_x",
                        "ws_clip_y",
                        "ws_clip_z",
                        "speed_clip",
                        "orient_clip",
                        "joint_clip",
                        "reject",
                    ]
                    safety_flags = np.array(
                        [ik_step.last_flags.get(k, 0) for k in flags_order], dtype=np.float32
                    )
                    ee_delta = ik_step.last_delta or EEDelta()

                    frame = {
                        "observation.state": _obs_to_array(obs),
                        "action": _action_to_array(action),
                        "action.ee_delta": np.array(
                            [
                                ee_delta.dx,
                                ee_delta.dy,
                                ee_delta.dz,
                                ee_delta.droll,
                                ee_delta.dpitch,
                                ee_delta.dyaw,
                                ee_delta.gripper,
                            ],
                            dtype=np.float32,
                        ),
                        "action.ee_target": _pose_to_array(ee_target_pose),
                        "observation.ee_pose": _pose_to_array(ee_obs_pose),
                        "safety.flags": safety_flags,
                        "task": task,
                    }
                    dataset.add_frame(frame)
                    frames.append(frame)
                    total_frames += 1

                    if getattr(state, "y_button", False):
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
        if teleop is not None:
            teleop.disconnect()
        if controller is not None:
            controller.disconnect()
        robot.disconnect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
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
        swap_xy=not args.no_swap_xy,
        strict_safety=not args.no_strict_safety,
        ik_vel_scale=args.ik_vel_scale,
    )


if __name__ == "__main__":
    main()
