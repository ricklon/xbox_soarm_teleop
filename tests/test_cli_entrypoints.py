"""Tests for packaged CLI entry points."""

import numpy as np

from xbox_soarm_teleop.cli.analyze_joint_diag import build_parser as build_analyze_parser
from xbox_soarm_teleop.cli.diagnose_robot import build_parser as build_diagnose_robot_parser
from xbox_soarm_teleop.cli.joint_rom_test import build_parser as build_joint_rom_parser
from xbox_soarm_teleop.cli.record_xbox import build_parser as build_record_xbox_parser
from xbox_soarm_teleop.cli.simulate_mujoco import (
    STACK_CUBE_COUNT,
    ChallengeManager,
    MuJoCoSimulator,
)
from xbox_soarm_teleop.cli.simulate_mujoco import (
    build_parser as build_simulate_mujoco_parser,
)
from xbox_soarm_teleop.cli.teleoperate_dual import build_parser as build_teleoperate_dual_parser
from xbox_soarm_teleop.cli.teleoperate_real import build_parser as build_teleoperate_real_parser
from xbox_soarm_teleop.cli.xbox_joint_diagnostic import build_parser as build_joint_diag_parser
from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES


def test_analyze_joint_diag_parser() -> None:
    parser = build_analyze_parser()
    args = parser.parse_args(["--input", "joint_diag.csv"])
    assert args.input == "joint_diag.csv"
    assert args.cmd_threshold == 5.0
    assert args.limit_margin == 1.0


def test_diagnose_robot_parser() -> None:
    parser = build_diagnose_robot_parser()
    args = parser.parse_args(["--port", "/dev/ttyUSB0", "--simple"])
    assert args.port == "/dev/ttyUSB0"
    assert args.simple is True
    assert args.motors is None


def test_xbox_joint_diagnostic_parser() -> None:
    parser = build_joint_diag_parser()
    args = parser.parse_args(["--port", "/dev/ttyACM0", "--max-vel", "80"])
    assert args.port == "/dev/ttyACM0"
    assert args.max_vel == 80.0
    assert args.hz == 30.0
    assert args.no_test_positions is False


def test_joint_rom_parser() -> None:
    parser = build_joint_rom_parser()
    args = parser.parse_args(["--sim", "--skip-gripper"])
    assert args.sim is True
    assert args.skip_gripper is True
    assert args.sweep_speed == 20.0


def test_teleoperate_real_parser() -> None:
    parser = build_teleoperate_real_parser()
    args = parser.parse_args(["--port", "/dev/ttyUSB0", "--mode", "cartesian"])
    assert args.port == "/dev/ttyUSB0"
    assert args.mode == "cartesian"
    assert args.controller == "xbox"


def test_simulate_mujoco_parser() -> None:
    parser = build_simulate_mujoco_parser()
    args = parser.parse_args(["--controller", "keyboard", "--mode", "joint"])
    assert args.controller == "keyboard"
    assert args.mode == "joint"
    assert args.no_controller is False
    assert args.challenge_layout == "random"
    assert args.camera_view == "front_right"


def test_simulate_mujoco_stack_layout_parser() -> None:
    parser = build_simulate_mujoco_parser()
    args = parser.parse_args(["--challenge", "--challenge-layout", "stack"])
    assert args.challenge is True
    assert args.challenge_layout == "stack"


class _FakeKinematics:
    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = np.array([0.20, 0.0, 0.20], dtype=float)
        return pose


def test_challenge_manager_builds_diagnostic_targets() -> None:
    limits_deg = {name: (-90.0, 90.0) for name in IK_JOINT_NAMES}
    challenge = ChallengeManager(
        kinematics=_FakeKinematics(),
        joint_limits_deg=limits_deg,
        layout="diagnostic",
    )
    labels = [label for label, _ in challenge.diagnostic_targets]
    assert labels == ["forward", "back", "left", "right", "up", "down"]


def test_challenge_collection_uses_box_boundary() -> None:
    limits_deg = {name: (-90.0, 90.0) for name in IK_JOINT_NAMES}
    challenge = ChallengeManager(
        kinematics=_FakeKinematics(),
        joint_limits_deg=limits_deg,
        target_size=0.02,
        collect_radius=0.04,
        layout="diagnostic",
    )
    target = challenge.active_targets[0] if challenge.active_targets else None
    if target is None:
        challenge.start()
        target = challenge.active_targets[0]

    touch_from_outside = target.position + np.array([0.039, 0.0, 0.0], dtype=float)
    collected = challenge.update(touch_from_outside, dt=0.02)
    assert len(collected) == 1
    assert collected[0].label == target.label


def test_challenge_collection_accepts_any_touch_probe() -> None:
    limits_deg = {name: (-90.0, 90.0) for name in IK_JOINT_NAMES}
    challenge = ChallengeManager(
        kinematics=_FakeKinematics(),
        joint_limits_deg=limits_deg,
        target_size=0.02,
        collect_radius=0.04,
        layout="diagnostic",
    )
    challenge.start()
    target = challenge.active_targets[0]
    center_miss = target.position + np.array([0.20, 0.0, 0.0], dtype=float)
    side_touch = target.position + np.array([0.039, 0.0, 0.0], dtype=float)
    collected = challenge.update_with_touch_points(
        center_miss,
        dt=0.02,
        touch_points=[side_touch],
    )
    assert len(collected) == 1
    assert collected[0].label == target.label


def test_stack_scene_compiles() -> None:
    sim = MuJoCoSimulator("assets/so101_abs.urdf", scene="stack")
    assert sim.has_stack_scene() is True
    assert len(sim.stack_cube_body_ids) == STACK_CUBE_COUNT


def test_teleoperate_dual_parser() -> None:
    parser = build_teleoperate_dual_parser()
    args = parser.parse_args(["--port", "/dev/ttyUSB0", "--challenge"])
    assert args.port == "/dev/ttyUSB0"
    assert args.challenge is True
    assert args.motion_routine is False


def test_record_xbox_parser() -> None:
    parser = build_record_xbox_parser()
    args = parser.parse_args(
        ["--repo-id", "user/dataset", "--robot-port", "/dev/ttyUSB0", "--task", "stack block"]
    )
    assert args.repo_id == "user/dataset"
    assert args.robot_port == "/dev/ttyUSB0"
    assert args.task == "stack block"
    assert args.mode == "crane"
