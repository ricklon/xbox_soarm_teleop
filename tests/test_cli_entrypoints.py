"""Tests for packaged CLI entry points."""

from xbox_soarm_teleop.cli.analyze_joint_diag import build_parser as build_analyze_parser
from xbox_soarm_teleop.cli.diagnose_robot import build_parser as build_diagnose_robot_parser
from xbox_soarm_teleop.cli.joint_rom_test import build_parser as build_joint_rom_parser
from xbox_soarm_teleop.cli.record_xbox import build_parser as build_record_xbox_parser
from xbox_soarm_teleop.cli.simulate_mujoco import build_parser as build_simulate_mujoco_parser
from xbox_soarm_teleop.cli.teleoperate_dual import build_parser as build_teleoperate_dual_parser
from xbox_soarm_teleop.cli.teleoperate_real import build_parser as build_teleoperate_real_parser
from xbox_soarm_teleop.cli.xbox_joint_diagnostic import build_parser as build_joint_diag_parser


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
