"""Tests for shared runtime/session helpers."""

from pathlib import Path

from xbox_soarm_teleop.config.modes import ControlMode
from xbox_soarm_teleop.processors.joint_direct import JointDirectProcessor
from xbox_soarm_teleop.processors.xbox_to_ee import MapDualJoyConToEEDelta
from xbox_soarm_teleop.runtime import control_help_lines
from xbox_soarm_teleop.runtime.session import build_control_runtime, controller_label
from xbox_soarm_teleop.teleoperators.keyboard import KeyboardController


def test_controller_label_values():
    assert controller_label("xbox") == "Xbox controller"
    assert controller_label("joycon") == "Joy-Con"
    assert controller_label("dual_joycon") == "Dual Joy-Con"
    assert controller_label("keyboard") == "keyboard"
    assert controller_label("custom") == "custom"


def test_build_control_runtime_joint_keyboard():
    runtime = build_control_runtime(
        controller_type="keyboard",
        mode="joint",
        deadzone=0.15,
        linear_scale=0.2,
        keyboard_grab=True,
        keyboard_record=None,
        keyboard_playback=None,
        loop_dt=1.0 / 30.0,
        urdf_path="unused",
    )

    assert runtime.control_mode is ControlMode.JOINT
    assert runtime.controller_label == "keyboard"
    assert isinstance(runtime.controller, KeyboardController)
    assert isinstance(runtime.processor, JointDirectProcessor)
    assert runtime.mapper is runtime.processor
    assert runtime.kinematics is None
    assert runtime.jacobian_controller is None
    assert runtime.processor_config.linear_scale == 0.2
    assert runtime.gripper_rate == 2.0


def test_build_control_runtime_cartesian_with_jacobian():
    urdf_path = Path(__file__).resolve().parents[1] / "assets" / "so101_abs.urdf"
    runtime = build_control_runtime(
        controller_type="xbox",
        mode="cartesian",
        deadzone=0.15,
        linear_scale=None,
        keyboard_grab=False,
        keyboard_record=None,
        keyboard_playback=None,
        loop_dt=1.0 / 30.0,
        urdf_path=str(urdf_path),
        use_jacobian=True,
    )

    assert runtime.control_mode is ControlMode.CARTESIAN
    assert runtime.kinematics is not None
    assert runtime.jacobian_controller is not None


def test_build_control_runtime_dual_joycon_cartesian():
    urdf_path = Path(__file__).resolve().parents[1] / "assets" / "so101_abs.urdf"
    runtime = build_control_runtime(
        controller_type="dual_joycon",
        mode="cartesian",
        deadzone=0.15,
        linear_scale=0.2,
        keyboard_grab=False,
        keyboard_record=None,
        keyboard_playback=None,
        loop_dt=1.0 / 30.0,
        urdf_path=str(urdf_path),
    )

    assert runtime.controller_label == "Dual Joy-Con"
    assert isinstance(runtime.mapper, MapDualJoyConToEEDelta)
    assert runtime.processor_config.linear_scale == 0.2
    assert runtime.processor_config.vertical_scale == 0.2


def test_control_help_lines_keyboard_cartesian_jacobian():
    lines = control_help_lines("keyboard", "cartesian", use_jacobian=True)
    assert lines[0] == "Controls:"
    assert "  Arrow keys      orientation disabled in touch mode" in lines
    assert "  H               home position" in lines


def test_control_help_lines_joycon_joint():
    lines = control_help_lines("joycon", "joint", exit_hint="Close window       exit")
    assert "  Stick left/right    drive selected joint" in lines
    assert "  (no joint cycle on Joy-Con — use cartesian mode)" in lines
    assert lines[-1] == "  Close window       exit"


def test_control_help_lines_dual_joycon_cartesian():
    lines = control_help_lines("dual_joycon", "cartesian")
    assert "  Hold ZL + left stick      translate X/Y" in lines
    assert "  Right Joy-Con IMU         wrist orientation" in lines


def test_control_help_lines_xbox_crane():
    lines = control_help_lines("xbox", "crane")
    assert "  Left stick X/Y            left-right / forward-back" in lines
    assert "  Right stick Y/X           up-down / wrist roll" in lines
    assert "  D-pad up/down             wrist up / down" in lines
    assert "  A button                  neutral crane pose" in lines


def test_control_help_lines_xbox_cartesian():
    lines = control_help_lines("xbox", "cartesian")
    assert "  D-pad                   orientation disabled in touch mode" in lines
