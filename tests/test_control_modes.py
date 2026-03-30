"""Tests for ControlMode enum, JointDirectProcessor, CraneProcessor, and make_processor."""

from __future__ import annotations

import pytest

from xbox_soarm_teleop.config.joints import HOME_POSITION_DEG
from xbox_soarm_teleop.config.modes import ControlMode
from xbox_soarm_teleop.processors.crane import CraneProcessor
from xbox_soarm_teleop.processors.factory import make_processor
from xbox_soarm_teleop.processors.joint_direct import JointCommand, JointDirectProcessor
from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta
from xbox_soarm_teleop.teleoperators.xbox import XboxState


def _state(**kwargs) -> XboxState:
    return XboxState(**kwargs)


# ---------------------------------------------------------------------------
# ControlMode enum
# ---------------------------------------------------------------------------


def test_control_mode_values():
    assert ControlMode.CARTESIAN.value == "cartesian"
    assert ControlMode.JOINT.value == "joint"
    assert ControlMode.CRANE.value == "crane"


def test_control_mode_from_string():
    assert ControlMode("cartesian") == ControlMode.CARTESIAN
    assert ControlMode("joint") == ControlMode.JOINT
    assert ControlMode("crane") == ControlMode.CRANE


def test_control_mode_invalid_raises():
    with pytest.raises(ValueError):
        ControlMode("invalid_mode")


# ---------------------------------------------------------------------------
# make_processor factory
# ---------------------------------------------------------------------------


def test_factory_cartesian_returns_mapper():
    p = make_processor(ControlMode.CARTESIAN)
    assert isinstance(p, MapXboxToEEDelta)


def test_factory_joint_returns_joint_processor():
    p = make_processor(ControlMode.JOINT)
    assert isinstance(p, JointDirectProcessor)


def test_factory_crane_returns_crane_processor():
    p = make_processor(ControlMode.CRANE)
    assert isinstance(p, CraneProcessor)


def test_factory_unknown_raises():
    with pytest.raises((ValueError, AttributeError)):
        make_processor("invalid_mode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# JointDirectProcessor
# ---------------------------------------------------------------------------


def test_joint_processor_initial_goals_are_home():
    proc = JointDirectProcessor()
    for name, val in HOME_POSITION_DEG.items():
        assert abs(proc._goals_deg[name] - val) < 0.01


def test_joint_processor_deadman_off_no_motion():
    proc = JointDirectProcessor(max_vel_deg_s=100.0, dt=1.0 / 30.0)
    selected = proc.selected_joint
    initial = proc._goals_deg[selected]
    state = _state(left_bumper=False, left_stick_x=1.0)
    cmd = proc(state)
    assert cmd.cmd_vel_deg_s == 0.0
    assert abs(cmd.goals_deg[selected] - initial) < 1e-9


def test_joint_processor_deadman_on_moves_selected_joint():
    proc = JointDirectProcessor(max_vel_deg_s=100.0, dt=1.0 / 30.0)
    selected = proc.selected_joint
    initial = proc._goals_deg[selected]
    state = _state(left_bumper=True, left_stick_x=1.0)
    cmd = proc(state)
    assert cmd.cmd_vel_deg_s > 0.0
    assert cmd.goals_deg[selected] != initial


def test_joint_processor_a_button_resets_home():
    proc = JointDirectProcessor()
    proc._goals_deg["shoulder_pan"] = 50.0
    state = _state(a_button_pressed=True)
    cmd = proc(state)
    assert abs(cmd.goals_deg["shoulder_pan"] - HOME_POSITION_DEG["shoulder_pan"]) < 0.01


def test_joint_processor_dpad_selects_next_joint():
    proc = JointDirectProcessor()
    first_joint = proc.selected_joint
    # rising edge: previous=0, current=1
    state = _state(dpad_x=1.0)
    proc(state)
    assert proc.selected_joint != first_joint


def test_joint_processor_dpad_wraps_around():
    proc = JointDirectProcessor()
    n = len(proc._goals_deg)
    # cycle all the way around
    for _ in range(n):
        proc._prev_dpad_x = 0.0
        proc(_state(dpad_x=1.0))
    assert proc._selected_idx == 0


def test_joint_processor_reset():
    proc = JointDirectProcessor()
    proc._selected_idx = 3
    proc._goals_deg["shoulder_pan"] = 99.0
    proc.reset()
    assert proc._selected_idx == 0
    assert abs(proc._goals_deg["shoulder_pan"] - HOME_POSITION_DEG["shoulder_pan"]) < 0.01


def test_joint_processor_returns_joint_command():
    proc = JointDirectProcessor()
    cmd = proc(_state())
    assert isinstance(cmd, JointCommand)
    assert isinstance(cmd.goals_deg, dict)
    assert cmd.selected_joint in cmd.goals_deg


# ---------------------------------------------------------------------------
# CraneProcessor
# ---------------------------------------------------------------------------


def test_crane_processor_returns_joint_command():
    proc = CraneProcessor()
    result = proc(_state(left_bumper=True, left_stick_x=1.0))
    assert isinstance(result, JointCommand)
    assert set(result.goals_deg.keys()) == {
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
    }


def test_crane_processor_deadman_off_holds_position():
    proc = CraneProcessor()
    initial = proc._pan_deg
    proc(_state(left_bumper=False, left_stick_x=1.0))
    assert proc._pan_deg == initial


def test_crane_processor_deadman_on_moves_pan():
    proc = CraneProcessor()
    initial = proc._pan_deg
    proc(_state(left_bumper=True, left_stick_x=1.0))
    assert proc._pan_deg != initial


def test_crane_processor_gripper_passthrough():
    from xbox_soarm_teleop.config.joints import JOINT_LIMITS_DEG

    proc = CraneProcessor()
    result = proc(_state(right_trigger=1.0))
    g_lo, _ = JOINT_LIMITS_DEG["gripper"]
    assert abs(result.goals_deg["gripper"] - g_lo) < 0.1


def test_crane_processor_a_button_resets_home():
    proc = CraneProcessor()
    proc._pan_deg = 50.0
    result = proc(_state(a_button_pressed=True))
    assert abs(result.goals_deg["shoulder_pan"] - HOME_POSITION_DEG["shoulder_pan"]) < 0.1


def test_crane_processor_reach_integrates():
    proc = CraneProcessor()
    initial_reach = proc._reach_m
    proc(_state(left_bumper=True, right_stick_y=-1.0))
    assert proc._reach_m > initial_reach


def test_crane_processor_height_integrates():
    proc = CraneProcessor()
    initial_height = proc._height_m
    proc(_state(left_bumper=True, left_stick_y=-1.0))
    assert proc._height_m > initial_height


def test_crane_processor_reset():
    proc = CraneProcessor()
    proc._pan_deg = 99.0
    proc.reset()
    assert abs(proc._pan_deg - HOME_POSITION_DEG["shoulder_pan"]) < 0.1


def test_crane_processor_handles_ik_none():
    class _DummyIK:
        def inverse_kinematics(self, *args, **kwargs):
            return None

    proc = CraneProcessor()
    proc._planar_ik = _DummyIK()
    # Force the IK path with deadman held
    result = proc(_state(left_bumper=True, right_stick_y=-1.0, left_stick_y=-1.0))
    assert isinstance(result, JointCommand)
