"""Unit tests for direct-joint diagnostic helpers."""

from xbox_soarm_teleop.diagnostics.xbox_joint_drive import (
    advance_goal,
    dpad_edge,
    map_trigger_to_gripper_deg,
)


def test_advance_goal_integrates_and_clamps():
    assert advance_goal(0.0, 10.0, 0.5, -20.0, 20.0) == 5.0
    assert advance_goal(19.0, 10.0, 1.0, -20.0, 20.0) == 20.0
    assert advance_goal(-19.0, -10.0, 1.0, -20.0, 20.0) == -20.0


def test_advance_goal_non_positive_dt_only_clamps():
    assert advance_goal(30.0, -100.0, 0.0, -10.0, 10.0) == 10.0
    assert advance_goal(-30.0, 100.0, -1.0, -10.0, 10.0) == -10.0


def test_map_trigger_to_gripper_deg_direction_and_bounds():
    low = -2.0
    high = 127.0
    assert map_trigger_to_gripper_deg(0.0, low, high) == high
    assert map_trigger_to_gripper_deg(1.0, low, high) == low
    assert map_trigger_to_gripper_deg(0.5, low, high) == (high + low) / 2.0
    assert map_trigger_to_gripper_deg(2.0, low, high) == low
    assert map_trigger_to_gripper_deg(-1.0, low, high) == high


def test_dpad_edge_detects_rising_crossings_only():
    assert dpad_edge(1.0, 0.0) == 1
    assert dpad_edge(-1.0, 0.0) == -1
    assert dpad_edge(1.0, 1.0) == 0
    assert dpad_edge(-1.0, -1.0) == 0
    assert dpad_edge(0.0, 1.0) == 0
