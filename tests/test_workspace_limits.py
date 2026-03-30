"""Tests for workspace limits loader."""

from xbox_soarm_teleop.config.workspace import load_workspace_limits


def test_workspace_limits_loaded_from_yaml():
    position, strict_position = load_workspace_limits()
    assert position["x"] == (0.05, 0.5)
    assert position["y"] == (-0.3, 0.3)
    assert position["z"] == (0.05, 0.45)
    assert strict_position["x"] == (0.1, 0.32)
    assert strict_position["y"] == (-0.2, 0.2)
    assert strict_position["z"] == (0.05, 0.3)
