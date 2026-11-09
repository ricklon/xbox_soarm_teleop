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


def test_make_workspace_clipper_matches_clip_workspace():
    import numpy as np

    from xbox_soarm_teleop.control.safety import make_workspace_clipper

    limits, _ = load_workspace_limits()
    clip = make_workspace_clipper(limits)
    clipped, flags = clip(np.array([0.6, -0.5, 0.0]))
    np.testing.assert_allclose(clipped, [0.5, -0.3, 0.05])
    assert flags == {"ws_clip_x": 1, "ws_clip_y": 1, "ws_clip_z": 1}


def test_simulator_enforces_the_configured_workspace():
    """Regression for #18: the sim left y unbounded while real/dual/YAML differed."""
    import numpy as np

    from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli

    limits, _ = load_workspace_limits()
    assert sim_cli.SIM_WORKSPACE_BOUNDS == limits
    clipped, _ = sim_cli._clip_to_sim_workspace(np.array([0.3, 0.9, 0.2]))
    assert clipped[1] == limits["y"][1]


def test_yaml_has_no_stale_home_position():
    """home_position used non-existent joint names and was never loaded."""
    from pathlib import Path

    import yaml

    import xbox_soarm_teleop.config as config_pkg

    data = yaml.safe_load((Path(config_pkg.__file__).parent / "workspace_limits.yaml").read_text())
    assert "home_position" not in data


def test_dual_teleop_has_no_inline_workspace_bounds():
    """teleoperate_dual allowed x down to -0.1 via inline clipping; it must use the config."""
    import inspect

    from xbox_soarm_teleop.cli import teleoperate_dual

    source = inspect.getsource(teleoperate_dual)
    assert "np.clip(pos[0]" not in source
    assert "load_workspace_limits" in source
