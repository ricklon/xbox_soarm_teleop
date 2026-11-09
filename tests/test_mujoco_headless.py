"""Headless demo routing must never launch a viewer."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli


class FakeKinematics:
    def __init__(self, **kwargs):
        pass

    def forward_kinematics(self, joints, **kwargs):
        pose = np.eye(4)
        pose[:3, 3] = [0.2, 0.0, 0.2]
        return pose

    def inverse_kinematics(self, seed, target, **kwargs):
        return seed.copy()


def test_headless_demo_steps_without_viewer(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "xbox_soarm_teleop.kinematics.constrained",
        SimpleNamespace(RobotKinematics=FakeKinematics),
    )
    monkeypatch.setattr(
        sim_cli.mujoco.viewer,
        "launch_passive",
        lambda *a, **kw: pytest.fail("headless launched a viewer"),
    )
    monkeypatch.setattr(sim_cli, "set_viewer_camera", lambda *a: pytest.fail("headless camera access"))
    monkeypatch.setattr(sim_cli.signal, "signal", lambda *a: None)
    monkeypatch.setattr(sim_cli.time, "sleep", lambda *a: None)
    sim = sim_cli.MuJoCoSimulator(str(Path(__file__).resolve().parents[1] / "assets/so101.urdf"))
    log = tmp_path / "routine.csv"
    code = sim_cli.run_demo_mode(
        sim, headless=True, routine_duration=0.04, routine_scale=0.0, ik_log_path=str(log)
    )
    assert code == 0
    assert len(log.read_text().splitlines()) == 3  # header + two 50 Hz samples


def test_main_forwards_headless_to_routine(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys, "argv", ["simulate-mujoco", "--headless", "--motion-routine", "--routine-duration", ".04"]
    )
    monkeypatch.setattr(sim_cli, "MuJoCoSimulator", lambda *a, **kw: object())
    monkeypatch.setattr(sim_cli, "run_demo_mode", lambda *a, **kw: calls.append(kw) or 0)
    with pytest.raises(SystemExit) as result:
        sim_cli.main()
    assert result.value.code == 0
    assert calls[0]["headless"] is True
