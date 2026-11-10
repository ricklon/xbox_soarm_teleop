"""Golden trace for the simulator's cartesian teleop loop.

Drives ``run_with_controller`` headless with a scripted Xbox controller and
records every joint/gripper command sent to the simulator. Refactors of the
control step (e.g. extracting it for the agent service) must reproduce it.

Regenerate only for an intended behavior change:
    XBOX_SOARM_REGEN_GOLDEN=1 uv run pytest tests/test_teleop_golden.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
from xbox_soarm_teleop.teleoperators.xbox import XboxController, XboxState

GOLDEN = Path(__file__).parent / "data" / "teleop_step_golden.json"
TICKS = 250


def scripted_state(tick: int) -> XboxState:
    """Exercise deadman, every axis, roll, pitch, gripper, homing and limits."""
    s = XboxState()
    if tick < 10:
        s.left_stick_y = -1.0  # no deadman: must not move
    elif tick < 50:
        s.left_bumper, s.left_stick_y = True, -1.0
    elif tick < 80:
        s.left_bumper, s.left_stick_x, s.right_stick_y = True, 0.7, -0.8
    elif tick < 100:
        s.left_bumper, s.right_stick_x = True, 1.0
    elif tick < 120:
        s.left_bumper, s.dpad_y = True, -1.0
    elif tick < 140:
        s.right_trigger = 1.0
        s.left_bumper = tick % 2 == 0
    elif tick == 140:
        s.a_button = s.a_button_pressed = True
    elif tick < 190:
        pass  # homing runs
    else:
        s.left_bumper, s.left_stick_y, s.right_stick_y = True, -1.0, 1.0  # into x/z bounds
    return s


def record_trace(monkeypatch, use_jacobian: bool) -> list[list[float]]:
    calls = {"n": 0}
    trace: list[list[float]] = []

    def fake_read(self):
        state = scripted_state(calls["n"])
        calls["n"] += 1
        return state

    monkeypatch.setattr(XboxController, "connect", lambda self: True)
    monkeypatch.setattr(XboxController, "read", fake_read)
    monkeypatch.setattr(XboxController, "disconnect", lambda self: None)
    monkeypatch.setattr(sim_cli._HeadlessViewer, "is_running", lambda self: calls["n"] < TICKS)
    monkeypatch.setattr(sim_cli.time, "sleep", lambda _s: None)

    sim = sim_cli.MuJoCoSimulator(str(sim_cli.URDF_PATH), scene="default")
    original_targets = sim.set_joint_targets
    original_gripper = sim.set_gripper

    def set_joint_targets(positions_deg):
        trace.append([float(v) for v in positions_deg])
        original_targets(positions_deg)

    def set_gripper(position):
        trace[-1].append(float(position))
        original_gripper(position)

    monkeypatch.setattr(sim, "set_joint_targets", set_joint_targets)
    monkeypatch.setattr(sim, "set_gripper", set_gripper)
    sim_cli.run_with_controller(
        sim, mode="cartesian", controller_type="xbox", headless=True, use_jacobian=use_jacobian
    )
    return trace


@pytest.mark.parametrize("use_jacobian", [False, True], ids=["ik", "jacobian"])
def test_cartesian_teleop_matches_golden(monkeypatch, use_jacobian):
    key = "jacobian" if use_jacobian else "ik"
    trace = record_trace(monkeypatch, use_jacobian)
    assert len(trace) == TICKS
    if os.environ.get("XBOX_SOARM_REGEN_GOLDEN") == "1":
        data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        data[key] = trace
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(data) + "\n")
        pytest.skip("golden regenerated")
    expected = json.loads(GOLDEN.read_text())[key]
    np.testing.assert_allclose(np.array(trace), np.array(expected), atol=1e-9)
