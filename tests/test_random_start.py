"""Agent trials from random start positions.

The A/B so far started the cube in the same spot every run, so its runs were
repeats of one placement. ``soarm-twin --start X,Y`` moves the target before
the first tick, and ``agent_trial.py --random-start --seed N`` gives run i the
same random spot in every variant, so the variants are compared on the same
placements.
"""

import importlib.util
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xbox_soarm_teleop.cli.twin import build_parser
from xbox_soarm_teleop.twin import build_twin, scenarios

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

_PATH = Path(__file__).parent.parent / "scripts" / "agent_trial.py"
_spec = importlib.util.spec_from_file_location("agent_trial", _PATH)
agent_trial = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_trial)


def test_start_spots_repeat_by_seed_and_run_and_stay_clear_of_the_box():
    goal = scenarios.GOALS["pick_place_basic"].center_xyz_m
    spots = [scenarios.start_spot(seed=7, run=run) for run in range(40)]
    assert spots == [scenarios.start_spot(seed=7, run=run) for run in range(40)]
    assert spots != [scenarios.start_spot(seed=8, run=run) for run in range(40)]
    assert len(set(spots)) == 40
    for x, y in spots:
        assert scenarios.TABLE_X[0] <= x <= scenarios.TABLE_X[1]
        assert scenarios.TABLE_Y[0] <= y <= scenarios.TABLE_Y[1]
        assert max(abs(x - goal[0]), abs(y - goal[1])) >= scenarios.START_CLEARANCE_M


def test_parse_start():
    assert scenarios.parse_start("0.3,-0.05") == (0.3, -0.05)
    for bad in ("0.3", "a,b", "0.3,0.1,0.2"):
        with pytest.raises(ValueError):
            scenarios.parse_start(bad)
    assert build_parser().parse_args(["--start", "0.3,-0.05"]).start == "0.3,-0.05"


def test_place_start_moves_only_the_target():
    twin = build_twin(scene="pick_place_basic", with_controller=False)
    try:
        others = {
            name: twin.grasp.position(name).copy() for name in twin.grasp.objects if name != "pick_cube"
        }
        scenarios.place_start(twin, "pick_cube", 0.33, -0.06)
        for _ in range(100):
            twin.tick()
        x, y, _ = twin.observe()["scene"]["objects"]["pick_cube"]["xyz_m"]
        assert (x, y) == pytest.approx((0.33, -0.06), abs=0.003)
        for name, before in others.items():
            np.testing.assert_allclose(twin.grasp.position(name), before, atol=1e-6)
    finally:
        twin.close()


def _args(**overrides):
    base = dict(
        scene="pick_place_basic",
        timeout=500.0,
        pointer="oracle",
        judge=None,
        random_start=False,
        seed=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_the_trial_passes_the_runs_start_to_the_twin(tmp_path):
    fixed = agent_trial.twin_command(_args(), port=9000, logs=tmp_path, run=3)
    assert "--start" not in fixed
    command = agent_trial.twin_command(
        _args(random_start=True, seed=5), port=9000, logs=tmp_path, run=3
    )
    x, y = scenarios.start_spot(seed=5, run=3)
    assert command[command.index("--start") + 1] == f"{x:.4f},{y:.4f}"
    assert agent_trial.start_of(_args(random_start=True, seed=5), 3) == [round(x, 4), round(y, 4)]
    assert agent_trial.start_of(_args(), 3) is None
