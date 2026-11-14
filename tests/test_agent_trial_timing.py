"""agent_trial.py splits a run's time: locate, arm motion, and the rest (the model)."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parent.parent / "scripts" / "agent_trial.py"
_spec = importlib.util.spec_from_file_location("agent_trial", _PATH)
agent_trial = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_trial)


def _line(kind, wall_s, action_id=None, **data):
    return {"kind": kind, "wall_s": wall_s, "t": 0.0, "frame": 0, "action_id": action_id, "data": data}


def test_time_is_split_between_locate_motion_and_the_model():
    lines = [
        _line("header", 0.0),
        _line("request", 4.0, type="locate"),
        _line("locate_result", 6.5, ok=True, seconds=2.5),
        _line("status", 8.0, "a-1", state="EXECUTING"),
        _line("status", 11.0, "a-1", state="SUCCEEDED"),
        _line("status", 15.0, "a-2", state="EXECUTING"),
        _line("status", 16.5, "a-2", state="FAILED"),
        _line("status", 20.0, "a-3", state="EXECUTING"),  # still moving when the run ended
        _line("shutdown", 22.0),
    ]
    split = agent_trial.time_split(lines)
    assert split["first_tool_s"] == pytest.approx(4.0)
    assert split["locate_s"] == pytest.approx(2.5)
    assert split["motion_s"] == pytest.approx(3.0 + 1.5 + 2.0)
    assert split["session_s"] == pytest.approx(22.0)
    assert split["other_s"] == pytest.approx(22.0 - 2.5 - 6.5)
    assert split["actions"] == 3
