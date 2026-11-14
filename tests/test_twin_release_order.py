"""After a release the open jaws still straddle the object; homing from there
drags it out of the box. In agent trials this undid 5 of 13 failed runs'
correct placements, and a 5 cm lift straight up before homing saved all 5
on replay. Every agent-facing text must say so, and check after the last move.
"""

import importlib.util
from pathlib import Path

import pytest

from xbox_soarm_teleop.twin import hub_server, mcp_server

_PATH = Path(__file__).parent.parent / "scripts" / "agent_trial.py"
_spec = importlib.util.spec_from_file_location("agent_trial", _PATH)
agent_trial = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_trial)


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("text", [mcp_server.INSTRUCTIONS, hub_server.INSTRUCTIONS])
def test_instructions_lift_clear_before_homing_then_check(text):
    text = _flat(text)
    assert "move_by(dx=0, dy=0, dz=0.05)" in text
    lift = text.index("lift straight up 5 cm")
    assert text.index("home", lift) < text.index("check_success(name)", lift)


def test_the_trial_prompt_asks_to_check_after_homing():
    task = _flat(agent_trial.PROMPT + agent_trial.REQUIRE_CHECK)
    assert "lift straight up 5 cm" in task
    assert "after homing" in task.lower()
