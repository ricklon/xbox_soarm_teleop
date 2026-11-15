"""Challenges from the command line: check and prove a file, run agents on it
(agent_trial.py --challenge), and turn the runs into a findings report."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from xbox_soarm_teleop.cli import challenge as challenge_cli
from xbox_soarm_teleop.cli import twin as twin_cli

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location("agent_trial", ROOT / "scripts" / "agent_trial.py")
agent_trial = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_trial)

SWAP = """\
name: red-swap
objects:
  red_cube:  {x: 0.275, y: -0.06}
  blue_cube: {x: 0.23, y: -0.16}
names: neutral
request:
  - Put the red cube in the blue bin and the blue cube in the red bin.
  - Swap them, red into blue, blue into red.
expect: {red_cube: blue_bin, blue_cube: red_bin}
"""


@pytest.fixture
def swap_file(tmp_path):
    path = tmp_path / "red_swap.yaml"
    path.write_text(SWAP)
    return path


def test_the_example_challenges_load(capsys):
    from xbox_soarm_teleop.twin.challenges import examples

    found = examples()
    assert {"crowded", "no_yellow_bin", "red_swap_tricky"} <= set(found)
    for name in found:
        assert challenge_cli.main(["check", name]) == 0, name  # by name, wherever installed


def test_check_prints_the_challenge_and_its_warnings(swap_file, capsys):
    assert challenge_cli.main(["check", str(swap_file)]) == 0
    out = capsys.readouterr().out
    assert "red-swap" in out and "neutral" in out and "blue_bin" in out


def test_check_explains_a_broken_file(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\nobjects: {purple_cube: {x: 0.27, y: 0}}\nrequest: Hi\n")
    assert challenge_cli.main(["check", str(path)]) == 1
    assert "purple_cube" in capsys.readouterr().out


def test_trials_run_a_challenge_with_its_request_names_and_grading(swap_file, tmp_path):
    args = SimpleNamespace(
        challenge=str(swap_file),
        task=None,
        scene="pick_place_basic",
        names="as_is",
        require_check=False,
        timeout=500.0,
        pointer="oracle",
        judge=None,
        random_start=False,
        seed=0,
        agent_sees=False,
    )
    agent_trial.apply_challenge(args)
    assert args.scene == "sort_colors" and args.names == "neutral"
    command = agent_trial.twin_command(args, port=9000, logs=tmp_path, run=0)
    assert command[command.index("--challenge") + 1] == str(swap_file)
    assert "red into blue" in agent_trial.prompt_for(args, 1)  # the next wording each run
    task = agent_trial.task_of(args)
    assert task.name == "red-swap" and task.expect == {"pick_cube": "blue_bin", "blue_cube": "red_bin"}


def test_the_twin_takes_a_challenge():
    args = twin_cli.build_parser().parse_args(["--challenge", "c.yaml"])
    assert args.challenge == "c.yaml"


def test_the_report_fills_the_template_with_the_runs(swap_file, tmp_path, capsys):
    runs = tmp_path / "runs.jsonl"
    rows = [
        {
            "task": "red-swap",
            "model": "google/gemma-4-31b-it",
            "agent_sees": False,
            "naming": "neutral",
            "request": "Swap them",
            "done": True,
            "right": 2,
            "wrong": 0,
            "unsorted": 0,
            "claim": "done",
            "claim_correct": True,
            "seconds": 120.5,
            "total_cost_usd": 0.009,
            "agent_usage": {"steps": 9},
            "stop_reason": "answered",
        },
        {
            "task": "red-swap",
            "model": "google/gemma-4-31b-it",
            "agent_sees": True,
            "naming": "neutral",
            "request": "Swap them",
            "done": False,
            "right": 1,
            "wrong": 1,
            "unsorted": 0,
            "claim": "done",
            "claim_correct": False,
            "seconds": 140.0,
            "total_cost_usd": 0.011,
            "agent_usage": {"steps": 12},
            "stop_reason": "answered",
        },
        {"task": "other", "model": "x", "done": True},  # another challenge's run: left out
        {"task": "red-swap", "model": "x", "agent_error": "HTTP 402 from x", "done": False},  # invalid
    ]
    runs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "report.md"
    assert challenge_cli.main(["report", str(swap_file), str(runs), "-o", str(out)]) == 0
    text = out.read_text()
    assert "# Findings: red-swap" in text
    assert "1 of 2" in text  # done, valid runs only
    assert "1 run left out" in text  # the invalid one, and why
    assert "| 2 | google/gemma-4-31b-it | sees |" in text
    for heading in (
        "What I expected to break",
        "What happened",
        "Kind of failure",
        "Evidence",
        "Proposed fix",
    ):
        assert heading in text
