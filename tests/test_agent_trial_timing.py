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


def test_opencode_json_events_give_cost_tokens_and_the_transcript():
    import json

    events = [
        {"type": "step_start", "part": {"type": "step-start"}},
        {
            "type": "tool_use",
            "part": {
                "type": "tool",
                "tool": "soarm-bench_locate",
                "state": {"input": {"name": "pick_cube"}},
            },
        },
        {
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "tokens": {
                    "input": 18000,
                    "output": 40,
                    "reasoning": 10,
                    "cache": {"read": 500, "write": 0},
                },
                "cost": 0.0007,
            },
        },
        {"type": "text", "part": {"type": "text", "text": "It is in the goal box."}},
        {
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "tokens": {"input": 19000, "output": 20, "reasoning": 0},
                "cost": 0.0008,
            },
        },
    ]
    raw = "\n".join(json.dumps(e) for e in events) + "\nnot json\n"
    parsed = agent_trial.parse_opencode_events(raw)
    assert parsed["usage"] == {
        "steps": 2,
        "input_tokens": 37000,
        "output_tokens": 60,
        "reasoning_tokens": 10,
        "cache_read_tokens": 500,
        "cost_usd": pytest.approx(0.0015),
    }
    assert parsed["last_text"] == "It is in the goal box."
    assert (
        "soarm-bench_locate" in parsed["transcript"]
        and "It is in the goal box." in parsed["transcript"]
    )


# ------------------------------------------------------------------ A/B scoring
def test_the_agents_claim_is_read_from_its_last_result_line():
    assert agent_trial.parse_claim("Done.\nRESULT: in_box") == "in_box"
    assert (
        agent_trial.parse_claim("RESULT: in_box ... then it fell. RESULT: not_in_box") == "not_in_box"
    )
    assert agent_trial.parse_claim("result: In_Box") == "in_box"
    assert agent_trial.parse_claim("It is in the box.") is None


def test_grading_adds_up_pointer_and_judge_cost_and_scores_the_claim(tmp_path):
    import json

    lines = [
        _line("header", 0.0),
        _line("locate_result", 2.0, ok=True, seconds=2.0, usage={"cost_usd": 0.004, "input_tokens": 1}),
        _line("locate_result", 5.0, ok=True, seconds=2.0, usage={"cost_usd": 0.005, "input_tokens": 1}),
        _line(
            "success_check", 9.0, ok=True, verdict="done", p_in_goal=0.95, correct=True, cost_usd=0.0002
        ),
        _line(
            "observation",
            9.5,
            scene={"objects": {"pick_cube": {"in_goal": True, "xyz_m": [0.3, 0.1, 0.14]}}},
        ),
        _line("shutdown", 10.0),
    ]
    log = tmp_path / "s.jsonl"
    log.write_text("\n".join(json.dumps(line) for line in lines))
    graded = agent_trial.grade(log, "pick_cube")
    assert graded["pointer_cost_usd"] == pytest.approx(0.009)
    assert graded["judge_cost_usd"] == pytest.approx(0.0002)
    assert graded["success_checks"] == [{"verdict": "done", "p_in_goal": 0.95, "correct": True}]
    scored = agent_trial.score_run(
        graded | {"in_goal": True}, final_text="RESULT: in_box", agent_cost=0.01, stopped=True
    )
    assert scored["claim"] == "in_box" and scored["claim_correct"] is True
    assert scored["total_cost_usd"] == pytest.approx(0.0192)
    assert scored["stopped_cleanly"] is True


def test_the_report_compares_conditions():
    _report_path = Path(__file__).parent.parent / "scripts" / "agent_report.py"
    spec = importlib.util.spec_from_file_location("agent_report", _report_path)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    rows = [
        {
            "model": "m",
            "harness": "loop",
            "variant": "no_judge",
            "in_goal": True,
            "stopped_cleanly": False,
            "claim_correct": True,
            "seconds": 100.0,
            "total_cost_usd": 0.05,
            "success_checks": [],
        },
        {
            "model": "m",
            "harness": "loop",
            "variant": "no_judge",
            "in_goal": False,
            "stopped_cleanly": True,
            "claim_correct": False,
            "seconds": 60.0,
            "total_cost_usd": 0.03,
            "success_checks": [],
        },
        {
            "model": "m",
            "harness": "loop",
            "variant": "judge",
            "in_goal": True,
            "stopped_cleanly": True,
            "claim_correct": True,
            "seconds": 50.0,
            "total_cost_usd": 0.02,
            "success_checks": [{"verdict": "done", "correct": True}],
        },
    ]
    table = {(r["model"], r["variant"]): r for r in report.summarize(rows)}
    base, judged = table[("m", "no_judge")], table[("m", "judge")]
    assert base["runs"] == 2 and base["in_box"] == 0.5 and base["stopped_cleanly"] == 0.5
    assert base["median_s"] == pytest.approx(80.0) and base["total_cost_usd"] == pytest.approx(0.08)
    assert judged["checks_per_run"] == 1.0 and judged["checks_right"] == 1.0


def test_failed_checks_are_counted_and_make_a_judged_run_invalid(tmp_path):
    import json

    lines = [
        _line("header", 0.0),
        _line("success_check", 3.0, ok=False, name="pick_cube", detail="HTTP 429 daily allocation"),
        _line(
            "success_check", 6.0, ok=True, verdict="done", p_in_goal=0.95, correct=True, cost_usd=0.0002
        ),
        _line("observation", 7.0, scene={"objects": {"pick_cube": {"in_goal": True}}}),
    ]
    log = tmp_path / "s.jsonl"
    log.write_text("\n".join(json.dumps(line) for line in lines))
    graded = agent_trial.grade(log, "pick_cube")
    assert graded["check_errors"] == 1
    assert graded["success_checks"] == [{"verdict": "done", "p_in_goal": 0.95, "correct": True}]

    spec = importlib.util.spec_from_file_location(
        "agent_report", Path(__file__).parent.parent / "scripts" / "agent_report.py"
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    rows = [
        {"model": "m", "variant": "judge", "in_goal": True, "check_errors": 2, "success_checks": []},
        {"model": "m", "variant": "judge", "in_goal": False, "check_errors": 0, "success_checks": []},
    ]
    (row,) = report.summarize(rows)
    assert row["runs"] == 1 and row["invalid_runs"] == 1 and row["in_box"] == 0.0


def test_a_check_that_never_answered_counts_as_an_error(tmp_path):
    """A check_success request with no success_check line crashed before logging."""
    import json

    lines = [
        _line("header", 0.0),
        _line("request", 1.0, type="check_success", name="pick_cube"),
        _line("request", 2.0, type="check_success", name="pick_cube"),
        _line(
            "success_check", 2.5, ok=True, verdict="done", p_in_goal=0.95, correct=True, cost_usd=0.0002
        ),
        _line("observation", 3.0, scene={"objects": {"pick_cube": {"in_goal": True}}}),
    ]
    log = tmp_path / "s.jsonl"
    log.write_text("\n".join(json.dumps(line) for line in lines))
    assert agent_trial.grade(log, "pick_cube")["check_errors"] == 1


def _task_args(**overrides):
    from types import SimpleNamespace

    base = dict(scene="pick_place_basic", task="swap", require_check=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_a_sort_task_run_asks_in_turn_in_each_wording():
    from xbox_soarm_teleop.twin.tasks import TASKS

    swap = TASKS["swap"]
    prompts = [agent_trial.prompt_for(_task_args(), run) for run in range(3)]
    assert swap.requests[0] in prompts[0] and swap.requests[1] in prompts[1]
    assert swap.requests[0] in prompts[2]  # and round again
    assert "RESULT: done" in prompts[0] and "red_bin" not in prompts[0]
    single = agent_trial.prompt_for(_task_args(task=None), 0)
    assert "goal box" in single and "RESULT: in_box" in single


def test_a_sort_task_is_graded_against_what_was_asked(tmp_path):
    import json

    from xbox_soarm_teleop.twin.tasks import TASKS

    objects = {
        "pick_cube": {"xyz_m": [0.41, 0.04, 0.14], "in_bin": "blue_bin"},
        "blue_cube": {"xyz_m": [0.41, 0.13, 0.14], "in_bin": "red_bin"},
        "green_cube": {"xyz_m": [0.41, -0.05, 0.14], "in_bin": "green_bin"},
    }
    lines = [
        _line("header", 0.0),
        _line("observation", 5.0, scene={"objects": objects}),
    ]
    log = tmp_path / "s.jsonl"
    log.write_text("\n".join(json.dumps(line) for line in lines))
    graded = agent_trial.grade_task(log, TASKS["swap"])
    assert graded["task"] == "swap" and graded["done"] is False  # the green cube was to stay
    assert (graded["right"], graded["wrong"], graded["unsorted"]) == (2, 1, 0)
    assert graded["in_bin"]["green_cube"] == "green_bin" and "in_goal" not in graded
    scored = agent_trial.score_run(graded, final_text="RESULT: done", agent_cost=0.01, stopped=True)
    assert scored["claim"] == "done" and scored["claim_correct"] is False


def test_runs_cut_short_by_the_model_service_are_invalid():
    """An agent stopped by HTTP 402 or 429 from its model service tested nothing."""
    spec = importlib.util.spec_from_file_location(
        "agent_report", Path(__file__).parent.parent / "scripts" / "agent_report.py"
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    rows = [
        {"model": "m", "variant": "no_judge", "in_goal": True, "agent_error": None},
        {"model": "m", "variant": "no_judge", "in_goal": False, "agent_error": "HTTP 402 from x"},
    ]
    (row,) = report.summarize(rows)
    assert row["runs"] == 1 and row["invalid_runs"] == 1 and row["in_box"] == 1.0


def test_trials_record_and_report_what_the_agent_was_given(tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(
        scene="sort_colors",
        timeout=500.0,
        pointer="oracle",
        judge=None,
        random_start=False,
        seed=0,
        names="misleading",
        agent_sees=True,
    )
    command = agent_trial.twin_command(args, port=9000, logs=tmp_path, run=0)
    assert command[command.index("--names") + 1] == "misleading"
    spec = importlib.util.spec_from_file_location(
        "agent_report", Path(__file__).parent.parent / "scripts" / "agent_report.py"
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    rows = [
        {"model": "m", "task": "by_colour", "naming": "misleading", "agent_sees": True, "done": True},
        {"model": "m", "task": "by_colour", "naming": "misleading", "agent_sees": False, "done": False},
    ]
    table = report.summarize(rows)
    assert {(r["naming"], r["sight"]) for r in table} == {
        ("misleading", "sees"),
        ("misleading", "blind"),
    }


def test_a_sort_task_can_require_the_tray_check():
    args = _task_args(task="by_colour", require_check=True)
    prompt = agent_trial.prompt_for(args, 0)
    assert "check_success(description, bin)" in prompt and "RESULT: done" in prompt
    assert "check_success" not in agent_trial.prompt_for(_task_args(task="by_colour"), 0)
