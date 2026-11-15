"""Clef asked "which tray is it in?", on the sorting desk.

Agents shown misleading names sorted every cube into the wrong tray and said
"done" every time; nothing outside the agent checked. Clef can answer a
multiple-choice question from a picture, so it is asked where an object is,
with the cell's own trays as options (built from ``scene.bins``) plus on the
table, in the gripper and on the floor. The truth comes from the simulator,
so the answers can be scored before any agent relies on them.
"""

import json
import logging
import warnings

import numpy as np
import pytest

from xbox_soarm_teleop.twin import build_twin, scenarios
from xbox_soarm_teleop.twin.judge import (
    ClefJudge,
    tray_metrics,
    tray_options,
    tray_questions,
    tray_state_text,
    tray_truth,
)

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

BINS = [("red_bin", "red bin"), ("blue_bin", "blue bin"), ("green_bin", "green bin")]


def test_the_options_are_the_cells_own_trays_and_the_rest():
    assert tray_options(BINS) == (
        "red_bin",
        "blue_bin",
        "green_bin",
        "on_table",
        "in_gripper",
        "on_floor",
    )
    question = tray_questions("red cube", BINS)["where"]
    assert question["type"] == "choice"
    assert question["criteria"]["red_bin"].startswith("inside the red bin")
    text = tray_state_text("red cube", ["overhead"], BINS, others=["blue cube"])
    assert "red bin" in text and "green goal box" not in text and "blue cube" in text


def _clef_answers(probabilities):
    asked = []

    def post(url, body, headers=None, timeout=None):
        asked.append(body)
        return {
            "success": True,
            "result": {
                "answers": {"where": {"type": "choice", "probabilities": probabilities}},
                "usage": {"input_tokens": 900},
            },
        }

    return ClefJudge(model="clef", post=post), asked


def test_clef_is_asked_and_its_answer_read(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cfTESTTOKEN0123456789abcdef")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "0123456789abcdef0123456789abcdef")
    judge, asked = _clef_answers({"blue_bin": 0.9, "red_bin": 0.05, "on_table": 0.05})
    answer = judge.judge_trays([b"png"], "red cube", BINS, views=["overhead"])
    assert answer["where"]["blue_bin"] == pytest.approx(0.9)
    assert answer["where"]["on_floor"] == 0.0  # options it left out
    assert answer["cost_usd"] == pytest.approx(900 * 0.24 / 1e6)
    assert set(asked[0]["questions"]) == {"where"}


def test_the_truth_says_which_tray():
    twin = build_twin(scene="sort_colors", with_controller=False)
    try:
        twin.tick()
        assert tray_truth(twin, "pick_cube") == "on_table"
        scenarios.place_in_bin(twin, "pick_cube", "blue_bin")
        for _ in range(30):
            twin.tick()
        assert tray_truth(twin, "pick_cube") == "blue_bin"
    finally:
        twin.close()


def test_scoring_tray_answers():
    options = tray_options(BINS)
    answers = [{"red_bin": 0.9, "blue_bin": 0.1}, {"red_bin": 0.6, "on_table": 0.4}]
    metrics = tray_metrics(answers, ["red_bin", "on_table"], options)
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion"]["on_table"]["red_bin"] == 1
    assert metrics["confident"]["n"] == 1 and metrics["confident"]["right"] == 1  # p >= 0.9


def test_check_success_asks_which_tray_and_logs_the_truth(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cfTESTTOKEN0123456789abcdef")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "0123456789abcdef0123456789abcdef")
    twin = build_twin(scene="sort_colors", with_controller=False)
    try:
        twin.tick()
        scenarios.place_in_bin(twin, "pick_cube", "blue_bin")  # the red cube, wrong tray
        for _ in range(30):
            twin.tick()
        # A Clef that sees it in the blue tray.
        twin.judge, asked = _clef_answers({"blue_bin": 0.95, "red_bin": 0.03, "on_table": 0.02})
        import threading

        result = {}
        worker = threading.Thread(
            target=lambda: result.update(twin.check_success("pick_cube", bin_name="red_bin"))
        )
        worker.start()
        while worker.is_alive():
            twin.tick()
        assert result["ok"] is True and result["where"] == "blue_bin"
        assert result["p_in_bin"] == pytest.approx(0.03) and result["verdict"] == "retry"
        assert "truth" not in str(result)
        assert set(asked[0]["questions"]) == {"where"} and len(asked[0]["images"]) == 1
        last = twin.success_checks()[-1]
        assert last["truth"]["where"] == "blue_bin" and last["correct"] is True
        bad = twin.check_success("pick_cube", bin_name="yellow_bin")
        assert bad["reason"] == "invalid_request"
    finally:
        twin.close()


def _check_in_thread(twin, **kwargs):
    import threading

    result = {}
    worker = threading.Thread(target=lambda: result.update(twin.check_success(**kwargs)))
    worker.start()
    while worker.is_alive():
        twin.tick()
    return result


def test_check_by_description_whatever_the_object_is_called(monkeypatch):
    """Lesson 24: asked by the agent's own (misleading) name, the check confirmed
    the agent's mistake. Asked in the request's words, Clef finds the object by
    how it looks, so a wrong name cannot fool it."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cfTESTTOKEN0123456789abcdef")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "0123456789abcdef0123456789abcdef")
    twin = build_twin(scene="sort_colors", with_controller=False, naming="misleading")
    try:
        twin.tick()
        scenarios.place_in_bin(twin, "pick_cube", "blue_bin")  # the red cube, wrong tray
        for _ in range(30):
            twin.tick()
        twin.judge, asked = _clef_answers({"blue_bin": 0.94, "red_bin": 0.03, "on_table": 0.03})
        result = _check_in_thread(twin, description="red cube", bin_name="red_bin")
        assert result["ok"] is True and result["verdict"] == "retry"
        assert result["where"] == "blue_bin" and "name" not in result
        assert "red cube" in asked[0]["questions"]["where"]["instructions"]
        last = twin.success_checks()[-1]
        assert last["matched"] == "pick_cube"  # the twin knows which object that is
        assert last["truth"]["where"] == "blue_bin" and last["correct"] is True

        nothing = _check_in_thread(twin, description="purple cube", bin_name="red_bin")
        assert nothing["ok"] is True  # Clef is asked anyway; nothing on the desk matches
        assert twin.success_checks()[-1]["matched"] is None
        assert twin.success_checks()[-1]["correct"] is None
        assert twin.check_success(bin_name="red_bin")["reason"] == "invalid_request"
    finally:
        twin.close()


def test_a_description_given_as_a_name_is_pointed_at_description():
    """An agent asked check_success(name="red cube"); the refusal now says how
    to ask by how it looks."""
    twin = build_twin(scene="sort_colors", with_controller=False)
    try:
        twin.tick()
        twin.judge, _ = _clef_answers({"red_bin": 0.9})
        answer = _check_in_thread(twin, name="red cube", bin_name="red_bin")
        assert answer["reason"] == "invalid_request"
        assert 'description="red cube"' in answer["detail"]
    finally:
        twin.close()


def test_the_judge_is_asked_about_a_movable_box_too(monkeypatch):
    """With the orange box out, it is one of the places Clef may answer, and
    the check is graded against where the box is now."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cfTESTTOKEN0123456789abcdef")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "0123456789abcdef0123456789abcdef")
    twin = build_twin(scene="sort_colors_box", with_controller=False)
    try:
        twin.tick()
        box = twin.observe_truth()["scene"]["objects"]["orange_box"]["xyz_m"]
        twin.grasp.place("pick_cube", np.array([box[0], box[1], box[2] + 0.004 + 0.02]))
        for _ in range(30):
            twin.tick()
        twin.judge, asked = _clef_answers({"orange_box": 0.93, "red_bin": 0.03, "on_table": 0.04})
        result = _check_in_thread(twin, name="pick_cube", bin_name="orange_box")
        assert result["ok"] is True and result["verdict"] == "done", result
        assert twin.success_checks()[-1]["correct"] is True  # it really is in the box
        assert "orange box" in json.dumps(asked[-1])
        # Without the box out, it is not an option.
        twin.load_scene("sort_colors")
        twin.tick()
        refused = _check_in_thread(twin, name="pick_cube", bin_name="orange_box")
        assert refused["reason"] == "invalid_request"
    finally:
        twin.close()
