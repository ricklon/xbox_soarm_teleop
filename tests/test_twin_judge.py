"""Success judge: Clef asked about camera images, graded by the twin's own state."""

import json
import logging
import math
import warnings

import numpy as np
import pytest

from xbox_soarm_teleop.twin import build_twin
from xbox_soarm_teleop.twin.judge import (
    QUESTION_IDS,
    WHERE_OPTIONS,
    ClefJudge,
    JudgeError,
    calibration_bins,
    noul_metrics,
    questions_for,
    truth_labels,
    where_metrics,
)
from xbox_soarm_teleop.twin.pointing import PointerError
from xbox_soarm_teleop.twin.scenarios import (
    grasp_and_lift,
    place_in_goal,
    place_near_goal,
    place_on_table,
    place_tipped,
)

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

FAKE_TOKEN = "cfTESTTOKEN0123456789abcdef"
FAKE_ACCOUNT = "0123456789abcdef0123456789abcdef"


# ------------------------------------------------------------------ the request
def _reply(in_goal=0.9, held=0.1, tipped=0.2, where="in_goal_box"):
    probabilities = {option: 0.01 for option in WHERE_OPTIONS}
    probabilities[where] = 0.97
    return {
        "result": {
            "model": "clef",
            "answers": {
                "in_goal": {"type": "noul", "noul": in_goal},
                "held": {"type": "noul", "noul": held},
                "tipped": {"type": "noul", "noul": tipped},
                "where": {"type": "choice", "choice": where, "probabilities": probabilities},
            },
            "usage": {"input_tokens": 600, "output_tokens": 0},
        },
        "success": True,
    }


def test_judge_sends_images_and_typed_questions(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", FAKE_ACCOUNT)
    sent = {}

    def post(url, body, headers=None, timeout=None):
        sent.update(url=url, body=body, headers=headers)
        return _reply()

    judge = ClefJudge(model="clef-flash", post=post)
    verdict = judge.judge([b"\x89PNG1", b"\x89PNG2"], "red cube", views=["front", "overhead"])
    assert sent["url"].endswith(f"/accounts/{FAKE_ACCOUNT}/ai/run/@cf/cloudflare/clef-flash")
    assert sent["headers"] == {"Authorization": f"Bearer {FAKE_TOKEN}"}
    body = sent["body"]
    assert body["model"] == "clef-flash"
    assert [image[:22] for image in body["images"]] == ["data:image/png;base64,"] * 2
    assert set(body["questions"]) == set(QUESTION_IDS)
    assert body["questions"]["where"]["type"] == "choice"
    assert set(body["questions"]["where"]["criteria"]) == set(WHERE_OPTIONS)
    assert "red cube" in body["questions"]["in_goal"]["instructions"]
    assert "front" in body["state"] and "overhead" in body["state"]  # says which picture is which
    assert verdict["in_goal"] == 0.9 and verdict["held"] == 0.1 and verdict["tipped"] == 0.2
    assert verdict["where"]["in_goal_box"] == pytest.approx(0.97)
    assert verdict["input_tokens"] == 600


def test_judge_needs_its_credentials(monkeypatch, tmp_path):
    from tests.test_twin_locate import _isolate_env_files

    _isolate_env_files(monkeypatch, tmp_path)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(JudgeError, match="CLOUDFLARE_API_TOKEN"):
        ClefJudge(post=lambda *a, **k: _reply()).judge([b"x"], "red cube")


def test_judge_errors_are_masked(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", FAKE_ACCOUNT)

    def post(*args, **kwargs):
        raise PointerError(f"HTTP 401 from .../accounts/{FAKE_ACCOUNT}/ai: token {FAKE_TOKEN}")

    with pytest.raises(JudgeError) as info:
        ClefJudge(post=post).judge([b"x"], "red cube")
    assert FAKE_TOKEN not in str(info.value) and FAKE_ACCOUNT not in str(info.value)
    bad = ClefJudge(post=lambda *a, **k: {"success": False, "errors": [{"message": "nope"}]})
    with pytest.raises(JudgeError, match="nope"):
        bad.judge([b"x"], "red cube")


def test_at_most_four_images():
    with pytest.raises(ValueError):
        ClefJudge(post=lambda *a, **k: _reply()).judge([b"x"] * 5, "red cube")


# ------------------------------------------------------------------ metrics
def test_noul_metrics():
    probs = [0.9, 0.8, 0.3, 0.1]
    truth = [True, False, False, False]
    m = noul_metrics(probs, truth)
    assert m["n"] == 4 and m["positives"] == 1
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx((0.01 + 0.64 + 0.09 + 0.01) / 4)
    expected_log = -(math.log(0.9) + math.log(0.2) + math.log(0.7) + math.log(0.9)) / 4
    assert m["log_loss"] == pytest.approx(expected_log)
    # A perfect, confident judge has no calibration error.
    assert noul_metrics([1.0, 0.0], [True, False])["ece"] == pytest.approx(0.0)


def test_calibration_bins_count_and_frequency():
    bins = calibration_bins([0.05, 0.07, 0.95, 0.92, 0.55], [False, True, True, True, False], n_bins=10)
    low = next(b for b in bins if b["lo"] == 0.0)
    assert low["n"] == 2 and low["observed"] == pytest.approx(0.5)
    high = next(b for b in bins if b["lo"] == pytest.approx(0.9))
    assert high["n"] == 2 and high["observed"] == 1.0
    assert sum(b["n"] for b in bins) == 5


def test_where_metrics():
    rows = [
        ({"in_goal_box": 0.7, "on_table": 0.1, "in_gripper": 0.1, "on_floor": 0.1}, "in_goal_box"),
        ({"in_goal_box": 0.2, "on_table": 0.6, "in_gripper": 0.1, "on_floor": 0.1}, "in_gripper"),
    ]
    m = where_metrics([r[0] for r in rows], [r[1] for r in rows])
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["p_true_mean"] == pytest.approx((0.7 + 0.1) / 2)
    assert m["confusion"]["in_gripper"]["on_table"] == 1


# ------------------------------------------------------------------ truth
@pytest.fixture
def twin():
    service = build_twin(scene="pick_place_basic", with_controller=False)
    service.tick()
    return service


def test_truth_on_the_table(twin):
    place_on_table(twin, "pick_cube", 0.33, -0.10, yaw=0.4)
    labels = truth_labels(twin, "pick_cube")
    assert labels == {"in_goal": False, "held": False, "tipped": False, "where": "on_table"}


def test_truth_in_the_goal_box_matches_observe(twin):
    place_in_goal(twin, "pick_cube", dx=0.01, dy=-0.01, yaw=0.3)
    labels = truth_labels(twin, "pick_cube")
    assert labels["in_goal"] is True and labels["where"] == "in_goal_box"
    twin.tick()  # observe's own verdict agrees
    assert twin.observe()["scene"]["objects"]["pick_cube"]["in_goal"] is True


def test_truth_just_outside_the_box_is_not_in_it(twin):
    rng = np.random.default_rng(1)
    place_near_goal(twin, "pick_cube", rng)
    labels = truth_labels(twin, "pick_cube")
    assert labels["in_goal"] is False and labels["where"] == "on_table"


def test_truth_tipped_onto_an_edge(twin):
    place_tipped(twin, "pick_cube", 0.30, -0.10, tilt=math.radians(40))
    labels = truth_labels(twin, "pick_cube")
    assert labels["tipped"] is True and labels["where"] == "on_table"
    # It rests on its edge: its lowest point is on the table top.
    z = twin.grasp.position("pick_cube")[2]
    assert z - twin.grasp.half_extents_world("pick_cube")[2] == pytest.approx(
        twin.table_top_z, abs=1e-3
    )


@pytest.mark.parametrize("name", ["pick_cube", "bean_bag"])
def test_truth_held_after_a_real_grasp(twin, name):
    place_on_table(twin, name, 0.30, -0.03)  # the bean bag starts parked in this layout
    assert grasp_and_lift(twin, name, lift_to=[0.28, 0.0, 0.24])
    labels = truth_labels(twin, name)
    assert labels["held"] is True and labels["where"] == "in_gripper" and labels["in_goal"] is False


def test_question_ids_are_stable():
    assert QUESTION_IDS == ("in_goal", "held", "tipped", "where")
    assert json.dumps(questions_for("blue object"))  # serializable


def test_random_spots_and_near_box_placements_are_on_the_table(twin):
    from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
    from xbox_soarm_teleop.twin.scenarios import random_table_spot

    center, half = sim_cli.PICK_PLACE_TABLE_CENTER, sim_cli.PICK_PLACE_TABLE_HALF_EXTENTS
    rng = np.random.default_rng(3)
    for name in ("pick_cube", "bean_bag"):
        for _ in range(30):
            for xyz in (
                place_on_table(twin, name, *random_table_spot(rng)),
                place_near_goal(twin, name, rng),
            ):
                extent = twin.grasp.half_extents_world(name)
                assert abs(xyz[0] - center[0]) + extent[0] <= half[0] + 1e-9, xyz
                assert abs(xyz[1] - center[1]) + extent[1] <= half[1] + 1e-9, xyz
        # Next to the box is never in it.
        for _ in range(30):
            place_near_goal(twin, name, rng)
            assert truth_labels(twin, name)["in_goal"] is False


def test_the_judge_is_told_every_object_on_the_table(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", FAKE_ACCOUNT)
    sent = {}

    def post(url, body, headers=None, timeout=None):
        sent.update(body)
        return _reply()

    ClefJudge(post=post).judge(
        [b"x"], "large red cube", views=["front"], others=["small red cube", "red bar"]
    )
    state = sent["body"]["state"] if "body" in sent else sent["state"]
    assert "large red cube" in state and "small red cube" in state and "red bar" in state
    assert "large red cube" in sent["questions"]["in_goal"]["instructions"]


def test_judge_cost_from_the_catalog_price(monkeypatch):
    from xbox_soarm_teleop.twin.judge import PRICE_PER_M_INPUT

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", FAKE_ACCOUNT)
    verdict = ClefJudge(model="clef-flash", post=lambda *a, **k: _reply()).judge([b"x"], "red cube")
    assert verdict["input_tokens"] == 600
    assert verdict["cost_usd"] == pytest.approx(600 * PRICE_PER_M_INPUT["clef-flash"] / 1e6)
    assert PRICE_PER_M_INPUT == {"clef": 0.24, "clef-flash": 0.038}
