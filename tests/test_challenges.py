"""Challenges: a student designs the desk, the request and the answer, then
tests the arm and its agents against it.

A challenge is a small file (YAML or a dict in a notebook) in the names a
person would use (``red_cube``), whatever names the agent is shown. It is
checked (errors for what cannot be set up, warnings for what breaks the
layout rules, since breaking them is a fair experiment), proved possible by
script, loaded on an arm, and graded against what the student expected.
"""

import math

import pytest

from xbox_soarm_teleop.sim.table_scenes import TABLE_SCENES
from xbox_soarm_teleop.twin.challenges import (
    Challenge,
    ChallengeError,
    check,
    grade_challenge,
    load_challenge,
    prove,
    verdict_of,
)
from xbox_soarm_teleop.twin.layouts import LAYOUTS
from xbox_soarm_teleop.twin.naming import shown_names

SWAP = {
    "name": "red-swap-tricky",
    "scene": "sort_colors",
    "objects": {
        "red_cube": {"x": 0.275, "y": -0.06},
        "blue_cube": {"x": 0.23, "y": -0.16},
        "yellow_cube": {"x": 0.30, "y": 0.07, "turn_deg": 30},
    },
    "names": "misleading",
    "request": "Put the red cube in the blue bin and the blue cube in the red bin. "
    "Leave the yellow cube alone.",
    "expect": {"red_cube": "blue_bin", "blue_cube": "red_bin", "yellow_cube": None},
}


def test_a_challenge_in_a_students_words_becomes_a_layout_and_a_task():
    challenge = load_challenge(SWAP)
    assert isinstance(challenge, Challenge)
    layout = challenge.layout()
    assert layout.model_scene == "sort_bins"
    assert set(layout.objects) == {"pick_cube", "blue_cube", "yellow_cube"}  # the twin's names
    x, y, z, yaw = layout.objects["yellow_cube"]
    assert (x, y) == (0.30, 0.07)
    assert yaw == pytest.approx(
        math.atan2(0.07, 0.30) + math.radians(30)
    )  # turned from facing the base
    assert z > TABLE_SCENES["sort_bins"].table.top_z  # resting on the desk
    task = challenge.task()
    assert task.requests == (SWAP["request"],)
    assert task.expect == {"pick_cube": "blue_bin", "blue_cube": "red_bin", "yellow_cube": None}
    assert challenge.names == "misleading" and challenge.possible is True
    # The layout's name tells an agent nothing about the challenge.
    assert "swap" not in layout.name and "tricky" not in layout.name


def test_yaml_text_loads_too_and_objects_left_out_of_expect_stay_out_of_the_bins(tmp_path):
    path = tmp_path / "only_red.yaml"
    path.write_text(
        "name: only-red\n"
        "objects:\n"
        "  red_cube: {x: 0.275, y: -0.06}\n"
        "  green_cube: {x: 0.30, y: 0.07}\n"
        "request:\n"
        "  - Put only the red cube away.\n"
        "  - Just the red one, into its bin.\n"
        "expect: {red_cube: red_bin}\n"
    )
    challenge = load_challenge(path)
    assert challenge.scene == "sort_colors" and challenge.names == "truthful"  # the defaults
    assert challenge.task().expect == {"pick_cube": "red_bin", "green_cube": None}
    assert len(challenge.task().requests) == 2
    assert any("green_cube" in note and "out of the bins" in note for note in check(challenge))


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"objects": {"purple_cube": {"x": 0.27, "y": 0.0}}}, "purple_cube"),
        ({"expect": {"red_cube": "yellow_bin"}}, "yellow_bin"),
        ({"expect": {"green_cube": "red_bin"}}, "green_cube"),  # not on the desk
        ({"names": "sneaky"}, "names"),
        ({"request": ""}, "request"),
        ({"objects": {"red_cube": {"x": 0.9, "y": 0.0}}}, "desk"),
        ({"objects": {"red_cube": {"x": 0.27}}}, "y"),
        (
            {"objects": {"red_cube": {"x": 0.27, "y": 0.0}, "blue_cube": {"x": 0.28, "y": 0.0}}},
            "overlap",
        ),
        ({"objects": {"red_cube": {"x": 0.1909, "y": 0.1909}}}, "red_bin"),  # inside a tray
        ({"scene": "pick_place_basic"}, "scene"),
        ({"request": [{"Swap them": "red into blue"}]}, "colon"),  # YAML read "a: b" as a mapping
    ],
)
def test_what_cannot_be_set_up_is_refused_with_a_reason(change, says):
    spec = {
        "name": "bad",
        "objects": {"red_cube": {"x": 0.275, "y": -0.06}},
        "request": "Put the red cube in the red bin.",
        "expect": {"red_cube": "red_bin"},
        **change,
    }
    with pytest.raises(ChallengeError, match=says):
        load_challenge(spec)


@pytest.mark.parametrize("layout", ["sort_colors", "sort_colors_decoy"])
def test_the_built_in_layouts_break_no_layout_rule(layout):
    """The rules the warnings use are the ones tests/test_twin_sort.py holds
    the built-in layouts to."""
    names = shown_names(LAYOUTS[layout].objects, "truthful")
    objects = {
        names[name]: {"x": pose[0], "y": pose[1]} for name, pose in LAYOUTS[layout].objects.items()
    }
    challenge = load_challenge(
        {
            "name": layout,
            "scene": layout,
            "objects": objects,
            "request": "Sort them.",
            "expect": {name: None for name in objects},
        }
    )
    assert [w for w in check(challenge) if "out of the bins" not in w] == []


def test_rule_breaking_layouts_are_warned_about_not_refused():
    challenge = load_challenge(
        {
            "name": "crowded",
            "objects": {
                "red_cube": {"x": 0.27, "y": 0.0},
                "blue_cube": {"x": 0.27, "y": 0.06},  # 6 cm apart: an open gripper spans 9
                "green_cube": {"x": 0.39, "y": -0.05},  # past the steady part of the reach
            },
            "request": "Sort the cubes by colour.",
            "expect": {"red_cube": "red_bin", "blue_cube": "blue_bin", "green_cube": "green_bin"},
        }
    )
    warnings = check(challenge)
    assert any("red_cube" in w and "blue_cube" in w and "apart" in w for w in warnings)
    assert any("green_cube" in w and "reach" in w for w in warnings)


def test_grades_are_in_the_students_names():
    challenge = load_challenge(SWAP)
    result = grade_challenge(
        challenge, {"pick_cube": "blue_bin", "blue_cube": "blue_bin", "yellow_cube": None}
    )
    assert result["done"] is False and result["right"] == 2 and result["wrong"] == 1
    assert result["in_bin"] == {"red_cube": "blue_bin", "blue_cube": "blue_bin", "yellow_cube": None}


def test_an_impossible_challenge_says_what_it_asks():
    challenge = load_challenge(
        {
            "name": "no-yellow-bin",
            "scene": "sort_colors_decoy",
            "objects": {"yellow_cube": {"x": 0.30, "y": 0.07}},
            "request": "Put the yellow cube in the yellow bin.",
            "possible": False,
            "asked": {"yellow_cube": "yellow_bin"},  # there is none: that is the point
            "expect": {"yellow_cube": None},
        }
    )
    assert challenge.possible is False
    proof = prove(challenge, aim_runs=0)
    assert proof["verdict"] == "impossible" and "yellow_bin" in proof["why"]


@pytest.mark.parametrize(
    ("possible", "true_done", "aim_done", "verdict"),
    [
        (True, True, [True, True, True], "doable"),
        (True, True, [True, False, True], "fragile"),
        (True, False, [], "not_doable"),
        (True, True, [], "doable"),
        (False, False, [], "impossible"),
        (False, True, [], "possible_after_all"),
    ],
)
def test_verdicts(possible, true_done, aim_done, verdict):
    assert verdict_of(possible, true_done, aim_done) == verdict


def test_the_swap_is_proved_doable_with_true_positions_and_aim_error():
    """The arm's own skills carry the expected arrangement out: once from true
    positions, once looking with a pointer whose aim is off as an agent's is."""
    proof = prove(load_challenge(SWAP), aim_runs=1)
    assert proof["true_run"]["done"] is True, proof
    assert proof["true_run"]["in_bin"] == {
        "red_cube": "blue_bin",
        "blue_cube": "red_bin",
        "yellow_cube": None,
    }
    assert len(proof["aim_runs"]) == 1
    assert proof["verdict"] in ("doable", "fragile")
    assert proof["hidden"] == []  # every object is in view of the front and overhead cameras


BOXED = {
    "name": "into-the-box",
    "scene": "sort_colors_box",
    "objects": {
        "red_cube": {"x": 0.275, "y": -0.06},
        "orange_box": {"x": 0.31, "y": 0.06, "turn_deg": 20},
    },
    "request": "Put the red cube in the orange box.",
    "expect": {"red_cube": "orange_box"},
}


def test_a_challenge_can_use_the_movable_box():
    challenge = load_challenge(BOXED)
    x, y, z, yaw = challenge.layout().objects["orange_box"]
    assert (x, y) == (0.31, 0.06)
    assert yaw == pytest.approx(math.radians(20))  # turned from square to the desk, not facing the arm
    assert z < 0.01  # its floor plate rests on the desk
    assert challenge.task().expect == {"pick_cube": "orange_box", "orange_box": None}
    assert not any("orange_box is turned" in w for w in check(challenge))
    with pytest.raises(ChallengeError, match="push"):
        load_challenge({**BOXED, "expect": {"orange_box": "red_bin"}})
    with pytest.raises(ChallengeError, match="orange_box"):
        load_challenge({**BOXED, "objects": {"red_cube": {"x": 0.275, "y": -0.06}}})  # no box out


def test_putting_a_cube_in_the_box_is_proved_doable():
    proof = prove(load_challenge(BOXED), aim_runs=1)
    assert proof["true_run"]["in_bin"]["red_cube"] == "orange_box", proof
    assert proof["verdict"] in ("doable", "fragile")


def test_a_note_for_an_object_hidden_under_the_arm():
    """An agent designing a challenge put the red cube straight ahead; the
    proof found it hard to see from overhead and not doable. Say so first."""
    ahead = load_challenge(
        {
            "name": "ahead",
            "objects": {"red_cube": {"x": 0.25, "y": 0.0}},
            "request": "Put it away.",
            "expect": {"red_cube": "red_bin"},
        }
    )
    assert any("straight ahead" in note for note in check(ahead))
