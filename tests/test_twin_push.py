"""push: the arm moves an object by pushing it along the table, without lifting.

Agents asked to push did not know how: one lowered the jaws onto the cube's
own spot and slid off, another picked and placed instead. Pushing is a skill:
close the jaws, get behind the object, go low, slide through, and look again,
since one push drifts (a cube slides off the narrow closed jaws). It uses only
what the arm can know: its looks (or observe when positions are shown).
"""

import threading
import time

import numpy as np
import pytest

from xbox_soarm_teleop.twin import build_twin
from xbox_soarm_teleop.twin.challenges import _with_aim_error
from xbox_soarm_teleop.twin.pointing import make_pointer

CLOSE_M = 0.015  # where it should end up, within this


def _running(service):
    alive = threading.Event()
    alive.set()

    def loop():
        try:
            while alive.is_set():
                service.tick()
        finally:
            service.close()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return alive, thread


@pytest.fixture(scope="module")
def shown():
    """The sorting desk with positions shown and no pointing model: fast."""
    service = build_twin(scene="sort_colors", with_controller=False)
    alive, thread = _running(service)
    yield service
    alive.clear()
    thread.join(timeout=10)


def _reset(service):
    service.reset_scene(client_id="test").result(timeout=10)
    time.sleep(0.2)


def _xy(service, name):
    return np.asarray(service.observe_truth()["scene"]["objects"][name]["xyz_m"][:2])


def test_push_moves_an_object_where_asked_and_leaves_the_others(shown):
    service = shown
    _reset(service)
    start = {name: _xy(service, name) for name in ("pick_cube", "blue_cube", "green_cube")}
    result = service.push("pick_cube", dx_m=0.06, dy_m=0.0, client_id="test")
    assert result["ok"] is True, result
    moved = _xy(service, "pick_cube") - start["pick_cube"]
    assert np.linalg.norm(moved - [0.06, 0.0]) <= CLOSE_M, moved
    for other in ("blue_cube", "green_cube"):
        assert np.linalg.norm(_xy(service, other) - start[other]) < 0.01, other
    assert service.observe_truth()["scene"]["objects"]["pick_cube"]["off_table"] is False
    assert result["pushes"] >= 1


def test_push_by_description_and_direction_words(shown):
    """ "Push the green cube 5 cm to the left": left is the arm's left (+y)."""
    service = shown
    _reset(service)
    start = _xy(service, "green_cube")
    result = service.push(description="green cube", direction="left", distance_m=0.05, client_id="test")
    assert result["ok"] is True, result
    assert result["pushed"] == "green cube"  # in the request's words, not a name (lesson 24)
    assert np.linalg.norm(_xy(service, "green_cube") - start - [0.0, 0.05]) <= CLOSE_M


@pytest.mark.parametrize(
    ("kwargs", "says"),
    [
        ({}, "Objects here"),
        ({"name": "pick_cube"}, "how far"),
        ({"name": "pick_cube", "direction": "sideways", "distance_m": 0.05}, "direction"),
        ({"name": "pick_cube", "direction": "left", "distance_m": 0.05, "dx_m": 0.05}, "either"),
        ({"name": "pick_cube", "dx_m": 0.4}, "at most"),
        ({"name": "nothing", "dx_m": 0.05}, "Unknown object"),
    ],
)
def test_unclear_pushes_are_refused_with_a_reason(shown, kwargs, says):
    result = shown.push(client_id="test", **kwargs)
    assert result["ok"] is False and result["reason"] == "invalid_request"
    assert says in result["detail"]


def test_push_with_an_agents_aim_error():
    """Looking through a pointer whose aim is off as an agent's is."""
    service = build_twin(
        scene="sort_colors", with_controller=False, hide_positions=True, pointer=make_pointer("oracle")
    )
    _with_aim_error(service, np.random.default_rng(3))
    alive, thread = _running(service)
    try:
        start = _xy(service, "blue_cube")
        result = service.push(
            description="blue cube", direction="forward", distance_m=0.06, client_id="test"
        )
        assert result["ok"] is True, result
        assert "xyz" not in str(result)  # nothing about where things really are
        end = _xy(service, "blue_cube")
        assert np.linalg.norm(end - start - [0.06, 0.0]) <= 0.025, end - start
    finally:
        alive.clear()
        thread.join(timeout=10)
