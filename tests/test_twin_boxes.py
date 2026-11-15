"""Movable boxes: an open box the arm can push about, with things in it.

A box is built like a tray (a floor and four walls) but as a free body, so it
slides when pushed and what is inside rides along. It lives in the sorting
desk's model, parked unless a layout puts it out (``sort_colors_box``), so
the other layouts, their names and their benchmarks are unchanged. Grading
follows the box: ``in_bin`` names it wherever it has been pushed.
"""

import threading
import time

import numpy as np
import pytest

from xbox_soarm_teleop.sim.table_scenes import TABLE_SCENES
from xbox_soarm_teleop.twin import build_twin
from xbox_soarm_teleop.twin.layouts import LAYOUTS, MOVABLE_BOXES, GoalBox


def test_the_box_is_part_of_the_sorting_desk_but_only_out_in_its_layout():
    assert [box.name for box in TABLE_SCENES["sort_bins"].boxes] == ["orange_box"]
    assert [box.name for box in MOVABLE_BOXES["sort_bins"]] == ["orange_box"]
    assert "orange_box" in LAYOUTS["sort_colors_box"].objects
    for name in ("sort_colors", "sort_colors_decoy"):
        assert "orange_box" not in LAYOUTS[name].objects


def test_a_turned_box_holds_what_is_inside_it():
    box = GoalBox("b", (0.3, 0.1, 0.01), (0.07, 0.07, 0.02), yaw=np.pi / 4)
    half = np.array([0.015, 0.015, 0.015])
    assert box.holds((0.3, 0.1, 0.025), half)
    # 2.8 cm out along x reaches into a corner of the turned box (a square box
    # would not hold it there); along the diagonal it is outside.
    assert box.holds((0.3 + 0.028, 0.1, 0.025), half)
    assert not GoalBox("b", (0.3, 0.1, 0.01), (0.07, 0.07, 0.02)).holds((0.3 + 0.028, 0.1, 0.025), half)
    assert not box.holds((0.3 + 0.028, 0.1 + 0.028, 0.025), half)


@pytest.fixture(scope="module")
def desk():
    service = build_twin(scene="sort_colors_box", with_controller=False)
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
    time.sleep(0.3)
    yield service
    alive.clear()
    thread.join(timeout=10)


def _reset(service):
    service.reset_scene(client_id="test").result(timeout=10)
    time.sleep(0.3)


def _truth(service):
    return service.observe_truth()["scene"]["objects"]


def test_the_box_rests_on_the_desk_and_is_seen_as_a_box(desk):
    _reset(desk)
    objects = desk.observe()["scene"]["objects"]
    assert objects["orange_box"]["box"] is True
    assert objects["orange_box"]["off_table"] is False
    size = objects["orange_box"]["size_m"]
    assert size[0] == pytest.approx(0.09, abs=0.002) and size[2] == pytest.approx(0.03, abs=0.004)
    assert "box" not in objects["pick_cube"]


def test_a_cube_placed_in_the_box_is_in_it_and_rides_along_when_pushed(desk):
    _reset(desk)
    assert desk.pick("pick_cube", client_id="test")["ok"] is True
    placed = desk.place("orange_box", client_id="test")
    assert placed["ok"] is True, placed
    time.sleep(0.3)
    assert _truth(desk)["pick_cube"]["in_bin"] == "orange_box"
    start = np.asarray(_truth(desk)["orange_box"]["xyz_m"][:2])
    pushed = desk.push("orange_box", direction="forward", distance_m=0.05, client_id="test")
    assert pushed["ok"] is True, pushed
    time.sleep(0.3)
    moved = np.asarray(_truth(desk)["orange_box"]["xyz_m"][:2]) - start
    assert moved[0] == pytest.approx(0.05, abs=0.02), moved
    assert _truth(desk)["pick_cube"]["in_bin"] == "orange_box"  # it went with the box


def test_reset_puts_the_box_back(desk):
    _reset(desk)
    home = LAYOUTS["sort_colors_box"].objects["orange_box"][:2]
    assert _truth(desk)["orange_box"]["xyz_m"][:2] == pytest.approx(list(home), abs=0.003)


def test_the_box_is_too_big_to_pick(desk):
    _reset(desk)
    refused = desk.pick("orange_box", client_id="test")
    assert refused["ok"] is False and refused["reason"] == "invalid_request"
    assert "push" in refused["detail"]
