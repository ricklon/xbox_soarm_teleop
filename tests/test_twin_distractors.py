"""Distractor scenes: the target among objects that look like it.

Pointing at "the red cube" is easy when it is the only thing on the table.
These scenes put objects beside it that differ in one way: colour (a blue and
a green cube), shape (a red bar), size (a small red cube), or all of them
(clutter). They are hidden from ``list_scenes`` (the workshop's scene picker
stays as it is) and loaded by name.
"""

import logging
import warnings

import numpy as np
import pytest

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
from xbox_soarm_teleop.twin import build_twin
from xbox_soarm_teleop.twin.judge import truth_labels
from xbox_soarm_teleop.twin.layouts import DISTRACTOR_LAYOUTS, LAYOUTS, layouts_for, object_description
from xbox_soarm_teleop.twin.scenarios import (
    grasp_and_lift,
    place_in_goal,
    place_with_distractors,
    reset,
)

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

NEW_OBJECTS = {"red_cube_small", "blue_cube", "red_bar", "green_cube", "yellow_cube"}


@pytest.fixture(scope="module")
def twin():
    service = build_twin(scene="pick_place_basic", with_controller=False)
    service.tick()
    return service


def _load(twin, name):
    future = twin.load_scene(name, client_id="t")
    twin.tick()
    return future.result(timeout=0)


def test_the_model_has_the_new_objects_and_the_old_scenes_park_them(twin):
    assert NEW_OBJECTS <= set(twin.grasp.objects)
    _load(twin, "pick_place_basic")
    assert list(twin.observe()["scene"]["objects"]) == ["pick_cube"]
    _load(twin, "pick_place_easy")
    assert list(twin.observe()["scene"]["objects"]) == ["bean_bag"]


def test_new_objects_have_the_sizes_the_grasp_reads(twin):
    assert twin.grasp.size("red_cube_small") == pytest.approx([0.03] * 3)
    assert twin.grasp.size("red_bar") == pytest.approx([0.03, 0.07, 0.03])
    for name in NEW_OBJECTS:
        assert not twin.grasp.objects[name].soft  # rigid, like the cube


def test_distractor_scenes_are_hidden_but_load_by_name(twin):
    listed = [scene["name"] for scene in twin.list_scenes()]
    assert listed == ["pick_place_easy", "pick_place_basic"]
    assert set(DISTRACTOR_LAYOUTS) == {
        "distractors_color",
        "distractors_shape",
        "distractors_size",
        "distractors_clutter",
    }
    for name in DISTRACTOR_LAYOUTS:
        layout = LAYOUTS[name]
        assert layout.target == "pick_cube" and not layout.listed
        summary = _load(twin, name)
        assert summary["name"] == name and summary["target"] == "pick_cube"
        assert set(twin.observe()["scene"]["objects"]) == set(layout.objects)
    assert [layout.name for layout in layouts_for("pick_place_basic")] == listed


def test_every_object_has_one_unambiguous_description_per_scene():
    for name in DISTRACTOR_LAYOUTS:
        layout = LAYOUTS[name]
        words = [object_description(obj, layout) for obj in layout.objects]
        assert len(set(words)) == len(words), (name, words)
    assert object_description("pick_cube", LAYOUTS["distractors_size"]) == "large red cube"
    assert object_description("pick_cube", LAYOUTS["distractors_color"]) == "red cube"
    assert object_description("pick_cube") == "red cube"


@pytest.mark.parametrize("scene", sorted(DISTRACTOR_LAYOUTS))
def test_the_target_can_be_picked_without_moving_the_distractors(scene):
    """The control: with true positions, the scripted pick takes the target
    and leaves everything else where it was. A scene failing this is
    unsolvable, not hard."""
    service = build_twin(scene=scene, with_controller=False)
    service.tick()
    others = [name for name in LAYOUTS[scene].objects if name != "pick_cube"]
    before = {name: service.grasp.position(name).copy() for name in others}
    x, y, z = service.grasp.position("pick_cube")
    assert grasp_and_lift(service, "pick_cube", [float(x), float(y), float(z) + 0.08])
    for name in others:
        moved = np.linalg.norm(service.grasp.position(name) - before[name])
        assert moved < 0.005, (scene, name, moved)


def test_random_placements_do_not_overlap_and_stay_on_the_table(twin):
    rng = np.random.default_rng(5)
    center, half = sim_cli.PICK_PLACE_TABLE_CENTER, sim_cli.PICK_PLACE_TABLE_HALF_EXTENTS
    names = ["pick_cube", "red_cube_small", "red_bar", "blue_cube", "yellow_cube"]
    for _ in range(10):
        placed = place_with_distractors(twin, "pick_cube", names[1:], rng)
        assert set(placed) == set(names)
        # Only these are on the table; the rest are parked.
        assert set(twin.grasp.positions()) == set(names)
        for name, xyz in placed.items():
            extent = twin.grasp.half_extents_world(name)
            assert abs(xyz[0] - center[0]) + extent[0] <= half[0] + 1e-9
            assert abs(xyz[1] - center[1]) + extent[1] <= half[1] + 1e-9
            assert truth_labels(twin, name)["in_goal"] is False
        # Footprints (as turned) keep a gap between every pair.
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ea, eb = twin.grasp.half_extents_world(a), twin.grasp.half_extents_world(b)
                gap_x = abs(placed[a][0] - placed[b][0]) - ea[0] - eb[0]
                gap_y = abs(placed[a][1] - placed[b][1]) - ea[1] - eb[1]
                assert max(gap_x, gap_y) >= 0.02, (a, b)


def test_the_same_seed_gives_the_same_scene(twin):
    a = place_with_distractors(twin, "pick_cube", ["red_bar", "blue_cube"], np.random.default_rng(9))
    b = place_with_distractors(twin, "pick_cube", ["red_bar", "blue_cube"], np.random.default_rng(9))
    for name in a:
        assert a[name] == pytest.approx(b[name])


def test_a_distractor_in_the_box_does_not_put_the_target_there(twin):
    reset(twin)
    place_with_distractors(twin, "pick_cube", ["blue_cube"], np.random.default_rng(2))
    place_in_goal(twin, "blue_cube", park_others=False)
    assert truth_labels(twin, "blue_cube")["in_goal"] is True
    assert truth_labels(twin, "pick_cube")["in_goal"] is False
    assert set(twin.grasp.positions()) == {"pick_cube", "blue_cube"}


@pytest.mark.parametrize("name", sorted(NEW_OBJECTS))
def test_every_new_object_can_be_picked_up(name):
    from xbox_soarm_teleop.twin.scenarios import place_on_table

    service = build_twin(scene="pick_place_basic", with_controller=False)
    service.tick()
    place_on_table(service, name, 0.30, -0.03)
    assert grasp_and_lift(service, name, [0.30, -0.03, 0.22])


def test_the_twin_command_offers_the_distractor_scenes():
    from xbox_soarm_teleop.cli import twin as twin_cli

    assert twin_cli.DISTRACTOR_SCENES == DISTRACTOR_LAYOUTS
    assert (
        twin_cli.build_parser().parse_args(["--scene", "distractors_size"]).scene == "distractors_size"
    )
