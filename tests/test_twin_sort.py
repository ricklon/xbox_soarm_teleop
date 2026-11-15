"""Sorting: a table scene with three bins, and tasks that say what goes where.

The scene recipe (``sim/table_scenes.py``) puts the bins on the table; a
layout (``twin/layouts.py``) puts the objects out. Neither says what goes
where: that is the request's job (``twin/tasks.py``), so the same scene serves
"sort by colour", "swap red and blue" and "only the red one", each asked in
several ways. ``observe`` says which bin each object is in (hidden in
no-coordinates mode) and never what the rule is. A scripted run with the true
positions must finish every task, so no task is impossible.
"""

import json
import logging
import warnings

import mujoco
import pytest

from tests.twin_helpers import jaws_around
from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
from xbox_soarm_teleop.cli.hub import build_parser as hub_parser
from xbox_soarm_teleop.cli.twin import SORT_SCENES
from xbox_soarm_teleop.sim.table_scenes import TABLE_SCENES
from xbox_soarm_teleop.twin import build_twin, scenarios
from xbox_soarm_teleop.twin.grasp import GraspSettings
from xbox_soarm_teleop.twin.layouts import BINS, GOALS, LAYOUTS, layouts_for
from xbox_soarm_teleop.twin.tasks import TASKS, grade

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

SORT_LAYOUTS = [layout.name for layout in layouts_for("sort_bins", include_hidden=True)]


def _geom_names(scene: str) -> set[str]:
    model = sim_cli.load_model_with_cameras(str(sim_cli.URDF_PATH), scene=scene)
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)}


def test_the_single_box_scene_is_unchanged():
    names = _geom_names("pick_place_basic")
    for part in ("base", "wall_left", "wall_right", "wall_back", "wall_front"):
        assert f"goal_{part}" in names
    assert not any("bin" in (name or "") for name in names)
    assert GOALS["pick_place_basic"].center_xyz_m == pytest.approx((0.29, 0.13, 0.125))
    assert GOALS["pick_place_basic"].inner_size_m == pytest.approx((0.09, 0.09, 0.06))
    assert "label" not in GOALS["pick_place_basic"].summary()


def test_the_sort_scene_has_three_bins_and_no_goal_box():
    names = _geom_names("sort_bins")
    for bin_ in TABLE_SCENES["sort_bins"].bins:
        assert f"{bin_.name}_base" in names
    assert "goal_base" not in names and "pick_place_table" in names
    assert [b.name for b in BINS["sort_bins"]] == ["red_bin", "blue_bin", "green_bin"]
    assert "sort_bins" not in GOALS


def test_the_command_lines_offer_the_sort_scenes():
    assert set(SORT_SCENES) == set(SORT_LAYOUTS)
    assert hub_parser().parse_args(["--scene", "sort_colors"]).scene == "sort_colors"


def test_tasks_are_asked_in_their_scenes_in_several_ways():
    assert {"by_colour", "swap", "only_red", "mismatch"} <= set(TASKS)
    for task in TASKS.values():
        assert task.scene in SORT_LAYOUTS and task.requests
        assert len(set(task.requests)) == len(task.requests)
        assert bool(task.expect) != bool(task.check), task.name
        objects = set(LAYOUTS[task.scene].objects)
        assert set(task.target()) == objects, task.name
        bins = {b.name for b in BINS["sort_bins"]}
        assert {b for b in task.target().values() if b} <= bins, task.name
        if task.check is not None:
            assert task.check(task.example) and task.check_text
    assert len(TASKS["by_colour"].requests) >= 3


def test_grading_against_an_expectation_and_a_check():
    by_colour = TASKS["by_colour"]
    half = grade(by_colour, {"pick_cube": "red_bin", "blue_cube": "green_bin", "green_cube": None})
    assert (half["done"], half["right"], half["wrong"], half["unsorted"]) == (False, 1, 1, 1)
    assert grade(by_colour, dict(by_colour.expect))["done"] is True
    swap = TASKS["swap"]  # moving the green cube is wrong: the request said leave it
    moved = {**swap.expect, "green_cube": "green_bin"}
    assert grade(swap, moved)["wrong"] == 1 and grade(swap, moved)["done"] is False
    mismatch = TASKS["mismatch"]
    assert grade(mismatch, mismatch.example)["done"] is True
    assert grade(mismatch, dict(by_colour.expect))["done"] is False
    two_in_one = {"pick_cube": "blue_bin", "blue_cube": "red_bin", "green_cube": "red_bin"}
    assert grade(mismatch, two_in_one)["done"] is False


@pytest.fixture(scope="module")
def twin():
    service = build_twin(scene="sort_colors", with_controller=False)
    service.tick()
    yield service
    service.close()


def test_observe_says_which_bin_each_object_is_in_and_never_the_rule(twin):
    scenarios.reset(twin)
    scene = twin.observe()["scene"]
    assert [b["name"] for b in scene["bins"]] == ["red_bin", "blue_bin", "green_bin"]
    assert scene["bins"][0]["label"] == "red bin"
    assert "goal" not in scene and "sorting" not in scene
    assert set(scene["objects"]) == {"pick_cube", "blue_cube", "green_cube"}
    assert all(entry["in_bin"] is None for entry in scene["objects"].values())
    assert twin.table_top_z == pytest.approx(0.12)
    listed = json.dumps(twin.list_scenes())
    assert "red_bin" not in listed and "sort_into" not in listed

    scenarios.place_in_bin(twin, "pick_cube", "red_bin")
    scenarios.place_in_bin(twin, "blue_cube", "green_bin")
    for _ in range(50):
        twin.tick()
    objects = twin.observe()["scene"]["objects"]
    assert objects["pick_cube"]["in_bin"] == "red_bin"
    assert objects["blue_cube"]["in_bin"] == "green_bin"
    assert objects["green_cube"]["in_bin"] is None


def test_no_coordinates_hides_the_bins_contents_but_not_the_bins():
    service = build_twin(scene="sort_colors", with_controller=False, hide_positions=True)
    try:
        service.tick()
        scene = service.observe()["scene"]
        assert len(scene["bins"]) == 3
        assert all("in_bin" not in entry for entry in scene["objects"].values())
        truth = service.observe_truth()["scene"]
        assert all(entry["in_bin"] is None for entry in truth["objects"].values())
    finally:
        service.close()


def test_check_success_does_not_pretend_to_judge_bins(twin):
    result = twin.check_success("pick_cube")
    assert result["ok"] is False and result["reason"] == "not_supported"
    assert "bin" in result["detail"]


def test_layouts_switch_within_the_sort_scene(twin):
    names = [scene["name"] for scene in twin.list_scenes()]
    assert "sort_colors" in names and "pick_place_basic" not in names
    future = twin.load_scene("sort_colors_decoy")
    twin.tick()
    future.result(timeout=0)
    assert "yellow_cube" in twin.observe()["scene"]["objects"]
    future = twin.load_scene("sort_colors")
    twin.tick()
    future.result(timeout=0)


def _do_by_script(service, target: dict) -> None:
    """Carry out an arrangement with the true positions: the proof a task can be done."""

    def run(request):
        future = service.submit(request)
        service.tick()
        status = future.result(timeout=0)
        for _ in range(3000):
            now = service.wait(status.id, 0)
            if now.state.terminal:
                assert now.state.value.lower() == "succeeded", (request, now)
                return
            service.tick()

    def jaw():
        return service.observe()["jaw_center_xyz_m"]

    bins = {b["name"]: b for b in service.observe()["scene"]["bins"]}
    for name, bin_name in target.items():
        if bin_name is None:
            continue
        jaws_around(run, service.observe, name)
        width = service.observe()["scene"]["objects"][name]["grip_width_m"]
        run({"type": "set_gripper", "params": {"position": GraspSettings().gripper_for_opening(width)}})
        assert service.grasp.attached == name
        run({"type": "move_ee_by", "params": {"dxyz_m": [0, 0, 0.22 - jaw()[2]]}})
        cx, cy, floor = bins[bin_name]["center_xyz_m"]
        for _ in range(2):
            run({"type": "move_ee_by", "params": {"dxyz_m": [cx - jaw()[0], cy - jaw()[1], 0]}})
        run({"type": "move_ee_by", "params": {"dxyz_m": [0, 0, floor + 0.04 - jaw()[2]]}})
        run({"type": "set_gripper", "params": {"position": 0.0}})
        run({"type": "move_ee_by", "params": {"dxyz_m": [0, 0, 0.05]}})
        run({"type": "home"})
        for _ in range(100):
            service.tick()


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_every_task_can_be_done(task_name):
    task = TASKS[task_name]
    service = build_twin(scene=task.scene, with_controller=False)
    try:
        service.tick()
        _do_by_script(service, task.target())
        final = {name: entry["in_bin"] for name, entry in service.observe()["scene"]["objects"].items()}
        # Every object where the arrangement says, so homing knocked nothing out.
        assert final == task.target()
        assert grade(task, final)["done"] is True
    finally:
        service.close()


def _visible_pixels(service, renderer, name: str) -> int:
    from xbox_soarm_teleop.twin.cameras import point_camera

    camera = mujoco.MjvCamera()
    point_camera(camera, service.sim.model, "front")
    renderer.update_scene(service.sim.data, camera=camera)
    seg = renderer.render()
    body = service.grasp.objects[name].body_id
    geoms = [i for i in range(service.sim.model.ngeom) if service.sim.model.geom_bodyid[i] == body]
    on_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
    return int((on_geom & (seg[..., 0][..., None] == geoms).any(-1)).sum())


@pytest.mark.parametrize("layout", SORT_LAYOUTS)
def test_the_front_camera_sees_every_object_at_the_start(layout):
    """locate looks from the front by default. The bins stand between that
    camera and the table, so they must be low enough to see past, and an object
    under the arm at home is hidden from it."""
    service = build_twin(scene=layout, with_controller=False)
    renderer = mujoco.Renderer(service.sim.model, 480, 640)
    renderer.enable_segmentation_rendering()
    try:
        service.tick()
        model = service.sim.model
        names = list(service.observe()["scene"]["objects"])
        with_bins = {name: _visible_pixels(service, renderer, name) for name in names}
        bin_geoms = [i for i in range(model.ngeom) if "_bin_" in (model.geom(i).name or "")]
        saved = model.geom_pos[bin_geoms].copy()
        model.geom_pos[bin_geoms, 2] -= 5.0  # out of the picture
        mujoco.mj_forward(model, service.sim.data)
        alone = {name: _visible_pixels(service, renderer, name) for name in names}
        model.geom_pos[bin_geoms] = saved
        mujoco.mj_forward(model, service.sim.data)
        for name in names:
            # Clearly in view (a 3 cm cube fully seen is about 900 pixels), so not
            # under the arm at home either; and the bins hide little of it.
            assert alone[name] >= 500, (name, alone[name])
            assert with_bins[name] >= 0.8 * alone[name], (name, with_bins[name], alone[name])
    finally:
        renderer.close()
        service.close()
