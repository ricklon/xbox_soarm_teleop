"""The desk cell: a work surface the size of the arm's reach, at the base's height.

The SO-101 grasps from above in a ring around its base, from about 17 to
36 cm out. The sort scene's recipe now carries its own table (a desk board at
the base's height, as on a real desk the arm is clamped to), its own trays
(at the arm's sides, inside the enforced workspace) and its own cameras. The
single-box scene keeps its raised table, trays and cameras exactly as they
were, so its benchmarks still compare.
"""

import logging
import math
import warnings

import mujoco
import numpy as np
import pytest

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
from xbox_soarm_teleop.sim.table_scenes import TABLE_SCENES
from xbox_soarm_teleop.twin import build_twin, scenarios, scene_workspace, table_top_z
from xbox_soarm_teleop.twin.cameras import FIXED_VIEWS, point_camera
from xbox_soarm_teleop.twin.projection import camera_geometry, pixel_to_xyz

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

DESK = TABLE_SCENES["sort_bins"]


def test_the_raised_table_scene_is_unchanged():
    assert table_top_z("pick_place_basic") == pytest.approx(0.12)
    assert scene_workspace("pick_place_basic")["z"][0] == pytest.approx(0.125)
    model = sim_cli.load_model_with_cameras(str(sim_cli.URDF_PATH), scene="pick_place_basic")
    table = model.geom("pick_place_table")
    np.testing.assert_allclose(table.size, [0.16, 0.14, 0.06])
    np.testing.assert_allclose(table.pos, [0.31, 0.04, 0.06])
    names = {model.camera(i).name for i in range(model.ncam)}
    assert names == {"wrist"}  # its views are still the shared presets


def test_the_desk_is_at_the_base_and_the_arm_can_reach_down_to_it():
    top = table_top_z("sort_bins")
    assert top == pytest.approx(DESK.table.top_z) and top < 0.01
    low = scene_workspace("sort_bins")["z"][0]
    assert low == pytest.approx(top + 0.005)  # the tool stops 5 mm over the desk, not at 5 cm


def test_the_trays_are_inside_the_enforced_workspace_and_apart():
    bounds = scene_workspace("sort_bins")
    for bin_ in DESK.bins:
        x, y = bin_.center_xy_m
        assert bounds["x"][0] + 0.03 <= x <= bounds["x"][1] - 0.03, bin_.name
        assert bounds["y"][0] + 0.03 <= y <= bounds["y"][1] - 0.03, bin_.name
        assert 0.20 <= math.hypot(x, y) <= 0.33, bin_.name
        assert bin_.inner_half_m[0] >= 0.05  # 10 cm inside: a cube lands turned at the sides
    for a in DESK.bins:
        for b in DESK.bins:
            if a is not b:
                assert math.dist(a.center_xy_m, b.center_xy_m) >= 0.12


def test_the_desk_has_its_own_cameras_aimed_at_it():
    model = sim_cli.load_model_with_cameras(str(sim_cli.URDF_PATH), scene="sort_bins")
    names = {model.camera(i).name for i in range(model.ncam)}
    assert {"front", "overhead", "side", "wrist"} <= names
    for view in ("front", "overhead", "side"):
        assert model.camera(view).fovy[0] == pytest.approx(model.vis.global_.fovy)
    camera = mujoco.MjvCamera()
    point_camera(camera, model, "overhead")
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FIXED
    assert camera.fixedcamid == model.camera("overhead").id
    point_camera(camera, sim_cli.load_model_with_cameras(str(sim_cli.URDF_PATH)), "overhead")
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert camera.elevation == FIXED_VIEWS["overhead"]["elevation"]


@pytest.fixture(scope="module")
def desk():
    service = build_twin(scene="sort_colors", with_controller=False)
    service.tick()
    yield service
    service.close()


def _yaw(service, name: str) -> float:
    rotation = np.zeros(9)
    obj = service.grasp.objects[name]
    mujoco.mju_quat2Mat(rotation, service.sim.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7])
    x_axis = rotation.reshape(3, 3)[:, 0]
    return math.degrees(math.atan2(x_axis[1], x_axis[0]))


def _square_to(angle: float, facing: float) -> float:
    """How far a cube turned ``angle`` is from square to a direction (0..45 deg)."""
    off = (angle - facing) % 90.0
    return min(off, 90.0 - off)


def test_objects_rest_on_the_desk_facing_the_base_after_a_reset(desk):
    for _ in range(2):
        for _ in range(60):
            desk.tick()
        for name, entry in desk.observe()["scene"]["objects"].items():
            x, y, z = entry["xyz_m"]
            half = desk.grasp.half_extents_world(name)[2]
            assert z == pytest.approx(desk.table_top_z + half, abs=0.002), name
            assert _square_to(_yaw(desk, name), math.degrees(math.atan2(y, x))) < 3.0, name
        scenarios.reset(desk)


@pytest.mark.parametrize("view", ["front", "overhead"])
def test_a_perfect_pixel_projects_back_onto_the_object(desk, view):
    """Pixel to xyz with the cell's own cameras: the centroid of an object's
    pixels, projected onto its rest height, lands on its true center."""
    model, data = desk.sim.model, desk.sim.data
    renderer = mujoco.Renderer(model, 480, 640)
    renderer.enable_segmentation_rendering()
    try:
        camera = mujoco.MjvCamera()
        point_camera(camera, model, view)
        renderer.update_scene(data, camera=camera)
        seg = renderer.render()
        geometry = camera_geometry(model, data, view, 640, 480)
        for name, entry in desk.observe()["scene"]["objects"].items():
            body = desk.grasp.objects[name].body_id
            geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body]
            mask = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], geoms)
            assert mask.sum() > 300, (view, name)
            rows, cols = np.nonzero(mask)
            truth = np.array(entry["xyz_m"])
            guess = pixel_to_xyz(geometry, cols.mean() + 0.5, rows.mean() + 0.5, truth[2])
            assert np.linalg.norm(guess[:2] - truth[:2]) < 0.012, (view, name, guess, truth)
    finally:
        renderer.close()
