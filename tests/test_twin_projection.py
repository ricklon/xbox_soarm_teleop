"""Pixel <-> xyz for the twin's cameras, checked against MuJoCo's own renderer.

A "perfect pointer" is simulated by segmentation rendering: the pixels that
show the object are known exactly, so their centroid is the pixel an ideal
pointing model would give. Turning that pixel back into xyz must land on the
object's true position.
"""

import logging
import warnings

import mujoco
import numpy as np
import pytest

from xbox_soarm_teleop.schemas.agent import ActionState
from xbox_soarm_teleop.twin import build_twin
from xbox_soarm_teleop.twin.cameras import point_camera
from xbox_soarm_teleop.twin.projection import camera_geometry, pixel_to_xyz, ray_to_plane

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

W, H = 640, 480
# Clear of the arm at home, so nothing hides it from the overhead camera.
CUBE_XY = (0.30, -0.12)


@pytest.fixture(scope="module")
def scene():
    twin = build_twin(scene="pick_place_basic", with_controller=False)
    z = twin.grasp.position("pick_cube")[2]
    twin.grasp.place("pick_cube", np.array([*CUBE_XY, z]))
    mujoco.mj_forward(twin.sim.model, twin.sim.data)
    renderer = mujoco.Renderer(twin.sim.model, H, W)
    renderer.enable_segmentation_rendering()
    yield twin, renderer
    renderer.close()


def _run(twin, request):
    future = twin.submit(request)
    twin.tick()
    status = future.result(timeout=0)
    for _ in range(1500):
        status = twin.wait(status.id, 0)
        if status.state.terminal:
            break
        twin.tick()
    assert status.state is ActionState.SUCCEEDED


def _mask(twin, renderer, view, width=W, height=H):
    model = twin.sim.model
    camera = mujoco.MjvCamera()
    point_camera(camera, model, view)
    renderer.update_scene(twin.sim.data, camera=camera)
    seg = renderer.render()
    body = twin.grasp.objects["pick_cube"].body_id
    geoms = np.where(model.geom_bodyid == body)[0]
    return (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], geoms)


def _views(twin):
    """Every view, the wrist one aimed at the cube from behind and above."""
    yield from ("overhead", "front", "side")
    x, y, _ = twin.grasp.position("pick_cube")
    _run(twin, {"type": "move_ee_to", "params": {"xyz_m": [float(x) - 0.10, float(y), 0.22]}})
    yield "wrist"


def test_ray_meets_plane():
    origin = np.array([0.0, 0.0, 1.0])
    down = np.array([0.0, 0.6, -0.8])
    assert ray_to_plane(origin, down, 0.2) == pytest.approx([0.0, 0.6, 0.2])
    assert ray_to_plane(origin, -down, 0.2) is None  # behind
    assert ray_to_plane(origin, np.array([1.0, 0.0, 0.0]), 0.2) is None  # parallel


def test_project_and_ray_are_inverse(scene):
    twin, _ = scene
    point = np.array([0.31, 0.04, 0.15])
    for view in ("overhead", "front", "side", "wrist"):
        geometry = camera_geometry(twin.sim.model, twin.sim.data, view, 320, 240)
        u, v = geometry.project(point)
        assert pixel_to_xyz(geometry, u, v, point[2]) == pytest.approx(point, abs=1e-9)


def test_projection_matches_the_renderer_and_a_perfect_pixel_finds_the_object(scene):
    twin, renderer = scene
    results = {}
    for view in _views(twin):
        truth = twin.grasp.position("pick_cube")
        geometry = camera_geometry(twin.sim.model, twin.sim.data, view, W, H)
        mask = _mask(twin, renderer, view)
        assert mask.sum() > 200, view  # the cube is in sight
        rows, cols = np.nonzero(mask)

        # The cube's corners project onto the edges of its drawn silhouette
        # (the wrist camera sees it past the jaw, which hides part of it).
        half = twin.grasp.size("pick_cube") / 2.0
        signs = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        corners = np.array([geometry.project(truth + half * np.array(s)) for s in signs])
        if view != "wrist":
            drawn = (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)
            box = (*corners.min(axis=0), *corners.max(axis=0))
            assert np.allclose(box, drawn, atol=1.5), (view, box, drawn)

        # Its true center projects onto the cube's own pixels.
        u, v = geometry.project(truth)
        assert mask[int(v), int(u)], view

        # The perfect pixel (centroid of what is seen) maps back near the truth.
        centroid = (cols.mean() + 0.5, rows.mean() + 0.5)
        estimate = pixel_to_xyz(geometry, *centroid, float(truth[2]))
        results[view] = float(np.linalg.norm(estimate - truth))
    assert set(results) == {"overhead", "front", "side", "wrist"}
    for view, error in results.items():
        assert error < 0.01, (view, error)


def test_small_images_scale_the_same(scene):
    twin, _ = scene
    big = camera_geometry(twin.sim.model, twin.sim.data, "front", 640, 480)
    small = camera_geometry(twin.sim.model, twin.sim.data, "front", 320, 240)
    point = twin.grasp.position("pick_cube")
    assert np.array(small.project(point)) * 2 == pytest.approx(big.project(point), abs=1e-6)
