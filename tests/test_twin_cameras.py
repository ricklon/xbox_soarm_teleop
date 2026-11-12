"""Camera views rendered by the twin service (offscreen, on the loop thread)."""

import io
import logging
import warnings

import numpy as np
import pytest
from PIL import Image

from xbox_soarm_teleop.twin import TwinClosed, build_twin
from xbox_soarm_teleop.twin.cameras import VIEWS

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)


@pytest.fixture(scope="module")
def twin():
    service = build_twin(scene="pick_place_basic", with_controller=False)
    service.tick()
    yield service
    service.close()


def shot(twin, view, **kw) -> np.ndarray:
    future = twin.render(view, **kw)
    twin.tick()
    return np.asarray(Image.open(io.BytesIO(future.result(timeout=0))).convert("RGB"), dtype=float)


@pytest.mark.parametrize("view", VIEWS)
def test_views_render_the_scene(twin, view):
    image = shot(twin, view)
    assert image.shape == (240, 320, 3)
    assert image.std() > 10  # not blank


def test_wrist_view_follows_the_gripper(twin):
    before = shot(twin, "wrist")
    future = twin.submit({"type": "move_ee_to", "params": {"xyz_m": [0.25, 0.12, 0.25]}})
    twin.tick()
    action = future.result(timeout=0)
    for _ in range(300):
        if twin.wait(action.id, 0).state.terminal:
            break
        twin.tick()
    after = shot(twin, "wrist")
    assert np.abs(after - before).mean() > 10
    overhead_png = twin.render("overhead", width=160, height=120, fmt="png")
    twin.tick()
    assert overhead_png.result(timeout=0).startswith(b"\x89PNG")


def test_one_render_per_tick(twin):
    first, second = twin.render("front"), twin.render("side")
    twin.tick()
    assert first.done() and not second.done()
    twin.tick()
    assert second.done()


def test_bad_requests(twin):
    with pytest.raises(ValueError):
        twin.render("ceiling")
    future = twin.render("wrist", width=4000, height=10)
    twin.tick()
    with pytest.raises(ValueError):
        future.result(timeout=0)


def test_render_after_close_fails():
    service = build_twin(scene="default", with_controller=False)
    pending = service.render("overhead")
    service.close()
    with pytest.raises(TwinClosed):
        pending.result(timeout=0)


# ------------------------------------------------- the viewer window's camera
def test_dpad_right_and_left_step_through_the_window_views():
    from xbox_soarm_teleop.twin.cameras import WINDOW_VIEWS, next_view

    assert WINDOW_VIEWS == ("free", "wrist", "overhead", "front", "side")
    assert next_view("free", 1.0, 0.0) == "wrist"
    assert next_view("wrist", -1.0, 0.0) == "free"
    assert next_view("side", 1.0, 0.0) == "free"  # wraps
    assert next_view("free", -1.0, 0.0) == "side"


def test_holding_the_dpad_switches_only_once():
    from xbox_soarm_teleop.twin.cameras import next_view

    assert next_view("wrist", 1.0, 1.0) == "wrist"
    assert next_view("wrist", 0.0, 1.0) == "wrist"


def test_point_camera_sets_each_view():
    import mujoco

    from xbox_soarm_teleop.twin import build_twin
    from xbox_soarm_teleop.twin.cameras import FIXED_VIEWS, point_camera

    service = build_twin(scene="pick_place_basic", with_controller=False)
    model = service.sim.model
    camera = mujoco.MjvCamera()
    point_camera(camera, model, "wrist")
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FIXED
    assert camera.fixedcamid == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    point_camera(camera, model, "overhead")
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE and camera.fixedcamid == -1
    assert camera.elevation == FIXED_VIEWS["overhead"]["elevation"]
    point_camera(camera, model, "free")  # back to the mouse, left where it was
    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert camera.elevation == FIXED_VIEWS["overhead"]["elevation"]
    service.close()


def test_window_camera_follows_the_dpad_without_taking_control():
    import mujoco

    from xbox_soarm_teleop.cli import twin as twin_cli
    from xbox_soarm_teleop.teleoperators.xbox import XboxState
    from xbox_soarm_teleop.twin import build_twin

    class Window:
        def __init__(self):
            self.cam = mujoco.MjvCamera()

        def lock(self):
            import contextlib

            return contextlib.nullcontext()

    service = build_twin(scene="pick_place_basic", with_controller=False)
    service.tick(XboxState(dpad_x=1.0))
    assert service.last_human.dpad_x == 1.0
    assert service.observe()["control"]["human_in_control"] is False
    window = Window()
    view = twin_cli._switch_view(window, service.sim.model, "free", 1.0, 0.0)
    assert view == "wrist" and window.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
    assert twin_cli._switch_view(object(), service.sim.model, "free", 1.0, 0.0) == "free"  # headless
    service.close()


def test_rig_can_draw_a_different_model_than_the_one_simulated(twin):
    """A rig that mirrors the simulator's state into its own model draws the same picture."""
    from xbox_soarm_teleop.cli.simulate_mujoco import URDF_PATH, load_model_with_cameras
    from xbox_soarm_teleop.twin.cameras import CameraRig

    twin.submit({"type": "move_ee_by", "params": {"dxyz_m": [0.0, 0.05, 0.04]}})
    for _ in range(80):
        twin.tick()
    simulated = twin.sim.model
    drawn = load_model_with_cameras(str(URDF_PATH), scene="pick_place_basic")  # a second copy
    direct = CameraRig(simulated, twin.sim.data)
    mirrored = CameraRig(drawn, state_model=simulated)
    try:
        for view in ("wrist", "front"):
            expected = direct.render(view, 320, 240).astype(float)
            actual = mirrored.render(view, 320, 240, twin.sim.data).astype(float)
            assert np.abs(actual - expected).mean() < 0.5, view
    finally:
        direct.close()
        mirrored.close()


def test_rig_refuses_a_state_model_with_a_different_layout(twin):
    from xbox_soarm_teleop.cli.simulate_mujoco import URDF_PATH, load_model_with_cameras
    from xbox_soarm_teleop.twin.cameras import CameraRig

    other = load_model_with_cameras(str(URDF_PATH), scene="stack")
    with pytest.raises(ValueError):
        CameraRig(other, state_model=twin.sim.model)
