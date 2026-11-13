"""Grasping by jaw width: close to fit an object, not as far as the jaws go."""

import logging
import warnings

import mujoco
import numpy as np
import pytest

from xbox_soarm_teleop.schemas.agent import ActionState
from xbox_soarm_teleop.teleoperators.xbox import XboxState
from xbox_soarm_teleop.twin import build_parts, build_twin
from xbox_soarm_teleop.twin.grasp import GraspSettings

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

CUBE_WIDTH = 0.04
SETTINGS = GraspSettings()


@pytest.fixture(scope="module")
def parts():
    return build_parts(scene="pick_place_basic")


@pytest.fixture
def twin(parts):
    service = build_twin(parts=parts, with_controller=False)
    service.tick()
    return service


def run(twin, request, max_ticks: int = 3000):
    future = twin.submit(request, client_id="agent")
    twin.tick()
    status = future.result(timeout=0)
    for _ in range(max_ticks):
        status = twin.wait(status.id, 0)
        if status.state.terminal:
            break
        twin.tick()
    return status


def move(twin, xyz):
    return run(twin, {"type": "move_ee_to", "params": {"xyz_m": [float(v) for v in xyz]}})


def grip(twin, position):
    return run(twin, {"type": "set_gripper", "params": {"position": float(position)}})


def over(twin, name="pick_cube"):
    """Put the open jaws around an object, clear of it."""
    xyz = twin.grasp.position(name)
    move(twin, xyz + [0.0, 0.0, 0.06])
    move(twin, xyz + [0.0, 0.0, 0.025])


def events(twin) -> list[str]:
    return twin.observe()["control"]["events"]


def test_opening_and_position_convert_both_ways():
    assert SETTINGS.jaw_opening_m(1.0) == pytest.approx(SETTINGS.jaw_gap_closed_m)
    assert SETTINGS.jaw_opening_m(0.0) > 0.12
    for opening in (0.01, 0.04, 0.08):
        assert SETTINGS.jaw_opening_m(SETTINGS.gripper_for_opening(opening)) == pytest.approx(opening)
    assert SETTINGS.gripper_for_opening(1.0) == 0.0 and SETTINGS.gripper_for_opening(0.0) == 1.0


def test_jaw_opening_matches_the_model(twin):
    """The settings' opening is the gap from the tool point to the moving jaw."""
    model, data = twin.sim.model, twin.sim.data
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1_link")
    hit = np.zeros(1, dtype=np.int32)
    for position in (0.75, 0.85, 1.0):
        twin.sim.set_gripper(position)
        mujoco.mj_forward(model, data)
        start, travelled = twin.sim.get_ee_position().copy(), 0.0
        axis = twin.grasp.closing_axis()
        for _ in range(20):  # step past the fixed jaw, which the tool point sits on
            distance = mujoco.mj_ray(model, data, start, axis, None, 1, -1, hit)
            assert distance >= 0
            if model.geom_bodyid[hit[0]] == jaw:
                break
            start, travelled = start + axis * (distance + 1e-5), travelled + distance + 1e-5
        assert travelled + distance == pytest.approx(SETTINGS.jaw_opening_m(position), abs=0.002)


def test_observation_gives_sizes_and_the_jaw_opening(twin):
    obs = twin.observe()
    assert obs["scene"]["objects"]["pick_cube"]["size_m"] == pytest.approx([0.04, 0.04, 0.04])
    assert obs["gripper_opening_m"] == pytest.approx(SETTINGS.jaw_opening_m(0.0), abs=1e-3)
    grip(twin, 0.9)
    assert twin.observe()["gripper_opening_m"] == pytest.approx(SETTINGS.jaw_opening_m(0.9), abs=5e-3)


def test_closing_to_the_cubes_width_holds_it(twin):
    over(twin)
    assert grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH)).state is ActionState.SUCCEEDED
    assert twin.grasp.attached == "pick_cube"
    move(twin, twin.grasp.position("pick_cube") + [0.0, 0.0, 0.08])
    assert twin.observe()["scene"]["objects"]["pick_cube"]["attached"] is True


def test_not_closing_far_enough_holds_nothing(twin):
    over(twin)
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH + 0.02))
    assert twin.grasp.attached is None
    assert not any("pick_cube" in event for event in events(twin))


def test_closing_all_the_way_squeezes_the_cube_out(twin):
    over(twin)
    grip(twin, 1.0)
    assert twin.grasp.attached is None
    assert any("squeezed out" in event for event in events(twin))
    # Closing again without opening grabs nothing; opening first re-arms.
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached is None


def test_squeezing_a_held_cube_drops_it(twin):
    over(twin)
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    move(twin, twin.grasp.position("pick_cube") + [0.0, 0.0, 0.08])
    held_z = twin.grasp.position("pick_cube")[2]
    grip(twin, 1.0)
    for _ in range(60):
        twin.tick()
    assert twin.grasp.attached is None
    assert twin.grasp.position("pick_cube")[2] < held_z - 0.03


def test_a_little_past_the_width_still_holds(twin):
    over(twin)
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH - SETTINGS.rigid_squeeze_m + 0.002))
    assert twin.grasp.attached == "pick_cube"


def test_jaws_off_square_let_the_cube_slip(twin):
    cube = twin.grasp.position("pick_cube")
    move(twin, cube + [0.0, 0.0, 0.10])
    run(twin, {"type": "set_wrist_roll", "params": {"angle_rad": float(np.deg2rad(45))}})
    over(twin)
    width, off_square = twin.grasp.fit("pick_cube")
    assert off_square > SETTINGS.rigid_align_tol_deg
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached is None
    assert any("slipped" in event for event in events(twin))


def test_square_jaws_report_the_cubes_width(twin):
    over(twin)
    width, off_square = twin.grasp.fit("pick_cube")
    assert width == pytest.approx(CUBE_WIDTH)
    assert off_square < SETTINGS.rigid_align_tol_deg


def test_the_bean_bag_takes_any_closure_and_angle():
    twin = build_twin(scene="pick_place_easy", with_controller=False)
    twin.tick()
    bag = twin.grasp.position("bean_bag")
    move(twin, bag + [0.0, 0.0, 0.08])
    run(twin, {"type": "set_wrist_roll", "params": {"angle_rad": float(np.deg2rad(45))}})
    move(twin, bag + [0.0, 0.0, 0.01])
    grip(twin, 1.0)
    assert twin.grasp.attached == "bean_bag"
    move(twin, bag + [0.0, 0.0, 0.10])
    assert twin.grasp.attached == "bean_bag"
    grip(twin, 0.0)
    assert twin.grasp.attached is None
    assert "bean_bag released" in events(twin)


def test_a_person_closing_fully_stops_on_the_cube(twin):
    """The trigger and the page's Close button go to 1.0; the jaws stop on the cube."""
    over(twin)
    for _ in range(4):  # let the agent's move settle and control lapse to nobody
        twin.tick()
    twin.tick(XboxState(left_bumper=True, right_trigger=0.0))  # trigger at rest, then squeezed
    human = XboxState(left_bumper=True, right_trigger=1.0)
    for _ in range(120):
        twin.tick(human)
    obs = twin.observe()
    assert twin.grasp.attached == "pick_cube"
    assert obs["gripper"] < 0.9
    assert not any("squeezed out" in event for event in obs["control"]["events"])
    for _ in range(120):  # still held while they keep squeezing
        twin.tick(human)
    assert twin.grasp.attached == "pick_cube"
    for _ in range(120):
        twin.tick(XboxState(left_bumper=True, right_trigger=0.0))
    assert twin.grasp.attached is None
