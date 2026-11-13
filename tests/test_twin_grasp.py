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


SIDEWAYS = -1.5  # wrist roll that puts the jaws side to side


def roll(twin, angle_rad):
    return run(twin, {"type": "set_wrist_roll", "params": {"angle_rad": float(angle_rad)}})


def over(twin, name="pick_cube", wrist=SIDEWAYS, dz=0.025):
    """Put the open jaws around an object, clear of it, side to side by default."""
    xyz = twin.grasp.position(name)
    move(twin, xyz + [0.0, 0.0, 0.06])
    roll(twin, wrist)
    move(twin, xyz + [0.0, 0.0, dz])


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


def test_top_and_bottom_do_not_count_on_a_rigid_object(twin):
    """At roll 0 the jaws close nearly top to bottom: refused, with the roll to use."""
    assert twin.grasp.grip_faces("pick_cube") == [0, 1]  # its sides, not top and bottom
    over(twin, wrist=0.0)
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached is None
    slipped = [event for event in events(twin) if "slipped" in event]
    assert slipped and "set_wrist_roll(angle_rad=" in slipped[-1]


@pytest.mark.parametrize("dz", [0.0, 0.025])  # at the cube's center, and the advised 2.5 cm above
def test_the_suggested_roll_makes_the_grasp_work(twin, dz):
    """What an agent does with a refusal: open, turn the wrist as told, close again."""
    over(twin, wrist=0.0, dz=dz)
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    suggested = twin.grasp.squaring_roll("pick_cube")
    assert twin.grasp.attached is None and suggested is not None
    assert abs(abs(suggested) - np.pi / 2) < 0.3
    grip(twin, 0.0)
    over(twin, wrist=suggested, dz=dz)
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached == "pick_cube"


def test_a_cube_turned_too_far_says_so(twin):
    """Skewed 45 degrees to the arm, no wrist roll can square the jaws to it."""
    xyz = twin.grasp.position("pick_cube")
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), np.deg2rad(45))
    obj = twin.grasp.objects["pick_cube"]
    twin.sim.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7] = quat
    mujoco.mj_forward(twin.sim.model, twin.sim.data)
    move(twin, xyz + [0.0, 0.0, 0.06])
    roll(twin, SIDEWAYS)
    move(twin, twin.grasp.position("pick_cube") + [0.0, 0.0, 0.025])
    assert twin.grasp.best_square_deg("pick_cube") > SETTINGS.rigid_align_tol_deg
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached is None
    assert any("turned too far" in event for event in events(twin))


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
    roll(twin, 0.0)  # top to bottom: fine for something soft
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


def test_observation_gives_the_goal_box(twin):
    goal = twin.observe()["scene"]["goal"]
    assert goal["name"] == "goal_box"
    assert goal["center_xyz_m"] == pytest.approx([0.29, 0.13, 0.125], abs=1e-3)
    assert goal["inner_size_m"] == pytest.approx([0.09, 0.09, 0.06])
    assert twin.observe()["scene"]["objects"]["pick_cube"]["in_goal"] is False


@pytest.mark.parametrize(
    "xy, inside",
    [((0.29, 0.13), True), ((0.31, 0.15), True), ((0.34, -0.03), False)],
)
def test_in_goal_only_when_resting_inside_the_walls(xy, inside):
    twin = build_twin(scene="pick_place_easy", with_controller=False)
    twin.tick()
    twin.grasp.place("bean_bag", np.array([*xy, 0.20]))
    mujoco.mj_forward(twin.sim.model, twin.sim.data)
    for _ in range(150):
        twin.tick()
    bag = twin.observe()["scene"]["objects"]["bean_bag"]
    assert bag["in_goal"] is inside, bag


def test_resting_on_a_wall_is_not_in_the_goal():
    """Where the bean bag ended up in a live session: across the box's left wall."""
    from xbox_soarm_teleop.twin.layouts import GOALS

    goal = GOALS["pick_place_basic"]
    bag_half = (0.03, 0.025, 0.012)
    assert goal.holds((0.259, 0.12, 0.155), bag_half) is False
    assert goal.holds((0.29, 0.13, 0.137), bag_half) is True
