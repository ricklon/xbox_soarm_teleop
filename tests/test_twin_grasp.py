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


def move_by(twin, dxyz):
    return run(twin, {"type": "move_ee_by", "params": {"dxyz_m": [float(v) for v in dxyz]}})


def jaw_center(twin):
    return np.array(twin.observe()["jaw_center_xyz_m"])


def over(twin, name="pick_cube", wrist=SIDEWAYS):
    """Put the open jaws around an object, the way the agent instructions say.

    Open a little wider than it, line the jaw center up above it, then come
    straight down until the jaw center is at its center (not below z 0.14).
    """
    width = float(min(twin.grasp.size(name)[:2]))
    grip(twin, SETTINGS.gripper_for_opening(width + 0.05))
    move(twin, twin.grasp.position(name) + [0.0, 0.0, 0.06])
    roll(twin, wrist)
    for _ in range(2):
        miss = twin.grasp.position(name) - jaw_center(twin)
        move_by(twin, [miss[0], miss[1], 0.0])
    move_by(twin, [0.0, 0.0, max(twin.grasp.position(name)[2], 0.14) - jaw_center(twin)[2]])


def tool_over(twin, name="pick_cube", wrist=SIDEWAYS, dz=0.025):
    """Put the tool point (not the jaw center) dz above an object, as agents first did."""
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
    tool_over(twin, wrist=0.0)
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached is None
    slipped = [event for event in events(twin) if "slipped" in event]
    assert slipped and "set_wrist_roll(angle_rad=" in slipped[-1]


@pytest.mark.parametrize("dz", [0.0, 0.025])
def test_the_suggested_roll_makes_the_grasp_work(twin, dz):
    """What an agent does with a refusal: open, turn the wrist as told, close again."""
    tool_over(twin, wrist=0.0, dz=dz)
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    suggested = twin.grasp.squaring_roll("pick_cube")
    assert twin.grasp.attached is None and suggested is not None
    assert abs(abs(suggested) - np.pi / 2) < 0.3
    grip(twin, 0.0)
    move(twin, twin.grasp.position("pick_cube") + [0.0, 0.0, 0.06])
    over(twin, wrist=suggested)
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached == "pick_cube"


@pytest.mark.parametrize("yaw_deg", [-30, 30])
def test_the_suggested_roll_turns_toward_square_not_away(twin, yaw_deg):
    """The closing axis is not square to the roll axis, so a roll tilts the jaws
    as well as turning them; the hint must come from trying rolls, not a plane."""
    xyz = twin.grasp.position("pick_cube")
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), np.deg2rad(yaw_deg))
    obj = twin.grasp.objects["pick_cube"]
    twin.sim.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7] = quat
    mujoco.mj_forward(twin.sim.model, twin.sim.data)
    over(twin)
    before = twin.grasp.fit("pick_cube")[1]
    suggested = twin.grasp.squaring_roll("pick_cube")
    predicted = twin.grasp.best_square_deg("pick_cube")
    assert predicted <= before
    move(twin, xyz + [0.0, 0.0, 0.10])  # turn the wrist well clear of it
    roll(twin, suggested)
    move(twin, xyz + [0.0, 0.0, 0.06])
    over(twin, wrist=suggested)
    assert twin.grasp.fit("pick_cube")[1] <= before + 1.0
    assert twin.grasp.fit("pick_cube")[1] == pytest.approx(predicted, abs=6.0)


@pytest.mark.parametrize("yaw_deg", [-20, 20])
def test_closing_on_a_slightly_turned_cube_squares_it(twin, yaw_deg):
    """Judged where the jaws first meet it, before a corner is caught and spun."""
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), np.deg2rad(yaw_deg))
    obj = twin.grasp.objects["pick_cube"]
    twin.sim.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7] = quat
    mujoco.mj_forward(twin.sim.model, twin.sim.data)
    over(twin)
    assert twin.grasp.extent_across("pick_cube") > CUBE_WIDTH + 0.01
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached == "pick_cube"
    assert twin.grasp.fit("pick_cube")[1] < 10.0  # turned square in the jaws


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


def follow(twin, event):
    """Do what a "missed" event says: open as told, then each move_by in turn."""
    import re

    opening = float(re.search(r"opening_m=([-\d.]+)", event).group(1))
    grip(twin, SETTINGS.gripper_for_opening(opening))
    for move in re.findall(r"move_by\(dx=([-\d.]+), dy=([-\d.]+), dz=([-\d.]+)\)", event):
        move_by(twin, [float(v) for v in move])


@pytest.mark.parametrize("wrist", [SIDEWAYS, -SIDEWAYS])
@pytest.mark.parametrize("dz", [0.0, 0.025, 0.05])
def test_the_tool_point_above_the_cube_misses_and_says_how_to_fix_it(twin, wrist, dz):
    """The tool point is the fixed jaw's face: putting it over the cube leaves the
    cube beside the pads. Doing what the refusal says centers the jaws on it."""
    tool_over(twin, wrist=wrist, dz=dz)
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached is None
    missed = [event for event in events(twin) if "missed" in event]
    assert missed and "set_gripper(opening_m=" in missed[-1] and "move_by(dx=" in missed[-1]
    follow(twin, missed[-1])
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached == "pick_cube"
    obs = twin.observe()
    held = np.array(obs["scene"]["objects"]["pick_cube"]["xyz_m"])
    assert np.linalg.norm(held - obs["jaw_center_xyz_m"]) < 0.01


def test_a_cube_past_the_fingertips_is_not_held(twin):
    """Held only by the last few millimetres of pad, as a live run once had it."""
    over(twin)
    data, body = twin.sim.data, twin.grasp._gripper_body
    rotation = data.xmat[body].reshape(3, 3)
    tips = data.xpos[body] + rotation @ np.array([0.012, 0.0, -0.118])  # 1.4 cm past the tips
    twin.grasp.place("pick_cube", tips)
    mujoco.mj_forward(twin.sim.model, twin.sim.data)
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached is None
    assert "missed" in events(twin)[-2]


def test_closing_near_nothing_says_where_the_object_is(twin):
    xyz = twin.grasp.position("pick_cube")
    move(twin, xyz + [-0.06, 0.04, 0.06])
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    missed = [event for event in events(twin) if "missed (" in event]
    assert missed and "cm away" in missed[-1] and "move_by(dx=" in missed[-1]


def test_grip_width_is_the_size_across_the_jaws():
    twin = build_twin(scene="pick_place_easy", with_controller=False)
    twin.tick()
    assert twin.observe()["scene"]["objects"]["bean_bag"]["grip_width_m"] == pytest.approx(0.024)
    roll(twin, SIDEWAYS)  # jaws side to side: across the bag's 5 cm, not its 6 cm
    assert twin.observe()["scene"]["objects"]["bean_bag"]["grip_width_m"] == pytest.approx(0.05)


@pytest.mark.parametrize("wrist", [SIDEWAYS, -SIDEWAYS])
def test_a_held_cube_sits_between_the_jaws_and_turns_with_them(twin, wrist):
    start = twin.grasp.position("pick_cube")
    over(twin, wrist=wrist)
    assert np.linalg.norm(twin.grasp.position("pick_cube") - start) < 0.006  # barely nudged
    grip(twin, SETTINGS.gripper_for_opening(CUBE_WIDTH))
    assert twin.grasp.attached == "pick_cube"

    def in_gripper():
        data, body = twin.sim.data, twin.grasp._gripper_body
        rotation = data.xmat[body].reshape(3, 3)
        return rotation.T @ (twin.grasp.position("pick_cube") - data.xpos[body])

    held = in_gripper()
    assert held[0] == pytest.approx(-0.008 + CUBE_WIDTH / 2, abs=1e-3)  # centered between pads
    move_by(twin, [0.0, 0.0, 0.10])
    move(twin, [0.333, -0.008, 0.30])  # where a live run saw it float beside the jaws
    roll(twin, 0.0)
    assert twin.grasp.attached == "pick_cube"
    np.testing.assert_allclose(in_gripper(), held, atol=1e-3)


def test_observation_gives_the_jaw_center(twin):
    obs = twin.observe()
    tool = np.array(obs["ee"]["xyz_m"])
    middle = np.array(obs["jaw_center_xyz_m"])
    assert 0.01 < np.linalg.norm(middle - tool) < 0.1  # not the tool point


def test_square_jaws_report_the_cubes_width(twin):
    over(twin)
    width, off_square = twin.grasp.fit("pick_cube")
    assert width == pytest.approx(CUBE_WIDTH)
    assert off_square < SETTINGS.rigid_align_tol_deg


def test_the_bean_bag_takes_any_closure_and_angle():
    twin = build_twin(scene="pick_place_easy", with_controller=False)
    twin.tick()
    bag = twin.grasp.position("bean_bag")
    over(twin, "bean_bag", wrist=2.3)  # well off square: fine for something soft
    assert twin.grasp.fit("bean_bag")[1] > SETTINGS.rigid_align_tol_deg
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
    for step in range(61):  # trigger squeezed from rest to fully in
        twin.tick(XboxState(left_bumper=True, right_trigger=step / 60))
    human = XboxState(left_bumper=True, right_trigger=1.0)
    for _ in range(60):
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
