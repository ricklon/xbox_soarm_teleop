"""Scene layouts: an easy bean bag and a hard cube, switchable on a running twin."""

import logging
import warnings

import numpy as np
import pytest

from xbox_soarm_teleop.schemas.agent import ActionState
from xbox_soarm_teleop.twin import ActionRejected, build_parts, build_twin
from xbox_soarm_teleop.twin.layouts import LAYOUTS, layouts_for, model_scene

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

TABLE_TOP = 0.12


def objects(twin) -> dict:
    return twin.observe()["scene"]["objects"]


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
    return run(twin, {"type": "set_gripper", "params": {"position": position}})


def load(twin, name):
    future = twin.load_scene(name, client_id="agent")
    twin.tick()
    return future.result(timeout=0)


@pytest.fixture(scope="module")
def parts():
    return build_parts(scene="pick_place_easy")


@pytest.fixture
def easy(parts):
    twin = build_twin(parts=parts, with_controller=False)
    twin.tick()
    return twin


def test_layout_names_map_to_one_compiled_scene():
    assert model_scene("pick_place_easy") == "pick_place_basic"
    assert model_scene("pick_place_basic") == "pick_place_basic"
    assert model_scene("stack") == "stack"
    assert [layout.name for layout in layouts_for("pick_place_basic")] == [
        "pick_place_easy",
        "pick_place_basic",
    ]
    assert layouts_for("stack") == []
    assert LAYOUTS["pick_place_easy"].difficulty == "easy"


def test_cube_scene_is_unchanged_by_default():
    twin = build_twin(scene="pick_place_basic", with_controller=False)
    twin.tick()
    obs = twin.observe()
    assert obs["scene"]["name"] == "pick_place_basic"
    assert list(obs["scene"]["objects"]) == ["pick_cube"]
    assert obs["scene"]["objects"]["pick_cube"]["xyz_m"] == pytest.approx([0.34, -0.03, 0.14], abs=2e-3)


def test_easy_scene_shows_only_the_bean_bag(easy):
    obs = easy.observe()
    assert obs["scene"]["name"] == "pick_place_easy"
    assert list(obs["scene"]["objects"]) == ["bean_bag"]
    bag = obs["scene"]["objects"]["bean_bag"]
    assert bag["xyz_m"][:2] == pytest.approx([0.34, -0.03], abs=2e-3)
    assert TABLE_TOP < bag["xyz_m"][2] < TABLE_TOP + 0.03
    assert bag["off_table"] is False


def test_bean_bag_rests_on_the_table_and_stays_put(easy):
    before = objects(easy)["bean_bag"]["xyz_m"]
    for _ in range(100):
        easy.tick()
    assert objects(easy)["bean_bag"]["xyz_m"] == pytest.approx(before, abs=1e-3)


def test_gripper_does_not_knock_the_bean_bag(easy):
    """Lowering the tool point right onto the bag's center leaves it where it is."""
    start = np.array(objects(easy)["bean_bag"]["xyz_m"])
    assert move(easy, [start[0], start[1], 0.20]).state is ActionState.SUCCEEDED
    assert move(easy, [start[0], start[1], 0.13]).state is ActionState.SUCCEEDED
    assert objects(easy)["bean_bag"]["xyz_m"] == pytest.approx(start.tolist(), abs=2e-3)
    assert grip(easy, 1.0).state is ActionState.SUCCEEDED
    assert objects(easy)["bean_bag"]["attached"] is True


def test_bean_bag_can_be_placed_in_the_goal_box(easy):
    start = objects(easy)["bean_bag"]["xyz_m"]
    move(easy, [start[0], start[1], 0.14])
    grip(easy, 1.0)
    move(easy, [start[0], start[1], 0.24])
    move(easy, [0.29, 0.13, 0.22])
    grip(easy, 0.0)
    for _ in range(100):
        easy.tick()
    bag = objects(easy)["bean_bag"]
    assert bag["attached"] is False and bag["off_table"] is False
    assert bag["xyz_m"][:2] == pytest.approx([0.29, 0.13], abs=0.03)
    assert bag["xyz_m"][2] < TABLE_TOP + 0.03  # resting on the box floor, not on a wall


def test_load_scene_swaps_the_object_and_homes_the_arm(easy):
    move(easy, [0.30, 0.05, 0.22])
    result = load(easy, "pick_place_basic")
    assert result["name"] == "pick_place_basic" and result["difficulty"] == "hard"
    easy.tick()
    obs = easy.observe()
    assert obs["scene"]["name"] == "pick_place_basic"
    assert list(obs["scene"]["objects"]) == ["pick_cube"]
    assert obs["scene"]["objects"]["pick_cube"]["xyz_m"] == pytest.approx([0.34, -0.03, 0.14], abs=2e-3)
    assert obs["ee"]["xyz_m"] == pytest.approx([0.391, 0.0, 0.226], abs=2e-3)
    assert any("pick_place_basic" in event for event in obs["control"]["events"])
    # ...and back again.
    load(easy, "pick_place_easy")
    easy.tick()
    assert list(objects(easy)) == ["bean_bag"]


def test_load_scene_drops_what_the_gripper_held(easy):
    start = objects(easy)["bean_bag"]["xyz_m"]
    move(easy, [start[0], start[1], 0.14])
    grip(easy, 1.0)
    assert objects(easy)["bean_bag"]["attached"] is True
    load(easy, "pick_place_basic")
    easy.tick()
    assert easy.grasp.attached is None
    assert objects(easy)["pick_cube"]["attached"] is False


def test_the_parked_object_cannot_be_grabbed_or_seen(easy):
    """With the bean bag out, closing on the cube's usual spot grabs nothing."""
    move(easy, [0.34, -0.03, 0.20])
    load(easy, "pick_place_basic")
    easy.tick()
    cube = objects(easy)["pick_cube"]["xyz_m"]
    move(easy, [cube[0], cube[1], cube[2] + 0.03])
    grip(easy, 1.0)
    assert easy.grasp.attached == "pick_cube"
    assert "bean_bag" not in easy.grasp.positions()


def test_reset_scene_keeps_the_loaded_layout(easy):
    load(easy, "pick_place_basic")
    future = easy.reset_scene(client_id="agent")
    easy.tick()
    assert future.result(timeout=0) is True
    assert list(objects(easy)) == ["pick_cube"]


def test_unknown_scene_is_rejected_with_the_choices(easy):
    future = easy.load_scene("moon_base", client_id="agent")
    easy.tick()
    with pytest.raises(ActionRejected) as info:
        future.result(timeout=0)
    assert "pick_place_easy" in info.value.detail and "pick_place_basic" in info.value.detail
    assert list(objects(easy)) == ["bean_bag"]


def test_list_scenes_marks_the_current_one(easy):
    scenes = easy.list_scenes()
    assert [scene["name"] for scene in scenes] == ["pick_place_easy", "pick_place_basic"]
    assert [scene["current"] for scene in scenes] == [True, False]
    assert scenes[0]["objects"] == ["bean_bag"] and scenes[1]["difficulty"] == "hard"


def test_scene_without_layouts_lists_itself():
    twin = build_twin(scene="stack", with_controller=False)
    twin.tick()
    scenes = twin.list_scenes()
    assert [scene["name"] for scene in scenes] == ["stack"]
    assert scenes[0]["current"] is True
    assert len(objects(twin)) == 3


def test_rooms_sharing_a_model_keep_their_own_layout(parts):
    one = build_twin(parts=parts, with_controller=False)
    two = build_twin(parts=parts, with_controller=False)
    one.tick()
    two.tick()
    load(one, "pick_place_basic")
    one.tick()
    two.tick()
    assert list(objects(one)) == ["pick_cube"]
    assert list(objects(two)) == ["bean_bag"]
