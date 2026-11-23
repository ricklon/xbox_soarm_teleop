"""Students switch games: pick and place on the raised table, sorting on the desk.

A hub that lets students switch compiles one model holding every table
scene's furniture (the "classroom" model, ``sim/table_scenes.py``): each
scene's table and bins ride on a mocap body, and a room poses its own game's
at the origin and parks the others. Switching game is then a reset, not a
recompile, and the hub's rooms still share one model and one renderer.

What must hold: a twin on the classroom model behaves as on its game's own
model (what ``observe`` says, what the cameras show, how far down the arm may
go), every task can still be done, and each room switches on its own.
"""

import logging
import time
import warnings

import mujoco
import numpy as np
import pytest

from tests.test_twin_sort import _do_by_script
from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
from xbox_soarm_teleop.sim.table_scenes import CLASSROOM, TABLE_SCENES
from xbox_soarm_teleop.twin import build_parts, build_twin
from xbox_soarm_teleop.twin.actions import ActionRejected
from xbox_soarm_teleop.twin.cameras import CameraRig
from xbox_soarm_teleop.twin.hub import Hub
from xbox_soarm_teleop.twin.layouts import LAYOUTS
from xbox_soarm_teleop.twin.tasks import TASKS, grade

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

LISTED = [name for name, layout in LAYOUTS.items() if layout.listed]


@pytest.fixture(scope="module")
def classroom_parts():
    return build_parts(scene="pick_place_easy", switch_games=True)


def _twin(parts, **kwargs):
    service = build_twin(parts=parts, with_controller=False, **kwargs)
    service.tick()
    return service


def _load(service, name: str):
    future = service.load_scene(name)
    service.tick()
    return future.result(timeout=0)


def _settle(service, ticks: int = 30) -> None:
    for _ in range(ticks):
        service.tick()


# --------------------------------------------------------------- the model
def test_the_classroom_model_holds_every_games_furniture(classroom_parts):
    model = classroom_parts.model
    assert model.nmocap == len(TABLE_SCENES)
    for scene in TABLE_SCENES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{scene}_table") >= 0
    for geom in ("goal_base", "red_bin_base", "blue_bin_base", "green_bin_base"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom) >= 0
    assert classroom_parts.scene == "pick_place_basic"  # the game the hub starts in


def test_the_games_own_models_are_unchanged():
    """The single-game models (the standalone twin, benchmarks) compile as before."""
    basic = sim_cli.load_model_with_cameras(str(sim_cli.URDF_PATH), scene="pick_place_basic")
    sort = sim_cli.load_model_with_cameras(str(sim_cli.URDF_PATH), scene="sort_bins")
    assert basic.nmocap == 0 and sort.nmocap == 0
    assert mujoco.mj_name2id(basic, mujoco.mjtObj.mjOBJ_GEOM, "pick_place_table") >= 0
    assert mujoco.mj_name2id(sort, mujoco.mjtObj.mjOBJ_CAMERA, "front") >= 0


@pytest.mark.parametrize("layout", LISTED)
def test_observe_matches_the_games_own_model(classroom_parts, layout):
    own = build_twin(scene=layout, with_controller=False)
    shared = _twin(classroom_parts)
    try:
        _load(shared, layout)
        own.tick()
        _settle(own)
        _settle(shared)
        mine, theirs = own.observe(), shared.observe()
        assert theirs["scene"] == mine["scene"]
        assert theirs["ee"]["xyz_m"] == mine["ee"]["xyz_m"]
        assert shared.table_top_z == own.table_top_z
        assert shared.workspace_bounds == own.workspace_bounds
    finally:
        own.close()
        shared.close()


@pytest.mark.parametrize("layout", ["pick_place_basic", "sort_colors"])
@pytest.mark.parametrize("view", ["overhead", "front", "side", "wrist"])
def test_the_cameras_show_what_the_games_own_model_shows(classroom_parts, layout, view):
    own = build_twin(scene=layout, with_controller=False)
    shared = _twin(classroom_parts)
    try:
        _load(shared, layout)
        own.tick()
        mine = CameraRig(own.sim.model, shadows=False).render(view, 320, 240, own.sim.data)
        rig = CameraRig(shared.sim.model, shadows=False)
        theirs = rig.render(view, 320, 240, shared.sim.data, scene=shared.sim.scene)
        differs = np.abs(mine.astype(int) - theirs.astype(int)).max(axis=2) > 24
        # Edges may differ by a pixel (one clipping range for every game); nothing more.
        assert differs.mean() < 0.005, f"{differs.sum()} pixels differ"
    finally:
        own.close()
        shared.close()


# ----------------------------------------------------------------- switching
def test_scenes_of_every_game_are_listed_with_their_game(classroom_parts):
    service = _twin(classroom_parts)
    try:
        listed = service.list_scenes()
        assert [scene["name"] for scene in listed] == LISTED
        games = {scene["name"]: scene["game"] for scene in listed}
        assert games["pick_place_easy"] == games["pick_place_basic"] == "pick_and_place"
        assert games["sort_colors"] == games["sort_colors_decoy"] == "sorting"
        assert [scene["name"] for scene in listed if scene["current"]] == ["pick_place_easy"]
    finally:
        service.close()


def test_a_twin_on_its_games_own_model_lists_only_that_game():
    service = build_twin(scene="pick_place_easy", with_controller=False)
    try:
        assert {scene["name"] for scene in service.list_scenes()} == {
            "pick_place_easy",
            "pick_place_basic",
        }
        future = service.load_scene("sort_colors")
        service.tick()
        with pytest.raises(ActionRejected):
            future.result(timeout=0)
    finally:
        service.close()


def test_switching_game_moves_the_furniture_and_the_floor(classroom_parts):
    service = _twin(classroom_parts)
    try:
        assert "goal" in service.observe()["scene"]
        assert service.workspace_bounds["z"][0] == pytest.approx(0.125)

        _load(service, "sort_colors")
        scene = service.observe()["scene"]
        assert service.sim.scene == "sort_bins"
        assert "goal" not in scene and {b["name"] for b in scene["bins"]} == {
            "red_bin",
            "blue_bin",
            "green_bin",
        }
        assert service.workspace_bounds["z"][0] == pytest.approx(0.007)
        assert service.table_top_z == pytest.approx(0.002)

        _load(service, "pick_place_basic")
        assert service.sim.scene == "pick_place_basic"
        assert "goal" in service.observe()["scene"]
        assert service.workspace_bounds["z"][0] == pytest.approx(0.125)
    finally:
        service.close()


def test_the_arm_cannot_go_into_the_raised_table_after_leaving_the_desk(classroom_parts):
    """Driven down as a person drives it (the page's Down button), after a game
    change: the swapped clipper stops the tool just over the raised table."""
    from xbox_soarm_teleop.teleoperators.xbox import XboxState

    # An arm that starts on the desk: its first limits are the desk's.
    service = _twin(build_parts(scene="sort_colors", switch_games=True))
    try:
        _load(service, "pick_place_basic")
        z_min = service.workspace_bounds["z"][0]
        for _ in range(150):  # hold LB, right stick down for 3 s
            service.tick(XboxState(left_bumper=True, right_stick_y=1.0))
        assert z_min - 1e-3 <= service.tool_xyz()[2] <= z_min + 0.01
        assert service.tool_xyz()[2] > 0.12
    finally:
        service.close()


def test_the_cube_can_be_put_in_the_goal_box_after_switching(classroom_parts):
    from xbox_soarm_teleop.twin import scenarios

    service = _twin(classroom_parts)
    try:
        _load(service, "sort_colors")
        _load(service, "pick_place_basic")
        gx, gy, floor = service.observe()["scene"]["goal"]["center_xyz_m"]
        assert scenarios.grasp_and_lift(service, "pick_cube", [gx, gy, floor + 0.12])
        jaw = service.observe()["jaw_center_xyz_m"]
        lower = {"type": "move_ee_by", "params": {"dxyz_m": [0, 0, floor + 0.035 - jaw[2]]}}
        assert scenarios.run(service, lower)
        assert scenarios.run(service, {"type": "set_gripper", "params": {"position": 0.0}})
        assert scenarios.run(service, {"type": "move_ee_by", "params": {"dxyz_m": [0, 0, 0.05]}})
        _settle(service, 50)
        assert service.observe()["scene"]["objects"]["pick_cube"]["in_goal"] is True
    finally:
        service.close()


@pytest.mark.parametrize("layout", ["pick_place_basic", "sort_colors"])
@pytest.mark.parametrize("view", ["overhead", "front", "side", "wrist"])
def test_locate_sees_through_the_same_cameras(classroom_parts, layout, view):
    """``locate`` turns a picture point into a place with the camera's geometry."""
    from xbox_soarm_teleop.twin.projection import camera_geometry

    own = build_twin(scene=layout, with_controller=False)
    shared = _twin(classroom_parts)
    try:
        _load(shared, "sort_colors" if layout != "sort_colors" else "pick_place_basic")
        _load(shared, layout)
        own.tick()
        mine = camera_geometry(own.sim.model, own.sim.data, view, 640, 480)
        theirs = camera_geometry(
            shared.sim.model, shared.sim.data, view, 640, 480, scene=shared.sim.scene
        )
        for part in ("position", "forward", "up"):
            np.testing.assert_allclose(getattr(theirs, part), getattr(mine, part), atol=1e-6)
        assert theirs.fovy_deg == pytest.approx(mine.fovy_deg)
    finally:
        own.close()
        shared.close()


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_every_sorting_task_can_be_done_after_switching(classroom_parts, task_name):
    task = TASKS[task_name]
    service = _twin(classroom_parts)
    try:
        _load(service, task.scene)
        _do_by_script(service, task.target())
        final = {name: entry["in_bin"] for name, entry in service.observe()["scene"]["objects"].items()}
        assert final == task.target()
        assert grade(task, final)["done"] is True
    finally:
        service.close()


def test_a_stopped_arm_does_not_switch_game(classroom_parts):
    service = _twin(classroom_parts)
    try:
        service.stop(by="human")
        service.tick()
        future = service.load_scene("sort_colors")
        service.tick()
        with pytest.raises(ActionRejected):
            future.result(timeout=0)
        assert service.sim.scene == "pick_place_basic"
    finally:
        service.close()


def test_a_game_switch_can_be_refused_by_the_host(classroom_parts):
    service = _twin(classroom_parts)
    try:
        service.switch_refusal = lambda: "your agent is running: stop it first"
        future = service.load_scene("sort_colors")
        service.tick()
        with pytest.raises(ActionRejected, match="agent is running"):
            future.result(timeout=0)
        _load(service, "pick_place_basic")  # the same game: still allowed
        assert service.layout.name == "pick_place_basic"
    finally:
        service.close()


def test_each_game_brings_its_own_rules(classroom_parts):
    service = _twin(
        classroom_parts, game_rules={"sort_bins": {"hide_positions": True, "naming": "truthful"}}
    )
    try:
        assert service.hide_positions is False and service.naming == "as_is"
        _load(service, "sort_colors")
        assert service.hide_positions is True and service.naming == "truthful"
        assert "xyz_m" not in service.observe()["scene"]["objects"]["pick_cube"]
        assert service.shown_names["pick_cube"] == "red_cube"
        _load(service, "pick_place_easy")
        assert service.hide_positions is False and service.naming == "as_is"
    finally:
        service.close()


def test_leaving_the_desk_ends_its_challenge(classroom_parts):
    import threading

    from xbox_soarm_teleop.twin.challenges import example_path, examples, load_challenge, set_up

    service = _twin(classroom_parts)
    ticking = threading.Event()
    ticking.set()

    def loop():
        while ticking.is_set():
            service.tick()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        challenge = load_challenge(example_path(sorted(examples())[0]))
        set_up(service, challenge, client_id="page")
        assert service.challenge is challenge
        service.load_scene("pick_place_basic").result(timeout=10)
        assert service.challenge is None and service.naming == "as_is"
    finally:
        ticking.clear()
        thread.join(timeout=5)
        service.close()


# ------------------------------------------------------------------------ hub
def test_each_room_of_a_hub_switches_on_its_own():
    hub = Hub(seats=2, scene="pick_place_easy", switch_games=True)
    try:
        first, second = hub.rooms.values()
        assert first.service.sim.model is second.service.sim.model  # one shared model
        future = first.service.load_scene("sort_colors")
        hub.tick()
        future.result(timeout=0)
        assert first.service.sim.scene == "sort_bins"
        assert second.service.sim.scene == "pick_place_basic"
        assert "bins" in first.service.observe()["scene"]
        assert "goal" in second.service.observe()["scene"]
        # Each room's pictures come from its own game's cameras.
        pictures = []
        for room in (first, second):
            frame = room.service.render("front", width=320, height=240, client_id="web")
            deadline = time.monotonic() + 20.0
            while not frame.done() and time.monotonic() < deadline:  # the painter is a thread
                hub.tick()
                time.sleep(0.01)
            pictures.append(frame.result(timeout=0))
        assert pictures[0] != pictures[1]
    finally:
        hub.close()


def test_a_hub_without_switching_keeps_its_games_own_model():
    hub = Hub(seats=1, scene="sort_colors")
    try:
        (room,) = hub.rooms.values()
        assert room.service.sim.model.nmocap == 0
        assert CLASSROOM not in {scene["name"] for scene in room.service.list_scenes()}
    finally:
        hub.close()


def test_the_hub_refuses_a_game_change_while_the_rooms_agent_runs():
    from xbox_soarm_teleop.twin.hub_agent import PanelRun
    from xbox_soarm_teleop.twin.hub_server import build_hub_app

    hub = Hub(seats=2, scene="pick_place_easy", switch_games=True)
    try:
        build_hub_app(hub)
        busy, free = hub.rooms.values()
        hub.agent_desk.runs[busy.token] = PanelRun(1, "sort the cubes", "a-model", False)
        refused = busy.service.load_scene("sort_colors")
        allowed = free.service.load_scene("sort_colors")
        hub.tick()
        with pytest.raises(ActionRejected, match="agent is running"):
            refused.result(timeout=0)
        allowed.result(timeout=0)
        assert (busy.service.sim.scene, free.service.sim.scene) == ("pick_place_basic", "sort_bins")
        hub.agent_desk.runs[busy.token].done = True
        again = busy.service.load_scene("sort_colors")
        hub.tick()
        again.result(timeout=0)
        assert busy.service.sim.scene == "sort_bins"
    finally:
        hub.close()


def test_the_sorting_preset_lets_students_switch_games():
    from xbox_soarm_teleop.cli.hub import apply_preset, build_parser, game_rules

    args = apply_preset(build_parser().parse_args(["--preset", "sorting"]))
    assert args.switch_games is True and args.scene == "sort_colors"
    assert game_rules(args) == {
        "pick_place_basic": {"hide_positions": False, "naming": "as_is"},
        "sort_bins": {"hide_positions": True, "naming": "truthful"},
    }
    # Options you give still win, for the sorting game.
    mine = apply_preset(build_parser().parse_args(["--preset", "sorting", "--names", "neutral"]))
    assert game_rules(mine)["sort_bins"]["naming"] == "neutral"
    # Off unless asked for.
    assert game_rules(build_parser().parse_args([])) is None
