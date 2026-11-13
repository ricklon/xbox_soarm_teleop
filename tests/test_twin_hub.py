"""The hub: one simulated arm per student, driven from a web page or an agent."""

import json
import logging
import warnings

import numpy as np
import pytest

from xbox_soarm_teleop.twin.hub import Hub, WebController
from xbox_soarm_teleop.twin.session_log import read_session_log

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


# ------------------------------------------------------------ web controller
def test_no_page_means_no_controller():
    pad = WebController(clock=Clock())
    assert pad.is_connected is False
    state = pad.read()
    assert state.left_bumper is False and state.left_stick_y == 0.0


def test_motion_engages_the_deadman_and_maps_to_the_sticks():
    pad = WebController(clock=Clock())
    pad.update({"fwd": 1, "left": -0.5, "up": 1, "roll": -1})
    state = pad.read()
    assert pad.is_connected and state.left_bumper is True
    # The Xbox mapping: stick up (negative y) is forward, stick left is +y, right stick up is +z.
    assert (state.left_stick_y, state.left_stick_x, state.right_stick_y) == (-1.0, 0.5, -1.0)
    assert state.right_stick_x == -1.0


def test_values_are_clamped_and_junk_is_ignored():
    pad = WebController(clock=Clock())
    pad.update({"fwd": 50, "left": "x", "up": None, "press": ["nonsense", 3]})
    state = pad.read()
    assert state.left_stick_y == -1.0 and state.left_stick_x == 0.0 and state.right_stick_y == 0.0
    assert not (state.b_button_pressed or state.a_button_pressed)


def test_a_page_that_goes_quiet_releases_the_deadman():
    clock = Clock()
    pad = WebController(clock=clock)
    pad.update({"fwd": 1})
    assert pad.read().left_bumper is True
    clock.now += WebController.STALE_S + 0.01
    state = pad.read()
    assert state.left_bumper is False and state.left_stick_y == 0.0
    assert pad.is_connected  # the page may just be slow
    clock.now += WebController.GONE_S
    assert pad.is_connected is False


def test_button_presses_are_delivered_exactly_once():
    pad = WebController(clock=Clock())
    pad.update({"press": ["stop"]})
    pad.update({})  # a later post must not lose the press
    first, second = pad.read(), pad.read()
    assert first.b_button_pressed and first.b_button
    assert not second.b_button_pressed
    pad.update({"press": ["release", "home", "reset"]})
    state = pad.read()
    assert state.start_button_pressed and state.a_button_pressed and state.y_button_pressed


def test_gripper_starts_from_where_it_is_then_goes_to_the_target():
    pad = WebController(clock=Clock())
    pad.gripper_feedback = lambda: 0.4
    pad.update({"hold": True, "grip": 1.0})
    assert pad.read().right_trigger == 0.4  # first: match the gripper, so nothing jumps
    assert pad.read().right_trigger == 1.0
    pad.update({"fwd": 1})  # moving without a gripper command keeps the grip
    assert pad.read().right_trigger == 1.0


# ---------------------------------------------------------------------- hub
@pytest.fixture(scope="module")
def hub(tmp_path_factory):
    root = tmp_path_factory.mktemp("hub")
    hub = Hub(seats=3, log_dir=root / "logs", roster_path=root / "roster.json")
    hub.tick()
    yield hub
    hub.close()


def tool(room) -> np.ndarray:
    return np.array(room.service.observe()["ee"]["xyz_m"])


def test_every_student_gets_a_room_with_its_own_secret_link(hub):
    rooms = list(hub.rooms.values())
    assert [room.name for room in rooms] == ["Arm 1", "Arm 2", "Arm 3"]
    tokens = {room.token for room in rooms} | {hub.admin_token}
    assert len(tokens) == 4 and all(len(token) >= 12 for token in tokens)
    assert hub.room(rooms[0].token) is rooms[0]
    assert hub.room("nope") is None and hub.room(hub.admin_token) is None


def test_rooms_share_the_heavy_parts_but_not_their_state(hub):
    a, b, _ = hub.rooms.values()
    assert a.service.sim.model is b.service.sim.model
    assert a.service.kinematics is b.service.kinematics
    assert a.service.sim.data is not b.service.sim.data


def test_driving_one_arm_from_the_web_leaves_the_others_alone(hub):
    a, b, _ = hub.rooms.values()
    start_a, start_b = tool(a), tool(b)
    for _ in range(50):
        a.controller.update({"up": -1})
        hub.tick()
    assert tool(a)[2] < start_a[2] - 0.05
    np.testing.assert_allclose(tool(b), start_b, atol=1e-6)
    assert a.service.observe()["control"]["human_in_control"] is True
    assert b.service.observe()["control"]["controller_connected"] is False


def test_web_stop_latches_and_only_release_clears_it(hub):
    _, b, _ = hub.rooms.values()
    b.controller.update({"press": ["stop"]})
    hub.tick()
    assert b.service.observe()["control"]["stop_latched_by"] == "human"
    held = tool(b)
    for _ in range(20):
        b.controller.update({"fwd": 1})
        hub.tick()
    np.testing.assert_allclose(tool(b), held, atol=1e-6)
    b.controller.update({"press": ["release"]})
    hub.tick()
    assert b.service.observe()["control"]["stop_latched_by"] is None


def test_web_gripper_closes_on_the_cube_and_carries_it(hub):
    _, _, c = hub.rooms.values()
    cube = c.service.observe()["scene"]["objects"]["pick_cube"]["xyz_m"]
    for request in (
        {"type": "move_ee_to", "params": {"xyz_m": [cube[0], cube[1], cube[2] + 0.06]}},
        {"type": "set_wrist_roll", "params": {"angle_rad": -1.5}},  # jaws side to side
        {"type": "move_ee_to", "params": {"xyz_m": [cube[0], cube[1], cube[2] + 0.025]}},
    ):
        future = c.service.submit(request)
        for _ in range(400):
            hub.tick()
        assert future.result(timeout=0)
    for _ in range(150):  # agent done; now the person closes the gripper from the page
        c.controller.update({"hold": True, "grip": 1.0})
        hub.tick()
    observation = c.service.observe()
    # Jaws a person drives shut stop on the cube instead of squeezing it out.
    assert 0.7 < observation["gripper"] < 0.85
    assert observation["scene"]["objects"]["pick_cube"]["attached"] is True
    for _ in range(60):
        c.controller.update({"up": 1})
        hub.tick()
    lifted = c.service.observe()["scene"]["objects"]["pick_cube"]
    assert lifted["attached"] is True and lifted["xyz_m"][2] > cube[2] + 0.05


def test_camera_requests_from_every_room_are_all_served(hub):
    import time

    futures = [room.service.render("overhead") for room in hub.rooms.values()]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not all(future.done() for future in futures):
        hub.tick()  # each tick hands the painter thread at most one frame
        time.sleep(0.005)
    images = [future.result(timeout=0) for future in futures]
    assert all(image[:2] == b"\xff\xd8" for image in images)  # JPEG


def test_camera_sizes_are_snapped_to_a_few_renderers(hub):
    import io
    import time

    from PIL import Image

    room = next(iter(hub.rooms.values()))
    future = room.service.render("wrist", width=300, height=200)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not future.done():
        hub.tick()
        time.sleep(0.005)
    assert Image.open(io.BytesIO(future.result(timeout=0))).size == (320, 240)


def test_each_room_writes_its_own_session_log(hub):
    for room in hub.rooms.values():
        room.service.log.flush()
        lines = read_session_log(room.log_path)
        assert lines[0]["kind"] == "header" and room.log_path.name.startswith("arm-")
    assert len({room.log_path for room in hub.rooms.values()}) == 3


def test_links_survive_a_restart(tmp_path):
    roster = tmp_path / "roster.json"
    first = Hub(seats=2, roster_path=roster)
    tokens = [room.token for room in first.rooms.values()]
    admin = first.admin_token
    first.close()
    saved = json.loads(roster.read_text())
    assert [seat["token"] for seat in saved["rooms"]] == tokens
    second = Hub(seats=3, roster_path=roster)  # one more student joined
    assert [room.token for room in second.rooms.values()][:2] == tokens
    assert second.admin_token == admin and len(second.rooms) == 3
    second.close()


def test_hub_keeps_time_with_many_rooms():
    import time

    hub = Hub(seats=12)
    for _ in range(5):
        hub.tick()
    started = time.perf_counter()
    for _ in range(50):
        hub.tick()
    per_tick = (time.perf_counter() - started) / 50
    hub.close()
    assert per_tick < 0.012  # 12 idle arms must fit well inside the 20 ms period


def test_hub_draws_simplified_meshes_but_simulates_the_originals(hub):
    room = next(iter(hub.rooms.values()))
    assert hub.cameras.model is not room.service.sim.model
    assert hub.cameras.model.nmeshface < room.service.sim.model.nmeshface / 8
    assert room.service.sim.model.nmeshface > 300_000


def test_pictures_follow_each_rooms_own_arm():
    import io
    import time

    from PIL import Image

    fresh = Hub(seats=2)

    def picture(room) -> np.ndarray:
        future = room.service.render("front", width=320, height=240)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not future.done():
            fresh.tick()
            time.sleep(0.005)
        return np.asarray(Image.open(io.BytesIO(future.result(timeout=0))).convert("RGB"), dtype=float)

    try:
        fresh.tick()
        a, b = fresh.rooms.values()
        still_a, still_b = picture(a), picture(b)
        assert np.abs(still_a - still_b).mean() < 0.5  # two untouched arms look alike
        for _ in range(60):
            a.controller.update({"up": -1, "left": 1})
            fresh.tick()
        assert np.abs(picture(a) - still_a).mean() > 1.0  # the moved arm's picture changed
        assert np.abs(picture(b) - still_b).mean() < 0.5  # the other one's did not
    finally:
        fresh.close()


def test_hub_can_draw_the_original_meshes():
    hub = Hub(seats=1, mesh_cell_mm=0)
    try:
        assert hub.cameras.model is next(iter(hub.rooms.values())).service.sim.model
    finally:
        hub.close()


def test_instructor_stop_is_released_only_by_the_instructor():
    from xbox_soarm_teleop.twin.actions import ActionRejected

    fresh = Hub(seats=1)
    try:
        room = next(iter(fresh.rooms.values()))
        fresh.tick()
        room.service.stop(by="instructor", client_id="instructor")
        fresh.tick()
        latched = lambda: room.service.observe()["control"]["stop_latched_by"]  # noqa: E731
        assert latched() == "instructor"
        room.controller.update({"press": ["release"]})  # the student tries
        fresh.tick()
        assert latched() == "instructor"
        room.controller.update({"press": ["stop"]})  # the student's own stop does not take it over
        fresh.tick()
        room.controller.update({"press": ["release"]})
        fresh.tick()
        assert latched() == "instructor"
        attempt = room.service.reset_stop(client_id="agent", confirm=True)  # an agent tries
        fresh.tick()
        with pytest.raises(ActionRejected) as refused:
            attempt.result(timeout=0)
        assert "instructor" in refused.value.detail
        held = tool(room)
        for _ in range(20):
            room.controller.update({"fwd": 1})
            fresh.tick()
        np.testing.assert_allclose(tool(room), held, atol=1e-6)
        room.service.release_stop(by="instructor")
        fresh.tick()
        assert latched() is None
    finally:
        fresh.close()


def test_home_from_the_page_opens_the_gripper():
    fresh = Hub(seats=1)
    try:
        room = next(iter(fresh.rooms.values()))
        fresh.tick()
        for _ in range(150):
            room.controller.update({"hold": True, "grip": 1.0})
            fresh.tick()
        assert room.service.observe()["gripper"] > 0.95
        for _ in range(150):  # hands off for three seconds
            room.controller.update({})
            fresh.tick()
        room.controller.update({"press": ["home"]})
        for _ in range(250):
            fresh.tick()
            room.controller.update({})
        assert room.service.observe()["gripper"] < 0.05
    finally:
        fresh.close()


def test_page_pictures_are_not_written_to_the_session_log(tmp_path):
    import time

    fresh = Hub(seats=1, log_dir=tmp_path)
    try:
        room = next(iter(fresh.rooms.values()))
        futures = [
            room.service.render("overhead", client_id="web", log=False),
            room.service.render("overhead", client_id="agent-1"),
        ]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not all(f.done() for f in futures):
            fresh.tick()
            time.sleep(0.005)
        room.service.log.flush()
        cameras = [
            line
            for line in read_session_log(room.log_path)
            if line["kind"] == "request" and line["data"].get("type") == "camera"
        ]
        assert [line["client_id"] for line in cameras] == ["agent-1"]
    finally:
        fresh.close()
