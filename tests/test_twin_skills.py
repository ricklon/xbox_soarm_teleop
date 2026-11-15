"""pick and place: skills the arm does by itself, so an agent decides what goes
where and the arm does the moving.

A language model steering the jaws a few centimetres at a time, every second
or two, with an aim off by a centimetre or two, misses most grasps (6 of 9 in
``test_an_agents_misses_do_not_push_neighbours_or_drop_anything``) and pushes
things about while it tries. ``pick`` does it the way a robot does: a coarse
look from overhead, a close look through the wrist camera to centre the jaws,
down, close, and a check by feel; after a miss it lifts, looks again and
retries. It uses only what the arm can know (its pointing model and whether
it holds something), never the twin's truth, so it works in no-coordinates
mode and its answer gives nothing away.
"""

import logging
import socket
import threading
import time
import warnings

import numpy as np
import pytest

from xbox_soarm_teleop.twin import build_twin, scenarios
from xbox_soarm_teleop.twin.mcp_server import McpThread, build_mcp_server
from xbox_soarm_teleop.twin.pointing import make_pointer

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

# Aim errors as measured in docs/vision_benchmark.md: a distant view is off by
# a centimetre or two; the wrist camera, close up, by about half a centimetre.
FAR_ERROR_M = 0.015
WRIST_ERROR_M = 0.005
# From overhead a cube and the floor of the tray of its colour are both
# squares of that colour: in the skills trial the pointer chose the tray. Here
# it does so half the time.
TRAY_MIXUP_RATE = 0.5
TRAY_OF = {"pick_cube": "red_bin", "blue_cube": "blue_bin", "green_cube": "green_bin"}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def running():
    """A sort desk in no-coordinates mode, its loop on its own thread, the oracle
    pointer behind locate with seeded aim errors on top."""
    service = build_twin(
        scene="sort_colors", with_controller=False, hide_positions=True, pointer=make_pointer("oracle")
    )
    rng = np.random.default_rng(11)
    real_locate = service.locate

    def noisy_locate(name, **kwargs):
        answer = real_locate(name, **kwargs)
        if not answer.get("ok"):
            return answer
        spread = WRIST_ERROR_M if kwargs.get("view") == "wrist" else FAR_ERROR_M
        dx, dy = rng.uniform(-spread, spread, size=2)
        if kwargs.get("view") == "overhead" and name in TRAY_OF and rng.random() < TRAY_MIXUP_RATE:
            bins = {b["name"]: b for b in service.observe()["scene"]["bins"]}
            cx, cy, _ = bins[TRAY_OF[name]]["center_xyz_m"]
            dx, dy = cx - answer["xyz_m"][0], cy - answer["xyz_m"][1]  # it pointed at the tray
        answer["xyz_m"] = [answer["xyz_m"][0] + dx, answer["xyz_m"][1] + dy, answer["xyz_m"][2]]
        if "jaw_move_m" in answer:
            answer["jaw_move_m"] = {
                **answer["jaw_move_m"],
                "dx": answer["jaw_move_m"]["dx"] + dx,
                "dy": answer["jaw_move_m"]["dy"] + dy,
            }
        return answer

    service.locate = noisy_locate
    alive = threading.Event()
    alive.set()

    def loop():
        while alive.is_set():
            start = time.monotonic()
            service.tick()
            time.sleep(max(0.0, service.dt - (time.monotonic() - start)))
        service.close()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    yield service
    alive.clear()
    thread.join(timeout=10)


def _reset(service):
    service.reset_scene(client_id="test").result(timeout=10)
    time.sleep(0.3)


def test_pick_beats_steering_by_hand(running):
    service = running
    picked = tries = attempts = 0
    for _ in range(3):
        _reset(service)
        for name in ("pick_cube", "blue_cube", "green_cube"):
            result = service.pick(name, client_id="test")
            tries += 1
            if result["ok"]:
                picked += 1
                attempts += result["attempts"]
                assert service.observe()["scene"]["objects"][name]["attached"] is True
                # Nothing about where things are: only what the arm itself knows.
                assert "xyz" not in str(result) and "error" not in str(result)
                service.submit({"type": "set_gripper", "params": {"position": 0.0}}, client_id="test")
                time.sleep(0.5)
                service.submit({"type": "home"}, client_id="test")
                time.sleep(3.0)
    print(
        f"\npick: {picked} of {tries}, {attempts} grasp attempts, with distant aim off by up "
        f"to 1.5 cm and the overhead look on the tray half the time"
    )
    assert picked >= 8  # steering by hand with the same aim: 3 of 9
    # Most picks close on the cube the first time: the first look is checked
    # against a second view, so a tray does not pass for a cube.
    assert attempts <= picked + 2


def test_place_puts_what_it_holds_in_the_tray(running):
    service = running
    _reset(service)
    assert service.pick("blue_cube", client_id="test")["ok"]
    result = service.place("blue_bin", client_id="test")
    assert result["ok"] is True and result["released"] == "blue_cube"
    time.sleep(0.5)
    truth = service.observe_truth()["scene"]["objects"]["blue_cube"]
    assert truth["in_bin"] == "blue_bin"
    assert service.observe()["scene"]["objects"]["blue_cube"]["attached"] is False


def test_skills_say_why_they_cannot(running):
    service = running
    _reset(service)
    assert service.pick("purple_cube", client_id="test")["reason"] == "invalid_request"
    assert service.place("red_bin", client_id="test")["reason"] == "not_holding"
    assert service.pick("pick_cube", client_id="test")["ok"]
    assert service.place("yellow_bin", client_id="test")["reason"] == "invalid_request"
    service.submit({"type": "set_gripper", "params": {"position": 0.0}}, client_id="test")
    time.sleep(0.5)


def test_a_stop_ends_a_skill(running):
    service = running
    _reset(service)
    done = {}
    worker = threading.Thread(
        target=lambda: done.update(service.pick("green_cube", client_id="test")), daemon=True
    )
    worker.start()
    time.sleep(1.0)
    service.stop(by="agent", client_id="test")
    worker.join(timeout=30)
    assert done["ok"] is False and done["reason"] == "stopped"
    service.reset_stop(client_id="test", confirm=True).result(timeout=5)


def test_pick_and_place_are_tools(running):
    import asyncio
    import json

    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    mcp = McpThread(build_mcp_server(running), port=_free_port())
    mcp.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", mcp.port)) == 0:
                break
        time.sleep(0.05)

    async def go():
        async with streamable_http_client(mcp.url) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                result = await session.call_tool("place", {"bin": "red_bin"})
                return init, tools, json.loads(result.content[0].text)

    _reset(running)
    init, tools, answer = asyncio.run(go())
    assert {"pick", "place"} <= tools
    assert answer["reason"] == "not_holding"
    assert "pick(name)" in (init.instructions or "")


def test_without_a_pointer_pick_uses_what_observe_shows():
    """With coordinates shown and no pointing model, the arm knows where things
    are, so pick uses that; in no-coordinates mode it has no way to look."""
    shown = build_twin(scene="sort_colors", with_controller=False)
    hidden = build_twin(scene="sort_colors", with_controller=False, hide_positions=True)
    try:
        for service in (shown, hidden):
            service.tick()
        assert hidden.pick("pick_cube", client_id="test")["reason"] == "no_pointer"
        assert scenarios is not None
    finally:
        shown.close()
        hidden.close()


def test_skills_log_on_a_twin_that_keeps_a_session_log(tmp_path):
    """check_success once crashed on logging twins (a field named like the log's
    own); every skill result is logged, so try both on one."""
    from xbox_soarm_teleop.twin.session_log import read_session_log

    service = build_twin(scene="sort_colors", with_controller=False, log_path=tmp_path / "s.jsonl")
    try:
        service.tick()
        assert service.place("red_bin", client_id="test")["reason"] == "not_holding"
        assert service.pick("nothing", client_id="test")["reason"] == "invalid_request"
    finally:
        service.close()
    kinds = [line for line in read_session_log(tmp_path / "s.jsonl") if line["kind"] == "skill_result"]
    assert [line["data"]["skill"] for line in kinds] == ["place", "pick"]


def test_pick_by_description_finds_it_by_looking_and_closes_by_feel(running):
    """The request's words, not a name: the arm looks for "the red cube",
    closes until it feels something, and says what it is holding."""
    service = running
    _reset(service)
    result = service.pick(description="red cube", client_id="test")
    assert result["ok"] is True and result["holding"] == "red cube"  # its words, not a name
    assert service.observe()["scene"]["objects"]["pick_cube"]["attached"] is True
    placed = service.place("red_bin", client_id="test")
    assert placed["ok"] is True and placed["released"] == "red cube"
    time.sleep(0.5)
    assert service.observe_truth()["scene"]["objects"]["pick_cube"]["in_bin"] == "red_bin"


def test_locate_by_description(running):
    service = running
    _reset(service)
    found = service.locate(None, description="green cube", client_id="test")
    assert found["ok"] is True and "name" not in found
    assert service.estimate_scores()[-1]["name"] == "green_cube"  # scored against what matches
    nothing = service.locate(None, description="purple cube", client_id="test")
    assert nothing["ok"] is False and nothing["reason"] == "not_found"
    assert service.pick(client_id="test")["reason"] == "invalid_request"
