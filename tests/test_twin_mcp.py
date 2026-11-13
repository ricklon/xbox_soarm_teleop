"""The twin's MCP server, exercised with the real MCP client over HTTP."""

import asyncio
import base64
import json
import logging
import socket
import threading
import time
import warnings

import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from xbox_soarm_teleop.twin import build_twin
from xbox_soarm_teleop.twin.mcp_server import McpThread, build_mcp_server

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def twin_url():
    service = build_twin(scene="pick_place_basic", with_controller=False)
    running = threading.Event()
    running.set()

    def loop():
        while running.is_set():
            start = time.monotonic()
            service.tick()
            time.sleep(max(0.0, service.dt - (time.monotonic() - start)))
        service.close()  # on the loop thread, which owns the GL context

    loop_thread = threading.Thread(target=loop, daemon=True)
    loop_thread.start()
    mcp = McpThread(build_mcp_server(service), port=_free_port())
    mcp.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", mcp.port)) == 0:
                break
        time.sleep(0.05)
    yield mcp.url, service
    running.clear()
    loop_thread.join(timeout=5)


def call(url, tool, args=None):
    return asyncio.run(_call(url, [(tool, args or {})]))[0]


async def _call(url, calls):
    async with streamable_http_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            results = []
            for tool, args in calls:
                results.append(await session.call_tool(tool, args))
            return results


def payload(result) -> dict:
    assert not result.is_error, result
    return json.loads(result.content[0].text)


def test_tools_listed(twin_url):
    url, _ = twin_url

    async def names():
        async with streamable_http_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                tools = await session.list_tools()
                return init, {tool.name for tool in tools.tools}

    init, tools = asyncio.run(names())
    assert {
        "observe",
        "move_to",
        "move_by",
        "set_gripper",
        "home",
        "stop",
        "reset_stop",
        "reset_scene",
        "camera",
        "wait_for_action",
        "cancel_action",
        "list_scenes",
        "load_scene",
    } <= tools
    assert "Assisted" in (init.instructions or "") or "assisted" in (init.instructions or "")


def test_observe_move_reject_and_camera(twin_url):
    url, _ = twin_url
    obs = payload(call(url, "observe"))
    assert obs["schema"] == "soarm.observation/v1"

    moved = payload(call(url, "move_to", {"x": 0.3, "y": 0.05, "z": 0.22}))
    assert moved["ok"] and moved["action"]["state"] == "SUCCEEDED"
    assert moved["ee_xyz_m"] == pytest.approx([0.3, 0.05, 0.22], abs=0.003)

    refused = payload(call(url, "move_to", {"x": 0.3, "y": 0.0, "z": -0.1}))
    assert refused == {"ok": False, "reason": "workspace_bound", "detail": refused["detail"]}
    assert refused["detail"].startswith("z_min")

    image = call(url, "camera", {"view": "overhead"}).content[0]
    assert image.type == "image" and image.mime_type == "image/jpeg"
    assert base64.b64decode(image.data)[:2] == b"\xff\xd8"


def test_stop_is_not_blocked_by_a_waiting_move(twin_url):
    url, service = twin_url

    async def scenario():
        async def far_move():
            async with streamable_http_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool("move_to", {"x": 0.42, "y": -0.2, "z": 0.15})

        mover = asyncio.create_task(far_move())
        await asyncio.sleep(0.6)  # the move is under way and its call is blocked waiting
        started = time.monotonic()
        stopped = (await _call(url, [("stop", {})]))[0]
        stop_latency = time.monotonic() - started
        return await mover, stopped, stop_latency

    moved, stopped, latency = asyncio.run(scenario())
    assert payload(stopped)["stop_latched"] is True
    assert latency < 0.5
    result = payload(moved)
    assert result["action"]["state"] == "STOPPED"

    refused = payload(call(url, "move_by", {"dx": 0.0, "dy": 0.0, "dz": 0.02}))
    assert refused["reason"] == "stopped"
    # A different MCP session latched nothing; the agent stop may be released by any agent.
    assert payload(call(url, "reset_stop", {"confirm": True}))["released"] is True
    assert payload(call(url, "home"))["action"]["state"] == "SUCCEEDED"
    assert service.observe()["stop_latched"] is False


def test_agent_lists_and_loads_scenes(twin_url):
    url, service = twin_url
    scenes = payload(call(url, "list_scenes"))["scenes"]
    assert [(scene["name"], scene["current"]) for scene in scenes] == [
        ("pick_place_easy", False),
        ("pick_place_basic", True),
    ]
    loaded = payload(call(url, "load_scene", {"name": "pick_place_easy"}))
    assert loaded["ok"] is True and loaded["scene"]["objects"] == ["bean_bag"]
    assert list(payload(call(url, "observe"))["scene"]["objects"]) == ["bean_bag"]

    unknown = payload(call(url, "load_scene", {"name": "moon_base"}))
    assert unknown["ok"] is False and unknown["reason"] == "invalid_request"
    assert "pick_place_easy" in unknown["detail"]

    assert payload(call(url, "load_scene", {"name": "pick_place_basic"}))["ok"] is True
    assert list(service.observe()["scene"]["objects"]) == ["pick_cube"]


def test_agent_grips_by_opening_and_sees_what_it_holds(twin_url):
    url, service = twin_url
    call(url, "reset_scene")
    cube = payload(call(url, "observe"))["scene"]["objects"]["pick_cube"]
    x, y, z = cube["xyz_m"]
    call(url, "set_gripper", {"opening_m": cube["size_m"][0] + 0.03})
    call(url, "move_to", {"x": x, "y": y, "z": z + 0.06})
    call(url, "set_wrist_roll", {"angle_rad": -1.5})
    for _ in range(2):  # the tool point is a jaw face: center the jaws, not the tool
        jaw = payload(call(url, "observe"))["jaw_center_xyz_m"]
        call(url, "move_by", {"dx": x - jaw[0], "dy": y - jaw[1], "dz": 0.0})
    jaw = payload(call(url, "observe"))["jaw_center_xyz_m"]
    moved = payload(call(url, "move_by", {"dx": 0.0, "dy": 0.0, "dz": z - jaw[2]}))
    assert moved["jaw_center_xyz_m"] == pytest.approx([x, y, z], abs=0.006)

    both = payload(call(url, "set_gripper", {"position": 1.0, "opening_m": 0.04}))
    assert both["ok"] is False and both["reason"] == "invalid_request"
    assert payload(call(url, "set_gripper"))["reason"] == "invalid_request"

    held = payload(call(url, "set_gripper", {"opening_m": cube["size_m"][0]}))
    assert held["ok"] is True and held["holding"] == "pick_cube"
    assert held["gripper_opening_m"] == pytest.approx(0.04, abs=0.006)

    crushed = payload(call(url, "set_gripper", {"position": 1.0}))
    assert crushed["holding"] is None
    assert any("squeezed out" in event for event in service.observe()["control"]["events"])
    call(url, "reset_scene")
