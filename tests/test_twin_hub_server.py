"""The hub over real HTTP: student page, instructor page and per-arm MCP."""

import asyncio
import json
import logging
import socket
import threading
import time
import warnings

import httpx2 as httpx
import pytest
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from xbox_soarm_teleop.twin.hub import Hub
from xbox_soarm_teleop.twin.hub_server import build_hub_app

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def served():
    port = _free_port()
    hub = Hub(seats=2)
    app = build_hub_app(hub, port=port, notebook_url="https://example.org/notebook")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    hub.start()
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.05)
    assert server.started
    yield hub, f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)
    hub.close()


def rooms(hub):
    return list(hub.rooms.values())


def test_student_page_needs_the_right_link(served):
    hub, base = served
    page = httpx.get(f"{base}/r/{rooms(hub)[0].token}")
    assert page.status_code == 200 and "Connect an AI agent" in page.text
    assert page.headers["referrer-policy"] == "no-referrer"  # the key is in the address
    assert httpx.get(f"{base}/r/not-a-room").status_code == 404
    assert httpx.get(f"{base}/r/{hub.admin_token}").status_code == 404


def test_page_drives_the_arm_and_reads_its_status(served):
    hub, base = served
    room = rooms(hub)[0]
    url = f"{base}/r/{room.token}/pad"
    before = httpx.post(url, json={}).json()
    assert before["name"] == "Arm 1" and before["status"][0] == "Ready"
    assert before["notebook_url"] == "https://example.org/notebook"
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        after = httpx.post(url, json={"up": -1}).json()
        time.sleep(0.05)
    assert after["ee"][2] < before["ee"][2] - 0.03
    assert after["status"][0] == "You are driving"


def test_page_stop_and_release(served):
    hub, base = served
    url = f"{base}/r/{rooms(hub)[0].token}/pad"
    httpx.post(url, json={"press": ["stop"]})
    time.sleep(0.1)
    stopped = httpx.post(url, json={}).json()
    assert stopped["stop"] is True and stopped["status"] == ["STOPPED", "Press Release stop to release"]
    httpx.post(url, json={"press": ["release"]})
    time.sleep(0.1)
    assert httpx.post(url, json={}).json()["stop"] is False


def test_bad_posts_are_refused(served):
    hub, base = served
    url = f"{base}/r/{rooms(hub)[0].token}/pad"
    assert httpx.post(url, content=b"not json").status_code == 400
    assert httpx.post(url, content=b"[1, 2]").status_code == 400
    assert httpx.post(url, content=b'{"x": "' + b"a" * 5000 + b'"}').status_code == 400
    assert httpx.post(f"{base}/r/nope/pad", json={}).status_code == 404


def test_camera_image(served):
    hub, base = served
    token = rooms(hub)[1].token
    image = httpx.get(f"{base}/r/{token}/camera.jpg", params={"view": "wrist", "w": 160, "h": 120})
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert image.content[:2] == b"\xff\xd8"
    assert httpx.get(f"{base}/r/{token}/camera.jpg", params={"view": "sky"}).status_code == 400
    assert httpx.get(f"{base}/r/{token}/camera.jpg", params={"w": "big"}).status_code == 400


def test_mcp_refuses_requests_without_a_room_key(served):
    _, base = served
    body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    headers = {"Accept": "application/json, text/event-stream"}
    assert httpx.post(f"{base}/mcp", json=body, headers=headers).status_code == 401
    wrong = {**headers, "Authorization": "Bearer nope"}
    refused = httpx.post(f"{base}/mcp", json=body, headers=wrong)
    assert refused.status_code == 401 and refused.headers["www-authenticate"] == "Bearer"


async def _call(base: str, token: str, name: str, **args):
    client = httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=30)
    async with client, streamable_http_client(f"{base}/mcp", http_client=client) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            return json.loads(result.content[0].text)


def test_each_agent_moves_only_its_own_arm(served):
    hub, base = served
    first, second = rooms(hub)
    untouched = second.service.observe()["ee"]["xyz_m"]
    # The page drove this arm a moment ago, so the person still has it.
    refused = asyncio.run(_call(base, first.token, "move_to", x=0.30, y=0.08, z=0.25))
    assert refused == {"ok": False, "reason": "not_owner", "detail": "human"}
    time.sleep(2.2)  # the page has been idle for 2 s: the agent may drive
    moved = asyncio.run(_call(base, first.token, "move_to", x=0.30, y=0.08, z=0.25))
    assert moved["ok"] and moved["action"]["state"] == "SUCCEEDED", moved
    assert first.service.observe()["ee"]["xyz_m"] == pytest.approx([0.30, 0.08, 0.25], abs=0.004)
    seen = asyncio.run(_call(base, second.token, "observe"))
    assert seen["ee"]["xyz_m"] == pytest.approx(untouched, abs=1e-6)
    assert seen["owner"] is None  # nobody has driven the second arm


def test_instructor_page_lists_arms_and_can_stop_them_all(served):
    hub, base = served
    admin = f"{base}/admin/{hub.admin_token}"
    assert "Workshop arms" in httpx.get(admin).text
    assert httpx.get(f"{base}/admin/{rooms(hub)[0].token}").status_code == 404
    listing = httpx.get(f"{admin}/rooms").json()
    assert [room["name"] for room in listing["rooms"]] == ["Arm 1", "Arm 2"]
    assert {room["token"] for room in listing["rooms"]} == set(hub.rooms)
    assert httpx.post(f"{admin}/stop_all").json() == {"ok": True, "rooms": 2}
    time.sleep(0.15)
    assert all(room["stop"] for room in httpx.get(f"{admin}/rooms").json()["rooms"])
    refused = asyncio.run(_call(base, rooms(hub)[1].token, "move_to", x=0.3, y=0.0, z=0.2))
    assert refused == {"ok": False, "reason": "stopped", "detail": "latched by human"}
    blocked = asyncio.run(_call(base, rooms(hub)[1].token, "reset_stop", confirm=True))
    assert (
        blocked["reason"] == "not_owner" and "Release stop on the arm's web page" in blocked["detail"]
    )
    httpx.post(f"{admin}/release_all")
    time.sleep(0.15)
    assert not any(room["stop"] for room in httpx.get(f"{admin}/rooms").json()["rooms"])
    assert httpx.post(f"{base}/admin/wrong/stop_all").status_code == 404
