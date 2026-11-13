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
    assert before["picture_ms"] == 100  # the page asks for 10 pictures a second by default
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


# ------------------------------------------------- the page's own agent tools (WebMCP)
def _page_tool(base: str, token: str, name: str, **args):
    reply = httpx.post(f"{base}/r/{token}/tools/{name}", json=args, timeout=40)
    assert reply.status_code == 200, reply.text
    return reply.json()


def _text(result: dict):
    return json.loads(result["content"][0]["text"])


def test_page_lists_the_same_tools_the_agent_address_serves(served):
    hub, base = served
    token = rooms(hub)[1].token

    async def over_mcp():
        client = httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=30)
        async with client, streamable_http_client(f"{base}/mcp", http_client=client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return {tool.name: tool.input_schema for tool in (await session.list_tools()).tools}

    expected = asyncio.run(over_mcp())
    listing = httpx.get(f"{base}/r/{token}/tools")
    assert listing.status_code == 200 and listing.headers["cache-control"] == "no-store"
    tools = {tool["name"]: tool for tool in listing.json()["tools"]}
    assert set(tools) == set(expected) and {"observe", "move_to", "camera", "stop"} <= set(tools)
    assert tools["move_to"]["inputSchema"] == expected["move_to"]
    assert "ctx" not in tools["move_to"]["inputSchema"]["properties"]
    assert all(tool["description"] for tool in tools.values())
    assert httpx.get(f"{base}/r/nope/tools").status_code == 404


def test_page_tool_drives_only_its_own_arm(served):
    hub, base = served
    first, second = rooms(hub)
    untouched = first.service.observe()["ee"]["xyz_m"]
    deadline = time.monotonic() + 4  # the instructor's release was a moment ago
    while True:
        moved = _text(_page_tool(base, second.token, "move_to", x=0.28, y=-0.05, z=0.24))
        if moved["ok"] or time.monotonic() > deadline:
            break
        time.sleep(0.3)
    assert moved["ok"] and moved["action"]["state"] == "SUCCEEDED", moved
    seen = _text(_page_tool(base, second.token, "observe"))
    assert seen["ee"]["xyz_m"] == pytest.approx([0.28, -0.05, 0.24], abs=0.004)
    assert seen["owner"] == "browser-agent"
    assert first.service.observe()["ee"]["xyz_m"] == pytest.approx(untouched, abs=1e-6)


def test_page_tool_refusals_are_results_not_errors(served):
    hub, base = served
    token = rooms(hub)[1].token
    refused = _text(_page_tool(base, token, "move_to", x=0.3, y=0.0, z=0.02))
    assert refused["ok"] is False and refused["reason"] == "workspace_bound"


def test_page_tool_camera_returns_a_picture(served):
    hub, base = served
    result = _page_tool(base, rooms(hub)[1].token, "camera", view="overhead", width=160, height=120)
    block = result["content"][0]
    assert block["type"] == "image" and block["mimeType"] == "image/jpeg"
    import base64

    assert base64.b64decode(block["data"])[:2] == b"\xff\xd8"


def test_page_tool_cannot_release_the_students_stop(served):
    hub, base = served
    room = rooms(hub)[1]
    httpx.post(f"{base}/r/{room.token}/pad", json={"press": ["stop"]})
    time.sleep(0.15)
    blocked = _text(_page_tool(base, room.token, "reset_stop", confirm=True))
    assert blocked["ok"] is False and blocked["reason"] == "not_owner"
    httpx.post(f"{base}/r/{room.token}/pad", json={"press": ["release"]})
    time.sleep(0.15)
    assert room.service.observe()["control"]["stop_latched_by"] is None


def test_bad_page_tool_calls_are_refused(served):
    hub, base = served
    url = f"{base}/r/{rooms(hub)[1].token}/tools"
    assert httpx.post(f"{url}/fly_away", json={}).status_code == 404
    assert httpx.post(f"{url}/move_to", content=b"not json").status_code == 400
    assert httpx.post(f"{url}/move_to", content=b"[1]").status_code == 400
    assert httpx.post(f"{url}/observe", content=b'{"x": "' + b"a" * 5000 + b'"}').status_code == 400
    wrong = httpx.post(f"{url}/move_to", json={"x": "far"})
    assert wrong.status_code == 400 and "error" in wrong.json()
    assert httpx.post(f"{base}/r/nope/tools/observe", json={}).status_code == 404


def test_student_page_registers_webmcp_tools_when_the_browser_has_them(served):
    hub, base = served
    page = httpx.get(f"{base}/r/{rooms(hub)[0].token}").text
    assert "document.modelContext" in page and "navigator.modelContext" in page
    assert "registerTool" in page


def test_picture_rate_is_told_to_the_page_and_kept_sane():
    hub = Hub(seats=1)
    try:
        from starlette.testclient import TestClient

        token = next(iter(hub.rooms))
        for asked, expected in ((20, 50), (1000, 33), (0.1, 1000)):
            client = TestClient(build_hub_app(hub, picture_hz=asked), base_url="http://127.0.0.1:8800")
            assert client.post(f"/r/{token}/pad", json={}).json()["picture_ms"] == expected
        page = TestClient(build_hub_app(hub), base_url="http://127.0.0.1:8800").get(f"/r/{token}").text
        assert "picture_ms" in page
    finally:
        hub.close()


def test_hub_command_options():
    from xbox_soarm_teleop.cli.hub import build_parser

    defaults = build_parser().parse_args([])
    assert defaults.mesh_detail_mm == 2.0 and defaults.picture_rate == 10.0
    chosen = build_parser().parse_args(["--mesh-detail-mm", "0", "--picture-rate", "15"])
    assert chosen.mesh_detail_mm == 0.0 and chosen.picture_rate == 15.0


# ------------------------------------------------------------------ keys stay hidden
def test_pages_can_be_shown_without_a_key_in_the_address(served):
    """The pages move to a key-free address once loaded; the server serves them there."""
    hub, base = served
    secrets_ = [hub.admin_token, *hub.rooms]
    for path, marker in (("/arm", "Connect an AI agent"), ("/instructor", "Workshop arms")):
        page = httpx.get(base + path)
        assert page.status_code == 200 and marker in page.text
        assert page.headers["cache-control"] == "no-store"
        assert not any(secret in page.text for secret in secrets_)
        assert "sessionStorage" in page.text and "history.replaceState" in page.text


def test_student_page_masks_the_key_until_asked(served):
    hub, base = served
    page = httpx.get(f"{base}/r/{rooms(hub)[0].token}").text
    assert 'id="showKey"' in page  # the key is revealed only on purpose
    assert "dataset.copy" in page  # copying still gives the real value
    assert '$("token").textContent = token' not in page


def test_instructor_page_has_no_links_to_hover_over(served):
    hub, base = served
    page = httpx.get(f"{base}/admin/{hub.admin_token}").text
    assert 'el.querySelector("a").href' not in page and "window.open" in page


def test_hub_start_up_text_keeps_keys_out_of_the_terminal():
    from xbox_soarm_teleop.cli.hub import build_parser, startup_lines

    args = build_parser().parse_args([])
    assert args.open_browser is True and args.show_links is False
    assert build_parser().parse_args(["--no-open"]).open_browser is False
    quiet = "\n".join(
        startup_lines("https://x.example", "ADMINKEY123", "roster.json", public=True, opened=True)
    )
    assert "ADMINKEY123" not in quiet and "roster.json" in quiet and "opened" in quiet
    closed = "\n".join(
        startup_lines("https://x.example", "ADMINKEY123", "roster.json", public=True, opened=False)
    )
    assert "ADMINKEY123" not in closed and "--show-links" in closed
    shown = "\n".join(
        startup_lines(
            "https://x.example", "ADMINKEY123", "roster.json", public=True, opened=False, show=True
        )
    )
    assert "https://x.example/admin/ADMINKEY123" in shown


def test_student_page_explains_a_local_only_arm(served):
    hub, base = served
    page = httpx.get(f"{base}/r/{rooms(hub)[0].token}").text
    assert 'id="localNote" hidden' in page and "--public-url" in page
