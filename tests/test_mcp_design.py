"""An attached agent designs a challenge: check, prove, load and clear it,
through MCP, on its own arm, when the hub allows it (``--agent-designs``).

The designer works in a person's names (``red_cube``), whatever names the
arm's agents are shown. Its replies say to solve with a fresh agent: the one
that designed it knows the answer.
"""

import asyncio
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

SWAP = """\
name: close-swap
objects:
  red_cube:  {x: 0.27, y: -0.05}
  blue_cube: {x: 0.27, y: 0.02}
names: misleading
request: Put the red cube in the blue bin and the blue cube in the red bin.
expect: {red_cube: blue_bin, blue_cube: red_bin}
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(designs: bool):
    service = build_twin(scene="sort_colors", with_controller=False, naming="truthful")
    alive = threading.Event()
    alive.set()

    def loop():
        try:
            while alive.is_set():
                start = time.monotonic()
                service.tick()
                time.sleep(max(0.0, service.dt - (time.monotonic() - start)))
        finally:
            service.close()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    mcp = McpThread(build_mcp_server(service, designs=designs), port=_free_port())
    mcp.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", mcp.port)) == 0:
                break
        time.sleep(0.05)
    return service, mcp.url, alive, thread


@pytest.fixture(scope="module")
def designing():
    service, url, alive, thread = _serve(designs=True)
    yield service, url
    alive.clear()
    thread.join(timeout=10)


def _calls(url, calls):
    async def go():
        async with streamable_http_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                answers = []
                for name, arguments in calls:
                    result = await session.call_tool(name, arguments)
                    answers.append(json.loads(result.content[0].text))
                return answers, [tool.name for tool in (await session.list_tools()).tools]

    return asyncio.run(go())


def test_the_design_tools_are_off_unless_the_hub_allows_them():
    service, url, alive, thread = _serve(designs=False)
    try:
        _, tools = _calls(url, [])
        assert not {"check_challenge", "prove_challenge", "load_challenge", "clear_challenge"} & set(
            tools
        )
    finally:
        alive.clear()
        thread.join(timeout=10)


def test_an_agent_checks_a_challenge_in_a_persons_names(designing):
    service, url = designing
    (checked, broken), tools = _calls(
        url,
        [
            ("check_challenge", {"challenge": SWAP}),
            (
                "check_challenge",
                {"challenge": "name: x\nobjects: {purple_cube: {x: 0.27, y: 0}}\nrequest: hi\n"},
            ),
        ],
    )
    assert {"check_challenge", "prove_challenge", "load_challenge", "clear_challenge"} <= set(tools)
    assert checked["ok"] is True
    assert any("red_cube and blue_cube" in note for note in checked["notes"])  # not renamed
    assert broken["ok"] is False and "purple_cube" in broken["error"]


def test_an_agent_loads_and_clears_a_challenge_on_its_arm(designing):
    service, url = designing
    (loaded,), _ = _calls(url, [("load_challenge", {"challenge": SWAP})])
    assert loaded["ok"] is True and loaded["name"] == "close-swap"
    assert "fresh" in loaded["next"]  # solve it with an agent that has not seen the answer
    time.sleep(0.3)
    assert service.challenge.name == "close-swap" and service.naming == "misleading"
    assert service.observe_truth()["scene"]["objects"]["blue_cube"]["xyz_m"][:2] == pytest.approx(
        [0.27, 0.02], abs=0.005
    )
    (cleared,), _ = _calls(url, [("clear_challenge", {})])
    assert cleared["ok"] is True
    time.sleep(0.3)
    assert service.challenge is None and service.naming == "truthful"
    assert "green_cube" in service.observe_truth()["scene"]["objects"]


def test_an_agent_asks_whether_the_arm_can_do_it(designing):
    _service, url = designing
    (proof,), _ = _calls(url, [("prove_challenge", {"challenge": SWAP})])
    assert proof["verdict"] in ("doable", "fragile"), proof
    assert proof["why"]
