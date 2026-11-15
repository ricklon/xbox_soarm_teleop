"""What an agent is told about objects is part of the experiment.

A blind agent sorted the cubes "by colour" from their names: blue_cube went
in the blue tray, and pick_cube, "likely the decoy/red", in the green one. A
real robot is not told names; it sees. So the twin can show objects under
other names, and the trial records which:

- ``as_is``: the twin's own names (pick_cube, blue_cube, ...);
- ``truthful``: names that say what each object looks like (red_cube);
- ``neutral``: object_1, object_2, ... (shuffled: the order says nothing);
- ``misleading``: each named for another object's colour (the red cube is
  "blue_cube"), to see whether an agent believes the words or its eyes.

Names change only at the agent's door (the MCP tools); the session log,
grading and the answer key keep the twin's own names. Under any renaming,
tool answers carry no ``description`` (it would say the true colour).
"""

import asyncio
import json
import logging
import re
import socket
import threading
import time
import warnings

import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from xbox_soarm_teleop.twin import build_twin
from xbox_soarm_teleop.twin.layouts import LAYOUTS
from xbox_soarm_teleop.twin.mcp_server import McpThread, build_mcp_server
from xbox_soarm_teleop.twin.naming import APPEARANCE, NAMINGS, shown_names
from xbox_soarm_teleop.twin.pointing import OraclePointer

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

SORT = list(LAYOUTS["sort_colors"].objects)
DECOY = list(LAYOUTS["sort_colors_decoy"].objects)


def test_the_conditions():
    assert set(NAMINGS) == {"as_is", "truthful", "neutral", "misleading"}
    assert shown_names(SORT, "as_is") is None
    assert shown_names(SORT, "truthful") == {
        "pick_cube": "red_cube",
        "blue_cube": "blue_cube",
        "green_cube": "green_cube",
    }
    neutral = shown_names(SORT, "neutral")
    assert sorted(neutral.values()) == ["object_1", "object_2", "object_3"]
    assert neutral != dict(zip(SORT, ["object_1", "object_2", "object_3"]))  # shuffled


@pytest.mark.parametrize("objects", [SORT, DECOY])
def test_misleading_names_name_another_colour_on_the_table(objects):
    names = shown_names(objects, "misleading")
    colours = {APPEARANCE[name][0] for name in objects}
    assert len(set(names.values())) == len(objects)
    for name, shown in names.items():
        said = shown.split("_")[0]
        assert said != APPEARANCE[name][0], (name, shown)
        assert said in colours, (name, shown)  # it points at a real other object


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def served():
    service = build_twin(
        scene="sort_colors",
        with_controller=False,
        hide_positions=True,
        pointer=OraclePointer(),
        naming="misleading",
    )
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
    mcp = McpThread(build_mcp_server(service), port=_free_port())
    mcp.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", mcp.port)) == 0:
                break
        time.sleep(0.05)
    yield service, mcp.url
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
                return answers

    return asyncio.run(go())


INTERNAL = re.compile(r"(?<![A-Za-z0-9_])(pick_cube)(?![A-Za-z0-9_])")


def test_the_agent_sees_only_the_shown_names(served):
    service, url = served
    names = service.shown_names
    observed, located, refused = _calls(
        url,
        [
            ("observe", {}),
            ("locate", {"name": names["pick_cube"]}),
            ("locate", {"name": "pick_cube"}),
        ],
    )
    assert set(observed["scene"]["objects"]) == set(names.values())
    text = json.dumps([observed, located])
    assert not INTERNAL.search(text)  # the twin's own name never shows
    assert located["ok"] is True and located["name"] == names["pick_cube"]
    assert "description" not in located  # it would say "red cube"
    assert refused["ok"] is False and refused["reason"] == "invalid_request"
    # The twin itself, its log and its grading keep their own names.
    assert "pick_cube" in service.observe()["scene"]["objects"]
    assert service.estimate_scores()[-1]["name"] == "pick_cube"
