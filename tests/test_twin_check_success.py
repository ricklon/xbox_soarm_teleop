"""check_success: the agent asks a judge (Clef) whether the task is done, by looking.

On the real arm nothing reports ``in_goal``; the agent has to look. The tool
takes a front picture, asks the judge, and answers with probabilities and a
verdict (done, look again, retry). The twin's truth is logged beside the
judge's answer, so the check itself can be graded, and is never returned.
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
from xbox_soarm_teleop.twin.judge import WHERE_OPTIONS, ClefJudge
from xbox_soarm_teleop.twin.mcp_server import McpThread, build_mcp_server
from xbox_soarm_teleop.twin.service import verdict_for

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

FAKE_TOKEN = "cfTESTTOKEN0123456789abcdef"
FAKE_ACCOUNT = "0123456789abcdef0123456789abcdef"


def clef_says(in_goal: float, held: float = 0.05, where: str = "in_goal_box"):
    """A fake Clef that always gives these answers, and keeps what it was asked."""
    asked = []

    def post(url, body, headers=None, timeout=None):
        asked.append(body)
        probabilities = {option: 0.01 for option in WHERE_OPTIONS}
        probabilities[where] = 0.97
        return {
            "success": True,
            "result": {
                "answers": {
                    "in_goal": {"type": "noul", "noul": in_goal},
                    "held": {"type": "noul", "noul": held},
                    "tipped": {"type": "noul", "noul": 0.02},
                    "where": {"type": "choice", "choice": where, "probabilities": probabilities},
                },
                "usage": {"input_tokens": 880},
            },
        }

    return ClefJudge(model="clef", post=post), asked


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def running():
    service = build_twin(scene="distractors_color", with_controller=False, hide_positions=True)
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
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def cloudflare_keys(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", FAKE_ACCOUNT)


def test_verdicts_follow_the_calibrated_thresholds():
    assert verdict_for(0.95) == "done"
    assert verdict_for(0.9) == "done"
    assert verdict_for(0.5) == "look_again"
    assert verdict_for(0.1) == "retry"
    assert verdict_for(0.02) == "retry"


def test_check_success_asks_the_judge_about_a_front_picture(running):
    service, _ = running
    service.judge, asked = clef_says(in_goal=0.96)
    result = service.check_success("pick_cube", client_id="t")
    assert result["ok"] is True and result["verdict"] == "done"
    assert result["p_in_goal"] == pytest.approx(0.96)
    assert result["description"] == "red cube" and result["judge"] == "clef"
    assert result["cost_usd"] == pytest.approx(880 * 0.24 / 1e6)
    # One front picture, and Clef told about the other objects on the table.
    body = asked[-1]
    assert len(body["images"]) == 1 and body["images"][0].startswith("data:image/png;base64,")
    assert "blue cube" in body["state"] and "green cube" in body["state"]
    assert "front" in body["state"]


def test_the_truth_is_logged_and_scored_but_never_returned(running):
    service, _ = running
    service.judge, _ = clef_says(in_goal=0.96)  # wrong: the cube is on the table
    before = len(service.success_checks())
    result = service.check_success("pick_cube", client_id="t")
    assert "truth" not in json.dumps(result) and "correct" not in json.dumps(result)
    checks = service.success_checks()
    assert len(checks) == before + 1
    last = checks[-1]
    assert last["truth"]["in_goal"] is False
    assert last["verdict"] == "done" and last["correct"] is False


def test_middle_answers_say_look_again_and_low_ones_retry(running):
    service, _ = running
    service.judge, _ = clef_says(in_goal=0.5)
    assert service.check_success("pick_cube")["verdict"] == "look_again"
    service.judge, _ = clef_says(in_goal=0.03, where="on_table")
    result = service.check_success("pick_cube")
    assert result["verdict"] == "retry" and result["where"] == "on_table"
    assert service.success_checks()[-1]["correct"] is True


def test_check_success_problems(running):
    service, _ = running
    service.judge = None
    assert service.check_success("pick_cube")["reason"] == "no_judge"
    service.judge, _ = clef_says(in_goal=0.9)
    assert service.check_success("moon")["reason"] == "invalid_request"

    def broken(*args, **kwargs):
        from xbox_soarm_teleop.twin.pointing import PointerError

        raise PointerError(f"HTTP 401: token {FAKE_TOKEN}")

    service.judge = ClefJudge(post=broken)
    failed = service.check_success("pick_cube")
    assert failed["reason"] == "judge_error" and FAKE_TOKEN not in failed["detail"]


def test_check_success_over_mcp(running):
    service, url = running
    service.judge, _ = clef_says(in_goal=0.04, where="on_table")

    async def go():
        async with streamable_http_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                result = await session.call_tool("check_success", {"name": "pick_cube"})
                return init, tools, json.loads(result.content[0].text)

    init, tools, result = asyncio.run(go())
    assert "check_success" in tools
    assert result["ok"] is True and result["verdict"] == "retry"
    assert "check_success" in (init.instructions or "")
