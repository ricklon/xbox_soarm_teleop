"""The lean agent loop: any OpenRouter model drives the twin through its MCP tools.

The model is faked here: a script of replies stands in for the LLM, so the
loop itself is what is tested (tool schemas, calling tools in order, feeding
results back, limits, usage). The twin and its MCP server are real.
"""

import json
import logging
import socket
import threading
import time
import warnings

import pytest

from xbox_soarm_teleop.twin import build_twin
from xbox_soarm_teleop.twin.agent_loop import AgentError, mcp_tool_to_function, run_agent
from xbox_soarm_teleop.twin.mcp_server import McpThread, build_mcp_server
from xbox_soarm_teleop.twin.pointing import OraclePointer

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

FAKE_KEY = "sk-or-TESTKEY0123456789abcdef"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def twin_url():
    service = build_twin(
        scene="pick_place_basic", with_controller=False, hide_positions=True, pointer=OraclePointer()
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
    yield mcp.url, service
    alive.clear()
    thread.join(timeout=5)


def _call(name, args, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _reply(content=None, calls=None, cost=0.001, prompt=1000, completion=50):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {
        "choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "cost": cost},
    }


class ScriptedModel:
    """Answers with the next scripted reply and keeps what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def __call__(self, url, body, headers=None, timeout=None):
        self.sent.append({"url": url, "body": json.loads(json.dumps(body)), "headers": headers})
        return self.replies.pop(0)


def test_mcp_tools_become_function_tools():
    class Tool:
        name = "move_by"
        description = "Move the tool point."
        input_schema = {
            "type": "object",
            "properties": {"dx": {"type": "number"}, "ctx": {"type": "object"}},
            "required": ["dx"],
        }

    function = mcp_tool_to_function(Tool())
    assert function["type"] == "function"
    assert function["function"]["name"] == "move_by"
    assert function["function"]["parameters"]["properties"]["dx"] == {"type": "number"}
    assert function["function"]["parameters"]["required"] == ["dx"]


def test_the_loop_calls_tools_in_order_and_feeds_results_back(twin_url, monkeypatch):
    url, service = twin_url
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    model = ScriptedModel(
        [
            # Two calls in one reply: run one after the other, never at once.
            _reply(calls=[_call("observe", {}, "c1"), _call("locate", {"name": "pick_cube"}, "c2")]),
            _reply(calls=[_call("move_to", {"x": 0.3, "y": 0.0, "z": 0.22}, "c3")]),
            _reply(content="Found it and moved over the table."),
        ]
    )
    run = run_agent(url, "z-ai/glm-5.3-flash", "Find the red cube.", post=model)
    assert run.stop_reason == "answered"
    assert run.final_text == "Found it and moved over the table."
    assert [call["name"] for call in run.tool_calls] == ["observe", "locate", "move_to"]
    assert all(call["ok"] for call in run.tool_calls)

    first = model.sent[0]
    assert first["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert first["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert first["body"]["model"] == "z-ai/glm-5.3-flash"
    assert first["body"]["usage"] == {"include": True}
    names = {tool["function"]["name"] for tool in first["body"]["tools"]}
    assert {"observe", "locate", "move_to", "set_gripper", "camera"} <= names
    system = first["body"]["messages"][0]
    assert (
        system["role"] == "system" and "No-coordinates mode" in system["content"]
    )  # the server's own instructions
    assert first["body"]["messages"][1] == {"role": "user", "content": "Find the red cube."}

    # The second request carries both tool results, answered in order.
    tool_messages = [m for m in model.sent[1]["body"]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
    located = json.loads(tool_messages[1]["content"])
    assert located["ok"] is True and len(located["xyz_m"]) == 3
    assert "xyz_m" not in json.loads(tool_messages[0]["content"])["scene"]["objects"]["pick_cube"]

    assert run.usage == {
        "steps": 3,
        "input_tokens": 3000,
        "output_tokens": 150,
        "cost_usd": pytest.approx(0.003),
    }
    assert run.time["model_s"] >= 0 and run.time["tools_s"] > 0
    assert service.estimate_scores()[-1]["source"].startswith("locate:oracle")


def test_bad_tool_calls_are_told_to_the_model(twin_url, monkeypatch):
    url, _ = twin_url
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    model = ScriptedModel(
        [
            _reply(
                calls=[
                    {
                        "id": "b1",
                        "type": "function",
                        "function": {"name": "teleport", "arguments": "{}"},
                    },
                    {
                        "id": "b2",
                        "type": "function",
                        "function": {"name": "move_by", "arguments": "{not json"},
                    },
                ]
            ),
            _reply(content="Giving up."),
        ]
    )
    run = run_agent(url, "m", "Do something.", post=model)
    errors = [
        json.loads(m["content"]) for m in model.sent[1]["body"]["messages"] if m["role"] == "tool"
    ]
    assert errors[0]["ok"] is False and "teleport" in errors[0]["detail"]
    assert errors[1]["ok"] is False and "JSON" in errors[1]["detail"]
    assert [call["ok"] for call in run.tool_calls] == [False, False]


def test_step_and_cost_limits_stop_the_run(twin_url, monkeypatch):
    url, _ = twin_url
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    looping = ScriptedModel([_reply(calls=[_call("observe", {}, f"s{i}")]) for i in range(10)])
    run = run_agent(url, "m", "Loop.", post=looping, max_steps=3)
    assert run.stop_reason == "max_steps" and run.usage["steps"] == 3

    pricey = ScriptedModel(
        [_reply(calls=[_call("observe", {}, f"p{i}")], cost=0.04) for i in range(10)]
    )
    run = run_agent(url, "m", "Loop.", post=pricey, max_cost_usd=0.1)
    assert run.stop_reason == "max_cost" and run.usage["steps"] == 3  # stops once past $0.10


def test_no_key_says_so(twin_url, monkeypatch, tmp_path):
    from tests.test_twin_locate import _isolate_env_files

    _isolate_env_files(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(AgentError, match="OPENROUTER_API_KEY"):
        run_agent(twin_url[0], "m", "Hi.", post=ScriptedModel([]))
