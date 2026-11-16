"""The agent panel: a student's page runs the benchmarked agent loop on its arm.

Students could already bring their own agent or use a notebook; the panel
gives them the loop we test (twin/agent_loop.py) without either. It runs on
the hub, on the instructor's OpenRouter key, with a model the instructor
allows, connects to the arm through the hub's own MCP address and key (so it
gets the same tools, names and instructions as any agent), and is limited by a
per-arm budget and a cap on runs across the hub. The model here is scripted:
nothing is spent.
"""

import json
import logging
import socket
import threading
import time
import warnings

import httpx2 as httpx
import pytest
import uvicorn

from xbox_soarm_teleop.twin.hub import Hub
from xbox_soarm_teleop.twin.hub_server import build_hub_app

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

FAKE_KEY = "sk-or-TESTKEY0123456789abcdef"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _call(name, args, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _reply(content=None, calls=None, cost=0.01):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "cost": cost},
    }


class Scripted:
    """A model that answers from a script; ``hold`` makes it wait before answering."""

    def __init__(self):
        self.replies = []
        self.bodies = []
        self.hold = threading.Event()
        self.hold.set()

    def __call__(self, url, body, headers=None, timeout=None):
        self.bodies.append(body)
        self.hold.wait(30)
        if not self.replies:
            return _reply(content="RESULT: done")
        return self.replies.pop(0)


@pytest.fixture(scope="module")
def served():
    import os

    os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
    port = _free_port()
    hub = Hub(seats=2, scene="sort_colors", log_dir=None)
    model = Scripted()
    app = build_hub_app(
        hub,
        port=port,
        agent_models=("google/gemma-4-31b-it", "z-ai/glm-5.3-flash"),
        agent_budget_usd=0.05,
        agent_post=model,
    )
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    hub.start()
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.05)
    yield hub, f"http://127.0.0.1:{port}", model
    server.should_exit = True
    thread.join(timeout=5)
    hub.close()


def _token(hub, index=0):
    return list(hub.rooms)[index]


def _wait_done(base, token, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = httpx.get(f"{base}/r/{token}/agent").json()
        if state["run"] and state["run"]["done"]:
            return state
        time.sleep(0.2)
    raise AssertionError("the run did not finish")


def test_the_panel_says_what_it_offers(served):
    hub, base, _ = served
    state = httpx.get(f"{base}/r/{_token(hub)}/agent").json()
    assert state["available"] is True
    assert state["models"] == ["google/gemma-4-31b-it", "z-ai/glm-5.3-flash"]
    assert state["budget_usd"] == pytest.approx(0.05) and state["spent_usd"] == 0
    assert FAKE_KEY not in json.dumps(state)
    assert httpx.get(f"{base}/r/not-a-room/agent").status_code == 404


def test_a_run_shows_its_steps_and_result(served):
    hub, base, model = served
    token = _token(hub)
    model.replies = [
        _reply(content="Looking first.", calls=[_call("observe", {}, "c1")]),
        _reply(content="Everything is on the table. RESULT: not_done"),
    ]
    started = httpx.post(
        f"{base}/r/{token}/agent/run",
        json={"task": "Look around.", "model": "google/gemma-4-31b-it", "sees": False},
    )
    assert started.status_code == 200 and started.json()["ok"] is True
    state = _wait_done(base, token)
    run = state["run"]
    assert run["stop_reason"] == "answered" and "RESULT: not_done" in run["final_text"]
    assert run["steps"][0]["text"] == "Looking first."
    call = run["steps"][0]["calls"][0]
    assert call["name"] == "observe" and call["ok"] is True and call["result"].startswith("{")
    assert state["spent_usd"] == pytest.approx(0.02)


def test_bad_requests_are_refused(served):
    hub, base, _ = served
    url = f"{base}/r/{_token(hub, 1)}/agent/run"
    unknown = httpx.post(url, json={"task": "Hi", "model": "some/other-model"})
    assert unknown.status_code == 400 and "model" in unknown.json()["error"]
    empty = httpx.post(url, json={"task": "  ", "model": "google/gemma-4-31b-it"})
    assert empty.status_code == 400


def test_one_run_at_a_time_and_stop_ends_it(served):
    hub, base, model = served
    token = _token(hub, 1)
    model.hold.clear()  # the model is "thinking"
    model.replies = [_reply(calls=[_call("observe", {}, f"c{i}")]) for i in range(20)]
    body = {"task": "Keep looking.", "model": "z-ai/glm-5.3-flash"}
    assert httpx.post(f"{base}/r/{token}/agent/run", json=body).status_code == 200
    busy = httpx.post(f"{base}/r/{token}/agent/run", json=body)
    assert busy.status_code == 409
    stopped = httpx.post(f"{base}/r/{token}/agent/stop")
    assert stopped.status_code == 200
    model.hold.set()
    state = _wait_done(base, token)
    assert state["run"]["stop_reason"] == "stopped"


def test_stop_ends_a_pick_in_the_middle(served):
    hub, base, model = served
    token = _token(hub)
    room = hub.rooms[token]
    model.replies = [_reply(calls=[_call("pick", {"name": "pick_cube"}, "p1")])]
    body = {"task": "Pick up the red cube.", "model": "google/gemma-4-31b-it"}
    assert httpx.post(f"{base}/r/{token}/agent/run", json=body).status_code == 200
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and room.service.action is None:
        time.sleep(0.05)  # the pick has started moving the arm
    asked = time.monotonic()
    httpx.post(f"{base}/r/{token}/agent/stop")
    state = _wait_done(base, token, timeout=30)
    assert time.monotonic() - asked < 10
    assert state["run"]["stop_reason"] == "stopped"
    assert room.service.action is None or room.service.action.state.terminal


def test_after_a_panel_run_another_agent_can_drive(served):
    hub, base, model = served
    token = _token(hub)
    model.replies = [_reply(calls=[_call("home", {}, "h1")]), _reply(content="RESULT: done")]
    body = {"task": "Go home.", "model": "google/gemma-4-31b-it"}
    assert httpx.post(f"{base}/r/{token}/agent/run", json=body).status_code == 200
    _wait_done(base, token)
    room = hub.rooms[token]
    status = room.service.submit({"type": "home"}, client_id="someone-else").result(timeout=5)
    assert status.id  # not refused as "not the owner"


def test_the_budget_runs_out(served):
    hub, base, model = served
    token = _token(hub)
    model.replies = [_reply(content="RESULT: done", cost=0.05)]
    body = {"task": "Say done.", "model": "google/gemma-4-31b-it"}
    httpx.post(f"{base}/r/{token}/agent/run", json=body)
    _wait_done(base, token)
    refused = httpx.post(f"{base}/r/{token}/agent/run", json=body)
    assert refused.status_code == 402 and "budget" in refused.json()["error"]


def test_the_panel_asks_the_agent_to_say_what_it_is_doing(served):
    """Issue 53: "I wish it could communicate to the operator more." The
    panel asks for a sentence before each action and a summary at the end,
    and the page shows the summary's verdict and what the arm can do."""
    hub, base, model = served
    token = _token(hub, 1)
    model.replies = [_reply(content="I looked. RESULT: done")]
    body = {"task": "Look around.", "model": "google/gemma-4-31b-it"}
    assert httpx.post(f"{base}/r/{token}/agent/run", json=body).status_code == 200
    state = _wait_done(base, token)
    asked = json.dumps(model.bodies[-1]["messages"])
    assert "Look around." in asked and "before each action" in asked and "summary" in asked
    assert "Never do something" in asked  # it once pushed a cube for a tray and said done
    assert state["run"]["task"] == "Look around."  # the page shows what the student typed
    assert state["run"]["verdict"] == "done"
    page = httpx.get(f"{base}/arm").text
    assert "What the arm can do" in page and "fixed to the desk" in page
