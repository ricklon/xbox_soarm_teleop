"""A challenge on a student's arm: posted from the notebook (or the hub's
command line), it sets the desk up, gives the agent its names, offers its
request on the page, and the answer key grades against what the student
expected, in the student's own names."""

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

SWAP_YAML = """\
name: red-swap
objects:
  red_cube:    {x: 0.27, y: -0.10}
  blue_cube:   {x: 0.24, y: 0.04}
  yellow_cube: {x: 0.31, y: 0.12, turn_deg: 30}
names: misleading
request: Put the red cube in the blue bin and the blue cube in the red bin.
expect: {red_cube: blue_bin, blue_cube: red_bin}
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def served():
    port = _free_port()
    hub = Hub(seats=2, scene="sort_colors", log_dir=None, naming="truthful")
    app = build_hub_app(hub, port=port, agent_models=())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    hub.start()
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.05)
    yield hub, f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)
    hub.close()


def _token(hub, index=0):
    return list(hub.rooms)[index]


def test_a_posted_challenge_sets_the_arm_up(served):
    hub, base = served
    token = _token(hub)
    room = hub.rooms[token]
    reply = httpx.post(
        f"{base}/r/{token}/challenge", content=SWAP_YAML, headers={"Content-Type": "text/yaml"}
    )
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["ok"] is True and body["challenge"]["name"] == "red-swap"
    assert any("turned" in w for w in body["warnings"])
    time.sleep(0.3)
    truth = room.service.observe_truth()["scene"]["objects"]
    assert set(truth) == {"pick_cube", "blue_cube", "yellow_cube"}
    assert truth["pick_cube"]["xyz_m"][:2] == pytest.approx([0.27, -0.10], abs=0.005)
    assert room.service.naming == "misleading"  # the agent is told what the student chose
    # The page offers the challenge's request first.
    prompts = httpx.get(f"{base}/r/{token}/prompts").json()["prompts"]
    assert "blue cube in the red bin" in str(prompts[0])
    assert httpx.get(f"{base}/r/{token}/challenge").json()["challenge"]["name"] == "red-swap"
    # The other arm is untouched.
    other = hub.rooms[_token(hub, 1)].service
    assert other.naming == "truthful" and "green_cube" in other.observe_truth()["scene"]["objects"]


def test_the_answer_key_grades_the_challenge_in_the_students_names(served):
    hub, base = served
    token = _token(hub)
    key = httpx.get(f"{base}/r/{token}/scores").json()
    graded = key["challenge"]
    assert graded["done"] is False and graded["unsorted"] == 2
    assert graded["in_bin"] == {"red_cube": None, "blue_cube": None, "yellow_cube": None}


def test_a_broken_challenge_is_refused_with_the_reason(served):
    hub, base = served
    token = _token(hub, 1)
    reply = httpx.post(
        f"{base}/r/{token}/challenge",
        json={"name": "x", "objects": {"purple_cube": {"x": 0.27, "y": 0}}, "request": "Hi"},
    )
    assert reply.status_code == 400 and "purple_cube" in reply.json()["error"]
    assert httpx.post(f"{base}/r/not-a-room/challenge", json={}).status_code == 404


def test_clearing_the_challenge_puts_the_arm_back(served):
    hub, base = served
    token = _token(hub)
    room = hub.rooms[token]
    assert httpx.delete(f"{base}/r/{token}/challenge").status_code == 200
    time.sleep(0.3)
    assert room.service.naming == "truthful"
    assert "green_cube" in room.service.observe_truth()["scene"]["objects"]
    assert httpx.get(f"{base}/r/{token}/challenge").json()["challenge"] is None
    assert "challenge" not in httpx.get(f"{base}/r/{token}/scores").json()


def test_the_page_offers_the_example_challenges(served):
    hub, base = served
    examples = httpx.get(f"{base}/r/{_token(hub)}/challenge/examples").json()["examples"]
    names = {example["name"] for example in examples}
    assert {"crowded", "no_yellow_bin", "red_swap_tricky"} <= names
    assert all("request" in example["text"] for example in examples)


def test_the_page_can_ask_whether_the_arm_can_do_it(served):
    hub, base = served
    token = _token(hub, 1)
    reply = httpx.post(f"{base}/r/{token}/challenge/prove", content=SWAP_YAML, timeout=120)
    assert reply.status_code == 200, reply.text
    proof = reply.json()
    assert proof["verdict"] in ("doable", "fragile") and proof["why"]
    assert proof["true_run"]["in_bin"]["red_cube"] == "blue_bin"
    bad = httpx.post(f"{base}/r/{token}/challenge/prove", content="name: x\nobjects: {}\n")
    assert bad.status_code == 400
    # Proving uses a twin of its own: this arm was not touched.
    assert hub.rooms[token].challenge is None


def test_a_scene_picked_after_a_challenge_is_not_graded_as_it(served):
    hub, base = served
    token = _token(hub)
    httpx.post(f"{base}/r/{token}/challenge", content=SWAP_YAML)
    room = hub.rooms[token]
    room.service.load_scene("sort_colors", client_id="student").result(timeout=10)
    assert "challenge" not in httpx.get(f"{base}/r/{token}/scores").json()
    httpx.delete(f"{base}/r/{token}/challenge")
