"""No-coordinates mode: agents must find objects by looking, and the twin grades their guesses.

With ``hide_positions`` the twin keeps every object's true position to itself
(and in the session log) and tells an agent only what a real arm could know:
object sizes, the goal box, its own pose and jaw center, and whether it holds
something. ``report_estimate`` scores a guessed position against the truth.
"""

import asyncio
import json
import logging
import socket
import threading
import time
import warnings

import numpy as np
import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from xbox_soarm_teleop.schemas.agent import ActionState
from xbox_soarm_teleop.twin import ActionRejected, build_twin
from xbox_soarm_teleop.twin.grasp import GraspSettings
from xbox_soarm_teleop.twin.mcp_server import McpThread, build_mcp_server
from xbox_soarm_teleop.twin.session_log import read_session_log

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

# What an object entry may carry when positions are hidden.
HIDDEN_OBJECT_KEYS = {"size_m", "grip_width_m", "attached"}
CUBE_FIT = GraspSettings().gripper_for_opening(0.04)


def _twin(**kwargs):
    service = build_twin(scene="pick_place_basic", with_controller=False, **kwargs)
    service.tick()
    return service


def _run(twin, request):
    future = twin.submit(request)
    twin.tick()
    status = future.result(timeout=0)
    for _ in range(1500):
        status = twin.wait(status.id, 0)
        if status.state.terminal:
            break
        twin.tick()
    assert status.state is ActionState.SUCCEEDED, (request, status)
    return status


def _resolve(twin, future):
    twin.tick()
    return future.result(timeout=0)


def _close_above_cube(twin):
    """Close the jaws 6 cm above the cube: a near miss the grasp explains."""
    x, y, z = twin.grasp.position("pick_cube")
    _run(twin, {"type": "move_ee_to", "params": {"xyz_m": [float(x), float(y), float(z) + 0.06]}})
    _run(twin, {"type": "set_wrist_roll", "params": {"angle_rad": -1.5}})
    open_wide = GraspSettings().gripper_for_opening(0.09)
    _run(twin, {"type": "set_gripper", "params": {"position": open_wide}})
    _run(twin, {"type": "set_gripper", "params": {"position": CUBE_FIT}})


# ------------------------------------------------------------------ observe
def test_normal_mode_shows_positions():
    twin = _twin()
    scene = twin.observe()["scene"]
    assert scene["object_positions"] == "shown"
    cube = scene["objects"]["pick_cube"]
    assert cube["xyz_m"] == pytest.approx(list(twin.grasp.position("pick_cube")), abs=1e-3)
    assert "in_goal" in cube and "off_table" in cube


def test_hidden_mode_leaves_positions_out_of_observe():
    twin = _twin(hide_positions=True)
    observation = twin.observe()
    scene = observation["scene"]
    assert scene["object_positions"] == "hidden"
    cube = scene["objects"]["pick_cube"]
    assert set(cube) == HIDDEN_OBJECT_KEYS  # no xyz_m, in_goal or off_table
    assert cube["size_m"] == pytest.approx([0.04, 0.04, 0.04], abs=1e-3)
    # What the arm knows about itself, and the goal box, are still there.
    assert scene["goal"]["center_xyz_m"]
    assert len(observation["jaw_center_xyz_m"]) == 3
    assert len(observation["ee"]["xyz_m"]) == 3
    # Nowhere in the observation does the true position appear.
    truth = [round(float(v), 4) for v in twin.grasp.position("pick_cube")]
    assert json.dumps(truth[:2])[1:-1] not in json.dumps(observation)


def test_hidden_mode_keeps_the_truth_for_the_grader():
    twin = _twin(hide_positions=True)
    truth = twin.observe_truth()["scene"]["objects"]["pick_cube"]
    assert truth["xyz_m"] == pytest.approx(list(twin.grasp.position("pick_cube")), abs=1e-3)
    assert "in_goal" in truth


def test_hidden_mode_session_log_keeps_true_positions(tmp_path):
    twin = _twin(hide_positions=True, log_path=tmp_path / "s.jsonl")
    for _ in range(10):
        twin.tick()
    twin.close()
    observations = [
        line for line in read_session_log(tmp_path / "s.jsonl") if line["kind"] == "observation"
    ]
    assert observations
    cube = observations[-1]["data"]["scene"]["objects"]["pick_cube"]
    assert len(cube["xyz_m"]) == 3


# ------------------------------------------------------------------ events
def test_normal_mode_missed_event_says_how_to_move():
    twin = _twin()
    _close_above_cube(twin)
    missed = [e for e in twin.observe()["control"]["events"] if "missed" in e]
    assert missed and "move_by(" in missed[-1]


def test_hidden_mode_missed_event_gives_no_offsets(tmp_path):
    twin = _twin(hide_positions=True, log_path=tmp_path / "s.jsonl")
    _close_above_cube(twin)
    events = twin.observe()["control"]["events"]
    missed = [e for e in events if "missed" in e]
    assert missed, events
    assert "move_by(" not in missed[-1] and "cm away" not in missed[-1]
    assert "camera" in missed[-1]  # tells the agent to look again
    twin.close()
    # The session log still has the full hint.
    logged = [
        line["data"]["text"]
        for line in read_session_log(tmp_path / "s.jsonl")
        if line["kind"] == "event" and "missed" in line["data"].get("text", "")
    ]
    assert logged and "move_by(" in logged[-1]


# --------------------------------------------------------------- estimates
def test_estimate_is_scored_against_the_truth(tmp_path):
    twin = _twin(hide_positions=True, log_path=tmp_path / "s.jsonl")
    truth = twin.grasp.position("pick_cube").copy()
    guess = truth + np.array([0.03, -0.04, 0.0])  # 5 cm off
    result = _resolve(twin, twin.report_estimate("pick_cube", guess.tolist(), client_id="a1"))
    # The agent learns its guess was recorded, not how far off it was.
    assert result == {"ok": True, "name": "pick_cube", "recorded": 1}

    scores = twin.estimate_scores()
    assert len(scores) == 1
    score = scores[0]
    assert score["name"] == "pick_cube" and score["client_id"] == "a1"
    assert score["error_cm"] == pytest.approx(5.0, abs=0.05)
    assert score["error_xy_cm"] == pytest.approx(5.0, abs=0.05)
    assert score["error_z_cm"] == pytest.approx(0.0, abs=0.05)
    assert score["truth_xyz_m"] == pytest.approx(truth.tolist(), abs=1e-4)
    assert score["estimate_xyz_m"] == pytest.approx(guess.tolist(), abs=1e-6)

    twin.close()
    logged = [line for line in read_session_log(tmp_path / "s.jsonl") if line["kind"] == "estimate"]
    assert len(logged) == 1
    assert logged[0]["data"]["error_cm"] == pytest.approx(5.0, abs=0.05)
    assert logged[0]["client_id"] == "a1"


def test_estimate_of_unknown_object_is_rejected():
    twin = _twin(hide_positions=True)
    with pytest.raises(ActionRejected) as info:
        _resolve(twin, twin.report_estimate("moon", [0.3, 0.0, 0.14]))
    assert "pick_cube" in info.value.detail
    with pytest.raises(ActionRejected):
        _resolve(twin, twin.report_estimate("pick_cube", [0.3, 0.0]))
    assert twin.estimate_scores() == []


def test_parked_objects_cannot_be_estimated():
    twin = _twin(hide_positions=True)
    # bean_bag belongs to the other layout and is parked out of sight.
    with pytest.raises(ActionRejected):
        _resolve(twin, twin.report_estimate("bean_bag", [0.3, 0.0, 0.14]))


# --------------------------------------------------------------- over MCP
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def hidden_url():
    service = build_twin(scene="pick_place_basic", with_controller=False, hide_positions=True)
    running = threading.Event()
    running.set()

    def loop():
        while running.is_set():
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
    running.clear()
    thread.join(timeout=5)


def _calls(url, calls):
    async def go():
        async with streamable_http_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return [await session.call_tool(tool, args) for tool, args in calls]

    return asyncio.run(go())


def _payload(result) -> dict:
    assert not result.is_error, result
    return json.loads(result.content[0].text)


def test_mcp_hides_positions_from_observe_and_motion_results(hidden_url):
    url, service = hidden_url
    observed, moved = _calls(url, [("observe", {}), ("move_to", {"x": 0.3, "y": 0.05, "z": 0.22})])
    observation = _payload(observed)
    assert observation["scene"]["object_positions"] == "hidden"
    for entry in observation["scene"]["objects"].values():
        assert set(entry) == HIDDEN_OBJECT_KEYS
    result = _payload(moved)
    assert result["ok"] and "xyz_m" not in json.dumps(result.get("events", []))
    truth = [round(float(v), 3) for v in service.grasp.position("pick_cube")]
    assert json.dumps(truth[:2])[1:-1] not in json.dumps(result)


def test_mcp_report_estimate_records_without_revealing_the_error(hidden_url):
    url, service = hidden_url
    before = len(service.estimate_scores())
    truth = service.grasp.position("pick_cube")
    (reported,) = _calls(
        url,
        [
            (
                "report_estimate",
                {
                    "name": "pick_cube",
                    "x": float(truth[0]) + 0.01,
                    "y": float(truth[1]),
                    "z": float(truth[2]),
                },
            )
        ],
    )
    result = _payload(reported)
    assert result["ok"] is True and result["name"] == "pick_cube"
    assert "error" not in json.dumps(result)
    scores = service.estimate_scores()
    assert len(scores) == before + 1
    assert scores[-1]["error_cm"] == pytest.approx(1.0, abs=0.1)

    (bad,) = _calls(url, [("report_estimate", {"name": "moon", "x": 0.3, "y": 0.0, "z": 0.14})])
    refused = _payload(bad)
    assert refused["ok"] is False and refused["reason"] == "invalid_request"


def test_mcp_instructions_describe_the_mode(hidden_url):
    from xbox_soarm_teleop.twin import hub_server, mcp_server

    for text in (mcp_server.INSTRUCTIONS, hub_server.INSTRUCTIONS):
        assert "object_positions" in text and "report_estimate" in text
        assert "locate(name)" in text


# ---------------------------------------------------------------- command line
def test_twin_and_hub_commands_take_the_flag():
    from xbox_soarm_teleop.cli import hub, twin

    assert twin.build_parser().parse_args(["--no-coordinates"]).no_coordinates is True
    assert twin.build_parser().parse_args([]).no_coordinates is False
    assert hub.build_parser().parse_args(["--no-coordinates"]).no_coordinates is True
    assert hub.build_parser().parse_args([]).no_coordinates is False


def test_hub_rooms_hide_positions(tmp_path):
    from xbox_soarm_teleop.twin.hub import Hub

    hub = Hub(seats=1, roster_path=tmp_path / "roster.json", hide_positions=True)
    try:
        room = next(iter(hub.rooms.values()))
        assert room.service.observe()["scene"]["object_positions"] == "hidden"
    finally:
        hub.close()
