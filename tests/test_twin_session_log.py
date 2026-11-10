"""Session JSONL log (docs/mcp_control_interface.md §10)."""

import json
import logging
import socket
import warnings

import pytest

from xbox_soarm_teleop.cli import twin as twin_cli
from xbox_soarm_teleop.teleoperators.xbox import XboxState
from xbox_soarm_teleop.twin import ActionRejected, build_twin
from xbox_soarm_teleop.twin.session_log import (
    SESSION_LOG_SCHEMA_ID,
    SessionLog,
    read_session_log,
)

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

LINE_KEYS = {"t", "wall_s", "frame", "kind", "client_id", "action_id", "data"}


def obs(frame: int, dt: float = 0.02, moving: bool = True) -> dict:
    gripper = frame / 1000 if moving else 0.0
    return {"frame": frame, "sim_time_s": frame * dt, "gripper": gripper, "workspace": {"x": [0, 1]}}


def kinds(lines: list[dict]) -> list[str]:
    return [line["kind"] for line in lines]


# ------------------------------------------------------------- the log file
def test_header_is_the_first_line_and_every_line_has_the_same_keys(tmp_path):
    log = SessionLog(tmp_path / "s.jsonl", {"scene": "pick_place_basic"}, dt=0.02)
    log.event("stop", t=0.1, frame=5, client_id="c1", by="agent")
    log.close()
    lines = read_session_log(log.path)
    assert kinds(lines) == ["header", "stop"]
    assert all(set(line) == LINE_KEYS for line in lines)
    assert lines[0]["data"]["schema"] == SESSION_LOG_SCHEMA_ID
    assert lines[0]["data"]["scene"] == "pick_place_basic"
    assert lines[1]["client_id"] == "c1" and lines[1]["data"] == {"by": "agent"}


def test_observations_are_downsampled_to_10_hz(tmp_path):
    log = SessionLog(tmp_path / "s.jsonl", {}, dt=0.02)
    for frame in range(100):  # 2 s at 50 Hz
        log.observation(obs(frame))
    log.close()
    frames = [line["frame"] for line in read_session_log(log.path) if line["kind"] == "observation"]
    assert frames == list(range(0, 100, 5))


def test_an_idle_arm_is_logged_at_1_hz(tmp_path):
    log = SessionLog(tmp_path / "s.jsonl", {}, dt=0.02)
    for frame in range(120):
        log.observation(obs(frame, moving=frame >= 100))
    log.close()
    frames = [line["frame"] for line in read_session_log(log.path) if line["kind"] == "observation"]
    assert frames == [0, 50, 100, 105, 110, 115]


def test_observations_run_at_full_rate_around_a_transition(tmp_path):
    log = SessionLog(tmp_path / "s.jsonl", {}, dt=0.02, full_rate_s=0.2)
    for frame in range(13):
        log.observation(obs(frame))
    log.event("stop", t=0.24, frame=12)
    for frame in range(13, 40):
        log.observation(obs(frame))
    log.close()
    lines = read_session_log(log.path)[1:]
    frames = [line["frame"] for line in lines if line["kind"] == "observation"]
    # 10 Hz, then the tick just before the stop, then every tick for 0.2 s.
    assert frames == [0, 5, 10, 12, *range(13, 23), 25, 30, 35]
    assert kinds(lines)[3:5] == ["observation", "stop"]


def test_logged_observation_omits_the_workspace_from_the_header(tmp_path):
    log = SessionLog(tmp_path / "s.jsonl", {}, dt=0.02)
    log.observation(obs(0))
    log.close()
    data = read_session_log(log.path)[1]["data"]
    assert "workspace" not in data and data["gripper"] == 0.0


def test_write_failure_disables_the_log_instead_of_raising(tmp_path):
    log = SessionLog(tmp_path / "s.jsonl", {}, dt=0.02)
    log._file.close()  # as if the disk went away
    log.event("stop", t=0.0, frame=0)
    log.observation(obs(0))
    assert log.error is not None
    log.close()


# ---------------------------------------------------------- twin service
@pytest.fixture
def twin(tmp_path):
    service = build_twin(
        scene="pick_place_basic", with_controller=False, log_path=tmp_path / "session.jsonl"
    )
    service.tick()
    yield service
    service.close()


def lines_of(twin) -> list[dict]:
    twin.log.flush()
    return read_session_log(twin.log.path)


def move(twin, xyz, client_id="agent"):
    return twin.submit({"type": "move_ee_to", "params": {"xyz_m": list(xyz)}}, client_id=client_id)


def run_until_done(twin, action_id, max_ticks=1500):
    for _ in range(max_ticks):
        if twin.wait(action_id, 0).state.terminal:
            break
        twin.tick()


def test_header_records_what_the_run_was(twin):
    header = lines_of(twin)[0]["data"]
    assert header["schemas"] == {
        "session_log": "soarm.session_log/v1",
        "observation": "soarm.observation/v1",
        "action": "soarm.action/v1",
    }
    assert header["scene"] == "pick_place_basic" and header["world"]
    assert header["backend"] == "mujoco"
    assert len(header["urdf_sha256"]) == 64
    assert header["limits"]["workspace"] == {k: list(v) for k, v in twin.workspace_bounds.items()}
    assert header["limits"]["dt_s"] == twin.dt
    assert "commit" in header["git"] and "calibration" in header


def test_milestone_exercise_is_in_the_log(twin):
    # move -> SUCCEEDED
    future = move(twin, [0.30, 0.05, 0.22])
    twin.tick()
    first = future.result(timeout=0)
    run_until_done(twin, first.id)
    # below the floor -> rejected
    future = move(twin, [0.30, 0.0, 0.0])
    twin.tick()
    with pytest.raises(ActionRejected):
        future.result(timeout=0)
    # stop mid-motion, rejected while latched, reset, resume
    future = move(twin, [0.40, -0.10, 0.15])
    twin.tick()
    second = future.result(timeout=0)
    for _ in range(10):
        twin.tick()
    twin.stop(by="agent", client_id="agent")
    twin.tick()
    future = move(twin, [0.3, 0.0, 0.2])
    twin.tick()
    twin.reset_stop(confirm=True)
    twin.tick()

    lines = lines_of(twin)

    def find(kind, **data):
        return [
            line
            for line in lines
            if line["kind"] == kind and all(line["data"].get(k) == v for k, v in data.items())
        ]

    requests = find("request", type="submit")
    assert len(requests) == 4 and requests[0]["client_id"] == "agent"
    assert requests[0]["data"]["request"]["params"]["xyz_m"] == [0.30, 0.05, 0.22]

    ok = find("response", type="submit", ok=True)
    assert [line["action_id"] for line in ok] == [first.id, second.id]
    refused = find("response", type="submit", ok=False)
    assert [line["data"]["reason"] for line in refused] == ["workspace_bound", "stopped"]
    assert refused[0]["data"]["detail"].startswith("z_min")

    states = [(line["action_id"], line["data"]["state"]) for line in find("status")]
    assert states == [
        (first.id, "EXECUTING"),
        (first.id, "SUCCEEDED"),
        (second.id, "EXECUTING"),
        (second.id, "STOPPED"),
    ]
    assert find("stop")[0]["data"]["by"] == "agent" and find("stop")[0]["client_id"] == "agent"
    assert find("stop_reset")[0]["client_id"] == "agent"
    assert find("ownership")[0]["data"]["owner"] == "agent"
    # every line is ordered in time
    assert [line["frame"] for line in lines] == sorted(line["frame"] for line in lines)


def test_human_override_and_handback_are_logged(twin):
    future = move(twin, [0.35, 0.0, 0.2])
    twin.tick()
    action = future.result(timeout=0)
    twin.tick(XboxState(left_bumper=True))
    for _ in range(120):  # controller idle > 2 s
        twin.tick(XboxState())
    lines = lines_of(twin)
    status = [line for line in lines if line["kind"] == "status" and line["action_id"] == action.id]
    assert status[-1]["data"]["state"] == "CANCELLED" and status[-1]["data"]["detail"] == "human"
    human = [line["data"]["human_in_control"] for line in lines if line["kind"] == "ownership"]
    assert human == [False, True, False]


def test_controller_stop_and_grasp_events_are_logged(twin):
    twin.tick(XboxState(b_button=True, b_button_pressed=True))
    twin.tick(XboxState(start_button=True, start_button_pressed=True))
    cube = twin.observe()["scene"]["objects"]["pick_cube"]["xyz_m"]
    future = move(twin, [cube[0], cube[1], cube[2] + 0.01])
    twin.tick()
    run_until_done(twin, future.result(timeout=0).id)
    future = twin.submit({"type": "set_gripper", "params": {"position": 1.0}})
    twin.tick()
    run_until_done(twin, future.result(timeout=0).id)
    lines = lines_of(twin)
    by_kind = {kind: [line for line in lines if line["kind"] == kind] for kind in set(kinds(lines))}
    assert by_kind["stop"][0]["data"]["by"] == "human"
    assert by_kind["stop_reset"][0]["data"]["by"] == "human"
    assert any("pick_cube" in line["data"]["text"] for line in by_kind["event"])


def test_twin_without_a_log_path_writes_nothing():
    service = build_twin(scene="pick_place_basic", with_controller=False)
    assert service.log is None
    service.tick()
    service.close()


def test_close_ends_the_log_with_a_shutdown_line(twin):
    twin.close()
    lines = read_session_log(twin.log.path)
    assert lines[-1]["kind"] == "shutdown"
    assert all(set(line) == LINE_KEYS for line in lines)
    json.dumps(lines)


# ------------------------------------------------------------------- CLI
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_cli_writes_a_session_log_by_default(tmp_path, capsys):
    args = ["--headless", "--no-controller", "--max-seconds", "0.3", "--port", str(_free_port())]
    assert twin_cli.main([*args, "--log-dir", str(tmp_path)]) == 0
    (path,) = tmp_path.glob("twin-*.jsonl")
    assert str(path) in capsys.readouterr().out
    lines = read_session_log(path)
    assert lines[0]["kind"] == "header" and lines[-1]["kind"] == "shutdown"


def test_cli_no_log(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = ["--headless", "--no-controller", "--max-seconds", "0.2", "--port", str(_free_port())]
    assert twin_cli.main([*args, "--no-log"]) == 0
    assert not list(tmp_path.rglob("*.jsonl"))
