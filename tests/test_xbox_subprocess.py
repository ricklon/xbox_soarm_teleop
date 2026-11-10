"""Out-of-process Xbox backend: needed when mjpython owns the macOS main thread."""

import subprocess
import sys
import textwrap
import time

import pytest

from xbox_soarm_teleop.config.xbox_config import XboxConfig
from xbox_soarm_teleop.teleoperators import xbox as xbox_mod
from xbox_soarm_teleop.teleoperators.xbox import XboxController

FAKE_HELPER = textwrap.dedent(
    """
    import json, sys, time
    mode = sys.argv[1]
    state = {"left_bumper": True, "left_stick_x": 0.5, "a_button": False}

    def emit(seq, connected=True, **extra):
        print(json.dumps({"seq": seq, "connected": connected, "state": {**state, **extra}}), flush=True)

    emit(0)
    if mode == "stall":
        time.sleep(30)
    elif mode == "exit":
        sys.exit(0)
    elif mode == "disconnect":
        emit(1, connected=False)
        time.sleep(30)
    else:
        seq = 1
        while True:
            emit(seq, a_button=seq % 2 == 1)
            seq += 1
            time.sleep(0.01)
    """
)


def _controller(monkeypatch, tmp_path, mode: str) -> XboxController:
    script = tmp_path / "fake_helper.py"
    script.write_text(FAKE_HELPER)
    monkeypatch.setattr(
        XboxController, "_helper_command", lambda self: [sys.executable, str(script), mode]
    )
    controller = XboxController(XboxConfig(backend="subprocess"))
    assert controller.connect()
    return controller


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_streamed_state_passes_through(monkeypatch, tmp_path):
    controller = _controller(monkeypatch, tmp_path, "stream")
    try:
        assert _wait_for(lambda: controller.read().left_bumper)
        state = controller.read()
        assert state.left_stick_x == pytest.approx(0.5)
        assert controller.is_connected
        # Edge detection is computed in the parent from level states.
        assert _wait_for(lambda: controller.read().a_button_pressed)
    finally:
        controller.disconnect()


@pytest.mark.parametrize("mode", ["stall", "exit", "disconnect"])
def test_deadman_releases_when_helper_stops_reporting(monkeypatch, tmp_path, mode):
    controller = _controller(monkeypatch, tmp_path, mode)
    try:
        time.sleep(xbox_mod.SUBPROCESS_STALE_S + 0.1)
        state = controller.read()
        assert not state.left_bumper
        assert state.left_stick_x == 0.0
        assert not controller.is_connected
    finally:
        controller.disconnect()


def test_disconnect_stops_helper(monkeypatch, tmp_path):
    controller = _controller(monkeypatch, tmp_path, "stream")
    process = controller._helper
    controller.disconnect()
    assert process.wait(timeout=2) is not None


def test_real_helper_exits_when_parent_closes_stdin():
    """The helper must never outlive the twin and hold the controller."""
    process = subprocess.Popen(
        [sys.executable, "-m", "xbox_soarm_teleop.teleoperators.xbox_helper", "--backend", "none"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline()  # first heartbeat
        process.stdin.close()
        assert process.wait(timeout=3) == 0
    finally:
        process.kill()


def test_auto_selects_subprocess_under_mjpython(monkeypatch):
    monkeypatch.setenv("XBOX_SOARM_MJPYTHON", "1")
    assert XboxController(XboxConfig())._resolve_backend() == "subprocess"
    monkeypatch.delenv("XBOX_SOARM_MJPYTHON")
    assert XboxController(XboxConfig())._resolve_backend() != "subprocess"
