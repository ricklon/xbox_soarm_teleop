"""check_success on a twin that writes a session log, as every agent-facing twin does.

Its own module: a second twin rendering beside test_twin_check_success's
running one would mean two OpenGL contexts on two threads at once.
"""

import logging
import threading
import time
import warnings

from tests.test_twin_check_success import clef_says  # noqa: F401
from xbox_soarm_teleop.twin import build_twin

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)


def test_check_success_works_when_the_twin_keeps_a_log(tmp_path, monkeypatch):
    """The twins agents use write a session log; the check must work there too."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cfTESTTOKEN0123456789abcdef")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "0123456789abcdef0123456789abcdef")
    from xbox_soarm_teleop.twin.session_log import read_session_log

    service = build_twin(
        scene="pick_place_basic",
        with_controller=False,
        hide_positions=True,
        log_path=tmp_path / "s.jsonl",
    )
    alive = threading.Event()
    alive.set()

    def loop():
        while alive.is_set():
            service.tick()
            time.sleep(service.dt)
        service.close()  # on this thread: its GL context belongs here

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        service.judge, _ = clef_says(in_goal=0.03, where="on_table")
        result = service.check_success("pick_cube")
        assert result["ok"] is True and result["verdict"] == "retry"
    finally:
        alive.clear()
        thread.join(timeout=5)
    logged = [
        line for line in read_session_log(tmp_path / "s.jsonl") if line["kind"] == "success_check"
    ]
    assert len(logged) == 1
    assert logged[0]["data"]["truth"]["in_goal"] is False and logged[0]["data"]["correct"] is True
