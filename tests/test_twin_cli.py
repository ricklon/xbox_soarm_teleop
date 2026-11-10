"""soarm-twin command: headless run with the MCP server."""

import socket

from xbox_soarm_teleop.cli import twin as twin_cli


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_headless_run_exits_cleanly(capsys):
    port = _free_port()
    code = twin_cli.main(
        ["--headless", "--no-controller", "--max-seconds", "0.5", "--port", str(port), "--no-log"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"http://127.0.0.1:{port}/mcp" in out and "Twin stopped." in out


def test_refuses_a_busy_port(capsys):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert twin_cli.main(["--headless", "--no-controller", "--port", str(port), "--no-log"]) == 2
    assert "in use" in capsys.readouterr().out


def test_new_event_count():
    assert twin_cli._new_count([], ["a"]) == 1
    assert twin_cli._new_count(["a", "b"], ["a", "b", "c"]) == 1
    assert twin_cli._new_count(["a", "b"], ["b", "c", "d"]) == 2  # deque rolled over
    assert twin_cli._new_count(["a"], ["a"]) == 0
