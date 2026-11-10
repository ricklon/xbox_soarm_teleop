"""soarm-twin: run the digital twin for the Xbox controller and AI agents.

One process runs everything, on its main thread: the control loop, the MuJoCo
viewer and offscreen cameras. The Xbox pad is read by a helper process on
macOS (the viewer owns the OS main thread there), and the MCP server listens
on localhost in a background thread.

    uv run soarm-twin                      # pick-and-place scene, viewer, Xbox, MCP
    uv run soarm-twin --scene stack
    uv run soarm-twin --headless --no-controller   # agent only

Each run is recorded to twin_logs/twin-*.jsonl (see docs/schemas.md); --no-log
turns that off.

Connect Claude Code:  claude mcp add --transport http soarm-twin http://127.0.0.1:8765/mcp
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
from pathlib import Path

SCENES = ("pick_place_basic", "stack", "default")


def _wait_for_port(host: str, port: int, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.05)
    return False


def _port_free(host: str, port: int) -> bool:
    with socket.socket() as sock:
        return sock.connect_ex((host, port)) != 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soarm-twin", description="SO-ARM101 digital twin for Xbox and AI agents."
    )
    parser.add_argument("--scene", choices=SCENES, default="pick_place_basic")
    parser.add_argument("--world", default=None, help="Lighting/floor preset (see simulate-mujoco)")
    parser.add_argument("--port", type=int, default=8765, help="MCP server port (localhost)")
    parser.add_argument("--no-mcp", action="store_true", help="Do not start the MCP server")
    parser.add_argument("--no-controller", action="store_true", help="No Xbox controller")
    parser.add_argument("--controller-backend", default="auto", help="auto, sdl, inputs, subprocess")
    parser.add_argument("--deadzone", type=float, default=0.15)
    parser.add_argument("--headless", action="store_true", help="No viewer window")
    parser.add_argument(
        "--log-dir", default="twin_logs", help="Directory for the session JSONL log (one per run)"
    )
    parser.add_argument("--no-log", action="store_true", help="Do not write a session log")
    parser.add_argument(
        "--max-seconds", type=float, default=None, help="Exit after this long (automation)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from xbox_soarm_teleop.cli.simulate_mujoco import _HeadlessViewer, _maybe_reexec_under_mjpython

    if argv is None:
        _maybe_reexec_under_mjpython(args, module="xbox_soarm_teleop.cli.twin")

    import mujoco.viewer

    from xbox_soarm_teleop.twin import build_twin
    from xbox_soarm_teleop.twin.mcp_server import McpThread, build_mcp_server
    from xbox_soarm_teleop.twin.status import MENU_BUTTON

    host = "127.0.0.1"
    if not args.no_mcp and not _port_free(host, args.port):
        print(f"ERROR: port {args.port} is in use (another twin running?). Use --port.", flush=True)
        return 2

    log_path = None
    if not args.no_log:
        log_path = Path(args.log_dir) / f"twin-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.jsonl"
    service = build_twin(
        log_path=log_path,
        scene=args.scene,
        world=args.world,
        controller_backend=args.controller_backend,
        deadzone=args.deadzone,
        with_controller=not args.no_controller,
    )
    if service.log is not None:
        print(f"Session log: {service.log.path}", flush=True)
    controller = service.controller
    if controller is not None:
        controller.connect()
        if not controller.is_connected:
            print(
                "Xbox controller: not found yet. Turn it on (Xbox button) or plug it in; "
                "it is picked up while the twin runs.",
                flush=True,
            )
        print(
            "  Hold LB + sticks: move | RT: gripper | A: home | Y: reset scene | "
            "B: STOP | Menu: release stop",
            flush=True,
        )
        print(f"  Nothing moves while the stop is latched: press the {MENU_BUTTON}.", flush=True)

    if not args.no_mcp:
        mcp = McpThread(build_mcp_server(service), host=host, port=args.port)
        mcp.start()
        if not _wait_for_port(host, args.port) or mcp.error is not None:
            print(f"ERROR: MCP server failed to start: {mcp.error}", flush=True)
            service.close()
            return 2
        print(f"MCP server: {mcp.url}", flush=True)
        print(f"  Claude Code: claude mcp add --transport http soarm-twin {mcp.url}", flush=True)

    running = True

    def on_signal(_sig, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)
    print(f"Scene: {args.scene}. Close the window or press Ctrl+C to quit.", flush=True)

    shown_events: list[str] = []
    shown_status: tuple[str, str] | None = None
    pad_connected = False
    started = time.monotonic()
    next_tick = started
    viewer_ctx = (
        _HeadlessViewer()
        if args.headless
        else mujoco.viewer.launch_passive(service.sim.model, service.sim.data)
    )
    try:
        with viewer_ctx as viewer:
            while running and viewer.is_running():
                tick_start = time.monotonic()
                service.tick()
                viewer.sync()
                observation = service.observe()
                shown_status = _show_status(viewer, observation, shown_status)
                if controller is not None:
                    now_connected = observation["control"]["controller_connected"]
                    if now_connected != pad_connected:
                        pad_connected = now_connected
                        state = "connected" if now_connected else "disconnected"
                        print(f"Xbox controller: {state}", flush=True)
                events = observation["control"]["events"]
                for event in events[len(events) - _new_count(shown_events, events) :]:
                    print(f"  {event}", flush=True)
                shown_events = list(events)
                if args.max_seconds is not None and tick_start - started >= args.max_seconds:
                    break
                next_tick = _next_deadline(next_tick, time.monotonic(), service.dt)
                time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        service.close()
        service.tick()  # release GL resources on this (the loop) thread
        if controller is not None:
            controller.disconnect()
    print("Twin stopped.", flush=True)
    return 0


def _show_status(viewer, observation: dict, shown: tuple[str, str] | None) -> tuple[str, str]:
    """Draw the status and the controls in the viewer when the status changes."""
    import mujoco

    from xbox_soarm_teleop.twin.status import CONTROLS_HELP, status_text

    status = status_text(observation)
    set_texts = getattr(viewer, "set_texts", None)
    if status != shown and callable(set_texts):
        set_texts(
            [
                (mujoco.mjtFontScale.mjFONTSCALE_200, mujoco.mjtGridPos.mjGRID_TOPLEFT, *status),
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    "Xbox controller",
                    CONTROLS_HELP,
                ),
            ]
        )
    return status


def _next_deadline(deadline: float, now: float, dt: float) -> float:
    """When the next tick is due, on a fixed grid so sleep overshoot cannot add up.

    Sleeping ``dt - work`` every tick runs slow by however much each sleep
    overshoots (about 15% at 50 Hz on macOS). A tick that is less than one
    period late is caught up; a longer stall restarts the grid instead of
    bursting.
    """
    deadline += dt
    return deadline if deadline > now - dt else now


def _new_count(previous: list[str], current: list[str]) -> int:
    """How many entries at the end of ``current`` were not in ``previous``."""
    for overlap in range(min(len(previous), len(current)), -1, -1):
        if previous[len(previous) - overlap :] == current[:overlap]:
            return len(current) - overlap
    return len(current)


if __name__ == "__main__":
    sys.exit(main())
