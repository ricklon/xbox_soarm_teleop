"""Read an Xbox controller in its own process and stream its state as JSON lines.

On macOS the MuJoCo viewer runs under mjpython, which keeps the OS main
thread for itself. SDL (pygame) must pump events on the main thread, so the
controller cannot be read inside the twin process. This helper owns its own
main thread, reads the controller with the normal in-process backends, and
writes one JSON line per sample to stdout:

    {"seq": 12, "connected": true, "state": {"left_bumper": true, ...}}

It exits when stdin reaches EOF, so it never outlives the twin.
"""

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict, fields

from xbox_soarm_teleop.config.xbox_config import XboxConfig
from xbox_soarm_teleop.teleoperators.xbox import XboxController, XboxState

# Level fields only; the parent derives button edges itself.
STATE_FIELDS = [f.name for f in fields(XboxState) if not f.name.endswith("_pressed")]
RECONNECT_INTERVAL_S = 1.0


def _watch_stdin(stop: threading.Event) -> None:
    """Set ``stop`` when the parent closes our stdin (or dies)."""
    try:
        while sys.stdin.read(1024):
            pass
    except (OSError, ValueError):
        pass
    stop.set()


def _emit(seq: int, connected: bool, state: XboxState | None) -> None:
    payload = {"seq": seq, "connected": connected}
    if state is not None:
        raw = asdict(state)
        payload["state"] = {name: raw[name] for name in STATE_FIELDS}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def run(config: XboxConfig, backend: str, rate_hz: float) -> int:
    """Stream controller state until stdin closes."""
    stop = threading.Event()
    threading.Thread(target=_watch_stdin, args=(stop,), daemon=True).start()
    controller = None
    if backend != "none":
        config.backend = backend
        controller = XboxController(config)
    next_connect = 0.0
    seq = 0
    period = 1.0 / rate_hz
    try:
        while not stop.is_set():
            now = time.monotonic()
            if controller is not None and not controller.is_connected and now >= next_connect:
                controller.connect()
                next_connect = now + RECONNECT_INTERVAL_S
            connected = controller is not None and controller.is_connected
            _emit(seq, connected, controller.read() if connected else None)
            seq += 1
            stop.wait(period)
    except BrokenPipeError:
        pass
    finally:
        if controller is not None:
            controller.disconnect()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="auto", help="auto, sdl, inputs, or none (testing)")
    parser.add_argument("--config", default="{}", help="XboxConfig fields as JSON")
    parser.add_argument("--rate", type=float, default=100.0, help="Samples per second")
    args = parser.parse_args(argv)
    config = XboxConfig(**json.loads(args.config))
    return run(config, args.backend, args.rate)


if __name__ == "__main__":
    sys.exit(main())
