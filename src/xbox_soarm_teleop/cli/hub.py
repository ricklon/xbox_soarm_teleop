"""soarm-hub: host one simulated arm per student for a remote workshop.

Every student gets a secret link to their own arm: a web page to watch and
drive it, and an MCP address for their AI agent. Nothing is installed on the
students' machines.

    uv run soarm-hub --seats 12
    uv run soarm-hub --seats 12 --public-url https://your-tunnel.example.com

The server listens on localhost only. To reach students over the internet,
put a tunnel or reverse proxy that provides HTTPS in front of it and pass its
address as --public-url. See docs/workshop_hosting.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_NOTEBOOK = (
    "https://colab.research.google.com/github/ricklon/xbox_soarm_teleop/"
    "blob/main/notebooks/workshop_agent.ipynb"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soarm-hub", description="Host one simulated SO-ARM101 per workshop student."
    )
    parser.add_argument("--seats", type=int, default=4, help="Number of students (arms)")
    parser.add_argument("--scene", choices=("pick_place_basic", "stack"), default="pick_place_basic")
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument(
        "--public-url",
        default=None,
        help="Address students use (your tunnel or proxy), e.g. https://...",
    )
    parser.add_argument("--log-dir", default="twin_logs", help="Directory for per-arm session logs")
    parser.add_argument("--no-log", action="store_true", help="Do not write session logs")
    parser.add_argument(
        "--roster",
        default="twin_logs/hub_roster.json",
        help="File that keeps the links, so they stay the same after a restart",
    )
    parser.add_argument(
        "--notebook-url", default=DEFAULT_NOTEBOOK, help="Notebook link shown to students"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seats < 1:
        print("ERROR: --seats must be at least 1", flush=True)
        return 2

    import uvicorn

    from xbox_soarm_teleop.cli.twin import _port_free
    from xbox_soarm_teleop.twin.hub import Hub
    from xbox_soarm_teleop.twin.hub_server import build_hub_app

    if not _port_free(args.host, args.port):
        print(f"ERROR: port {args.port} is in use. Use --port.", flush=True)
        return 2

    print(f"Building {args.seats} arms...", flush=True)
    hub = Hub(
        seats=args.seats,
        scene=args.scene,
        log_dir=None if args.no_log else args.log_dir,
        roster_path=args.roster,
    )
    app = build_hub_app(
        hub,
        host=args.host,
        port=args.port,
        public_url=args.public_url,
        notebook_url=args.notebook_url or None,
    )
    base = (args.public_url or f"http://{args.host}:{args.port}").rstrip("/")
    print(f"Instructor page (keep private): {base}/admin/{hub.admin_token}", flush=True)
    print("  Student links are on that page (Copy all student links)", flush=True)
    print(f"  and in {Path(args.roster)}; they stay the same after a restart.", flush=True)
    if args.public_url is None:
        print("  Only this computer can open these links. For students, see --public-url.", flush=True)
    print(f"Agents connect to {base}/mcp with their arm's key. Press Ctrl+C to quit.", flush=True)

    hub.start()
    try:
        # access_log off: request paths contain the room keys.
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
    finally:
        hub.close()
    print("Hub stopped.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
