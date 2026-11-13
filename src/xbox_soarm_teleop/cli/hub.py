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
    parser.add_argument(
        "--mesh-detail-mm",
        type=float,
        default=2.0,
        help="Draw camera pictures from meshes simplified this much; 0 draws the originals "
        "(slower). The arms are always simulated with the originals",
    )
    parser.add_argument(
        "--picture-rate",
        type=float,
        default=10.0,
        help="Camera pictures a second each student page asks for (1 to 30). "
        "Lower it if your upload speed is limited: 10 is about 1.3 Mbit/s per student",
    )
    parser.add_argument(
        "--no-open",
        dest="open_browser",
        action="store_false",
        help="Do not open the instructor page in your browser when the hub starts",
    )
    parser.add_argument(
        "--no-find-tunnel",
        dest="find_tunnel",
        action="store_false",
        help="Without --public-url, do not look for a cloudflared quick tunnel running on this "
        "computer to use as the students' address",
    )
    parser.add_argument(
        "--show-links",
        action="store_true",
        help="Print the instructor link, key included, in the terminal. Off by default so a "
        "shared screen does not show it",
    )
    return parser


def startup_lines(
    base: str, admin_token: str, roster: str | Path, *, public: bool, opened: bool, show: bool = False
) -> list[str]:
    """What the hub prints once it is ready. Keys are left out unless ``show``."""
    lines = []
    if show:
        lines.append(f"Instructor page (keep private): {base}/admin/{admin_token}")
    elif opened:
        lines.append("Instructor page: opened in your browser. Keys are not printed here.")
    else:
        lines.append("Instructor page: its link is in the roster file. Keys are not printed here;")
        lines.append("  run again with --show-links to print it, or without --no-open to open it.")
    lines.append("  Student links are on that page (Copy all student links)")
    lines.append(f"  and in {Path(roster)}; they stay the same after a restart.")
    if not public:
        lines.append("  Only this computer can open these links. For students, see --public-url.")
    lines.append(f"Agents connect to {base}/mcp with their arm's key. Press Ctrl+C to quit.")
    return lines


QUICK_TUNNEL_PORTS = range(20241, 20246)  # where cloudflared serves its own status


def find_quick_tunnel(ports=QUICK_TUNNEL_PORTS, timeout_s: float = 0.4) -> str | None:
    """The address of a cloudflared quick tunnel running on this computer, if any.

    ``cloudflared tunnel --url ...`` reports its public host name on a local
    status port. It does not say which local port it forwards to, so the
    caller must tell the user what it found.
    """
    import json
    import urllib.request

    for port in ports:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/quicktunnel", timeout=timeout_s
            ) as reply:
                name = json.loads(reply.read().decode("utf-8")).get("hostname")
        except (OSError, ValueError):
            continue
        if isinstance(name, str) and name and "/" not in name:
            return f"https://{name}"
    return None


def _open_when_ready(url: str, host: str, port: int) -> None:
    """Open ``url`` in the default browser once the server accepts connections."""
    import socket
    import threading
    import time
    import webbrowser

    def wait_then_open() -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            return
        webbrowser.open(url)

    threading.Thread(target=wait_then_open, name="open-instructor-page", daemon=True).start()


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

    if args.public_url is None and args.find_tunnel and args.host in ("127.0.0.1", "localhost"):
        args.public_url = find_quick_tunnel()
        if args.public_url:
            print(f"Found a cloudflared quick tunnel: {args.public_url}", flush=True)
            print(
                f"  Using it as the address students use. It must forward to port {args.port}: "
                f"cloudflared tunnel --url http://localhost:{args.port}",
                flush=True,
            )
            print("  Pass --public-url to use another address, or --no-find-tunnel.", flush=True)

    print(f"Building {args.seats} arms...", flush=True)
    hub = Hub(
        seats=args.seats,
        scene=args.scene,
        log_dir=None if args.no_log else args.log_dir,
        roster_path=args.roster,
        mesh_cell_mm=max(args.mesh_detail_mm, 0.0),
    )
    app = build_hub_app(
        hub,
        host=args.host,
        port=args.port,
        public_url=args.public_url,
        notebook_url=args.notebook_url or None,
        picture_hz=args.picture_rate,
    )
    base = (args.public_url or f"http://{args.host}:{args.port}").rstrip("/")
    for line in startup_lines(
        base,
        hub.admin_token,
        args.roster,
        public=args.public_url is not None,
        opened=args.open_browser,
        show=args.show_links,
    ):
        print(line, flush=True)
    if args.open_browser:
        # Opened on this computer's own address: it works before the tunnel does.
        local_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        local = f"http://{local_host}:{args.port}/admin/{hub.admin_token}"
        _open_when_ready(local, local_host, args.port)

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
