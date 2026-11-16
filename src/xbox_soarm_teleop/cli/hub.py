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
SORTING_NOTEBOOK = (
    "https://colab.research.google.com/github/ricklon/xbox_soarm_teleop/"
    "blob/main/notebooks/workshop_sorting.ipynb"
)


# Whole sessions in one option; anything given explicitly still wins.
PRESETS: dict[str, dict] = {
    # The sorting desk, as students meet it: no coordinates, so agents must
    # look (locate, with Gemma 4 31B: about 0.02 cents a look), and Clef to
    # check which tray things ended up in (measured: docs/building_scenes.md).
    "sorting": {
        "scene": "sort_colors",
        "no_coordinates": True,
        "pointer": "openrouter:google/gemma-4-31b-it",
        "judge": "clef",
        # Names that say what each cube looks like ("red_cube"): the twin's own
        # ("pick_cube") misled an agent. Neutral or misleading names are for
        # exploring that lesson (--names).
        "names": "truthful",
        "notebook_url": SORTING_NOTEBOOK,
        # Students' feedback goes here once the instructor posts it (gh login).
        "feedback_repo": "ricklon/xbox_soarm_teleop",
        # Students' own agents may design challenges for their arm (docs/students.md).
        "agent_designs": True,
    },
}


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in the preset's settings wherever the command line left the default."""
    preset = PRESETS.get(getattr(args, "preset", None) or "", {})
    defaults = build_parser().parse_args([])
    for key, value in preset.items():
        if getattr(args, key) == getattr(defaults, key):
            setattr(args, key, value)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soarm-hub", description="Host one simulated SO-ARM101 per workshop student."
    )
    parser.add_argument("--seats", type=int, default=4, help="Number of students (arms)")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="A whole session in one option. sorting: the sorting desk, no coordinates, "
        "locate with Gemma 4 31B (OPENROUTER_API_KEY), check_success with Clef "
        "(CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID). Options you give still win",
    )
    parser.add_argument(
        "--agent-models",
        default="google/gemma-4-31b-it,z-ai/glm-5.3-flash,google/gemini-3.5-flash",
        help="OpenRouter models the page's agent panel may use, comma-separated; empty turns "
        "the panel off. It runs on this computer's OPENROUTER_API_KEY",
    )
    parser.add_argument(
        "--agent-budget",
        type=float,
        default=0.50,
        help="Each arm's budget (USD) for the agent panel's model, for the whole session",
    )
    parser.add_argument(
        "--agent-designs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Give agents tools to check, prove, load and clear challenges on their own arm "
        "(students design challenges with their own agent). On in --preset sorting",
    )
    parser.add_argument(
        "--feedback-repo",
        default=None,
        metavar="OWNER/NAME",
        help="GitHub repository student feedback is posted to, as issues, once the instructor "
        "reviews it on the instructor page (uses this computer's gh login). Default: posting off",
    )
    parser.add_argument(
        "--challenge",
        default=None,
        metavar="FILE",
        help="Start every arm with a challenge file (twin/challenges.py): its desk, its "
        "request on the page, its names for agents. Students can post their own to "
        "/r/{key}/challenge (the sorting notebook shows how)",
    )
    parser.add_argument(
        "--names",
        default="as_is",
        choices=("as_is", "truthful", "neutral", "misleading"),
        help="The names agents are shown for objects: the twin's own, what each looks like "
        "(red_cube), object_1 ..., or another object's colour (twin/naming.py)",
    )
    parser.add_argument(
        "--scene",
        choices=("pick_place_easy", "pick_place_basic", "sort_colors", "stack"),
        default="pick_place_easy",
        help="What every arm starts with. pick_place_easy: a soft bean bag (default). "
        "pick_place_basic: a rigid cube, harder to grasp. Students' agents can switch "
        "between the two with load_scene. sort_colors: three cubes and a bin of each "
        "colour, on a different table; agents switch only between the sort scenes there.",
    )
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
        "--no-coordinates",
        action="store_true",
        help="Do not tell agents where objects are: observe leaves out their positions, so "
        "agents must find them with the cameras (a perception benchmark). The session log "
        "keeps the true positions, and report_estimate scores agents' guesses",
    )
    parser.add_argument(
        "--judge",
        default=None,
        choices=("clef", "clef-flash", "none"),
        help="Success judge behind the check_success tool: clef or clef-flash on Cloudflare "
        "Workers AI (CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID). Default: none",
    )
    parser.add_argument(
        "--pointer",
        default=None,
        help="Pointing model behind the locate tool: oracle (the truth; for tests), "
        "gemini[:model] (GEMINI_API_KEY), molmo[:url] (run scripts/molmo_server.py), "
        "ollama[:model], openrouter[:model] (OPENROUTER_API_KEY). Default: none (no locate)",
    )
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
        lines.append("Instructor page: opening in your browser. Keys are not printed here.")
    else:
        lines.append("Instructor page: its link is in the roster file. Keys are not printed here;")
        lines.append("  run again with --show-links to print it, or without --no-open to open it.")
    lines.append("  To use it at another address or on another device: Copy instructor link on that")
    lines.append("  page, then paste it into the box that address shows.")
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


def hub_answers_at(base: str, instance_id: str, *, attempts: int = 8, pause_s: float = 1.0) -> bool:
    """Whether ``base`` (a tunnel or proxy address) reaches this running hub.

    Asks ``base``/ping for the hub's instance id. A new quick tunnel can take
    a few seconds to answer, so it tries a few times.
    """
    import json
    import time
    import urllib.request

    request = urllib.request.Request(f"{base.rstrip('/')}/ping", headers={"User-Agent": "soarm-hub"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=4) as reply:
                if json.loads(reply.read().decode("utf-8")).get("hub") == instance_id:
                    return True
        except (OSError, ValueError):
            pass
        if attempt + 1 < attempts:
            time.sleep(pause_s)
    return False


def _open_when_ready(
    local: str, public: str | None, admin_token: str, instance_id: str, host: str, port: int
) -> None:
    """Open the instructor page in the default browser once the hub is serving.

    At the public address when there is one and it reaches this hub, so the
    instructor is where the students' links are; otherwise at the local one.
    """
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
        base = local
        if public and public.rstrip("/") != local:
            if hub_answers_at(public, instance_id):
                base = public.rstrip("/")
            else:
                print(
                    f"  {public} did not reach this hub. Check the tunnel forwards to port {port}."
                    f"\n  Opening the instructor page on this computer instead.",
                    flush=True,
                )
        print(f"Instructor page: {base}/instructor", flush=True)
        webbrowser.open(f"{base}/admin/{admin_token}")

    threading.Thread(target=wait_then_open, name="open-instructor-page", daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    args = apply_preset(build_parser().parse_args(argv))
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

    challenge = None
    if args.challenge:
        from xbox_soarm_teleop.twin.challenges import ChallengeError, load_challenge

        try:
            challenge = load_challenge(Path(args.challenge))
        except (ChallengeError, OSError) as exc:
            print(f"ERROR: --challenge: {exc}", flush=True)
            return 2
        args.scene = challenge.scene

    print(f"Building {args.seats} arms...", flush=True)
    hub = Hub(
        seats=args.seats,
        scene=args.scene,
        log_dir=None if args.no_log else args.log_dir,
        roster_path=args.roster,
        mesh_cell_mm=max(args.mesh_detail_mm, 0.0),
        hide_positions=args.no_coordinates,
        pointer=args.pointer,
        judge=args.judge,
        naming=args.names,
    )
    app = build_hub_app(
        hub,
        host=args.host,
        port=args.port,
        public_url=args.public_url,
        notebook_url=args.notebook_url or None,
        picture_hz=args.picture_rate,
        agent_models=tuple(m.strip() for m in args.agent_models.split(",") if m.strip()),
        agent_budget_usd=args.agent_budget,
        feedback_repo=args.feedback_repo or None,
        agent_designs=args.agent_designs,
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
        local_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        _open_when_ready(
            f"http://{local_host}:{args.port}",
            args.public_url,
            hub.admin_token,
            app.instance_id,
            local_host,
            args.port,
        )

    if challenge is not None:  # every arm starts with it; students may post their own
        for room in hub.rooms.values():
            before = room.service.layout  # what clearing the challenge puts back
            future = room.service.load_layout(challenge.layout(), client_id="challenge")
            room.service.tick()
            future.result(timeout=0)
            room.service.layout_before = before
            room.naming_before, room.service.naming = room.service.naming, challenge.names
            room.challenge = challenge
        print(f"Every arm starts with the challenge {challenge.name}.", flush=True)

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
