"""HTTP front of the hub: a web page and an MCP endpoint for every student arm.

One server, three audiences:

- ``/arm`` and ``/instructor``: the same two pages without a key in the
  address. A page opened by its link remembers the key for that browser tab
  and moves here at once, so the key is not left in the address bar where a
  shared screen would show it. Opened directly, they have no key and say so.
- ``/r/<room token>``: the student's page (live camera, hold-to-move
  controls, stop) and the small JSON API it polls. ``/r/<room token>/tools``
  serves the agent tools to the page itself, which registers them with the
  browser (WebMCP) for an agent running in the student's browser.
- ``/mcp``: the MCP endpoint for agents. The room is chosen by the request's
  ``Authorization: Bearer <room token>`` header; requests without a valid
  token are refused with 401 before they reach the MCP server.
- ``/admin/<admin token>``: the instructor's overview of every arm.

Tokens are the only access control, so links must be shared privately. The
server binds to localhost unless told otherwise; to reach students it is put
behind a tunnel or reverse proxy that terminates HTTPS.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from contextvars import ContextVar
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from xbox_soarm_teleop.twin.cameras import MAX_SIZE, VIEWS
from xbox_soarm_teleop.twin.hub import Hub, Room
from xbox_soarm_teleop.twin.mcp_server import _client_id, build_mcp_server
from xbox_soarm_teleop.twin.service import TwinService
from xbox_soarm_teleop.twin.status import status_text

RENDER_TIMEOUT_S = 3.0
MAX_PAD_BYTES = 2048
DEFAULT_PICTURE_HZ = 10.0  # camera pictures a second each student page asks for
PICTURE_HZ_RANGE = (1.0, 30.0)
AGENT_THREADS = 200  # agents block a worker thread while a move finishes
PAGE_AGENT = "browser-agent"  # who the session log says acted, for tools called from the page
PAGE_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}

INSTRUCTIONS = """\
You control a simulated SO-ARM101 robot arm (a digital twin) on a table. It
belongs to one workshop student, who is watching it in a web page.

Frame: robot base frame in meters. +x is forward (away from the base), +y is
the robot's left, +z is up. The table top is at z = 0.12 m. The tool point
(what move_to steers, `ee`) is the inner face of the fixed jaw near its tip,
not the middle of the jaws: `observe` gives that as `jaw_center_xyz_m`.

Start with `observe` to see the arm, the objects and the workspace bounds.
Motion tools wait until the move finishes (pass wait=false to return early)
and report SUCCEEDED, or FAILED with a reason such as workspace_bound,
out_of_reach or joint_limit. Moves outside the workspace are rejected.

Grasping is assisted: an object is held when the jaws close to its width
with it between their pads. To pick one up:
1. move_to 6 cm above it, then set_wrist_roll(angle_rad=-1.5) (or 1.5): at
   roll 0 the jaws close top to bottom, and a rigid object must be gripped by
   its sides.
2. observe: the object's `grip_width_m` is its size across the jaws as they
   are turned now (not always size_m[0]). set_gripper(opening_m=grip_width_m
   + 0.05): room for an object that sits a little turned.
3. Center the jaws over it: move_by(dx, dy, dz=0) by the object's x, y minus
   `jaw_center_xyz_m`; do it twice (the gripper tilts a little as it moves).
4. Go straight down: move_by(dz = object z - jaw center z), but keep the jaw
   center at z 0.14 or higher (lower, the jaw tips hit the table). Moving the
   jaws sideways or at an angle near a rigid object knocks it away.
5. set_gripper(opening_m=grip_width_m). The result's `holding` names it.
Every motion result gives `jaw_center_xyz_m` and the last few events.
Closed a little past the width it still holds; much further, a rigid object is
squeezed out and falls. A refused grasp's event says exactly what to do: the
wrist roll that squares the jaws ("slipped"), or the opening and moves that
center the jaws on it ("missed"). Do those, in order, then close again. A soft
object takes any closure at any roll. set_gripper(position=0) releases. A held
object rides at the jaw center, so to drop it somewhere put
`jaw_center_xyz_m` over that spot.

`observe` also gives the goal box (`scene.goal`: the center of its floor and
its inside size) and, for each object, `in_goal`. Release an object with the
jaw center a few centimeters above the box floor, inside its walls, and check
`in_goal`.

Scenes differ in how forgiving their objects are. `list_scenes` shows them,
easiest first, and `load_scene` switches (it homes the arm and swaps the
objects). A soft object gives way to the gripper and takes any closure; a
rigid one is pushed away if the gripper touches it before it closes.

The student can take over from their page at any time (reason
`preempted`/`human`); wait until they are done. If the stop is latched, motion
is refused until it is reset; a stop the student latched must be released by
the student. Use `camera` to look at the scene.
"""


def _bearer(value: str | None) -> str | None:
    if not value:
        return None
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def _page(name: str) -> str:
    return resources.files("xbox_soarm_teleop.twin").joinpath("web", name).read_text("utf-8")


def room_summary(room: Room) -> dict[str, Any]:
    """What a page needs to show about a room."""
    observation = room.service.observe()
    control = observation["control"]
    return {
        "name": room.name,
        "scene": observation["scene"]["name"],
        "status": list(status_text(observation, web=True)),
        "stop": control["stop_latched_by"] is not None,
        "ee": observation["ee"]["xyz_m"],
        "gripper": observation["gripper"],
        "objects": observation["scene"]["objects"],
        "events": control["events"],
    }


class _Gate:
    """ASGI wrapper: refuse MCP requests that carry no valid room token."""

    def __init__(self, app, hub: Hub, mcp_path: str = "/mcp") -> None:
        self.app = app
        self.hub = hub
        self.mcp_path = mcp_path

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            import anyio.to_thread

            anyio.to_thread.current_default_thread_limiter().total_tokens = AGENT_THREADS
        elif scope["type"] == "http" and scope["path"].rstrip("/") == self.mcp_path:
            headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
            token = _bearer(headers.get("authorization"))
            if token is None or self.hub.room(token) is None:
                response = JSONResponse(
                    {"error": "unauthorized", "detail": "Send 'Authorization: Bearer <your key>'."},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_hub_app(
    hub: Hub,
    *,
    host: str = "127.0.0.1",
    port: int = 8800,
    public_url: str | None = None,
    notebook_url: str | None = None,
    picture_hz: float = DEFAULT_PICTURE_HZ,
):
    """Build the ASGI app serving ``hub``.

    Args:
        hub: The hub (its loop must be started by the caller).
        host: Address the server will bind to.
        port: Port the server will bind to.
        public_url: The address students use (a tunnel or proxy in front of
            this server). Its host is accepted in addition to localhost.
        notebook_url: Link to the workshop notebook, shown on each page.
        picture_hz: Camera pictures a second each student page asks for, at
            most (kept within 1 to 30). Each picture is about 17 kB, so 10 a
            second is roughly 1.3 Mbit/s of upload per student.
    """
    picture_ms = round(1000 / min(max(picture_hz, PICTURE_HZ_RANGE[0]), PICTURE_HZ_RANGE[1]))

    page_room: ContextVar[Room | None] = ContextVar("page_room", default=None)

    def twin_for(ctx: Context | None) -> TwinService:
        from_page = page_room.get()
        if from_page is not None:  # called by the room's own page, already checked by its link
            return from_page.service
        headers = (ctx.headers if ctx is not None else None) or {}
        room = hub.room(_bearer(headers.get("authorization")) or "")
        if room is None:
            raise PermissionError("unknown room key")
        return room.service

    def client_for(ctx: Context | None) -> str:
        return PAGE_AGENT if page_room.get() is not None else _client_id(ctx)

    server = build_mcp_server(
        twin_for, name="soarm-arm", instructions=INSTRUCTIONS, client_id=client_for
    )

    def room_of(request: Request) -> Room | None:
        return hub.room(request.path_params["token"])

    not_found = Response("Not found", status_code=404, headers=PAGE_HEADERS)

    @server.custom_route("/", methods=["GET"])
    async def index(_request: Request) -> Response:
        text = "SO-ARM workshop. Open the link your instructor sent you."
        return Response(text, media_type="text/plain", headers=PAGE_HEADERS)

    instance_id = secrets.token_hex(6)  # not a key: tells one running hub from another

    @server.custom_route("/ping", methods=["GET"])
    async def ping(_request: Request) -> Response:
        return JSONResponse({"hub": instance_id}, headers=PAGE_HEADERS)

    @server.custom_route("/arm", methods=["GET"])
    async def room_page_without_key(_request: Request) -> Response:
        return HTMLResponse(_page("room.html"), headers=PAGE_HEADERS)

    @server.custom_route("/instructor", methods=["GET"])
    async def admin_page_without_key(_request: Request) -> Response:
        return HTMLResponse(_page("admin.html"), headers=PAGE_HEADERS)

    @server.custom_route("/r/{token}", methods=["GET"])
    async def room_page(request: Request) -> Response:
        if room_of(request) is None:
            return not_found
        return HTMLResponse(_page("room.html"), headers=PAGE_HEADERS)

    @server.custom_route("/r/{token}/pad", methods=["POST"])
    async def pad(request: Request) -> Response:
        room = room_of(request)
        if room is None:
            return not_found
        body = await request.body()
        try:
            wanted = json.loads(body) if len(body) <= MAX_PAD_BYTES else None
        except ValueError:
            wanted = None
        if not isinstance(wanted, dict):
            return JSONResponse({"error": "expected a small JSON object"}, status_code=400)
        room.controller.update(wanted)
        summary = room_summary(room)
        summary["notebook_url"] = notebook_url
        summary["picture_ms"] = picture_ms
        return JSONResponse(summary, headers=PAGE_HEADERS)

    @server.custom_route("/r/{token}/observe", methods=["GET"])
    async def observe(request: Request) -> Response:
        room = room_of(request)
        if room is None:
            return not_found
        return JSONResponse(room.service.observe(), headers=PAGE_HEADERS)

    @server.custom_route("/r/{token}/camera.jpg", methods=["GET"])
    async def camera(request: Request) -> Response:
        room = room_of(request)
        if room is None:
            return not_found
        view = request.query_params.get("view", "overhead")
        try:
            width = min(max(int(request.query_params.get("w", 480)), 16), MAX_SIZE[0])
            height = min(max(int(request.query_params.get("h", 360)), 16), MAX_SIZE[1])
        except ValueError:
            return JSONResponse({"error": "w and h must be whole numbers"}, status_code=400)
        if view not in VIEWS:
            return JSONResponse({"error": f"view must be one of {', '.join(VIEWS)}"}, status_code=400)
        future = room.service.render(view, width=width, height=height, client_id="web", log=False)
        try:
            image = await asyncio.wait_for(asyncio.wrap_future(future), RENDER_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - busy or shutting down; the page just retries
            return Response("camera busy", status_code=503, headers=PAGE_HEADERS)
        return Response(image, media_type="image/jpeg", headers=PAGE_HEADERS)

    @server.custom_route("/r/{token}/tools", methods=["GET"])
    async def page_tools(request: Request) -> Response:
        if room_of(request) is None:
            return not_found
        tools = [
            {"name": tool.name, "description": tool.description, "inputSchema": tool.input_schema}
            for tool in await server.list_tools()
        ]
        return JSONResponse({"tools": tools, "instructions": INSTRUCTIONS}, headers=PAGE_HEADERS)

    @server.custom_route("/r/{token}/tools/{name}", methods=["POST"])
    async def page_tool(request: Request) -> Response:
        room = room_of(request)
        name = request.path_params["name"]
        if room is None or name not in {tool.name for tool in await server.list_tools()}:
            return not_found
        body = await request.body()
        try:
            arguments = json.loads(body or b"{}") if len(body) <= MAX_PAD_BYTES else None
        except ValueError:
            arguments = None
        if not isinstance(arguments, dict):
            return JSONResponse({"error": "expected a small JSON object"}, status_code=400)
        chosen = page_room.set(room)
        try:
            result = await server.call_tool(name, arguments)
        except ToolError as exc:  # arguments that do not fit the tool
            return JSONResponse({"error": str(exc)}, status_code=400, headers=PAGE_HEADERS)
        finally:
            page_room.reset(chosen)
        content = [
            block.model_dump(mode="json", by_alias=True, exclude_none=True) for block in result.content
        ]
        return JSONResponse({"content": content}, headers=PAGE_HEADERS)

    def is_admin(request: Request) -> bool:
        return request.path_params["token"] == hub.admin_token

    @server.custom_route("/admin/{token}", methods=["GET"])
    async def admin_page(request: Request) -> Response:
        if not is_admin(request):
            return not_found
        return HTMLResponse(_page("admin.html"), headers=PAGE_HEADERS)

    @server.custom_route("/admin/{token}/rooms", methods=["GET"])
    async def admin_rooms(request: Request) -> Response:
        if not is_admin(request):
            return not_found
        rooms = []
        for room in hub.rooms.values():
            summary = room_summary(room)
            seen = status_text(room.service.observe(), web=True, instructor=True)
            rooms.append(
                {
                    "name": room.name,
                    "token": room.token,
                    "status": list(seen),
                    "stop": summary["stop"],
                    "page_open": room.controller.is_connected,
                    "agent": room.service.observe()["owner"] is not None,
                    "last_event": summary["events"][-1] if summary["events"] else "",
                }
            )
        listing = {"rooms": rooms, "loop_hz": hub.loop_hz, "public_url": public_url}
        return JSONResponse(listing, headers=PAGE_HEADERS)

    done = lambda: JSONResponse({"ok": True, "rooms": len(hub.rooms)}, headers=PAGE_HEADERS)  # noqa: E731

    @server.custom_route("/admin/{token}/stop_all", methods=["POST"])
    async def stop_all(request: Request) -> Response:
        if not is_admin(request):
            return not_found
        # The instructor's own stop: students and agents cannot release it.
        for room in hub.rooms.values():
            room.service.stop(by="instructor", client_id="instructor")
        return done()

    @server.custom_route("/admin/{token}/release_all", methods=["POST"])
    async def release_all(request: Request) -> Response:
        if not is_admin(request):
            return not_found
        for room in hub.rooms.values():
            room.service.release_stop(by="instructor")
        return done()

    hosts = [f"127.0.0.1:{port}", f"localhost:{port}", f"{host}:{port}"]
    origins = [f"http://127.0.0.1:{port}", f"http://localhost:{port}"]
    if public_url:
        parts = urlsplit(public_url)
        hosts.append(parts.netloc)
        origins.append(f"{parts.scheme}://{parts.netloc}")
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
    )
    app = server.streamable_http_app(transport_security=security, host=host)
    gate = _Gate(app, hub)
    gate.instance_id = instance_id
    return gate
