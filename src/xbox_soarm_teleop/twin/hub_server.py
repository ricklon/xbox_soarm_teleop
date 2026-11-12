"""HTTP front of the hub: a web page and an MCP endpoint for every student arm.

One server, three audiences:

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
AGENT_THREADS = 200  # agents block a worker thread while a move finishes
PAGE_AGENT = "browser-agent"  # who the session log says acted, for tools called from the page
PAGE_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}

INSTRUCTIONS = """\
You control a simulated SO-ARM101 robot arm (a digital twin) on a table. It
belongs to one workshop student, who is watching it in a web page.

Frame: robot base frame in meters. +x is forward (away from the base), +y is
the robot's left, +z is up. The table top is at z = 0.12 m. The tool point is
between the gripper jaws.

Start with `observe` to see the arm, the objects and the workspace bounds.
Motion tools wait until the move finishes (pass wait=false to return early)
and report SUCCEEDED, or FAILED with a reason such as workspace_bound,
out_of_reach or joint_limit. Moves outside the workspace are rejected.

Grasping is assisted: move the tool point to within about 4 cm of an object's
center, then set_gripper(1.0) to grab it; set_gripper(0.0) releases it.

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
):
    """Build the ASGI app serving ``hub``.

    Args:
        hub: The hub (its loop must be started by the caller).
        host: Address the server will bind to.
        port: Port the server will bind to.
        public_url: The address students use (a tunnel or proxy in front of
            this server). Its host is accepted in addition to localhost.
        notebook_url: Link to the workshop notebook, shown on each page.
    """

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
        future = room.service.render(view, width=width, height=height, client_id="web")
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
            rooms.append(
                {
                    "name": room.name,
                    "token": room.token,
                    "status": summary["status"],
                    "stop": summary["stop"],
                    "page_open": room.controller.is_connected,
                    "agent": room.service.observe()["owner"] is not None,
                    "last_event": summary["events"][-1] if summary["events"] else "",
                }
            )
        return JSONResponse({"rooms": rooms, "loop_hz": hub.loop_hz}, headers=PAGE_HEADERS)

    def press_everywhere(button: str) -> Response:
        for room in hub.rooms.values():
            room.controller.press(button)
        return JSONResponse({"ok": True, "rooms": len(hub.rooms)}, headers=PAGE_HEADERS)

    @server.custom_route("/admin/{token}/stop_all", methods=["POST"])
    async def stop_all(request: Request) -> Response:
        return press_everywhere("stop") if is_admin(request) else not_found

    @server.custom_route("/admin/{token}/release_all", methods=["POST"])
    async def release_all(request: Request) -> Response:
        return press_everywhere("release") if is_admin(request) else not_found

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
    return _Gate(app, hub)
