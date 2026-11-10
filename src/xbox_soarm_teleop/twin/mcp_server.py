"""MCP server for the twin: a thin adapter over ``TwinService`` (design D1).

No control logic lives here. Every tool enqueues a command or reads a
snapshot; the twin loop owns physics. Rejections are returned as results
(``{"ok": false, "reason": ..., "detail": ...}``) so an agent can see why.
Clients are identified by their MCP session id.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from mcp.server.mcpserver import Context, Image, MCPServer

from xbox_soarm_teleop.twin.actions import ActionRejected
from xbox_soarm_teleop.twin.service import TwinClosed, TwinService

COMMAND_TIMEOUT_S = 2.0  # the loop resolves commands within one tick
MAX_WAIT_S = 30.0

INSTRUCTIONS = """\
You control a simulated SO-ARM101 robot arm (a digital twin) on a table.

Frame: robot base frame in meters. +x is forward (away from the base), +y is
the robot's left, +z is up. The table top is at z = 0.12 m. The tool point is
between the gripper jaws.

Start with `observe` to see the arm, the objects and the workspace bounds.
Motion tools wait until the move finishes (pass wait=false to return early)
and report SUCCEEDED, or FAILED with a reason such as workspace_bound,
out_of_reach or joint_limit. Moves outside the workspace are rejected.

Grasping is assisted: move the tool point to within about 4 cm of an object's
center, then set_gripper(1.0) to grab it; set_gripper(0.0) releases it.

A human with an Xbox controller can take over at any time (reason
`preempted`/`human`); wait until they are done. If the stop is latched, motion
is refused until it is reset; a stop latched on the controller must be
released there (Menu button, also called Start). Use `camera` to look at the scene.
"""


def _client_id(ctx: Context | None) -> str:
    headers = (ctx.headers if ctx is not None else None) or {}
    return headers.get("mcp-session-id") or "agent"


def _rejection(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ActionRejected):
        return {"ok": False, "reason": exc.reason.value, "detail": exc.detail}
    if isinstance(exc, TwinClosed):
        return {"ok": False, "reason": "twin_stopped", "detail": str(exc)}
    return {"ok": False, "reason": "error", "detail": str(exc)}


def _status_dict(status) -> dict[str, Any]:
    return {
        "id": status.id,
        "type": status.type,
        "state": status.state.value,
        "reason": status.reason.value if status.reason else None,
        "detail": status.detail,
        "progress": status.progress,
    }


def build_mcp_server(service: TwinService) -> MCPServer:
    """Create the MCP server exposing ``service``."""
    server = MCPServer("soarm-twin", instructions=INSTRUCTIONS)

    def run_command(future) -> tuple[bool, Any]:
        try:
            return True, future.result(timeout=COMMAND_TIMEOUT_S)
        except FutureTimeout:
            return False, {"ok": False, "reason": "timeout", "detail": "twin loop not responding"}
        except Exception as exc:  # noqa: BLE001 - reported to the agent
            return False, _rejection(exc)

    def submit(request: dict[str, Any], ctx: Context | None, wait: bool, preempt: bool = False):
        ok, result = run_command(service.submit(request, client_id=_client_id(ctx), preempt=preempt))
        if not ok:
            return result
        status = result
        if wait:
            status = service.wait(status.id, MAX_WAIT_S)
        observation = service.observe()
        return {
            "ok": status.state.value in {"EXECUTING", "HELD", "SUCCEEDED"},
            "action": _status_dict(status),
            "ee_xyz_m": observation["ee"]["xyz_m"],
            "gripper": observation["gripper"],
            "events": observation["control"]["events"][-3:],
        }

    def speed_and_timeout(max_speed_m_s: float | None, timeout_s: float | None) -> dict:
        extra = {}
        if max_speed_m_s is not None:
            extra["max_speed_m_s"] = max_speed_m_s
        if timeout_s is not None:
            extra["timeout_s"] = timeout_s
        return extra

    @server.tool()
    def observe() -> dict[str, Any]:
        """Current state: tool pose, joints (deg), gripper (0 open..1 closed),
        objects and whether one is attached, workspace bounds, the current
        action, stop status and who is in control."""
        return service.observe()

    @server.tool()
    def move_to(
        x: float,
        y: float,
        z: float,
        max_speed_m_s: float | None = None,
        timeout_s: float | None = None,
        wait: bool = True,
        preempt: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Move the tool point to (x, y, z) meters in the base frame, in a
        straight line at up to 0.1 m/s (max_speed_m_s can only lower it).
        preempt=true replaces a running action instead of failing as busy."""
        request = {"type": "move_ee_to", "params": {"xyz_m": [x, y, z]}}
        request.update(speed_and_timeout(max_speed_m_s, timeout_s))
        return submit(request, ctx, wait, preempt)

    @server.tool()
    def move_by(
        dx: float,
        dy: float,
        dz: float,
        max_speed_m_s: float | None = None,
        timeout_s: float | None = None,
        wait: bool = True,
        preempt: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Move the tool point by (dx, dy, dz) meters relative to where it is now."""
        request = {"type": "move_ee_by", "params": {"dxyz_m": [dx, dy, dz]}}
        request.update(speed_and_timeout(max_speed_m_s, timeout_s))
        return submit(request, ctx, wait, preempt)

    @server.tool()
    def set_gripper(position: float, wait: bool = True, ctx: Context | None = None) -> dict[str, Any]:
        """Set the gripper: 0 = open, 1 = closed. Closing near an object grabs it."""
        return submit({"type": "set_gripper", "params": {"position": position}}, ctx, wait)

    @server.tool()
    def set_wrist_roll(
        angle_rad: float, wait: bool = True, ctx: Context | None = None
    ) -> dict[str, Any]:
        """Rotate the wrist to an absolute roll angle in radians (0 = level)."""
        return submit({"type": "set_wrist_roll", "params": {"angle_rad": angle_rad}}, ctx, wait)

    @server.tool()
    def home(wait: bool = True, ctx: Context | None = None) -> dict[str, Any]:
        """Return the arm to its home pose and open the gripper."""
        return submit({"type": "home"}, ctx, wait)

    @server.tool()
    def wait_for_action(action_id: str, timeout_s: float = 10.0) -> dict[str, Any]:
        """Wait (up to timeout_s, max 30) for an action to finish; return its status."""
        try:
            status = service.wait(action_id, min(max(timeout_s, 0.0), MAX_WAIT_S))
        except KeyError as exc:
            return {"ok": False, "reason": "invalid_request", "detail": str(exc)}
        return {"ok": True, "action": _status_dict(status)}

    @server.tool()
    def cancel_action(ctx: Context | None = None) -> dict[str, Any]:
        """Cancel your running action; the arm holds where it is."""
        ok, result = run_command(service.cancel(client_id=_client_id(ctx)))
        if not ok:
            return result
        return {"ok": True, "action": _status_dict(result) if result else None}

    @server.tool()
    def stop(ctx: Context | None = None) -> dict[str, Any]:
        """Latch the virtual emergency stop: cancels motion and refuses new
        motion until reset_stop. Always allowed."""
        service.stop(by="agent", client_id=_client_id(ctx))
        return {"ok": True, "stop_latched": True}

    @server.tool()
    def reset_stop(confirm: bool = False, ctx: Context | None = None) -> dict[str, Any]:
        """Release a stop you latched. Requires confirm=true. A stop latched on
        the Xbox controller can only be released there (Menu button, also called Start)."""
        ok, result = run_command(service.reset_stop(client_id=_client_id(ctx), confirm=confirm))
        return {"ok": True, "released": bool(result)} if ok else result

    @server.tool()
    def reset_scene(ctx: Context | None = None) -> dict[str, Any]:
        """Home the arm and put every object back where it started."""
        ok, result = run_command(service.reset_scene(client_id=_client_id(ctx)))
        return {"ok": True} if ok else result

    @server.tool()
    def release_control(ctx: Context | None = None) -> dict[str, Any]:
        """Give up control so another agent can drive the arm."""
        ok, result = run_command(service.release_control(client_id=_client_id(ctx)))
        return {"ok": True, "released": bool(result)} if ok else result

    @server.tool()
    def camera(
        view: str = "wrist", width: int = 320, height: int = 240, ctx: Context | None = None
    ) -> Image | dict[str, Any]:
        """Look at the scene. Views: "wrist" (on the gripper, looking past the
        jaw), "overhead" (map view: robot base at the bottom, forward up, robot's
        left on the left), "front" (facing the robot, so its left is on your
        right), "side" (from the robot's left). Max 640x480."""
        try:
            data = service.render(view, width=width, height=height, client_id=_client_id(ctx)).result(
                timeout=COMMAND_TIMEOUT_S
            )
        except Exception as exc:  # noqa: BLE001 - reported to the agent
            return _rejection(exc)
        return Image(data=data, format="jpeg")

    return server


class McpThread:
    """Run the MCP server (streamable HTTP, localhost only) in a daemon thread."""

    def __init__(self, server: MCPServer, *, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.server = server
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/mcp"
        self._thread = threading.Thread(target=self._run, name="mcp-server", daemon=True)
        self.error: BaseException | None = None

    def _run(self) -> None:
        try:
            asyncio.run(self.server.run_streamable_http_async(host=self.host, port=self.port))
        except BaseException as exc:  # noqa: BLE001 - surfaced to the main thread
            self.error = exc

    def start(self) -> None:
        self._thread.start()
