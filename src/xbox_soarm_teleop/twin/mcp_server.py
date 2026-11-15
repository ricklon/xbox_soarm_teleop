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
from typing import Any, Callable

from mcp.server.mcpserver import Context, Image, MCPServer

from xbox_soarm_teleop.twin.actions import ActionRejected
from xbox_soarm_teleop.twin.cameras import VIEWS, default_size
from xbox_soarm_teleop.twin.service import TwinClosed, TwinService

COMMAND_TIMEOUT_S = 2.0  # the loop resolves commands within one tick
MAX_WAIT_S = 30.0

INSTRUCTIONS = """\
You control a simulated SO-ARM101 robot arm (a digital twin) on a table.

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
   center 1.5 cm or more above the workspace's lowest z (0.14 on the table;
   lower, the jaw tips hit it). Moving the
   jaws sideways or at an angle near a rigid object knocks it away.
5. set_gripper(opening_m=grip_width_m). The result's `holding` names it.
Every motion result gives `jaw_center_xyz_m` and the last few events.
Closed a little past the width it still holds; much further, a rigid object is
squeezed out and falls. A refused grasp's event says exactly what to do: the
wrist roll that squares the jaws ("slipped"), or the opening and moves that
center the jaws on it ("missed"). If both are wrong, the roll comes first. Do
those, in order, then close again. A soft object takes any closure at any roll.
A held object rides near the jaw center, so to put it somewhere, move
`jaw_center_xyz_m` over that spot, lower until the object's own z is a few mm
above where it will rest (as low as the workspace allows), then
set_gripper(position=0). Opening only a little does not let go: the jaws taper
in toward the wrist and still touch it ("still held"). A held object turns
with the gripper, so it may land on an edge and rock flat.

`observe` also gives the goal box (`scene.goal`: the center of its floor and
its inside size) and, for each object, `in_goal`. Release an object with the
jaw center a few centimeters above the box floor, inside its walls. Then lift
straight up 5 cm with move_by(dx=0, dy=0, dz=0.05) before any other move: the
open jaws still straddle the object, and homing from there drags it out of the
box. Then check `in_goal`.

Sorting scenes (sort_colors) have bins instead of a goal box: `scene.bins`
gives each bin's name, label, floor center and inside size, and each object's
`in_bin` says which bin it is in (null: none). Which object goes in which bin
is up to the request you were given; if it leaves something open, say what
you assumed. Release into a bin as into the goal box, then lift straight up
5 cm before any other move. check_success does not judge bins.

Skills: pick(name) picks an object up by itself (it looks, centres the
jaws, grasps, checks it holds it and retries after a miss) and place(bin)
puts what it holds into a bin, lifts clear and goes home. Prefer them to
steering the jaws step by step: decide what goes where, and let the arm do
the moving.

No-coordinates mode: when `observe` says `scene.object_positions` is
"hidden", the twin does not tell you where objects are, as on the real arm.
Each object's `xyz_m`, `in_goal`, `in_bin` and `off_table` are left out, and a "missed"
event says only that the jaws missed. You still get each object's `size_m`
and `grip_width_m`, whether you hold it, the goal box, `ee` and
`jaw_center_xyz_m`. Find an object with locate(name): a vision model looks
at a camera view and the result's `xyz_m` is its estimate of the object's
center, often a centimetre or so off; `jaw_move_m` is the move_by that puts
the jaw center on that estimate. locate looks from the front view by default,
which the arm does not block; if it answers not_found, try view="side" or
view="overhead". To pick an object up: locate it; move_to 6 cm above its
`xyz_m`; set_wrist_roll(angle_rad=-1.5); set_gripper(opening_m=grip_width_m
+ 0.05); locate again and move_by(dx, dy, dz=0) from `jaw_move_m`, twice
(locate before each); locate once more and move_by(dx=0, dy=0, dz) with its
`jaw_move_m.dz`; then set_gripper(opening_m=grip_width_m). Make one tool call
at a time and wait for its result: two motion calls at once are refused as
busy. A "missed" event means the estimate was off: open, lift 5 cm, locate
again and repeat. After releasing, lift straight up 5 cm
and home, then check the placement with check_success(name), after the last
move: a judge looks at a front picture and says "done" (it is
in the box), "retry" (it is not: locate it and pick again) or "look_again"
(unsure: ask again or look with `camera`). Without a judge (reason no_judge),
use locate(name, rest_z_m=the goal box floor z) or `camera`. Without locate
(reason no_pointer), use `camera`: the overhead view is a map, so judge an
object against things whose positions you know (the jaw center, the goal
box). locate's answers are scored against the truth for a benchmark; when you
work a position out yourself, call report_estimate(name, x, y, z) with its
center. Neither tells you the true answer.

Scenes differ in how forgiving their objects are. `list_scenes` shows them,
easiest first, and `load_scene` switches (it homes the arm and swaps the
objects). A soft object gives way to the gripper and takes any closure; a
rigid one is pushed away if the gripper touches it before it closes.

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


def build_mcp_server(
    service: TwinService | Callable[[Context | None], TwinService],
    *,
    name: str = "soarm-twin",
    instructions: str = INSTRUCTIONS,
    client_id: Callable[[Context | None], str] | None = None,
) -> MCPServer:
    """Create the MCP server exposing a twin.

    Args:
        service: The twin, or a function that picks the twin for a request
            from its context (the hub chooses a student's arm by the bearer
            token the request carries).
        name: Server name shown to clients.
        instructions: Server instructions for the agent.
        client_id: Names the caller of a request; by default its MCP session id.
    """
    server = MCPServer(name, instructions=instructions)
    client_of = client_id or _client_id

    def twin(ctx: Context | None) -> TwinService:
        return service(ctx) if callable(service) else service

    def run_command(future) -> tuple[bool, Any]:
        try:
            return True, future.result(timeout=COMMAND_TIMEOUT_S)
        except FutureTimeout:
            return False, {"ok": False, "reason": "timeout", "detail": "twin loop not responding"}
        except Exception as exc:  # noqa: BLE001 - reported to the agent
            return False, _rejection(exc)

    def submit(request: dict[str, Any], ctx: Context | None, wait: bool, preempt: bool = False):
        service = twin(ctx)
        ok, result = run_command(service.submit(request, client_id=client_of(ctx), preempt=preempt))
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
            "gripper_opening_m": observation["gripper_opening_m"],
            "jaw_center_xyz_m": observation["jaw_center_xyz_m"],
            "holding": service.grasp.attached,
            "events": observation["control"]["events"][-5:],
        }

    def speed_and_timeout(max_speed_m_s: float | None, timeout_s: float | None) -> dict:
        extra = {}
        if max_speed_m_s is not None:
            extra["max_speed_m_s"] = max_speed_m_s
        if timeout_s is not None:
            extra["timeout_s"] = timeout_s
        return extra

    @server.tool()
    def observe(ctx: Context | None = None) -> dict[str, Any]:
        """Current state: tool pose, joints (deg), gripper (0 open..1 closed),
        objects and whether one is attached, workspace bounds, the current
        action, stop status and who is in control."""
        return twin(ctx).observe()

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
    def set_gripper(
        position: float | None = None,
        opening_m: float | None = None,
        wait: bool = True,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Set the gripper, either by position (0 = open, 1 = closed) or by
        opening_m, the gap to leave between the jaws in meters. Closing to an
        object's width grabs it; the result's `holding` names what is held.
        Closing much further on a rigid object squeezes it out."""
        if (position is None) == (opening_m is None):
            return {
                "ok": False,
                "reason": "invalid_request",
                "detail": "give exactly one of position (0..1) or opening_m (meters)",
            }
        if opening_m is not None:
            position = twin(ctx).grasp.settings.gripper_for_opening(opening_m)
        return submit({"type": "set_gripper", "params": {"position": position}}, ctx, wait)

    @server.tool()
    def set_wrist_roll(
        angle_rad: float, wait: bool = True, ctx: Context | None = None
    ) -> dict[str, Any]:
        """Rotate the wrist to an absolute roll angle in radians (0 = level)."""
        return submit({"type": "set_wrist_roll", "params": {"angle_rad": angle_rad}}, ctx, wait)

    @server.tool()
    def home(wait: bool = True, ctx: Context | None = None) -> dict[str, Any]:
        """Return the arm to its home pose and open the gripper. Right after a
        release, lift straight up 5 cm first: from low down, homing drags the
        object the jaws still straddle."""
        return submit({"type": "home"}, ctx, wait)

    @server.tool()
    def wait_for_action(
        action_id: str, timeout_s: float = 10.0, ctx: Context | None = None
    ) -> dict[str, Any]:
        """Wait (up to timeout_s, max 30) for an action to finish; return its status."""
        try:
            status = twin(ctx).wait(action_id, min(max(timeout_s, 0.0), MAX_WAIT_S))
        except KeyError as exc:
            return {"ok": False, "reason": "invalid_request", "detail": str(exc)}
        return {"ok": True, "action": _status_dict(status)}

    @server.tool()
    def cancel_action(ctx: Context | None = None) -> dict[str, Any]:
        """Cancel your running action; the arm holds where it is."""
        ok, result = run_command(twin(ctx).cancel(client_id=client_of(ctx)))
        if not ok:
            return result
        return {"ok": True, "action": _status_dict(result) if result else None}

    @server.tool()
    def stop(ctx: Context | None = None) -> dict[str, Any]:
        """Latch the virtual emergency stop: cancels motion and refuses new
        motion until reset_stop. Always allowed."""
        twin(ctx).stop(by="agent", client_id=client_of(ctx))
        return {"ok": True, "stop_latched": True}

    @server.tool()
    def reset_stop(confirm: bool = False, ctx: Context | None = None) -> dict[str, Any]:
        """Release a stop you latched. Requires confirm=true. A stop latched on
        the Xbox controller can only be released there (Menu button, also called Start)."""
        ok, result = run_command(twin(ctx).reset_stop(client_id=client_of(ctx), confirm=confirm))
        return {"ok": True, "released": bool(result)} if ok else result

    @server.tool()
    def reset_scene(ctx: Context | None = None) -> dict[str, Any]:
        """Home the arm and put every object back where it started."""
        ok, result = run_command(twin(ctx).reset_scene(client_id=client_of(ctx)))
        return {"ok": True} if ok else result

    @server.tool()
    def list_scenes(ctx: Context | None = None) -> dict[str, Any]:
        """Scenes you can load, easiest first: the objects in each, how hard
        they are to grasp, and which scene is current."""
        return {"ok": True, "scenes": twin(ctx).list_scenes()}

    @server.tool()
    def load_scene(name: str, ctx: Context | None = None) -> dict[str, Any]:
        """Switch to another scene from list_scenes. Homes the arm, opens the
        gripper and replaces the objects on the table."""
        ok, result = run_command(twin(ctx).load_scene(name, client_id=client_of(ctx)))
        return {"ok": True, "scene": result} if ok else result

    @server.tool()
    def report_estimate(
        name: str, x: float, y: float, z: float, ctx: Context | None = None
    ) -> dict[str, Any]:
        """Report where you think an object's center is (m, base frame). The
        twin scores it against the true position for a perception benchmark;
        the result only says it was recorded. Mainly for no-coordinates mode
        (`scene.object_positions` is "hidden")."""
        _, result = run_command(twin(ctx).report_estimate(name, [x, y, z], client_id=client_of(ctx)))
        return result

    @server.tool()
    def pick(name: str, description: str | None = None, ctx: Context | None = None) -> dict[str, Any]:
        """Pick an object up, by itself: the arm looks (overhead, then close up
        through the wrist camera), centres its jaws, grasps, checks it holds it,
        and retries after a miss, then lifts it to carrying height. Uses only what
        the arm can see and feel. description overrides how the object is
        described to the pointing model. Takes 20 to 60 seconds."""
        return twin(ctx).pick(name, client_id=client_of(ctx), description=description)

    @server.tool()
    def place(bin: str, ctx: Context | None = None) -> dict[str, Any]:
        """Put what the gripper holds into a bin (a name from scene.bins, or the
        goal box's), let go, lift clear and go home."""
        return twin(ctx).place(bin, client_id=client_of(ctx))

    @server.tool()
    def check_success(name: str, ctx: Context | None = None) -> dict[str, Any]:
        """Is the object in the goal box? A judge model (Clef) looks at a front
        picture and answers with probabilities and a verdict: "done" (p_in_goal
        0.9 or more), "retry" (0.1 or less: it is not in the box; locate it and
        pick again) or "look_again" (unsure: check again after a moment, or
        look with camera). Use it after releasing an object, since no
        coordinates say where it ended up."""
        return twin(ctx).check_success(name, client_id=client_of(ctx))

    @server.tool()
    def locate(
        name: str,
        view: str = "front",
        description: str | None = None,
        rest_z_m: float | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Find an object by looking: a pointing model is shown a camera view and
        points at the object; that pixel is turned into the object's center xyz
        (m, base frame), assuming it rests on the table (or on rest_z_m, e.g.
        the goal box floor). The answer is the model's estimate, not the truth:
        it may be off by a centimetre or more. Views: front (default: the arm
        does not block it), side, overhead, wrist. description overrides how the object is described to the
        model (by default e.g. "red cube"). Keep the arm still while it looks."""
        return twin(ctx).locate(
            name, view=view, description=description, rest_z_m=rest_z_m, client_id=client_of(ctx)
        )

    @server.tool()
    def release_control(ctx: Context | None = None) -> dict[str, Any]:
        """Give up control so another agent can drive the arm."""
        ok, result = run_command(twin(ctx).release_control(client_id=client_of(ctx)))
        return {"ok": True, "released": bool(result)} if ok else result

    @server.tool()
    def camera(
        view: str = "wrist",
        width: int | None = None,
        height: int | None = None,
        ctx: Context | None = None,
    ) -> Image | dict[str, Any]:
        """Look at the scene. Views: "wrist" (on a post beside the gripper: the
        two jaw tips come in from the left and right with what they grip
        between them; 432x243 by default), "overhead" (map view: robot base at
        the bottom, forward up, robot's left on the left), "front" (facing the
        robot, so its left is on your right), "side" (from the robot's left).
        320x240 by default; max 640x480."""
        if width is None or height is None:
            width, height = default_size(view) if view in VIEWS else (320, 240)
        try:
            data = (
                twin(ctx)
                .render(view, width=width, height=height, client_id=client_of(ctx))
                .result(timeout=COMMAND_TIMEOUT_S)
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
