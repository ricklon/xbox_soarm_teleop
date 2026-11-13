"""The digital twin service: one owner of the simulator for humans and agents.

``TwinService`` owns the MuJoCo simulator and the cartesian control state.
Exactly one thread -- the loop that calls :meth:`TwinService.tick` -- steps
physics or mutates state (design D3). Any other thread (the MCP server, tests)
only:

- enqueues commands, which run at the start of the next tick and resolve a
  ``concurrent.futures.Future``;
- reads :meth:`observe`, an immutable snapshot published after each tick;
- calls :meth:`stop`, which takes effect before any queued command runs.

Human and agent share one motion path (``control.step.cartesian_step``):

- The human (Xbox) wins: holding LB, or pressing A (home) / Y (reset scene),
  cancels the agent's action (``preempted``/``human``) and blocks agent motion
  until the controller has been idle for ``HUMAN_IDLE_S``.
- B latches the virtual stop; Menu (Start) releases it. A stop latched on the
  controller can only be released on the controller.
- Agents take control automatically when it is free; a different client can
  take over whenever the owner has no running action, or after the lease
  (``LEASE_TIMEOUT_S``) lapses. Expiry only frees ownership; actions keep
  their own timeouts.

Time is ``frame * dt`` (simulated control time), so timeouts are deterministic
in tests and do not fire early if the loop runs slow.
"""

from __future__ import annotations

import math
import queue
import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, replace
from typing import Any, Callable

import mujoco
import numpy as np

from xbox_soarm_teleop.config.joints import IK_JOINT_VEL_LIMITS_ARRAY
from xbox_soarm_teleop.control.cartesian import make_cartesian_state, sync_cartesian_state
from xbox_soarm_teleop.control.pose import rotation_matrix_to_euler
from xbox_soarm_teleop.control.step import CartesianStepConfig, CartesianStepResult, cartesian_step
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta, apply_axis_mapping
from xbox_soarm_teleop.schemas.agent import (
    ActionRequest,
    ActionState,
    ActionStatus,
    EEPose,
    IKStatus,
    Observation,
    ReasonCode,
)
from xbox_soarm_teleop.teleoperators.xbox import XboxState
from xbox_soarm_teleop.twin.actions import (
    Action,
    ActionRejected,
    command_for,
    start_action,
    update_action,
)
from xbox_soarm_teleop.twin.cameras import DEFAULT_SIZE, VIEWS, CameraRig, encode
from xbox_soarm_teleop.twin.grasp import AssistedGrasp
from xbox_soarm_teleop.twin.layouts import Layout, layouts_for
from xbox_soarm_teleop.twin.session_log import SessionLog

ARM_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
HUMAN_IDLE_S = 2.0
LEASE_TIMEOUT_S = 60.0
GRIPPER_PICKUP_TOLERANCE = 0.05
RECENT_EVENTS = 8
OFF_TABLE_MARGIN_M = 0.01
KEPT_ACTIONS = 200


class TwinClosed(RuntimeError):
    """The twin loop has shut down."""


@dataclass
class RenderRequest:
    """A camera image somebody is waiting for."""

    view: str
    width: int
    height: int
    fmt: str
    future: Future


_STOP_RANK = {"agent": 0, "api": 0, "human": 1, "instructor": 2}


class TwinService:
    """Owns the simulator; arbitrates human (Xbox) and agent control.

    Args:
        sim: ``MuJoCoSimulator`` (any scene).
        kinematics: Constrained ``RobotKinematics`` for the IK chain.
        mapper: Xbox state -> ``EEDelta`` mapper (``MapXboxToEEDelta``).
        workspace_bounds: Bounds the control step enforces, ``{"x": (lo, hi), ...}``.
        clip_position: The matching ``clip_position`` callback.
        gripper_rate: Gripper change per second (0..1 range).
        dt: Control period (s).
        jacobian_ctrl: Optional Jacobian controller (otherwise IK).
        substeps: Physics steps per tick; default keeps sim time = control time.
        table_top_z: Height of the scene's table top (m), if it has one;
            objects that fall below it are reported ``off_table``.
        controller: Optional Xbox controller read each tick when ``tick`` is
            not given a state explicitly.
        log: Optional session log (design §10). Every request, response,
            status transition, stop, ownership change and a downsampled
            observation stream are written to it.
        cameras: Optional camera rig shared with other twins of the same
            model. The caller owns it: this service draws its own state with
            it and never closes it. Created on first use when None.
        layout: Which of the scene's objects start on the table
            (``twin.layouts``); ``load_scene`` changes it. None leaves every
            object where the model has it.
    """

    def __init__(
        self,
        sim,
        kinematics,
        *,
        mapper,
        workspace_bounds: dict,
        clip_position: Callable,
        gripper_rate: float = 2.0,
        dt: float = 0.02,
        jacobian_ctrl=None,
        substeps: int | None = None,
        controller=None,
        table_top_z: float | None = None,
        log: SessionLog | None = None,
        cameras: CameraRig | None = None,
        layout: Layout | None = None,
    ) -> None:
        self.log = log
        self._shared_cameras = cameras is not None
        self._status_logged: tuple | None = None
        self._control_logged: tuple = (None, False)
        self._log_error_reported = False
        self.sim = sim
        self.kinematics = kinematics
        self.mapper = mapper
        self.controller = controller
        self.jacobian_ctrl = jacobian_ctrl
        self.dt = dt
        self.workspace_bounds = workspace_bounds
        self.table_top_z = table_top_z
        self.step_config = CartesianStepConfig(
            dt=dt,
            joint_vel_limits_deg=IK_JOINT_VEL_LIMITS_ARRAY,
            gripper_rate=gripper_rate,
            clip_position=clip_position,
            workspace_bounds=workspace_bounds,
        )
        if substeps is None:
            substeps = max(1, round(dt / sim.model.opt.timestep)) if sim.physics_scene_active else 1
        self.substeps = substeps
        self.grasp = AssistedGrasp(sim.model, sim.data)
        self.layout = layout
        self._arrange_objects()
        self.state = make_cartesian_state(kinematics, np.zeros(4))

        self.frame = 0
        self.homing = False
        self.action: Action | None = None
        self._actions: dict[str, Action] = {}
        self.agent_gripper = 0.0
        self.owner: str | None = None
        self._owner_seen_s = -math.inf
        self._human_active_s = -math.inf
        self._gripper_following = False
        self.stop_latched_by: str | None = None
        self.stop_release_hint = "press the controller's Menu/Start button"
        """Told to an agent that tries to release a stop a person latched."""
        self.home_opens_gripper = False
        """A person's Home also opens the gripper. For controllers with no
        trigger to let go of (the hub's web page); a pad's trigger decides."""
        self._opening_for_home = False
        self._stop_flag = threading.Event()
        self._stop_by = "api"
        self._stop_client: str | None = None
        self._commands: queue.SimpleQueue[tuple[Callable[[], Any], Future]] = queue.SimpleQueue()
        self._renders: queue.SimpleQueue[RenderRequest] = queue.SimpleQueue()
        self._cameras: CameraRig | None = cameras
        self._loop_thread: int | None = None
        self._closed = False
        self.controller_connected = False
        self.last_human: XboxState | None = None
        self.events: deque[str] = deque(maxlen=RECENT_EVENTS)
        self.last_step: CartesianStepResult | None = None
        self._last_cmd = (np.zeros(5), 0.0)
        self._apply_command(np.zeros(5), 0.0)
        self._snapshot = self._build_observation()
        if log is not None:
            log.observation(self._snapshot)

    # ------------------------------------------------------------------ time
    @property
    def now_s(self) -> float:
        """Control time: ticks elapsed times the control period."""
        return self.frame * self.dt

    # ------------------------------------------------- thread-safe client API
    def call(self, fn: Callable[[], Any]) -> Future:
        """Run ``fn`` on the loop thread at the next tick; return its Future."""
        future: Future = Future()
        if self._closed:
            future.set_exception(TwinClosed("twin is not running"))
            return future
        self._commands.put((fn, future))
        return future

    def submit(
        self, request: dict | ActionRequest, *, client_id: str = "agent", preempt: bool = False
    ) -> Future:
        """Start an action. Resolves to its ``ActionStatus`` or raises ``ActionRejected``."""
        payload = {
            "request": request.to_dict() if isinstance(request, ActionRequest) else request,
            "preempt": preempt,
        }
        return self.call(
            self._logged(
                "submit", client_id, payload, lambda: self._submit(request, client_id, preempt)
            )
        )

    def cancel(self, *, client_id: str = "agent") -> Future:
        """Cancel the running action, if the caller owns it."""
        return self.call(self._logged("cancel", client_id, {}, lambda: self._cancel(client_id)))

    def stop(self, *, by: str = "agent", client_id: str | None = None) -> None:
        """Latch the virtual stop. Takes effect before any queued command."""
        self._stop_by = by
        self._stop_client = client_id
        self._stop_flag.set()

    def release_stop(self, *, by: str = "instructor") -> Future:
        """Release the stop whoever latched it. For the person in charge, not for agents."""

        def release() -> bool:
            if self.stop_latched_by is None:
                return False
            self._release_stop(by)
            return True

        return self.call(release)

    def reset_stop(self, *, client_id: str = "agent", confirm: bool = False) -> Future:
        """Release a stop latched by an agent (requires ``confirm=True``)."""
        return self.call(
            self._logged(
                "reset_stop",
                client_id,
                {"confirm": confirm},
                lambda: self._reset_stop(client_id, confirm),
            )
        )

    def acquire_control(self, *, client_id: str, force: bool = False) -> Future:
        return self.call(
            self._logged(
                "acquire_control", client_id, {"force": force}, lambda: self._acquire(client_id, force)
            )
        )

    def release_control(self, *, client_id: str) -> Future:
        return self.call(
            self._logged("release_control", client_id, {}, lambda: self._release(client_id))
        )

    def reset_scene(self, *, client_id: str = "agent") -> Future:
        """Home the arm and restore the scene's objects."""
        return self.call(
            self._logged("reset_scene", client_id, {}, lambda: self._reset_scene_cmd(client_id))
        )

    def load_scene(self, name: str, *, client_id: str = "agent") -> Future:
        """Switch to another layout of this scene: home the arm, swap the objects.

        Resolves to the layout's summary; an unknown name is rejected with the
        names that exist.
        """
        return self.call(
            self._logged(
                "load_scene", client_id, {"name": name}, lambda: self._load_scene(name, client_id)
            )
        )

    def list_scenes(self) -> list[dict[str, Any]]:
        """The scenes ``load_scene`` accepts, easiest first, marking the current one."""
        layouts = layouts_for(self.sim.scene)
        if not layouts:
            return [
                {
                    "name": self.sim.scene,
                    "difficulty": None,
                    "description": "",
                    "objects": list(self.grasp.objects),
                    "current": True,
                }
            ]
        current = self.layout.name if self.layout is not None else None
        return [{**layout.summary(), "current": layout.name == current} for layout in layouts]

    def render(
        self,
        view: str = "wrist",
        *,
        width: int = DEFAULT_SIZE[0],
        height: int = DEFAULT_SIZE[1],
        fmt: str = "jpeg",
        client_id: str | None = None,
        log: bool = True,
    ) -> Future:
        """Render a camera view after the next tick; resolves to encoded bytes.

        Views: ``wrist`` (on the gripper), ``overhead``, ``front``, ``side``.
        At most one image is rendered per tick to protect the control rate.
        ``log=False`` keeps the request out of the session log: a web page
        asks many times a second, which would bury everything else.
        """
        if view not in VIEWS:
            raise ValueError(f"Unknown view '{view}'. Known: {', '.join(VIEWS)}")
        future: Future = Future()
        if self._closed:
            future.set_exception(TwinClosed("twin is not running"))
            return future

        if log:
            self._log("request", client_id=client_id, type="camera", view=view, size=[width, height])
        self._renders.put(RenderRequest(view, width, height, fmt, future))
        return future

    def observe(self) -> dict[str, Any]:
        """Latest ``soarm.observation/v1`` snapshot (do not mutate)."""
        return self._snapshot

    def wait(self, action_id: str, timeout_s: float) -> ActionStatus:
        """Block until an action ends or ``timeout_s`` elapses; return its status."""
        action = self._actions.get(action_id)
        if action is None:
            raise KeyError(f"unknown action '{action_id}'")
        action.done.wait(timeout_s)
        return action.status()

    def close(self) -> None:
        """Shut down: fail pending commands and wake every waiter.

        Safe from any thread. OpenGL resources belong to the loop thread, so
        when called elsewhere they are released by the loop's next ``tick``.
        """
        if self._closed:
            return
        self._closed = True
        self._fail_pending()
        if self._loop_thread in (None, threading.get_ident()):
            self._close_cameras()
        for action in list(self._actions.values()):
            action.finish(ActionState.CANCELLED, ReasonCode.CANCELLED, "shutdown")
        if self.log is not None:
            self._log_status(self.action)
            self._log("shutdown")
            self.log.close()

    # ------------------------------------------------------------- the loop
    @property
    def render_waiting(self) -> bool:
        """True when a camera image has been requested and not drawn yet."""
        return not self._renders.empty()

    def take_render_request(self) -> RenderRequest | None:
        """Hand over the oldest waiting camera request, for a host that draws
        for several twins itself (``twin.hub``). The host must resolve its future.
        """
        try:
            return self._renders.get_nowait()
        except queue.Empty:
            return None

    def render_pending(self) -> bool:
        """Draw one requested camera image now; False if none was waiting.

        Call only from the loop thread. For loops that tick several twins and
        ration rendering between them (``tick(render=False)``).
        """
        request = self.take_render_request()
        if request is None:
            return False
        if request.future.set_running_or_notify_cancel():
            try:
                if self._cameras is None:
                    self._cameras = CameraRig(self.sim.model, self.sim.data)
                pixels = self._cameras.render(
                    request.view, request.width, request.height, self.sim.data
                )
                request.future.set_result(encode(pixels, request.fmt))
            except BaseException as exc:  # noqa: BLE001 - delivered to the caller
                request.future.set_exception(exc)
        return True

    def tick(self, human: XboxState | None = None, *, render: bool = True) -> None:
        """Advance one control period. Call only from the loop thread.

        Args:
            human: Controller state this tick. If None and a controller was
                given, it is read here; with neither, no human is present.
            render: Draw at most one requested camera image after the step.
                Pass False to draw later with :meth:`render_pending`.
        """
        self._loop_thread = threading.get_ident()
        if self._closed:
            self._close_cameras()
            return
        if human is None and self.controller is not None:
            human = self.controller.read()
            self.controller_connected = bool(self.controller.is_connected)
            if not self.controller_connected:
                human = None
        else:
            self.controller_connected = human is not None
        self.last_human = human
        now = self.now_s
        if self._stop_flag.is_set():
            self._stop_flag.clear()
            self._latch_stop(self._stop_by)
        self._drain_commands()

        delta = self._decide_command(human, now)
        step = cartesian_step(
            self.state,
            self.kinematics,
            delta,
            self.step_config,
            homing=self.homing,
            jacobian_ctrl=self.jacobian_ctrl,
        )
        self.homing = step.homing
        self.last_step = step
        self._apply_command(step.joints_deg, step.gripper)

        action = self.action
        if action is not None and not action.state.terminal:
            update_action(
                action,
                step,
                now_s=now + self.dt,
                tool_xyz=self.tool_xyz(),
                wrist_roll_rad=math.radians(self.state.wrist_roll_deg),
                gripper=self.state.gripper_pos,
                gripper_target=self.agent_gripper,
            )
            if action.type == "home" and action.state.terminal:
                self.homing = False
            if action.state.terminal:
                self.events.append(f"{action.id} {action.type}: {action.state.value}")
            self._log_status(action)
        if (
            self.owner is not None
            and now - self._owner_seen_s > LEASE_TIMEOUT_S
            and not self._owner_busy()
        ):
            self.owner = None
            self._log_control("lease_expired")
        self._log_control("human")
        self.frame += 1
        self._snapshot = self._build_observation()
        if self.log is not None:
            self.log.observation(self._snapshot)
            if self.log.error is not None and not self._log_error_reported:
                self._log_error_reported = True
                self.events.append(f"session log stopped: {self.log.error}")
        if render:
            self.render_pending()

    # --------------------------------------------------------------- helpers
    def tool_xyz(self) -> np.ndarray:
        """Control tool point (m, base frame): what actions target."""
        return self.state.ee_pose[:3, 3].copy()

    def _log(
        self, kind: str, *, client_id: str | None = None, action_id: str | None = None, **data: Any
    ) -> None:
        if self.log is not None:
            self.log.event(
                kind, t=self.now_s, frame=self.frame, client_id=client_id, action_id=action_id, **data
            )

    def _logged(
        self, name: str, client_id: str, payload: dict[str, Any], fn: Callable[[], Any]
    ) -> Callable[[], Any]:
        """Wrap a command body so its request and response are logged."""
        if self.log is None:
            return fn

        def run() -> Any:
            self._log("request", client_id=client_id, type=name, **payload)
            try:
                result = fn()
            except ActionRejected as exc:
                self._log(
                    "response",
                    client_id=client_id,
                    type=name,
                    ok=False,
                    reason=exc.reason.value,
                    detail=exc.detail,
                )
                raise
            finally:
                self._log_control(name)
            self._log(
                "response",
                client_id=client_id,
                action_id=result.id if isinstance(result, ActionStatus) else None,
                type=name,
                ok=True,
                result=result,
            )
            return result

        return run

    def _log_status(self, action: Action | None) -> None:
        """Log an action's state if it changed since the last status line."""
        if self.log is None or action is None:
            return
        status = action.status()
        key = (status.id, status.state, status.reason)
        if key == self._status_logged:
            return
        self._status_logged = key
        self._log(
            "status",
            client_id=action.client_id,
            action_id=status.id,
            type=status.type,
            state=status.state,
            reason=status.reason,
            detail=status.detail,
            progress=round(status.progress, 4),
        )

    def _log_control(self, cause: str) -> None:
        """Log who is in control (agent owner, human override) when it changes."""
        if self.log is None:
            return
        current = (self.owner, self._human_in_control(self.now_s))
        if current == self._control_logged:
            return
        previous, self._control_logged = self._control_logged, current
        self._log(
            "ownership",
            client_id=self.owner,
            owner=self.owner,
            previous_owner=previous[0],
            human_in_control=current[1],
            cause=cause,
        )

    def _event(self, text: str) -> None:
        self.events.append(text)
        self._log("event", text=text)

    def _drain_commands(self) -> None:
        while self._run_one(self._commands):
            pass

    def _close_cameras(self) -> None:
        if self._cameras is not None and not self._shared_cameras:
            self._cameras.close()
            self._cameras = None

    @staticmethod
    def _run_one(pending: queue.SimpleQueue) -> bool:
        """Run one queued callable into its Future; False if none was waiting."""
        try:
            fn, future = pending.get_nowait()
        except queue.Empty:
            return False
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - delivered to the caller
                future.set_exception(exc)
        return True

    def _fail_pending(self) -> None:
        waiting = []
        for pending in (self._commands, self._renders):
            while True:
                try:
                    item = pending.get_nowait()
                except queue.Empty:
                    break
                waiting.append(item.future if isinstance(item, RenderRequest) else item[1])
        for future in waiting:
            if future.set_running_or_notify_cancel():
                future.set_exception(TwinClosed("twin stopped"))

    def _human_in_control(self, now: float) -> bool:
        return now - self._human_active_s < HUMAN_IDLE_S

    def _owner_busy(self) -> bool:
        action = self.action
        return action is not None and not action.state.terminal and action.client_id == self.owner

    def _ensure_owner(self, client_id: str) -> None:
        now = self.now_s
        if self._human_in_control(now):
            raise ActionRejected(ReasonCode.NOT_OWNER, "human")
        if self.owner is not None and self.owner != client_id:
            if self._owner_busy() and now - self._owner_seen_s < LEASE_TIMEOUT_S:
                raise ActionRejected(ReasonCode.NOT_OWNER, self.owner)
        self.owner = client_id
        self._owner_seen_s = now

    def _end_action(self, state: ActionState, reason: ReasonCode, detail: str = "") -> None:
        action = self.action
        if action is None or action.state.terminal:
            return
        action.finish(state, reason, detail)
        if action.type == "home":
            self.homing = False
        self.events.append(f"{action.id} {action.type}: {state.value} ({reason.value})")
        self._log_status(action)

    def _latch_stop(self, by: str) -> None:
        # Whoever ranks higher keeps the stop: instructor, then the person, then an agent.
        if _STOP_RANK.get(by, 0) >= _STOP_RANK.get(self.stop_latched_by, -1):
            self.stop_latched_by = by
        self._end_action(ActionState.STOPPED, ReasonCode.STOPPED, by)
        self.homing = False
        self.events.append(f"stop latched by {by}")
        client_id = None if by == "human" else self._stop_client
        self._log("stop", client_id=client_id, by=by, latched_by=self.stop_latched_by)

    def _release_stop(self, by: str) -> None:
        self.stop_latched_by = None
        self.events.append(f"stop released by {by}")
        self._log("stop_reset", client_id=None if by == "human" else by, by=by)

    # ------------------------------------------------------ command bodies
    def _submit(self, request: dict | ActionRequest, client_id: str, preempt: bool) -> ActionStatus:
        if not isinstance(request, ActionRequest):
            try:
                request = ActionRequest.from_dict(request)
            except ValueError as exc:
                raise ActionRejected(ReasonCode.INVALID_REQUEST, str(exc)) from exc
        if self.stop_latched_by is not None:
            raise ActionRejected(ReasonCode.STOPPED, f"latched by {self.stop_latched_by}")
        self._ensure_owner(client_id)
        if self.action is not None and not self.action.state.terminal:
            if not preempt:
                raise ActionRejected(ReasonCode.BUSY, self.action.id)
            self._end_action(ActionState.CANCELLED, ReasonCode.PREEMPTED, "agent")
        action = start_action(
            request,
            client_id=client_id,
            now_s=self.now_s,
            tool_xyz=self.tool_xyz(),
            wrist_roll_rad=math.radians(self.state.wrist_roll_deg),
            workspace_bounds=self.workspace_bounds,
        )
        if request.type == "set_gripper":
            self.agent_gripper = float(request.params["position"])
        if request.type == "home":
            self.homing = True
        self.action = action
        self._actions[action.id] = action
        self._log_status(action)
        while len(self._actions) > KEPT_ACTIONS:
            self._actions.pop(next(iter(self._actions)))
        return action.status()

    def _cancel(self, client_id: str) -> ActionStatus | None:
        if self.action is None or self.action.state.terminal:
            return self.action.status() if self.action else None
        if self.action.client_id != client_id:
            raise ActionRejected(ReasonCode.NOT_OWNER, self.action.client_id)
        self._end_action(ActionState.CANCELLED, ReasonCode.CANCELLED)
        return self.action.status()

    def _reset_stop(self, client_id: str, confirm: bool) -> bool:
        if self.stop_latched_by is None:
            return False
        if not confirm:
            raise ActionRejected(ReasonCode.INVALID_REQUEST, "reset_stop requires confirm=true")
        if self.stop_latched_by == "instructor":
            raise ActionRejected(
                ReasonCode.NOT_OWNER,
                "instructor (the instructor latched this stop and must release it)",
            )
        if self.stop_latched_by == "human":
            raise ActionRejected(
                ReasonCode.NOT_OWNER,
                f"human (a person latched this stop and must release it: {self.stop_release_hint})",
            )
        self._ensure_owner(client_id)
        self._release_stop(client_id)
        return True

    def _acquire(self, client_id: str, force: bool) -> str:
        if force and not self._human_in_control(self.now_s):
            self.owner = None
        self._ensure_owner(client_id)
        return client_id

    def _release(self, client_id: str) -> bool:
        if self.owner != client_id:
            return False
        self.owner = None
        return True

    def _reset_scene_cmd(self, client_id: str) -> bool:
        if self.stop_latched_by is not None:
            raise ActionRejected(ReasonCode.STOPPED, f"latched by {self.stop_latched_by}")
        self._ensure_owner(client_id)
        self._reset_scene()
        return True

    def _load_scene(self, name: str, client_id: str) -> dict[str, Any]:
        if self.stop_latched_by is not None:
            raise ActionRejected(ReasonCode.STOPPED, f"latched by {self.stop_latched_by}")
        layouts = {layout.name: layout for layout in layouts_for(self.sim.scene)}
        if name not in layouts:
            known = ", ".join(scene["name"] for scene in self.list_scenes())
            raise ActionRejected(ReasonCode.INVALID_REQUEST, f"Unknown scene '{name}'. Known: {known}")
        self._ensure_owner(client_id)
        self.layout = layouts[name]
        self._reset_scene()
        self._event(f"scene loaded: {name}")
        return {**self.layout.summary(), "current": True}

    def _arrange_objects(self) -> None:
        """Put the layout's objects at their starts and park the others."""
        if self.layout is None:
            return
        for name, xyz in self.layout.objects.items():
            self.grasp.place(name, np.asarray(xyz, dtype=float))
        self.grasp.park(name for name in self.grasp.objects if name not in self.layout.objects)
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def _reset_scene(self) -> None:
        self._end_action(ActionState.CANCELLED, ReasonCode.CANCELLED, "scene reset")
        self.grasp.reset()
        self.homing = False
        self.agent_gripper = 0.0
        sim = self.sim
        sim.go_home()
        if sim.has_stack_scene():
            sim.reset_stack_scene()
        elif sim.has_pick_place_scene():
            sim.reset_pick_place_scene()
        self._arrange_objects()
        sync_cartesian_state(
            self.state,
            self.kinematics,
            np.zeros(4),
            wrist_roll_deg=0.0,
            target_pitch=0.0,
            target_yaw=0.0,
            gripper_pos=0.0,
        )
        self.state.orientation_active = False
        self._last_cmd = (np.zeros(5), 0.0)
        self._apply_command(np.zeros(5), 0.0)
        self._event("scene reset")

    # ---------------------------------------------------- per-tick decision
    def _decide_command(self, human: XboxState | None, now: float) -> EEDelta:
        hold = EEDelta(gripper=self.state.gripper_pos)
        if human is not None:
            if human.b_button_pressed:
                self._latch_stop("human")
            if human.start_button_pressed and self.stop_latched_by not in (None, "instructor"):
                self._release_stop("human")
            engaged = human.left_bumper or human.a_button_pressed or human.y_button_pressed
            if engaged and self.stop_latched_by is None:
                self._human_active_s = now
                if self.action is not None and not self.action.state.terminal:
                    self._end_action(ActionState.CANCELLED, ReasonCode.PREEMPTED, "human")
        if self.stop_latched_by is not None:
            self.homing = False
            return hold

        if human is not None and self._human_in_control(now):
            if human.y_button_pressed:
                self._reset_scene()
                return EEDelta(gripper=self.state.gripper_pos)
            if human.a_button_pressed:
                self.homing = True
                if self.home_opens_gripper:
                    self._opening_for_home = True
                    self.agent_gripper = 0.0  # keeps opening after control passes back
            delta = apply_axis_mapping(self.mapper(human), swap_xy=False)
            if self._opening_for_home:
                if human.left_bumper or self.state.gripper_pos <= 0.01:
                    self._opening_for_home = False  # done, or the person took the gripper back
                else:
                    return replace(delta, gripper=0.0)
            # Jaws a person drives shut stop on a rigid object instead of
            # squeezing it out: nobody holds a trigger to the millimetre.
            limit = self.grasp.hold_limit()
            if limit is not None and delta.gripper > limit:
                delta = replace(delta, gripper=max(limit, min(delta.gripper, self.state.gripper_pos)))
                self._gripper_following = True
            # Soft takeover: the trigger is absolute, so only follow it once it
            # reaches the current gripper position (no sudden drop of a cube).
            if not self._gripper_following:
                if abs(delta.gripper - self.state.gripper_pos) <= GRIPPER_PICKUP_TOLERANCE:
                    self._gripper_following = True
                else:
                    delta = replace(delta, gripper=self.state.gripper_pos)
            return delta

        if self._gripper_following:
            # Control passes back: the agent keeps whatever the human left.
            self._gripper_following = False
            if not self._opening_for_home:
                self.agent_gripper = self.state.gripper_pos
        self._opening_for_home = False
        action = self.action
        if action is not None and not action.state.terminal:
            return command_for(
                action,
                tool_xyz=self.tool_xyz(),
                wrist_roll_rad=math.radians(self.state.wrist_roll_deg),
                gripper=self.agent_gripper,
            )
        return EEDelta(gripper=self.agent_gripper)

    # ------------------------------------------------------------ physics
    def _apply_command(self, joints_deg: np.ndarray, gripper: float) -> None:
        """Send the arm command, interpolated across physics substeps."""
        sim = self.sim
        prev_joints, prev_gripper = self._last_cmd
        joints_deg = np.asarray(joints_deg, dtype=float)
        for k in range(1, self.substeps + 1):
            alpha = k / self.substeps
            sim.set_joint_targets(prev_joints + (joints_deg - prev_joints) * alpha)
            sim.set_gripper(prev_gripper + (gripper - prev_gripper) * alpha)
            mujoco.mj_kinematics(sim.model, sim.data)
            self.grasp.carry(sim.get_ee_position())
            self.grasp.hold_parked()
            sim.step()
        self._last_cmd = (joints_deg.copy(), float(gripper))
        for event in self.grasp.update(sim.get_ee_position(), gripper):
            self._event(event)

    # -------------------------------------------------------- observation
    def _build_observation(self) -> dict[str, Any]:
        pose = self.state.ee_pose
        joints = self.sim.get_joint_positions_deg()
        ik = getattr(self.kinematics, "last_ik_result", None)
        step = self.last_step
        observation = Observation(
            frame=self.frame,
            sim_time_s=round(self.now_s, 6),
            mode="cartesian",
            ee=EEPose(
                xyz_m=tuple(round(float(v), 5) for v in pose[:3, 3]),
                rpy_rad=tuple(round(v, 4) for v in rotation_matrix_to_euler(pose[:3, :3])),
            ),
            joints_deg={n: round(float(v), 3) for n, v in zip(ARM_JOINT_NAMES, joints)},
            gripper=round(float(self.state.gripper_pos), 4),
            stop_latched=self.stop_latched_by is not None,
            owner=self.owner,
            action=self.action.status() if self.action is not None else None,
            last_ik=(
                IKStatus(bool(ik.accepted), str(ik.reason), float(ik.position_error_m))
                if ik is not None
                else None
            ),
            limit=step.limit if step is not None else "",
            workspace=self.workspace_bounds,
            scene={
                "name": self.layout.name if self.layout is not None else self.sim.scene,
                "objects": {
                    name: {
                        "xyz_m": [round(float(v), 4) for v in xyz],
                        "size_m": [round(float(v), 4) for v in self.grasp.size(name)],
                        "attached": name == self.grasp.attached,
                        "off_table": self.table_top_z is not None
                        and float(xyz[2]) < self.table_top_z - OFF_TABLE_MARGIN_M,
                    }
                    for name, xyz in self.grasp.positions().items()
                },
            },
            control={
                "stop_latched_by": self.stop_latched_by,
                "human_in_control": self._human_in_control(self.now_s),
                "controller_connected": self.controller_connected,
                "events": list(self.events),
            },
            gripper_opening_m=round(self.grasp.settings.jaw_opening_m(self.state.gripper_pos), 4),
        )
        return observation.to_dict()
