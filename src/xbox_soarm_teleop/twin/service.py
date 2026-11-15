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
import re
import threading
import time
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
from xbox_soarm_teleop.twin.layouts import (
    BINS,
    GOALS,
    Layout,
    layouts_for,
    object_description,
)
from xbox_soarm_teleop.twin.pointing import Pointer, PointRequest, mask_secrets
from xbox_soarm_teleop.twin.projection import CameraGeometry, camera_geometry, pixel_to_xyz
from xbox_soarm_teleop.twin.session_log import SessionLog

ARM_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
HUMAN_IDLE_S = 2.0
LEASE_TIMEOUT_S = 60.0
GRIPPER_PICKUP_TOLERANCE = 0.05
RECENT_EVENTS = 8
OFF_TABLE_MARGIN_M = 0.01
KEPT_ACTIONS = 200
LOCATE_SIZE = (640, 480)  # pointing models do better on the largest picture
LOCATE_TIMEOUT_S = 5.0
# check_success: the judge's P(in the goal box) at or above DONE_AT says done, at
# or below RETRY_AT says retry, and anything between says look again. Clef's
# answers at 0.9 or more were right 93% of the time, at 0.1 or less always
# (docs/vision_benchmark.md).
DONE_AT = 0.9
RETRY_AT = 0.1
CHECK_SIZE = (640, 480)
JAW_FLOOR_CLEARANCE_M = 0.015  # the jaw center no lower than this over the workspace floor  # for the loop to answer; the model itself may take longer


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
        hide_positions: No-coordinates mode, for testing perception. ``observe``
            (and so every agent tool) leaves out each object's position and
            what follows from it (``xyz_m``, ``in_goal``, ``off_table``), and a
            refused grasp's event no longer says how far to move. An agent has
            to find objects with the cameras, as on the real arm. The truth is
            kept: in the session log, in :meth:`observe_truth`, and as the
            grader of :meth:`report_estimate`.
        pointer: The pointing model behind :meth:`locate` (``twin.pointing``).
        judge: The success judge behind :meth:`check_success` (``twin.judge.ClefJudge``).
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
        hide_positions: bool = False,
        pointer: Pointer | None = None,
        judge: Any = None,
        naming: str = "as_is",
    ) -> None:
        self.log = log
        self.naming = naming
        """The names agents are shown for objects (``twin.naming``)."""
        self.hide_positions = hide_positions
        self._estimates: list[dict[str, Any]] = []
        self._shown: dict[str, Any] | None = None
        self.pointer = pointer
        """The pointing model ``locate`` asks (``twin.pointing``); None: no ``locate``."""
        self.judge = judge
        """The judge ``check_success`` asks (``twin.judge.ClefJudge``); None: no checks."""
        self._success_checks: list[dict[str, Any]] = []
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

    def abort_action(self, detail: str = "") -> Future:
        """End the running action, whoever owns it, as a person taking over
        does (for the page's agent Stop). The arm holds where it is."""
        return self.call(lambda: self._end_action(ActionState.CANCELLED, ReasonCode.PREEMPTED, detail))

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

    def load_layout(self, layout, *, client_id: str = "agent") -> Future:
        """Set the desk up as ``layout`` says (a challenge's own arrangement,
        ``twin.challenges``): home the arm, swap the objects. Reset restores it.

        Resolves to the layout's summary; refused if ``layout`` is for another
        compiled scene.
        """
        return self.call(
            self._logged(
                "load_layout",
                client_id,
                {"objects": {name: list(pose) for name, pose in layout.objects.items()}},
                lambda: self._load_layout(layout, client_id),
            )
        )

    def _load_layout(self, layout, client_id: str) -> dict[str, Any]:
        if self.stop_latched_by is not None:
            raise ActionRejected(ReasonCode.STOPPED, f"latched by {self.stop_latched_by}")
        if layout.model_scene != self.sim.scene:
            raise ActionRejected(
                ReasonCode.INVALID_REQUEST,
                f"that arrangement is for the {layout.model_scene} scene, this arm has {self.sim.scene}",
            )
        self._ensure_owner(client_id)
        self.layout = layout
        self._reset_scene()
        self._event("layout loaded")
        return {**self.layout.summary(), "current": True}

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
        """Latest ``soarm.observation/v1`` snapshot (do not mutate).

        In no-coordinates mode (``hide_positions``) object positions are left
        out; :meth:`observe_truth` has them.
        """
        snapshot = self._snapshot
        if not self.hide_positions:
            return snapshot
        shown = self._shown
        if shown is None or shown["frame"] != snapshot["frame"]:
            shown = self._shown = hide_object_positions(snapshot)
        return shown

    @property
    def shown_names(self) -> dict[str, str] | None:
        """The name agents see for each object on the table now (None: the
        twin's own). Changes with the layout; see ``twin.naming``."""
        from xbox_soarm_teleop.twin.naming import shown_names

        objects = self.layout.objects if self.layout is not None else self.grasp.positions()
        return shown_names(objects, self.naming)

    def observe_truth(self) -> dict[str, Any]:
        """The latest snapshot with every object's true position, whatever the mode.

        For graders, the session log and the people running the twin; never
        hand it to an agent in no-coordinates mode.
        """
        return self._snapshot

    def report_estimate(self, name: str, xyz_m, *, client_id: str = "agent") -> Future:
        """Score a guess at an object's center (m, base frame) against the truth.

        Resolves to what the agent may see (that it was recorded), never the
        error: an agent told how far off it was could home in on the answer.
        The score goes to :meth:`estimate_scores` and the session log
        (``estimate`` lines). Unknown or parked objects are rejected.
        """
        payload = {"name": name, "xyz_m": list(xyz_m) if xyz_m is not None else None}
        return self.call(
            self._logged(
                "report_estimate",
                client_id,
                payload,
                lambda: self._report_estimate(name, xyz_m, client_id),
            )
        )

    def estimate_scores(self) -> list[dict[str, Any]]:
        """Every scored estimate so far, oldest first (copies)."""
        return [dict(score) for score in self._estimates]

    def success_checks(self) -> list[dict[str, Any]]:
        """Every ``check_success`` so far, with the truth beside the judge's answer (copies)."""
        return [dict(check) for check in self._success_checks]

    def pick(
        self, name: str | None = None, *, client_id: str = "agent", description: str | None = None
    ) -> dict[str, Any]:
        """Pick ``name`` up by itself (``twin.skills.pick``): look, centre, grasp,
        check by feel, retry. Blocks; call it from a client thread."""
        from xbox_soarm_teleop.twin import skills

        self._log("request", client_id=client_id, type="pick", name=name)
        result = skills.pick(self, name, client_id=client_id, description=description)
        self._log("skill_result", client_id=client_id, skill="pick", target=name, **result)
        return result

    def place(self, bin_name: str, *, client_id: str = "agent") -> dict[str, Any]:
        """Put what the gripper holds in ``bin_name`` (``twin.skills.place``), lift
        clear and go home. Blocks; call it from a client thread."""
        from xbox_soarm_teleop.twin import skills

        self._log("request", client_id=client_id, type="place", target_bin=bin_name)
        result = skills.place(self, bin_name, client_id=client_id)
        # The result has its own "bin": the log line names the request's target_bin.
        self._log("skill_result", client_id=client_id, skill="place", target_bin=bin_name, **result)
        return result

    def push(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        direction: str | None = None,
        distance_m: float | None = None,
        dx_m: float | None = None,
        dy_m: float | None = None,
        client_id: str = "agent",
    ) -> dict[str, Any]:
        """Slide an object along the table (``twin.skills.push``). Blocks; call it
        from a client thread."""
        from xbox_soarm_teleop.twin import skills

        asked = {"direction": direction, "distance_m": distance_m, "dx_m": dx_m, "dy_m": dy_m}
        self._log(
            "request", client_id=client_id, type="push", name=name, description=description, **asked
        )
        result = skills.push(self, name, description=description, client_id=client_id, **asked)
        self._log("skill_result", client_id=client_id, skill="push", target=name, **result)
        return result

    def check_success(
        self,
        name: str | None = None,
        *,
        bin_name: str | None = None,
        description: str | None = None,
        client_id: str = "agent",
        timeout_s: float = LOCATE_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Is ``name`` in the goal box (or, in a sorting scene, in ``bin_name``)?
        Ask the judge about a front picture.

        Returns the judge's probabilities and a verdict: ``done`` at
        ``DONE_AT`` or more, ``retry`` at ``RETRY_AT`` or less, ``look_again``
        between. The twin's truth is logged beside the answer (``success_check``
        lines, :meth:`success_checks`) and never returned. Blocks while the
        judge answers; call it from a client thread.
        """
        if name is None and not description:
            return {
                "ok": False,
                "reason": "invalid_request",
                "detail": "say which object: its name, or (in sorting scenes) a description "
                'in the request\'s words, like description="red cube"',
            }
        if GOALS.get(self.sim.scene) is None and BINS.get(self.sim.scene):
            return self._check_trays(name, bin_name, client_id, timeout_s, description)
        if name is None:
            return {
                "ok": False,
                "reason": "invalid_request",
                "detail": "checking by description works in sorting scenes; give the name",
            }
        judge = self.judge
        if judge is None:
            return {
                "ok": False,
                "reason": "no_judge",
                "detail": "this twin has no success judge (start it with --judge)",
            }
        try:
            self._log("request", client_id=client_id, type="check_success", name=name)
            setup = self.call(lambda: self._check_setup(name)).result(timeout=timeout_s)
            picture = self.render(
                "front", width=CHECK_SIZE[0], height=CHECK_SIZE[1], fmt="png", client_id=client_id
            ).result(timeout=timeout_s)
        except ActionRejected as exc:
            return {"ok": False, "reason": exc.reason.value, "detail": exc.detail}
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            return {"ok": False, "reason": "error", "detail": str(exc)}
        started = time.monotonic()
        try:
            answer = judge.judge(
                [picture], setup["description"], views=["front"], others=setup["others"]
            )
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            detail = mask_secrets(str(exc))
            self._log("success_check", client_id=client_id, ok=False, name=name, detail=detail)
            return {"ok": False, "reason": "judge_error", "detail": detail}
        seconds = round(time.monotonic() - started, 3)
        p_in_goal = float(answer["in_goal"])
        verdict = verdict_for(p_in_goal)
        where = max(answer["where"], key=answer["where"].get)
        result = {
            "ok": True,
            "name": name,
            "description": setup["description"],
            "verdict": verdict,
            "p_in_goal": round(p_in_goal, 4),
            "p_held": round(float(answer["held"]), 4),
            "p_tipped": round(float(answer["tipped"]), 4),
            "where": where,
            "judge": getattr(judge, "model", "judge"),
            "seconds": seconds,
            "cost_usd": answer.get("cost_usd"),
        }
        truth = setup["truth"]
        record = {
            **result,
            "client_id": client_id,
            "sim_time_s": round(self.now_s, 3),  # not "t": the log line has its own
            "truth": truth,
            # A verdict is right when "done" means in the box and "retry" means not;
            # "look again" claims nothing.
            "correct": None
            if verdict == "look_again"
            else (verdict == "done") == bool(truth["in_goal"]),
        }
        self._success_checks.append(record)
        self._log(
            "success_check",
            client_id=client_id,
            **{k: v for k, v in record.items() if k != "client_id"},
        )
        return result

    def _check_trays(
        self,
        name: str | None,
        bin_name: str | None,
        client_id: str,
        timeout_s: float,
        description: str | None = None,
    ) -> dict[str, Any]:
        """check_success in a sorting scene: which tray is ``name`` in?

        Asked by ``description`` instead ("red cube", the request's words),
        Clef finds the object by how it looks, so a name that misleads the
        agent cannot mislead the check (lesson 24 in docs/lessons_learned.md).
        The twin works out which object matches, for grading only.

        Clef answers from the front picture with the cell's bins as options
        (measured on the sorting desk: 95% right, and every answer of 0.9 or
        more right; overhead was far worse, scripts/tray_judge_benchmark.py).
        """
        from xbox_soarm_teleop.twin.judge import tray_truth

        judge = self.judge
        if judge is None:
            return {
                "ok": False,
                "reason": "no_judge",
                "detail": "this twin has no success judge (start it with --judge)",
            }
        boxes = [(box.name, box.label or box.name) for box in BINS[self.sim.scene]]
        if bin_name is not None and bin_name not in dict(boxes):
            known = ", ".join(name_ for name_, _ in boxes)
            return {
                "ok": False,
                "reason": "invalid_request",
                "detail": f"Unknown bin '{bin_name}'. Here: {known}",
            }
        try:
            self._log(
                "request",
                client_id=client_id,
                type="check_success",
                name=name,
                description=description,
            )
            if name is not None:
                setup = self.call(lambda: self._check_setup(name, tray_truth)).result(timeout=timeout_s)
                setup["matched"] = name
            else:
                setup = self.call(lambda: self._describe_setup(description)).result(timeout=timeout_s)
            picture = self.render(
                "front", width=CHECK_SIZE[0], height=CHECK_SIZE[1], fmt="png", client_id=client_id
            ).result(timeout=timeout_s)
        except ActionRejected as exc:
            return {"ok": False, "reason": exc.reason.value, "detail": exc.detail}
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            return {"ok": False, "reason": "error", "detail": str(exc)}
        started = time.monotonic()
        try:
            answer = judge.judge_trays(
                [picture], setup["description"], boxes, views=["front"], others=setup["others"]
            )
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            detail = mask_secrets(str(exc))
            self._log("success_check", client_id=client_id, ok=False, name=name, detail=detail)
            return {"ok": False, "reason": "judge_error", "detail": detail}
        probabilities = {k: round(float(v), 4) for k, v in answer["where"].items()}
        where = max(probabilities, key=probabilities.get)
        result = {
            "ok": True,
            **({"name": name} if name is not None else {}),
            "description": setup["description"],
            "where": where,
            "p_where": probabilities,
            "judge": getattr(judge, "model", "judge"),
            "seconds": round(time.monotonic() - started, 3),
            "cost_usd": answer.get("cost_usd"),
        }
        truth = setup["truth"]
        if bin_name is not None:
            p_in_bin = probabilities.get(bin_name, 0.0)
            verdict = verdict_for(p_in_bin)
            result.update(bin=bin_name, p_in_bin=p_in_bin, verdict=verdict)
            correct = (
                None
                if verdict == "look_again" or truth is None
                else (verdict == "done") == (truth == bin_name)
            )
        else:
            correct = None if truth is None else where == truth
        record = {
            **result,
            "client_id": client_id,
            "sim_time_s": round(self.now_s, 3),
            "matched": setup["matched"],
            "truth": None
            if truth is None
            else {"where": truth, "in_goal": truth == bin_name if bin_name else None},
            "correct": correct,
        }
        self._success_checks.append(record)
        self._log(
            "success_check",
            client_id=client_id,
            **{k: v for k, v in record.items() if k != "client_id"},
        )
        return result

    def _describe_setup(self, description: str) -> dict[str, Any]:
        """On the loop thread: which object a description means (for grading
        only), the others on the desk, and where the matching one is."""
        from xbox_soarm_teleop.twin.judge import tray_truth

        positions = self.grasp.positions()
        wanted = description.strip().lower()
        matched = next(
            (n for n in positions if object_description(n, self.layout).lower() == wanted), None
        )
        return {
            "description": description.strip(),
            "others": [object_description(o, self.layout) for o in positions if o != matched],
            "matched": matched,
            "truth": tray_truth(self, matched) if matched is not None else None,
        }

    def _check_setup(self, name: str, truth_of=None) -> dict[str, Any]:
        """On the loop thread: check the name, word the question, read the truth."""
        from xbox_soarm_teleop.twin.judge import truth_labels

        positions = self.grasp.positions()
        if name not in positions:
            known = ", ".join(positions) or "none"
            hint = f'; to ask by how it looks, use description="{name}"' if " " in name else ""
            raise ActionRejected(
                ReasonCode.INVALID_REQUEST, f"Unknown object '{name}'. On the table: {known}{hint}"
            )
        return {
            "description": object_description(name, self.layout),
            "others": [object_description(other, self.layout) for other in positions if other != name],
            "truth": (truth_of or truth_labels)(self, name),
        }

    def locate(
        self,
        name: str | None = None,
        *,
        view: str = "front",
        description: str | None = None,
        rest_z_m: float | None = None,
        client_id: str = "agent",
        width: int = LOCATE_SIZE[0],
        height: int = LOCATE_SIZE[1],
        timeout_s: float = LOCATE_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Find an object by looking: ask the pointer for its pixel, return xyz.

        Takes a picture from ``view``, asks :attr:`pointer` where ``description``
        (by default the object's own, e.g. "red cube") is, and follows that
        pixel's ray down to the plane the object's center sits at when it
        rests on ``rest_z_m`` (default: the table top). The guess is scored
        like :meth:`report_estimate` (``source`` = ``locate:<backend>:<view>``),
        and the answer, which is only the model's estimate, goes back to the
        caller.

        Blocks while the model answers, so call it from a client thread, not
        the loop thread. Keep the arm still: the wrist view's pose is read
        just before the picture is taken.

        Returns ``{"ok": True, "xyz_m": [...], "pixel": [u, v], ...}`` or
        ``{"ok": False, "reason": ..., "detail": ...}``.

        Without ``name``, ``description`` alone says what to look for (the
        request's words: "red cube"). The guess is then scored against the
        object that matches it, if one does; nothing is returned about which.
        """
        pointer = self.pointer
        if pointer is None:
            return {
                "ok": False,
                "reason": "no_pointer",
                "detail": "this twin has no pointing model (start it with --pointer)",
            }
        if name is None and not description:
            return {
                "ok": False,
                "reason": "invalid_request",
                "detail": 'say what to look for: a name, or description="red cube"',
            }
        try:
            self._log(
                "request",
                client_id=client_id,
                type="locate",
                name=name,
                description=description,
                view=view,
            )
            setup = self.call(
                lambda: self._locate_setup(name, view, description, rest_z_m, width, height)
            ).result(timeout=timeout_s)
            image = self.render(
                view, width=width, height=height, fmt="png", client_id=client_id
            ).result(timeout=timeout_s)
        except ActionRejected as exc:
            return {"ok": False, "reason": exc.reason.value, "detail": exc.detail}
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            return {"ok": False, "reason": "error", "detail": str(exc)}

        geometry: CameraGeometry = setup["geometry"]
        request = PointRequest(
            image=image,
            mime="image/png",
            width=width,
            height=height,
            description=setup["description"],
            truth_uv=setup["truth_uv"],
        )
        started = time.monotonic()
        try:
            point = pointer.point(request)
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            detail = mask_secrets(str(exc))
            self._log(
                "locate_result", client_id=client_id, ok=False, backend=pointer.name, detail=detail
            )
            return {"ok": False, "reason": "pointer_error", "detail": detail}
        seconds = round(time.monotonic() - started, 3)
        answer = {
            **({"name": name} if name is not None else {}),
            "view": view,
            "description": setup["description"],
            "backend": pointer.name,
            "seconds": seconds,
        }
        if point is None:
            self._log("locate_result", client_id=client_id, ok=False, **answer)
            return {
                "ok": False,
                "reason": "not_found",
                "detail": f"the {pointer.name} model did not find the {setup['description']} "
                f"in the {view} view; try another view",
                **answer,
            }
        xyz = pixel_to_xyz(geometry, point.u, point.v, setup["plane_z_m"])
        answer.update(pixel=[round(point.u, 1), round(point.v, 1)], image_size=[width, height])
        if xyz is None:
            self._log("locate_result", client_id=client_id, ok=False, **answer)
            return {
                "ok": False,
                "reason": "not_found",
                "detail": "the pixel it pointed at does not look down at the table",
                **answer,
            }
        source = f"locate:{pointer.name}:{view}"
        scored = setup["matched"]
        if scored is not None:
            try:
                self.call(
                    lambda: self._report_estimate(scored, xyz.tolist(), client_id, source)
                ).result(timeout=timeout_s)
            except Exception:  # noqa: BLE001 - the object vanished meanwhile: still answer
                pass
        answer.update(xyz_m=[round(float(v), 4) for v in xyz], plane_z_m=round(setup["plane_z_m"], 4))
        if setup["jaw_center"] is not None:
            answer["jaw_move_m"] = self._jaw_move(np.asarray(setup["jaw_center"], dtype=float), xyz)
        self._log(
            "locate_result",
            client_id=client_id,
            ok=True,
            raw=point.raw[:300],
            usage=getattr(pointer, "last_usage", None),
            **answer,
        )
        return {"ok": True, **answer}

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
        """Tell agents about something; the log always gets the full text."""
        self.events.append(hide_event_position(text) if self.hide_positions else text)
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
        layouts = {layout.name: layout for layout in layouts_for(self.sim.scene, include_hidden=True)}
        if name not in layouts:
            known = ", ".join(scene["name"] for scene in self.list_scenes())
            raise ActionRejected(ReasonCode.INVALID_REQUEST, f"Unknown scene '{name}'. Known: {known}")
        self._ensure_owner(client_id)
        self.layout = layouts[name]
        self._reset_scene()
        self._event(f"scene loaded: {name}")
        return {**self.layout.summary(), "current": True}

    def _report_estimate(
        self, name: str, xyz_m, client_id: str, source: str = "agent"
    ) -> dict[str, Any]:
        positions = self.grasp.positions()  # parked objects are left out
        if name not in positions:
            known = ", ".join(positions) or "none"
            raise ActionRejected(
                ReasonCode.INVALID_REQUEST, f"Unknown object '{name}'. On the table: {known}"
            )
        try:
            guess = np.asarray(xyz_m, dtype=float).reshape(3)
        except (TypeError, ValueError) as exc:
            raise ActionRejected(
                ReasonCode.INVALID_REQUEST, "xyz_m must be three numbers (m, base frame)"
            ) from exc
        if not np.all(np.isfinite(guess)):
            raise ActionRejected(ReasonCode.INVALID_REQUEST, "xyz_m must be finite")
        truth = np.asarray(positions[name], dtype=float)
        error = guess - truth
        score = {
            "t": round(self.now_s, 6),
            "frame": self.frame,
            "client_id": client_id,
            "name": name,
            "source": source,
            "estimate_xyz_m": [float(v) for v in guess],
            "truth_xyz_m": [round(float(v), 5) for v in truth],
            "error_cm": round(float(np.linalg.norm(error)) * 100.0, 3),
            "error_xy_cm": round(float(np.linalg.norm(error[:2])) * 100.0, 3),
            "error_z_cm": round(float(error[2]) * 100.0, 3),
            "attached": name == self.grasp.attached,
            "positions_hidden": self.hide_positions,
        }
        self._estimates.append(score)
        self._log(
            "estimate",
            client_id=client_id,
            **{k: v for k, v in score.items() if k not in ("t", "frame", "client_id")},
        )
        return {"ok": True, "name": name, "recorded": len(self._estimates)}

    def _jaw_move(self, jaw: np.ndarray, xyz: np.ndarray) -> dict[str, float]:
        """The move_by that puts the jaw center on an estimate (from the estimate, not the truth).

        Down only as far as ``JAW_FLOOR_CLEARANCE_M`` above the workspace floor,
        where the jaw tips would touch the table.
        """
        floor = float(self.workspace_bounds["z"][0]) + JAW_FLOOR_CLEARANCE_M
        dx, dy = (float(v) for v in (xyz[:2] - jaw[:2]))
        dz = max(float(xyz[2]), floor) - float(jaw[2])
        return {"dx": round(dx, 4) + 0.0, "dy": round(dy, 4) + 0.0, "dz": round(dz, 4) + 0.0}

    def _locate_setup(
        self,
        name: str | None,
        view: str,
        description: str | None,
        rest_z_m: float | None,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        """On the loop thread: check the request, read the camera and the truth."""
        positions = self.grasp.positions()
        if name is None:
            return self._locate_by_description(positions, view, description, rest_z_m, width, height)
        if name not in positions:
            known = ", ".join(positions) or "none"
            raise ActionRejected(
                ReasonCode.INVALID_REQUEST, f"Unknown object '{name}'. On the table: {known}"
            )
        if view not in VIEWS:
            raise ActionRejected(
                ReasonCode.INVALID_REQUEST, f"Unknown view '{view}'. Known: {', '.join(VIEWS)}"
            )
        if name == self.grasp.attached:
            raise ActionRejected(
                ReasonCode.INVALID_REQUEST,
                f"{name} is held: it rides between the jaws, at jaw_center_xyz_m",
            )
        if rest_z_m is None:
            rest_z_m = self.table_top_z if self.table_top_z is not None else 0.0
        plane_z = float(rest_z_m) + float(self.grasp.half_extents_world(name)[2])
        mujoco.mj_kinematics(self.sim.model, self.sim.data)
        mujoco.mj_camlight(self.sim.model, self.sim.data)
        geometry = camera_geometry(self.sim.model, self.sim.data, view, width, height)
        return {
            "description": description or object_description(name, self.layout),
            "plane_z_m": plane_z,
            "geometry": geometry,
            "truth_uv": geometry.project(positions[name]),
            "jaw_center": self._snapshot.get("jaw_center_xyz_m"),
            "matched": name,
        }

    def _locate_by_description(
        self, positions, view: str, description: str, rest_z_m, width: int, height: int
    ) -> dict[str, Any]:
        """Setup for looking by description only. The object that matches it is
        found for scoring and for the oracle pointer only; the plane it rests on
        is guessed from the typical height of what is on the table, since the
        arm does not know which object the words mean."""
        if view not in VIEWS:
            raise ActionRejected(
                ReasonCode.INVALID_REQUEST, f"Unknown view '{view}'. Known: {', '.join(VIEWS)}"
            )
        loose = [n for n in positions if n != self.grasp.attached]
        wanted = description.strip().lower()
        matched = next((n for n in loose if object_description(n, self.layout).lower() == wanted), None)
        if rest_z_m is None:
            rest_z_m = self.table_top_z if self.table_top_z is not None else 0.0
        halves = [float(self.grasp.half_extents_world(n)[2]) for n in loose] or [0.02]
        plane_z = float(rest_z_m) + float(np.median(halves))
        mujoco.mj_kinematics(self.sim.model, self.sim.data)
        mujoco.mj_camlight(self.sim.model, self.sim.data)
        geometry = camera_geometry(self.sim.model, self.sim.data, view, width, height)
        return {
            "description": description.strip(),
            "plane_z_m": plane_z,
            "geometry": geometry,
            "truth_uv": geometry.project(positions[matched]) if matched else None,
            "jaw_center": self._snapshot.get("jaw_center_xyz_m"),
            "matched": matched,
        }

    def _arrange_objects(self) -> None:
        """Put the layout's objects at their starts and park the others."""
        if self.layout is None:
            return
        for name, pose in self.layout.objects.items():
            # (x, y, z), or (x, y, z, yaw) for an object that starts turned.
            yaw = float(pose[3]) if len(pose) > 3 else 0.0
            self.grasp.place(name, np.asarray(pose[:3], dtype=float), yaw)
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
            self.grasp.carry()
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
        goal = GOALS.get(self.sim.scene)
        # A scene with several bins (a sorting scene) says which bin each object is in.
        bins = BINS.get(self.sim.scene, ()) if goal is None else ()
        opening = self.grasp.settings.jaw_opening_m(self.state.gripper_pos)
        jaw_center = self.grasp.jaw_center(opening)
        objects = {}
        for name, xyz in self.grasp.positions().items():
            entry = {
                "xyz_m": [round(float(v), 4) for v in xyz],
                "size_m": [round(float(v), 4) for v in self.grasp.size(name)],
                # Its size across the jaws as the gripper is turned now: close to this.
                "grip_width_m": round(self.grasp.fit(name)[0], 4),
                "attached": name == self.grasp.attached,
                "off_table": self.table_top_z is not None
                and float(xyz[2]) < self.table_top_z - OFF_TABLE_MARGIN_M,
            }
            if goal is not None:
                entry["in_goal"] = name != self.grasp.attached and goal.holds(
                    xyz, self.grasp.half_extents_world(name)
                )
            if bins:
                half = self.grasp.half_extents_world(name)
                entry["in_bin"] = next(
                    (box.name for box in bins if name != self.grasp.attached and box.holds(xyz, half)),
                    None,
                )
            objects[name] = entry
        scene = {
            "name": self.layout.name if self.layout is not None else self.sim.scene,
            "object_positions": "hidden" if self.hide_positions else "shown",
            "objects": objects,
        }
        if goal is not None:
            scene["goal"] = goal.summary()
        if bins:  # where each object should go is the request's business, not the scene's
            scene["bins"] = [box.summary() for box in bins]
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
            scene=scene,
            control={
                "stop_latched_by": self.stop_latched_by,
                "human_in_control": self._human_in_control(self.now_s),
                "controller_connected": self.controller_connected,
                "events": list(self.events),
            },
            gripper_opening_m=round(opening, 4),
            jaw_center_xyz_m=(
                [round(float(v), 4) for v in jaw_center] if jaw_center is not None else None
            ),
        )
        return observation.to_dict()


# ------------------------------------------------- no-coordinates mode helpers
# Object fields that give away where it is (or follow from where it is).
POSITION_FIELDS = ("xyz_m", "in_goal", "in_bin", "off_table")

_MISSED = re.compile(r"^(?P<name>\S+) missed(?: \(\d+ cm away\))?: ")


def verdict_for(p_in_goal: float) -> str:
    """``done``, ``look_again`` or ``retry`` for the judge's P(in the goal box)."""
    if p_in_goal >= DONE_AT:
        return "done"
    if p_in_goal <= RETRY_AT:
        return "retry"
    return "look_again"


def hide_object_positions(observation: dict[str, Any]) -> dict[str, Any]:
    """A copy of an observation without object positions (no-coordinates mode).

    Sizes, ``grip_width_m`` and ``attached`` stay: the arm knows what it was
    told to pick up and can feel whether it holds something. The goal box,
    the arm's own pose and the jaw center stay too.
    """
    scene = observation["scene"]
    objects = {
        name: {k: v for k, v in entry.items() if k not in POSITION_FIELDS}
        for name, entry in scene["objects"].items()
    }
    return {
        **observation,
        "scene": {**scene, "object_positions": "hidden", "objects": objects},
    }


def hide_event_position(text: str) -> str:
    """An event as an agent may see it when positions are hidden.

    A refused grasp's "missed" event says exactly how far to move, which with
    the jaw center gives the object's position away; this keeps only the fact.
    """
    match = _MISSED.match(text)
    if match is None:
        return text
    return (
        f"{match['name']} missed: it is not between the jaw pads. Open wider, lift the jaws "
        "clear (move_by dz=0.05), locate it again (or look with camera), move the jaws by the "
        "result's jaw_move_m across, then straight down"
    )
