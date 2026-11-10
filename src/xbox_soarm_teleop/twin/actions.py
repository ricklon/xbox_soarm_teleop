"""Goal actions for agents: validation, per-tick commands and the state machine.

An action is a goal executed over many control ticks (design §4). Each tick the
executor turns the goal into an ``EEDelta`` -- the same input shape a joystick
produces -- which the shared ``cartesian_step`` executes. The step's limit
feedback drives the state machine (design §5):

    EXECUTING -> SUCCEEDED | HELD -> (recovers) | FAILED(reason)
    any       -> CANCELLED | STOPPED | FAILED(timeout)
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

import numpy as np

from xbox_soarm_teleop.config.joints import JOINT_LIMITS_DEG
from xbox_soarm_teleop.control.step import CartesianStepResult
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta
from xbox_soarm_teleop.schemas.agent import (
    ActionRequest,
    ActionState,
    ActionStatus,
    ReasonCode,
    reason_from_feedback,
)

LINEAR_SPEED_MAX_M_S = 0.1  # matches the Xbox full-stick speed
APPROACH_GAIN_PER_S = 2.0  # slow down within ~speed/gain of the goal
GOAL_TOLERANCE_M = 0.002
ROLL_SPEED_MAX_RAD_S = 1.0
ROLL_TOLERANCE_RAD = np.deg2rad(0.5)
GRIPPER_TOLERANCE = 0.01
HOLD_TIMEOUT_S = 1.0  # a limit blocking progress this long fails the action
TIMEOUT_CAP_S = 30.0

_ids = itertools.count(1)


class ActionRejected(Exception):
    """A request refused at submission (never started)."""

    def __init__(self, reason: ReasonCode, detail: str = "") -> None:
        super().__init__(f"{reason.value}{': ' + detail if detail else ''}")
        self.reason = reason
        self.detail = detail


@dataclass
class Action:
    """A submitted action and its live status."""

    request: ActionRequest
    client_id: str
    started_s: float
    deadline_s: float
    speed_m_s: float = LINEAR_SPEED_MAX_M_S
    target_xyz: np.ndarray | None = None
    start_distance: float = 0.0
    id: str = field(default_factory=lambda: f"a-{next(_ids)}")
    state: ActionState = ActionState.EXECUTING
    reason: ReasonCode | None = None
    detail: str = ""
    progress: float = 0.0
    held_since_s: float | None = None
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def type(self) -> str:
        return self.request.type

    def status(self) -> ActionStatus:
        return ActionStatus(
            id=self.id,
            type=self.type,
            state=self.state,
            reason=self.reason,
            detail=self.detail,
            progress=round(self.progress, 3),
        )

    def finish(self, state: ActionState, reason: ReasonCode | None = None, detail: str = "") -> None:
        """Move to a terminal state (once) and wake waiters."""
        if self.state.terminal:
            return
        self.state, self.reason, self.detail = state, reason, detail
        if state is ActionState.SUCCEEDED:
            self.progress = 1.0
        self.done.set()


def _check_workspace(xyz: np.ndarray, bounds: dict) -> None:
    for idx, axis in enumerate("xyz"):
        low, high = bounds.get(axis, (None, None))
        if low is not None and xyz[idx] < low:
            raise ActionRejected(ReasonCode.WORKSPACE_BOUND, f"{axis}_min ({axis} >= {low} m)")
        if high is not None and xyz[idx] > high:
            raise ActionRejected(ReasonCode.WORKSPACE_BOUND, f"{axis}_max ({axis} <= {high} m)")


def start_action(
    request: ActionRequest,
    *,
    client_id: str,
    now_s: float,
    tool_xyz: np.ndarray,
    wrist_roll_rad: float,
    workspace_bounds: dict,
) -> Action:
    """Validate a request against the current state and create the action.

    Raises:
        ActionRejected: Target outside the enforced workspace.
    """
    speed = LINEAR_SPEED_MAX_M_S
    if request.max_speed_m_s is not None:
        speed = min(speed, request.max_speed_m_s)
    target = None
    default_timeout = 3.0
    if request.type in {"move_ee_to", "move_ee_by"}:
        if request.type == "move_ee_to":
            target = np.array(request.params["xyz_m"], dtype=float)
        else:
            target = tool_xyz + np.array(request.params["dxyz_m"], dtype=float)
        _check_workspace(target, workspace_bounds)
        default_timeout = float(np.linalg.norm(target - tool_xyz)) / speed * 2.0 + 2.0
    elif request.type == "set_wrist_roll":
        low, high = np.deg2rad(JOINT_LIMITS_DEG["wrist_roll"])
        if not low <= request.params["angle_rad"] <= high:
            raise ActionRejected(ReasonCode.JOINT_LIMIT, f"wrist_roll ({low:.2f}..{high:.2f} rad)")
        default_timeout = abs(request.params["angle_rad"] - wrist_roll_rad) / ROLL_SPEED_MAX_RAD_S + 2.0
    elif request.type == "home":
        default_timeout = 10.0
    elif request.type == "hold":
        default_timeout = request.params["duration_s"] + 1.0
    timeout = request.timeout_s if request.timeout_s is not None else default_timeout
    action = Action(
        request=request,
        client_id=client_id,
        started_s=now_s,
        deadline_s=now_s + min(timeout, TIMEOUT_CAP_S),
        speed_m_s=speed,
        target_xyz=target,
    )
    if target is not None:
        action.start_distance = float(np.linalg.norm(target - tool_xyz))
    return action


def command_for(
    action: Action, *, tool_xyz: np.ndarray, wrist_roll_rad: float, gripper: float
) -> EEDelta:
    """The end-effector command this tick (gripper = the agent's held target)."""
    if action.target_xyz is not None:
        error = action.target_xyz - tool_xyz
        distance = float(np.linalg.norm(error))
        if distance <= GOAL_TOLERANCE_M:
            return EEDelta(gripper=gripper)
        speed = min(action.speed_m_s, APPROACH_GAIN_PER_S * distance)
        velocity = error / distance * speed
        return EEDelta(dx=velocity[0], dy=velocity[1], dz=velocity[2], gripper=gripper)
    if action.type == "set_wrist_roll":
        error = action.request.params["angle_rad"] - wrist_roll_rad
        if abs(error) <= ROLL_TOLERANCE_RAD:
            return EEDelta(gripper=gripper)
        droll = float(np.clip(APPROACH_GAIN_PER_S * error, -ROLL_SPEED_MAX_RAD_S, ROLL_SPEED_MAX_RAD_S))
        return EEDelta(droll=droll, gripper=gripper)
    return EEDelta(gripper=gripper)


def update_action(
    action: Action,
    step: CartesianStepResult,
    *,
    now_s: float,
    tool_xyz: np.ndarray,
    wrist_roll_rad: float,
    gripper: float,
    gripper_target: float,
) -> None:
    """Advance the state machine after the control step ran."""
    if action.state.terminal:
        return
    if action.target_xyz is not None:
        distance = float(np.linalg.norm(action.target_xyz - tool_xyz))
        if action.start_distance > 0:
            action.progress = float(np.clip(1.0 - distance / action.start_distance, 0.0, 1.0))
        if distance <= GOAL_TOLERANCE_M:
            action.finish(ActionState.SUCCEEDED)
            return
        reason, detail = reason_from_feedback(step.limit, step.ik_result)
        if reason is not None:
            if action.held_since_s is None:
                action.held_since_s = now_s
            action.state, action.reason, action.detail = ActionState.HELD, reason, detail
            if now_s - action.held_since_s >= HOLD_TIMEOUT_S:
                action.finish(ActionState.FAILED, reason, detail)
                return
        else:
            action.held_since_s = None
            action.state, action.reason, action.detail = ActionState.EXECUTING, None, ""
    elif action.type == "set_wrist_roll":
        if abs(action.request.params["angle_rad"] - wrist_roll_rad) <= ROLL_TOLERANCE_RAD:
            action.finish(ActionState.SUCCEEDED)
            return
    elif action.type == "set_gripper":
        if abs(gripper - gripper_target) <= GRIPPER_TOLERANCE:
            action.finish(ActionState.SUCCEEDED)
            return
    elif action.type == "home":
        if not step.homing:
            action.finish(ActionState.SUCCEEDED)
            return
    elif action.type == "hold":
        duration = action.request.params["duration_s"]
        action.progress = float(np.clip((now_s - action.started_s) / duration, 0.0, 1.0))
        if now_s - action.started_s >= duration:
            action.finish(ActionState.SUCCEEDED)
            return
    if now_s >= action.deadline_s:
        action.finish(ActionState.FAILED, ReasonCode.TIMEOUT)
