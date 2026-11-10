"""Agent-facing observation and action schemas.

Contracts for the agent control interface and MCP server
(``docs/mcp_control_interface.md``, ``docs/schemas.md``):

- ``soarm.observation/v1`` — one snapshot of robot and controller state.
- ``soarm.action/v1``      — a goal request (``move_ee_to``, ``home``, ...).

Units on the wire are SI (m, rad, s) except joint angles, which are degrees to
match the rest of the codebase. Positions are in the robot base frame (+x
forward, +z up). This module validates request *shape* (types, finiteness,
ranges that do not depend on the robot); workspace bounds are enforced by the
executor, which knows the limits in force.

Change the schema IDs (``/v2``) for any incompatible change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from xbox_soarm_teleop.config.modes import ControlMode

OBSERVATION_SCHEMA_ID = "soarm.observation/v1"
ACTION_SCHEMA_ID = "soarm.action/v1"


class ActionState(str, Enum):
    """Lifecycle state of an action (see design §5)."""

    IDLE = "IDLE"
    EXECUTING = "EXECUTING"
    HELD = "HELD"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    STOPPED = "STOPPED"

    @property
    def terminal(self) -> bool:
        """True once the action can no longer change state."""
        return self in {
            ActionState.SUCCEEDED,
            ActionState.FAILED,
            ActionState.CANCELLED,
            ActionState.STOPPED,
        }


class ReasonCode(str, Enum):
    """Machine-readable reason attached to HELD/FAILED/rejected outcomes."""

    WORKSPACE_BOUND = "workspace_bound"
    JOINT_LIMIT = "joint_limit"
    OUT_OF_REACH = "out_of_reach"
    IK_ERROR = "ik_error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    STOPPED = "stopped"
    PREEMPTED = "preempted"
    INVALID_REQUEST = "invalid_request"
    NOT_OWNER = "not_owner"
    BUSY = "busy"


# Action catalogue: parameter specs are (kind, required). Kinds: "vec3", "number", "unit".
ACTION_TYPES: dict[str, dict[str, Any]] = {
    "move_ee_to": {
        "description": "Move the tool point to an absolute position (m, base frame).",
        "params": {"xyz_m": ("vec3", True)},
    },
    "move_ee_by": {
        "description": "Move the tool point by a relative offset (m, base frame).",
        "params": {"dxyz_m": ("vec3", True)},
    },
    "set_wrist_roll": {
        "description": "Rotate the wrist roll joint to an absolute angle (rad).",
        "params": {"angle_rad": ("number", True)},
    },
    "set_gripper": {
        "description": "Set the gripper position, 0 = open .. 1 = closed.",
        "params": {"position": ("unit", True)},
    },
    "home": {
        "description": "Return to the home pose (joint-space move).",
        "params": {},
    },
    "hold": {
        "description": "Hold the current pose for a duration (s).",
        "params": {"duration_s": ("positive", True)},
    },
}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _parse_param(kind: str, value: Any, name: str) -> Any:
    if kind == "vec3":
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name} must be a list of 3 values")
        return tuple(_finite(v, name) for v in value)
    if kind == "number":
        return _finite(value, name)
    if kind == "positive":
        return _positive(value, name)
    if kind == "unit":
        number = _finite(value, name)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be within 0..1")
        return number
    raise AssertionError(f"unknown param kind {kind}")


@dataclass(frozen=True)
class ActionRequest:
    """A validated ``soarm.action/v1`` goal request.

    Attributes:
        type: Key of ``ACTION_TYPES``.
        params: Parsed parameters (vectors as tuples of floats).
        max_speed_m_s: Optional speed cap; can only lower configured limits.
        timeout_s: Optional timeout; the executor applies a default if None.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    max_speed_m_s: float | None = None
    timeout_s: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActionRequest:
        """Parse and validate a request payload.

        Raises:
            ValueError: Unknown type, missing/unexpected params, or invalid values.
        """
        action_type = payload.get("type")
        if action_type not in ACTION_TYPES:
            raise ValueError(
                f"Unknown action type '{action_type}'. Known: {', '.join(ACTION_TYPES)}"
            )
        spec = ACTION_TYPES[action_type]["params"]
        raw_params = payload.get("params") or {}
        if not isinstance(raw_params, dict):
            raise ValueError("params must be an object")
        unexpected = set(raw_params) - set(spec)
        if unexpected:
            raise ValueError(f"Unexpected params for {action_type}: {', '.join(sorted(unexpected))}")
        params: dict[str, Any] = {}
        for name, (kind, required) in spec.items():
            if name not in raw_params:
                if required:
                    raise ValueError(f"{action_type} requires param '{name}'")
                continue
            params[name] = _parse_param(kind, raw_params[name], name)

        max_speed = payload.get("max_speed_m_s")
        timeout = payload.get("timeout_s")
        return cls(
            type=action_type,
            params=params,
            max_speed_m_s=None if max_speed is None else _positive(max_speed, "max_speed_m_s"),
            timeout_s=None if timeout is None else _positive(timeout, "timeout_s"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a JSON-ready ``soarm.action/v1`` payload."""
        return {
            "schema": ACTION_SCHEMA_ID,
            "type": self.type,
            "params": {k: list(v) if isinstance(v, tuple) else v for k, v in self.params.items()},
            "max_speed_m_s": self.max_speed_m_s,
            "timeout_s": self.timeout_s,
        }


@dataclass(frozen=True)
class EEPose:
    """Tool pose: position (m) and roll/pitch/yaw (rad), robot base frame."""

    xyz_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]


@dataclass(frozen=True)
class IKStatus:
    """Outcome of the most recent IK solve (``kinematics.constrained.IKResult``)."""

    accepted: bool
    reason: str
    position_error_m: float


@dataclass(frozen=True)
class ActionStatus:
    """Status of the current or most recent action."""

    id: str
    type: str
    state: ActionState
    reason: ReasonCode | None = None
    detail: str = ""
    progress: float = 0.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    return value


@dataclass(frozen=True)
class Observation:
    """One ``soarm.observation/v1`` snapshot (design §8).

    Attributes:
        frame: Control tick counter.
        sim_time_s: Simulator (or wall) time in seconds.
        mode: Control mode value (``ControlMode``).
        ee: Tool pose.
        joints_deg: Arm joint angles in degrees, by joint name.
        gripper: Gripper position 0 (open) .. 1 (closed).
        stop_latched: Virtual stop is latched.
        owner: Client holding the control lease, if any.
        action: Current or most recent action status.
        last_ik: Most recent IK outcome, if IK ran.
        limit: Human-readable stop reason (``describe_motion_limit``), or "".
        workspace: Enforced workspace bounds ``{"x"|"y"|"z": (min, max)}`` in m.
        scene: Scene name and object positions.
    """

    frame: int
    sim_time_s: float
    mode: str
    ee: EEPose
    joints_deg: dict[str, float]
    gripper: float
    stop_latched: bool
    owner: str | None
    action: ActionStatus | None
    last_ik: IKStatus | None
    limit: str
    workspace: dict[str, tuple[float, float]]
    scene: dict[str, Any]

    def __post_init__(self) -> None:
        valid_modes = {m.value for m in ControlMode}
        if self.mode not in valid_modes:
            raise ValueError(f"Unknown mode '{self.mode}'. Known: {', '.join(sorted(valid_modes))}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a JSON-ready dict tagged with the schema ID."""
        data = {"schema": OBSERVATION_SCHEMA_ID}
        data.update(_jsonable(self))
        return data


# Mapping from motion_feedback limit messages to (reason, detail).
_LIMIT_REASONS = {
    "at floor": (ReasonCode.WORKSPACE_BOUND, "z_min"),
    "at ceiling": (ReasonCode.WORKSPACE_BOUND, "z_max"),
    "at front limit": (ReasonCode.WORKSPACE_BOUND, "x_max"),
    "at back limit": (ReasonCode.WORKSPACE_BOUND, "x_min"),
    "at side limit": (ReasonCode.WORKSPACE_BOUND, "y"),
    "out of reach": (ReasonCode.OUT_OF_REACH, ""),
    "joint limit": (ReasonCode.JOINT_LIMIT, ""),
    "IK error": (ReasonCode.IK_ERROR, ""),
}


def reason_from_feedback(limit: str, ik_result) -> tuple[ReasonCode | None, str]:
    """Map control-step feedback to a reason code and detail.

    Args:
        limit: Message from ``control.motion_feedback.describe_motion_limit``.
        ik_result: ``IKResult`` from this step, or None if IK did not run.

    Returns:
        ``(reason, detail)``; ``(None, "")`` when nothing is blocking motion.
    """
    if limit in _LIMIT_REASONS:
        return _LIMIT_REASONS[limit]
    if limit.startswith("joint limit: "):
        return ReasonCode.JOINT_LIMIT, limit.removeprefix("joint limit: ")
    if ik_result is not None and not ik_result.accepted:
        if ik_result.reason == "not_converged":
            return ReasonCode.OUT_OF_REACH, ""
        if ik_result.reason == "joint_limit":
            return ReasonCode.JOINT_LIMIT, ""
        return ReasonCode.IK_ERROR, ik_result.reason
    return None, ""
