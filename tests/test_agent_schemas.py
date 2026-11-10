"""Agent-facing observation/action schemas (soarm.observation/v1, soarm.action/v1)."""

import json
import math

import pytest

from xbox_soarm_teleop.kinematics.constrained import IKResult
from xbox_soarm_teleop.schemas.agent import (
    ACTION_SCHEMA_ID,
    ACTION_TYPES,
    OBSERVATION_SCHEMA_ID,
    ActionRequest,
    ActionState,
    ActionStatus,
    EEPose,
    IKStatus,
    Observation,
    ReasonCode,
    reason_from_feedback,
)


def _observation(**overrides):
    kwargs = dict(
        frame=7,
        sim_time_s=0.14,
        mode="cartesian",
        ee=EEPose(xyz_m=(0.39, 0.0, 0.23), rpy_rad=(0.0, 0.0, 0.0)),
        joints_deg={"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
                    "wrist_flex": 0.0, "wrist_roll": 0.0},
        gripper=0.0,
        stop_latched=False,
        owner=None,
        action=None,
        last_ik=IKStatus(accepted=True, reason="accepted", position_error_m=0.0),
        limit="",
        workspace={"x": (0.05, 0.5), "y": (-0.3, 0.3), "z": (0.05, 0.45)},
        scene={"name": "default", "objects": {}},
    )
    kwargs.update(overrides)
    return Observation(**kwargs)


class TestObservation:
    def test_to_dict_is_json_and_tagged(self):
        data = _observation().to_dict()
        assert data["schema"] == OBSERVATION_SCHEMA_ID == "soarm.observation/v1"
        text = json.dumps(data)
        assert json.loads(text)["ee"]["xyz_m"] == [0.39, 0.0, 0.23]
        assert data["workspace"]["y"] == [-0.3, 0.3]

    def test_action_status_serializes_enums_as_strings(self):
        status = ActionStatus(id="a-1", type="move_ee_to", state=ActionState.HELD,
                              reason=ReasonCode.WORKSPACE_BOUND, detail="z_min", progress=0.4)
        data = _observation(action=status).to_dict()
        assert data["action"]["state"] == "HELD"
        assert data["action"]["reason"] == "workspace_bound"
        json.dumps(data)

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            _observation(mode="telepathy")


class TestActionState:
    def test_terminal_states(self):
        terminal = {s for s in ActionState if s.terminal}
        assert terminal == {ActionState.SUCCEEDED, ActionState.FAILED,
                            ActionState.CANCELLED, ActionState.STOPPED}


class TestActionRequest:
    def test_valid_move(self):
        req = ActionRequest.from_dict({"type": "move_ee_to", "params": {"xyz_m": [0.3, 0.0, 0.2]},
                                       "max_speed_m_s": 0.05, "timeout_s": 5})
        assert req.type == "move_ee_to"
        assert req.params["xyz_m"] == (0.3, 0.0, 0.2)
        assert req.to_dict()["schema"] == ACTION_SCHEMA_ID == "soarm.action/v1"

    def test_every_action_type_has_documented_params(self):
        assert set(ACTION_TYPES) == {"move_ee_to", "move_ee_by", "set_wrist_roll",
                                     "set_gripper", "home", "hold"}
        for spec in ACTION_TYPES.values():
            assert "description" in spec and "params" in spec

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"type": "teleport", "params": {}}, "Unknown action"),
            ({"type": "move_ee_to", "params": {}}, "xyz_m"),
            ({"type": "move_ee_to", "params": {"xyz_m": [0.3, 0.0]}}, "3 values"),
            ({"type": "move_ee_to", "params": {"xyz_m": [0.3, math.nan, 0.2]}}, "finite"),
            ({"type": "move_ee_to", "params": {"xyz_m": [0.3, 0.0, 0.2], "warp": 1}}, "Unexpected"),
            ({"type": "set_gripper", "params": {"position": 1.5}}, "0..1"),
            ({"type": "hold", "params": {"duration_s": -1}}, "positive"),
            ({"type": "move_ee_to", "params": {"xyz_m": [0.3, 0.0, 0.2]}, "max_speed_m_s": 0}, "positive"),
            ({"type": "home", "params": {}, "timeout_s": math.inf}, "finite"),
        ],
    )
    def test_invalid_requests_are_rejected(self, payload, match):
        with pytest.raises(ValueError, match=match):
            ActionRequest.from_dict(payload)

    def test_actions_without_params(self):
        assert ActionRequest.from_dict({"type": "home"}).params == {}


class TestReasonFromFeedback:
    @pytest.mark.parametrize(
        ("limit", "expected"),
        [
            ("at floor", (ReasonCode.WORKSPACE_BOUND, "z_min")),
            ("at ceiling", (ReasonCode.WORKSPACE_BOUND, "z_max")),
            ("at front limit", (ReasonCode.WORKSPACE_BOUND, "x_max")),
            ("at back limit", (ReasonCode.WORKSPACE_BOUND, "x_min")),
            ("at side limit", (ReasonCode.WORKSPACE_BOUND, "y")),
            ("joint limit: wrist_flex", (ReasonCode.JOINT_LIMIT, "wrist_flex")),
            ("out of reach", (ReasonCode.OUT_OF_REACH, "")),
            ("joint limit", (ReasonCode.JOINT_LIMIT, "")),
            ("IK error", (ReasonCode.IK_ERROR, "")),
        ],
    )
    def test_limit_messages(self, limit, expected):
        assert reason_from_feedback(limit, None) == expected

    def test_ik_results(self):
        assert reason_from_feedback("", IKResult(False, "not_converged", 30, 0.01, 0.0)) == (
            ReasonCode.OUT_OF_REACH, "")
        assert reason_from_feedback("", IKResult(False, "solver_error", 0, math.inf, 0.0)) == (
            ReasonCode.IK_ERROR, "solver_error")
        assert reason_from_feedback("", IKResult(True, "accepted", 1, 0.0, 0.0)) == (None, "")

    def test_no_feedback_and_homing_are_not_failures(self):
        assert reason_from_feedback("", None) == (None, "")
        assert reason_from_feedback("homing", None) == (None, "")


def test_every_motion_feedback_message_maps_to_a_reason():
    """Adding a new stop message to motion_feedback requires mapping it here."""
    from xbox_soarm_teleop.control import motion_feedback as mf

    messages = {label for *_, label in mf._BOUND_LABELS} | set(mf._IK_REASON_LABELS.values())
    messages.add("joint limit: shoulder_pan")
    for message in messages:
        reason, _ = reason_from_feedback(message, None)
        assert reason is not None, message
