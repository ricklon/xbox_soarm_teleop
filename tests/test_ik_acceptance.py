"""IK must explicitly distinguish a valid solution from a held command."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("placo")
from lerobot.model.kinematics import RobotKinematics as UpstreamKinematics

from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES
from xbox_soarm_teleop.kinematics.constrained import RobotKinematics


@pytest.fixture
def model():
    return RobotKinematics(
        str(Path(__file__).resolve().parents[1] / "assets/so101.urdf"), joint_names=IK_JOINT_NAMES
    )


Q = np.array([10.0, 30.0, -45.0, 20.0])


def test_unreachable_target_holds_seed_and_restores_model(model):
    original = model.forward_kinematics(Q, wrist_roll_deg=45).copy()
    target = original.copy()
    target[0, 3] = 10
    result = model.inverse_kinematics(Q, target, orientation_weight=0)
    np.testing.assert_array_equal(result, Q)
    assert not model.last_ik_result.accepted
    assert model.last_ik_result.reason in {"not_converged", "joint_limit"}
    np.testing.assert_allclose(model.robot.get_T_world_frame("gripper_frame_link"), original)


def test_reachable_target_reports_acceptance(model):
    target = model.forward_kinematics(Q + [1, -1, 1, 0]).copy()
    result = model.inverse_kinematics(Q, target, orientation_weight=0)
    assert model.last_ik_result.accepted
    assert model.last_ik_result.position_error_m <= 1e-5
    np.testing.assert_allclose(model.forward_kinematics(result)[:3, 3], target[:3, 3], atol=1e-5)


@pytest.mark.parametrize(
    "bad_result",
    [None, np.array([np.nan] * 4), np.array([0.0, 0.0]), np.array([1000.0, 0.0, 0.0, 0.0])],
)
def test_invalid_solver_output_holds_seed(model, monkeypatch, bad_result):
    target = model.forward_kinematics(Q).copy()
    target[0, 3] += 0.01
    monkeypatch.setattr(UpstreamKinematics, "inverse_kinematics", lambda *a, **kw: bad_result)
    np.testing.assert_array_equal(model.inverse_kinematics(Q, target), Q)
    assert not model.last_ik_result.accepted


def test_solver_exception_restores_state(model, monkeypatch):
    original = model.forward_kinematics(Q).copy()
    target = original.copy()
    target[0, 3] += 0.01

    def fail(*args, **kwargs):
        model.robot.set_joint("elbow_flex", 1)
        raise RuntimeError("solver failed")

    monkeypatch.setattr(UpstreamKinematics, "inverse_kinematics", fail)
    np.testing.assert_array_equal(model.inverse_kinematics(Q, target), Q)
    assert model.last_ik_result.reason == "solver_error"
    np.testing.assert_allclose(model.robot.get_T_world_frame("gripper_frame_link"), original)


@pytest.mark.parametrize("bad", [np.full((4, 4), np.nan), np.eye(3), np.zeros((4, 4))])
def test_invalid_target_rejected_before_solving(model, bad):
    with pytest.raises(ValueError):
        model.inverse_kinematics(Q, bad)


def test_stalled_solver_is_bounded_and_cannot_report_success(model, monkeypatch):
    target = model.forward_kinematics(Q).copy()
    target[0, 3] += 0.01
    calls = []

    def stalled(*args, **kwargs):
        calls.append(1)
        return Q.copy()

    monkeypatch.setattr(UpstreamKinematics, "inverse_kinematics", stalled)
    np.testing.assert_array_equal(model.inverse_kinematics(Q, target, orientation_weight=0), Q)
    assert len(calls) == 30
    assert model.last_ik_result.iterations == 30
    assert model.last_ik_result.reason == "not_converged"


def test_orientation_task_cannot_pass_on_position_alone(model, monkeypatch):
    target = model.forward_kinematics(Q).copy()
    target[:3, :3] = target[:3, :3] @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    monkeypatch.setattr(UpstreamKinematics, "inverse_kinematics", lambda *a, **kw: Q.copy())
    model.inverse_kinematics(Q, target, orientation_weight=0.1)
    assert not model.last_ik_result.accepted
    assert model.last_ik_result.orientation_error_rad == pytest.approx(np.pi / 2)
    model.inverse_kinematics(Q, target, orientation_weight=0)
    assert model.last_ik_result.accepted
