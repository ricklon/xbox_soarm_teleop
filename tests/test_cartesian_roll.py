"""Joint and roll commands must describe the same solved tool target."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("placo")
from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES
from xbox_soarm_teleop.control.cartesian import make_cartesian_state, solve_cartesian_motion
from xbox_soarm_teleop.kinematics.constrained import RobotKinematics
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta


@pytest.fixture
def model():
    return RobotKinematics(
        str(Path(__file__).resolve().parents[1] / "assets/so101.urdf"), joint_names=IK_JOINT_NAMES
    )


def test_translation_and_roll_reach_same_tool_target(model):
    q = np.array([10.0, 30.0, -45.0, 20.0])
    state = make_cartesian_state(model, q)
    target = state.ee_pose.copy()
    target[0, 3] += 0.001
    joints, roll = solve_cartesian_motion(
        state, model, target, EEDelta(droll=0.5), dt=0.1, orientation_weight=0.0
    )
    assert roll == pytest.approx(np.rad2deg(0.05))
    assert model.last_ik_result.accepted
    # Planning restores the current model and does not commit controller state.
    assert model.robot.get_joint("wrist_roll") == pytest.approx(0.0)
    pose = model.forward_kinematics(joints, wrist_roll_deg=roll)
    assert np.linalg.norm(pose[:3, 3] - target[:3, 3]) <= 1e-5
    assert state.wrist_roll_deg == 0.0
    assert model.robot.get_joint("wrist_roll") == pytest.approx(np.deg2rad(roll))


def test_rejected_coupled_target_holds_roll_too(model):
    q = np.array([10.0, 30.0, -45.0, 20.0])
    state = make_cartesian_state(model, q, wrist_roll_deg=20)
    target = state.ee_pose.copy()
    target[0, 3] = 10
    joints, roll = solve_cartesian_motion(
        state, model, target, EEDelta(droll=0.5), dt=0.1, orientation_weight=0.0
    )
    np.testing.assert_array_equal(joints, q)
    assert roll == 20
    assert model.robot.get_joint("wrist_roll") == pytest.approx(np.deg2rad(20))


def test_velocity_limited_step_scales_roll_and_joints_together(model):
    q = np.array([10.0, 30.0, -45.0, 20.0])
    state = make_cartesian_state(model, q)
    target = state.ee_pose.copy()
    target[0, 3] += 0.001
    delta = EEDelta(droll=0.5)
    full, full_roll = solve_cartesian_motion(state, model, target, delta, dt=0.1, orientation_weight=0)
    limited, roll = solve_cartesian_motion(
        state, model, target, delta, dt=0.1, orientation_weight=0, max_joint_step_deg=np.full(4, 0.001)
    )
    assert np.max(np.abs(limited - q)) <= 0.001 + 1e-12
    np.testing.assert_allclose(limited - q, (full - q) * (roll / full_roll), atol=1e-12)


def test_recording_processor_uses_coupled_motion(model):
    from xbox_soarm_teleop.lerobot_steps.cartesian_ik import SoArmCartesianIKProcessor

    processor = SoArmCartesianIKProcessor(
        urdf_path=str(Path(__file__).resolve().parents[1] / "assets/so101.urdf"),
        strict_safety=False,
        seed_from_observation=False,
    )
    q = np.array([10.0, 30.0, -45.0, 20.0])
    processor._ik_joint_pos_deg = q.copy()
    processor._wrist_roll_deg = 0.0
    processor._workspace_limits = {axis: (-1.0, 1.0) for axis in "xyz"}
    processor._current_transition = {"observation": {}}
    processor.action({"dx": 0.01, "droll": 0.5})
    assert processor._kinematics.last_ik_result.accepted
    assert processor._wrist_roll_deg > 0
    achieved = model.forward_kinematics(
        processor._ik_joint_pos_deg, wrist_roll_deg=processor._wrist_roll_deg
    )
    assert np.linalg.norm(achieved[:3, 3] - processor.last_target_pose[:3, 3]) <= 1e-5


def test_clamp_to_ik_limits_uses_asymmetric_configured_limits():
    from xbox_soarm_teleop.config.joints import JOINT_LIMITS_DEG
    from xbox_soarm_teleop.control.cartesian import clamp_to_ik_limits

    clamped = clamp_to_ik_limits(np.array([180.0, 90.0, 90.0, 90.0]))
    # shoulder_pan ~+-110, shoulder_lift ~+-100, elbow ~+-96.8, wrist_flex ~[-95, 72]
    np.testing.assert_allclose(
        clamped,
        [JOINT_LIMITS_DEG["shoulder_pan"][1], 90.0, 90.0, JOINT_LIMITS_DEG["wrist_flex"][1]],
    )
    assert clamped[3] < 72.0 + 1e-6


def test_out_of_range_seed_recovers_instead_of_raising(model):
    """Jacobian mode could leave wrist_flex past its 72 deg limit; IK must not crash."""
    q = np.array([10.0, 30.0, -45.0, 90.0])
    state = make_cartesian_state(model, q)
    target = state.ee_pose.copy()
    step = np.full(4, 5.0)
    joints, roll = solve_cartesian_motion(
        state, model, target, EEDelta(droll=0.5), dt=0.1, orientation_weight=0.0,
        max_joint_step_deg=step,
    )
    assert np.all(np.isfinite(joints))
    # Motion from the true (out-of-range) position stays rate limited...
    assert np.all(np.abs(joints - q) <= step + 1e-9)
    # ...and never moves further out of range.
    assert joints[3] <= q[3]
    assert np.isfinite(roll)
