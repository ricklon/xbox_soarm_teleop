"""Regression contracts for the production reduced-joint model."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("placo")

from xbox_soarm_teleop.control.cartesian import (
    apply_ik_solution,
    make_cartesian_state,
    step_cartesian_home,
    sync_cartesian_state,
)
from xbox_soarm_teleop.kinematics.constrained import RobotKinematics

URDF = str(Path(__file__).resolve().parents[1] / "assets" / "so101.urdf")
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]


@pytest.mark.parametrize(
    "names, joints", [(ARM, [10, 30, -45, 20]), (["shoulder_lift", "elbow_flex"], [30, -45])]
)
def test_solver_preserves_all_excluded_joints(names, joints):
    model = RobotKinematics(URDF, joint_names=names)
    excluded = set(model.robot.joint_names()) - set(names)
    fixed = {name: 0.15 for name in excluded}
    for name, angle in fixed.items():
        model.robot.set_joint(name, angle)
    q = np.array(joints, dtype=float)
    target = model.forward_kinematics(q).copy()
    target[0, 3] += 0.01
    for _ in range(10):
        q = model.inverse_kinematics(q, target, orientation_weight=0.0)
        for name, angle in fixed.items():
            assert model.robot.get_joint(name) == pytest.approx(angle, abs=1e-12)
    # A fresh model must reproduce the pose without hidden solver motion.
    independent = RobotKinematics(URDF, joint_names=names)
    for name, angle in fixed.items():
        independent.robot.set_joint(name, angle)
    np.testing.assert_allclose(
        model.robot.get_T_world_frame("gripper_frame_link"),
        independent.forward_kinematics(q),
        atol=1e-10,
    )


def test_cartesian_state_uses_roll_for_every_pose_update():
    model = RobotKinematics(URDF, joint_names=ARM)
    reference = RobotKinematics(URDF, joint_names=ARM + ["wrist_roll"])
    q = np.array([10.0, 30.0, -45.0, 20.0])
    state = make_cartesian_state(model, q, wrist_roll_deg=90)
    np.testing.assert_allclose(state.ee_pose, reference.forward_kinematics(np.r_[q, 90]))
    sync_cartesian_state(state, model, q, wrist_roll_deg=-45)
    np.testing.assert_allclose(state.ee_pose, reference.forward_kinematics(np.r_[q, -45]))
    apply_ik_solution(state, model, q, wrist_roll_deg=30)
    np.testing.assert_allclose(state.ee_pose, reference.forward_kinematics(np.r_[q, 30]))
    step_cartesian_home(
        state,
        model,
        q,
        home_wrist_roll_deg=0,
        home_gripper_pos=0,
        ik_joint_max_step_deg=np.ones(4),
        wrist_roll_vel_deg_s=10,
        gripper_rate=1,
        dt=0.1,
    )
    np.testing.assert_allclose(state.ee_pose, reference.forward_kinematics(np.r_[q, 29]))
    np.testing.assert_allclose(state.last_target_pose, reference.forward_kinematics(np.r_[q, 0]))
    assert model.robot.get_joint("wrist_roll") == pytest.approx(np.deg2rad(29))


@pytest.mark.parametrize("roll", [-90.0, 0.0, 90.0])
def test_reachable_target_converges_with_fixed_roll(roll):
    model = RobotKinematics(URDF, joint_names=ARM)
    reference = RobotKinematics(URDF, joint_names=ARM + ["wrist_roll"])
    q = np.array([10.0, 30.0, -45.0, 20.0])
    target = reference.forward_kinematics(np.r_[q + [1.0, -1.0, 1.0, 0.0], roll]).copy()
    model.forward_kinematics(q, wrist_roll_deg=roll)
    for _ in range(30):
        q = model.inverse_kinematics(q, target, orientation_weight=0.0)
    actual = reference.forward_kinematics(np.r_[q, roll])
    assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) < 1e-5
    assert model.robot.get_joint("wrist_roll") == pytest.approx(np.deg2rad(roll), abs=1e-12)
