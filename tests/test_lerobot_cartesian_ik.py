"""Tests for the LeRobot cartesian IK processor step."""

from __future__ import annotations

import numpy as np

from xbox_soarm_teleop.config.joints import HOME_POSITION_DEG, IK_JOINT_NAMES
from xbox_soarm_teleop.lerobot_steps.cartesian_ik import SoArmCartesianIKProcessor


class _FakeKinematics:
    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[0, 3] = float(joints[0])
        pose[1, 3] = float(joints[1])
        pose[2, 3] = float(joints[2])
        return pose

    def inverse_kinematics(self, joints: np.ndarray, target_pose: np.ndarray, **kwargs) -> np.ndarray:
        return np.array(
            [
                target_pose[0, 3],
                target_pose[1, 3],
                target_pose[2, 3],
                0.0,
            ],
            dtype=float,
        )


def test_cartesian_ik_processor_homing_is_rate_limited() -> None:
    proc = SoArmCartesianIKProcessor.__new__(SoArmCartesianIKProcessor)
    proc.dt = 0.1
    proc.swap_xy = True
    proc.strict_safety = False
    proc.max_linear_speed = 0.02
    proc.max_angular_speed = 0.25
    proc.allow_orientation = False
    proc.orientation_weight = 0.1
    proc.position_weight = 1.0
    proc.ik_vel_scale = 1.0
    proc.gripper_rate = 2.0
    proc.seed_from_observation = False
    proc._kinematics = _FakeKinematics()
    proc._workspace_limits = {}
    proc._ik_joint_vel_limits = np.ones(4, dtype=float)
    proc._ik_joint_pos_deg = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    proc._wrist_roll_deg = 30.0
    proc._target_pitch = 1.0
    proc._target_yaw = -1.0
    proc._gripper_pos = 1.0
    proc._homing_active = False
    proc.last_target_pose = None
    proc.last_obs_pose = None
    proc.last_delta = None
    proc.last_flags = {}
    proc._current_transition = {"observation": {}}

    proc.start_homing()
    before = proc._ik_joint_pos_deg.copy()
    proc.action({})

    assert proc._homing_active is True
    home = np.array([HOME_POSITION_DEG[name] for name in IK_JOINT_NAMES], dtype=float)
    assert np.all(np.abs(proc._ik_joint_pos_deg - home) < np.abs(before - home))
    assert np.all(np.abs(proc._ik_joint_pos_deg - before) <= 0.1 + 1e-9)
    assert proc._wrist_roll_deg == 21.0
    assert proc._gripper_pos == 0.8
    assert proc._target_pitch < 1.0
    assert proc._target_yaw > -1.0
