"""Tests for shared cartesian control helpers."""

from __future__ import annotations

import numpy as np

from xbox_soarm_teleop.control.cartesian import (
    advance_cartesian_target,
    apply_ik_solution,
    full_joint_positions,
    make_cartesian_state,
    step_cartesian_home,
    step_gripper_toward,
    step_wrist_roll,
    sync_cartesian_state,
)
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta


class _FakeKinematics:
    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[0, 3] = float(joints[0])
        pose[1, 3] = float(joints[1])
        pose[2, 3] = float(joints[2])
        return pose


def test_make_and_sync_cartesian_state() -> None:
    kin = _FakeKinematics()
    state = make_cartesian_state(kin, np.array([0.1, 0.2, 0.3, 0.4]), wrist_roll_deg=5.0)
    assert state.ee_pose[0, 3] == 0.1
    assert state.wrist_roll_deg == 5.0

    sync_cartesian_state(
        state,
        kin,
        np.array([1.0, 2.0, 3.0, 4.0]),
        wrist_roll_deg=10.0,
        target_pitch=0.2,
        target_yaw=-0.3,
        gripper_pos=0.6,
    )
    assert state.ee_pose[0, 3] == 1.0
    assert state.ee_pose[1, 3] == 2.0
    assert state.target_pitch == 0.2
    assert state.target_yaw == -0.3
    assert state.gripper_pos == 0.6


def test_step_gripper_toward_and_roll() -> None:
    assert step_gripper_toward(0.0, 1.0, gripper_rate=2.0, dt=0.1) == 0.2
    assert step_gripper_toward(0.8, 1.0, gripper_rate=5.0, dt=0.1) == 1.0
    assert step_wrist_roll(179.0, np.deg2rad(20.0), dt=0.1) == 180.0
    assert step_wrist_roll(0.0, 0.0, dt=0.1, roll_target=np.deg2rad(45.0)) == 45.0


def test_advance_cartesian_target_and_apply_solution() -> None:
    kin = _FakeKinematics()
    state = make_cartesian_state(kin, np.array([0.1, 0.2, 0.3, 0.4]))
    delta = EEDelta(dx=1.0, dy=-1.0, dz=0.5, dpitch=10.0, dyaw=20.0)

    target_pose, target_pos, flags = advance_cartesian_target(
        state,
        delta,
        dt=0.1,
        clip_position=lambda pos: (np.clip(pos, -0.2, 0.2), {"clipped": True}),
        pitch_limit_rad=0.5,
        yaw_limit_rad=0.5,
    )

    assert np.allclose(target_pos, np.array([0.2, 0.1, 0.2]))
    assert flags["clipped"] is True
    assert flags["pitch_clipped"] is True
    assert flags["yaw_clipped"] is True
    assert flags["has_orientation_target"] is True
    assert flags["orientation_weight"] == 0.1
    assert not np.allclose(target_pose[:3, :3], np.eye(3))


def test_advance_cartesian_target_uses_absolute_orientation_targets() -> None:
    kin = _FakeKinematics()
    state = make_cartesian_state(kin, np.array([0.0, 0.0, 0.0, 0.0]))
    delta = EEDelta(
        dx=0.0,
        dy=0.0,
        dz=0.0,
        pitch_target=0.3,
        yaw_target=-0.4,
    )

    target_pose, _target_pos, flags = advance_cartesian_target(
        state,
        delta,
        dt=0.1,
        clip_position=lambda pos: (pos, {}),
        pitch_limit_rad=0.5,
        yaw_limit_rad=0.5,
    )

    assert state.target_pitch == 0.3
    assert state.target_yaw == -0.4
    assert flags["pitch_clipped"] is False
    assert flags["yaw_clipped"] is False
    assert flags["has_orientation_target"] is True
    assert not np.allclose(target_pose[:3, :3], np.eye(3))

    apply_ik_solution(
        state,
        kin,
        np.array([0.5, 0.6, 0.7, 0.8]),
        wrist_roll_deg=12.0,
        target_pose=target_pose,
    )
    assert state.ik_joint_pos_deg[0] == 0.5
    assert state.wrist_roll_deg == 12.0
    assert state.last_target_pose[0, 3] == target_pose[0, 3]


def test_full_joint_positions() -> None:
    joints = full_joint_positions(np.array([1.0, 2.0, 3.0, 4.0]), 5.0)
    assert np.allclose(joints, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_step_cartesian_home_is_rate_limited() -> None:
    kin = _FakeKinematics()
    state = make_cartesian_state(
        kin,
        np.array([1.0, 2.0, 3.0, 4.0]),
        wrist_roll_deg=30.0,
        target_pitch=1.0,
        target_yaw=-1.0,
        gripper_pos=1.0,
    )

    done = step_cartesian_home(
        state,
        kin,
        np.zeros(4, dtype=float),
        home_wrist_roll_deg=0.0,
        home_gripper_pos=0.0,
        ik_joint_max_step_deg=np.array([1.0, 1.0, 1.0, 1.0]),
        wrist_roll_vel_deg_s=10.0,
        gripper_rate=2.0,
        dt=0.1,
    )

    assert done is False
    assert np.allclose(state.ik_joint_pos_deg, np.array([0.9, 1.9, 2.9, 3.9]))
    assert state.wrist_roll_deg == 29.0
    assert state.gripper_pos == 0.8
    assert state.target_pitch < 1.0
    assert state.target_yaw > -1.0
