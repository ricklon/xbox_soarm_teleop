"""Shared cartesian control state and step helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from xbox_soarm_teleop.control.home import (
    array_reached,
    scalar_reached,
    step_array_toward,
    step_scalar_toward,
)
from xbox_soarm_teleop.control.pose import euler_to_rotation_matrix
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta

PositionClipper = Callable[[np.ndarray], tuple[np.ndarray, dict[str, bool]]]


@dataclass
class CartesianControlState:
    """Mutable cartesian control state shared by runtime loops."""

    ik_joint_pos_deg: np.ndarray
    wrist_roll_deg: float
    target_pitch: float
    target_yaw: float
    ee_pose: np.ndarray
    last_target_pose: np.ndarray
    last_ee_pose: np.ndarray
    gripper_pos: float


def make_cartesian_state(
    kinematics,
    ik_joint_pos_deg: np.ndarray,
    *,
    wrist_roll_deg: float = 0.0,
    target_pitch: float = 0.0,
    target_yaw: float = 0.0,
    gripper_pos: float = 0.0,
) -> CartesianControlState:
    """Create cartesian control state from the current IK joint position."""
    joints = np.asarray(ik_joint_pos_deg, dtype=float).copy()
    ee_pose = kinematics.forward_kinematics(joints)
    return CartesianControlState(
        ik_joint_pos_deg=joints,
        wrist_roll_deg=float(wrist_roll_deg),
        target_pitch=float(target_pitch),
        target_yaw=float(target_yaw),
        ee_pose=ee_pose.copy(),
        last_target_pose=ee_pose.copy(),
        last_ee_pose=ee_pose.copy(),
        gripper_pos=float(gripper_pos),
    )


def sync_cartesian_state(
    state: CartesianControlState,
    kinematics,
    ik_joint_pos_deg: np.ndarray,
    *,
    wrist_roll_deg: float | None = None,
    target_pitch: float | None = None,
    target_yaw: float | None = None,
    gripper_pos: float | None = None,
) -> None:
    """Resync cartesian state from joint-space state."""
    state.ik_joint_pos_deg = np.asarray(ik_joint_pos_deg, dtype=float).copy()
    if wrist_roll_deg is not None:
        state.wrist_roll_deg = float(wrist_roll_deg)
    if target_pitch is not None:
        state.target_pitch = float(target_pitch)
    if target_yaw is not None:
        state.target_yaw = float(target_yaw)
    if gripper_pos is not None:
        state.gripper_pos = float(gripper_pos)
    state.ee_pose = kinematics.forward_kinematics(state.ik_joint_pos_deg)
    state.last_target_pose = state.ee_pose.copy()
    state.last_ee_pose = state.ee_pose.copy()


def step_gripper_toward(current: float, target: float, *, gripper_rate: float, dt: float) -> float:
    """Rate-limit gripper motion toward a target position."""
    delta = target - current
    max_delta = gripper_rate * dt
    if abs(delta) > max_delta:
        return current + (max_delta if delta > 0 else -max_delta)
    return target


def step_wrist_roll(current_deg: float, droll_rad_s: float, *, dt: float) -> float:
    """Integrate wrist roll and clamp it to a conservative full-turn range."""
    if abs(droll_rad_s) <= 0.001:
        return current_deg
    next_deg = current_deg + np.rad2deg(droll_rad_s * dt)
    return float(np.clip(next_deg, -180.0, 180.0))


def advance_cartesian_target(
    state: CartesianControlState,
    ee_delta: EEDelta,
    *,
    dt: float,
    clip_position: PositionClipper,
    pitch_limit_rad: float,
    yaw_limit_rad: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, bool | float]]:
    """Integrate one EE delta into a target pose.

    Returns:
        target_pose, target_pos, flags dict containing workspace/orientation clipping
        and orientation metadata.
    """
    target_pos = state.ee_pose[:3, 3].copy()
    target_pos[0] += ee_delta.dx * dt
    target_pos[1] += ee_delta.dy * dt
    target_pos[2] += ee_delta.dz * dt
    target_pos, clip_flags = clip_position(target_pos)

    pitch_clipped = False
    if abs(ee_delta.dpitch) > 0.001:
        next_pitch = state.target_pitch + ee_delta.dpitch * dt
        clipped_pitch = float(np.clip(next_pitch, -pitch_limit_rad, pitch_limit_rad))
        pitch_clipped = clipped_pitch != next_pitch
        state.target_pitch = clipped_pitch

    yaw_clipped = False
    if abs(ee_delta.dyaw) > 0.001:
        next_yaw = state.target_yaw + ee_delta.dyaw * dt
        clipped_yaw = float(np.clip(next_yaw, -yaw_limit_rad, yaw_limit_rad))
        yaw_clipped = clipped_yaw != next_yaw
        state.target_yaw = clipped_yaw

    target_pose = state.ee_pose.copy()
    target_pose[:3, 3] = target_pos
    has_orientation_target = abs(state.target_pitch) > 0.01 or abs(state.target_yaw) > 0.01
    orientation_weight = 0.0
    if has_orientation_target:
        target_pose[:3, :3] = euler_to_rotation_matrix(0.0, state.target_pitch, state.target_yaw)
        orientation_weight = 0.1

    flags: dict[str, bool | float] = dict(clip_flags)
    flags["pitch_clipped"] = pitch_clipped
    flags["yaw_clipped"] = yaw_clipped
    flags["has_orientation_target"] = has_orientation_target
    flags["orientation_weight"] = orientation_weight
    return target_pose, target_pos, flags


def apply_ik_solution(
    state: CartesianControlState,
    kinematics,
    ik_joint_pos_deg: np.ndarray,
    *,
    wrist_roll_deg: float | None = None,
    target_pose: np.ndarray | None = None,
) -> None:
    """Update cartesian state after accepting a new joint-space solution."""
    state.ik_joint_pos_deg = np.asarray(ik_joint_pos_deg, dtype=float).copy()
    if wrist_roll_deg is not None:
        state.wrist_roll_deg = float(wrist_roll_deg)
    state.ee_pose = kinematics.forward_kinematics(state.ik_joint_pos_deg)
    state.last_ee_pose = state.ee_pose.copy()
    state.last_target_pose = target_pose.copy() if target_pose is not None else state.ee_pose.copy()


def full_joint_positions(ik_joint_pos_deg: np.ndarray, wrist_roll_deg: float) -> np.ndarray:
    """Combine IK joints plus wrist roll into the 5-joint arm vector."""
    joints = np.asarray(ik_joint_pos_deg, dtype=float)
    return np.array([joints[0], joints[1], joints[2], joints[3], float(wrist_roll_deg)])


def step_cartesian_home(
    state: CartesianControlState,
    kinematics,
    home_ik_joint_pos_deg: np.ndarray,
    *,
    home_wrist_roll_deg: float,
    home_gripper_pos: float,
    ik_joint_max_step_deg: np.ndarray,
    wrist_roll_vel_deg_s: float,
    gripper_rate: float,
    dt: float,
    target_pitch: float = 0.0,
    target_yaw: float = 0.0,
) -> bool:
    """Advance cartesian control state toward a home pose without snapping."""
    home_joints = np.asarray(home_ik_joint_pos_deg, dtype=float)
    next_joints = step_array_toward(state.ik_joint_pos_deg, home_joints, ik_joint_max_step_deg * dt)
    next_wrist_roll = step_scalar_toward(
        state.wrist_roll_deg,
        home_wrist_roll_deg,
        wrist_roll_vel_deg_s * dt,
    )
    state.gripper_pos = step_gripper_toward(
        state.gripper_pos,
        home_gripper_pos,
        gripper_rate=gripper_rate,
        dt=dt,
    )
    state.target_pitch = step_scalar_toward(state.target_pitch, target_pitch, np.pi * dt)
    state.target_yaw = step_scalar_toward(state.target_yaw, target_yaw, np.pi * dt)

    home_pose = kinematics.forward_kinematics(home_joints)
    apply_ik_solution(
        state,
        kinematics,
        next_joints,
        wrist_roll_deg=next_wrist_roll,
        target_pose=home_pose,
    )
    return (
        array_reached(state.ik_joint_pos_deg, home_joints)
        and scalar_reached(state.wrist_roll_deg, home_wrist_roll_deg)
        and scalar_reached(state.gripper_pos, home_gripper_pos)
        and scalar_reached(state.target_pitch, target_pitch)
        and scalar_reached(state.target_yaw, target_yaw)
    )
