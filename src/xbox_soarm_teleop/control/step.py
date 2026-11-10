"""One tick of cartesian control, shared by human teleop and agent actions.

Both the Xbox loop and the agent action executor turn their input into an
``EEDelta`` (m/s, rad/s, gripper 0-1) and call :func:`cartesian_step`, so
workspace clipping, joint velocity limits, IK acceptance and limit reporting
behave identically for people and agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from xbox_soarm_teleop.control.cartesian import (
    CartesianControlState,
    PositionClipper,
    advance_cartesian_target,
    apply_ik_solution,
    full_joint_positions,
    jacobian_joint_step,
    solve_cartesian_motion,
    step_cartesian_home,
    step_gripper_toward,
    use_jacobian_step,
)
from xbox_soarm_teleop.control.motion_feedback import commanded_progress, describe_motion_limit
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta


@dataclass
class CartesianStepConfig:
    """Fixed parameters of the cartesian control step."""

    dt: float
    joint_vel_limits_deg: np.ndarray
    gripper_rate: float
    clip_position: PositionClipper
    workspace_bounds: dict[str, tuple[float | None, float | None]]
    pitch_limit_rad: float = np.pi / 2
    yaw_limit_rad: float = np.pi
    home_wrist_roll_vel_deg_s: float = 90.0
    home_ik_joint_pos_deg: np.ndarray = field(default_factory=lambda: np.zeros(4))


@dataclass
class CartesianStepResult:
    """What one control tick decided."""

    homing: bool
    """Whether homing is still in progress after this tick."""
    phase: str
    """``"idle"``, ``"moving"`` or ``"homing"``."""
    limit: str
    """Why motion is limited this tick (``describe_motion_limit``), or empty."""
    joints_deg: np.ndarray
    """Arm command: IK chain plus wrist roll (5 joints, degrees)."""
    gripper: float
    target_pose: np.ndarray | None = None
    target_pos: np.ndarray | None = None
    ik_result: Any = None


def cartesian_step(
    state: CartesianControlState,
    kinematics,
    ee_delta: EEDelta,
    config: CartesianStepConfig,
    *,
    homing: bool,
    jacobian_ctrl=None,
) -> CartesianStepResult:
    """Advance cartesian control by one tick and update ``state`` in place.

    Args:
        state: Control state; updated with the accepted joints and gripper.
        kinematics: Constrained ``RobotKinematics`` for the IK chain.
        ee_delta: Commanded end-effector velocity and gripper target.
        config: Loop period, limits and workspace.
        homing: True while a home move is in progress.
        jacobian_ctrl: Use the Jacobian step for position/pitch when given.
    """
    if jacobian_ctrl is not None and abs(ee_delta.dyaw) > 1e-6:
        # Jacobian mode is position + pitch only; drop yaw like the real-robot path.
        ee_delta = replace(ee_delta, dyaw=0.0)
    phase = "homing" if homing else "idle"
    limit = "homing" if homing else ""
    if homing:
        homing = not step_cartesian_home(
            state,
            kinematics,
            config.home_ik_joint_pos_deg,
            home_wrist_roll_deg=0.0,
            home_gripper_pos=0.0,
            ik_joint_max_step_deg=config.joint_vel_limits_deg,
            wrist_roll_vel_deg_s=config.home_wrist_roll_vel_deg_s,
            gripper_rate=config.gripper_rate,
            dt=config.dt,
        )
        ee_delta = EEDelta(gripper=state.gripper_pos)
    else:
        state.gripper_pos = step_gripper_toward(
            state.gripper_pos, ee_delta.gripper, gripper_rate=config.gripper_rate, dt=config.dt
        )

    result = CartesianStepResult(
        homing=homing, phase=phase, limit=limit, joints_deg=np.zeros(5), gripper=0.0
    )
    if not homing and not ee_delta.is_zero_motion():
        dt = config.dt
        limits = config.joint_vel_limits_deg
        target_pose, target_pos, target_flags = advance_cartesian_target(
            state,
            ee_delta,
            dt=dt,
            clip_position=config.clip_position,
            pitch_limit_rad=config.pitch_limit_rad,
            yaw_limit_rad=config.yaw_limit_rad,
        )
        orientation_weight = float(target_flags["orientation_weight"])
        ran_ik = False
        if jacobian_ctrl is not None and use_jacobian_step(state, ee_delta):
            next_wrist_roll_deg = state.wrist_roll_deg
            candidate_joints = jacobian_joint_step(
                state,
                jacobian_ctrl,
                ee_delta,
                dt=dt,
                joint_vel_limits_deg=limits,
                clip_position=config.clip_position,
            )
        else:
            ran_ik = True
            new_joints, next_wrist_roll_deg = solve_cartesian_motion(
                state,
                kinematics,
                target_pose,
                ee_delta,
                dt=dt,
                orientation_weight=orientation_weight,
                max_joint_step_deg=limits * dt,
            )
            max_delta = limits * dt
            joint_delta = new_joints[:4] - state.ik_joint_pos_deg
            candidate_joints = state.ik_joint_pos_deg + np.clip(joint_delta, -max_delta, max_delta)
        pos_before = state.ee_pose[:3, 3].copy()
        apply_ik_solution(
            state,
            kinematics,
            candidate_joints,
            wrist_roll_deg=next_wrist_roll_deg,
            target_pose=target_pose,
        )
        ik_result = getattr(kinematics, "last_ik_result", None) if ran_ik else None
        result.phase = "moving"
        result.limit = describe_motion_limit(
            homing=False,
            delta=ee_delta,
            target_pos=target_pos,
            workspace_bounds=config.workspace_bounds,
            ik_joints_deg=state.ik_joint_pos_deg,
            ik_result=ik_result,
            progress=commanded_progress(pos_before, state.ee_pose[:3, 3], ee_delta, dt),
        )
        result.target_pose = target_pose
        result.target_pos = target_pos
        result.ik_result = ik_result

    result.joints_deg = full_joint_positions(state.ik_joint_pos_deg, state.wrist_roll_deg)
    result.gripper = state.gripper_pos
    return result
