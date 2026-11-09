"""Jacobian joint stepping must respect configured limits and hand off to IK safely."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("placo")
from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES, JOINT_LIMITS_DEG
from xbox_soarm_teleop.control.cartesian import (
    jacobian_joint_step,
    make_cartesian_state,
    solve_cartesian_motion,
    use_jacobian_step,
)
from xbox_soarm_teleop.kinematics.constrained import RobotKinematics
from xbox_soarm_teleop.kinematics.jacobian import JacobianController
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta

URDF = str(Path(__file__).resolve().parents[1] / "assets/so101.urdf")
VEL_LIMITS = np.array([120.0, 90.0, 90.0, 90.0])
DT = 0.02


@pytest.fixture
def model():
    return RobotKinematics(URDF, joint_names=IK_JOINT_NAMES)


def test_use_jacobian_step_defers_roll_and_orientation_to_ik(model):
    state = make_cartesian_state(model, np.array([0.0, 30.0, -45.0, 20.0]))
    assert use_jacobian_step(state, EEDelta(dx=0.05))
    assert not use_jacobian_step(state, EEDelta(dx=0.05, droll=0.5))
    assert not use_jacobian_step(state, EEDelta(roll_target=0.1))
    state.orientation_active = True
    assert not use_jacobian_step(state, EEDelta(dx=0.05))


def test_jacobian_step_never_exceeds_configured_limits(model):
    """Driving pitch hard into the wrist limit stops at ~72 deg, not the old 90."""
    ctrl = JacobianController(model, damping=0.05)
    state = make_cartesian_state(model, np.array([0.0, 30.0, -45.0, 60.0]))
    wf_upper = JOINT_LIMITS_DEG["wrist_flex"][1]
    for _ in range(200):
        for sign in (1.0, -1.0):
            joints = jacobian_joint_step(
                state, ctrl, EEDelta(dpitch=sign * 3.0), dt=DT, joint_vel_limits_deg=VEL_LIMITS
            )
            lower = np.array([JOINT_LIMITS_DEG[n][0] for n in IK_JOINT_NAMES])
            upper = np.array([JOINT_LIMITS_DEG[n][1] for n in IK_JOINT_NAMES])
            assert np.all(joints >= lower - 1e-9) and np.all(joints <= upper + 1e-9)
            assert np.all(np.abs(joints - state.ik_joint_pos_deg) <= VEL_LIMITS * DT + 1e-9)
        state.ik_joint_pos_deg = jacobian_joint_step(
            state, ctrl, EEDelta(dpitch=3.0), dt=DT, joint_vel_limits_deg=VEL_LIMITS
        )
    assert state.ik_joint_pos_deg[3] <= wf_upper + 1e-9


def test_roll_after_jacobian_reaches_limit_does_not_crash(model):
    """The original crash: Jacobian drives to a limit, then roll hands off to IK."""
    ctrl = JacobianController(model, damping=0.05)
    state = make_cartesian_state(model, np.array([100.0, 30.0, -45.0, 60.0]))
    for _ in range(200):
        state.ik_joint_pos_deg = jacobian_joint_step(
            state, ctrl, EEDelta(dy=0.2, dpitch=3.0), dt=DT, joint_vel_limits_deg=VEL_LIMITS
        )
    state.ee_pose = model.forward_kinematics(state.ik_joint_pos_deg, wrist_roll_deg=0.0)
    delta = EEDelta(droll=0.5)
    assert not use_jacobian_step(state, delta)
    joints, roll = solve_cartesian_motion(
        state, model, state.ee_pose.copy(), delta, dt=DT, orientation_weight=0.0,
        max_joint_step_deg=VEL_LIMITS * DT,
    )
    assert np.all(np.isfinite(joints)) and np.isfinite(roll)


def test_simulate_mujoco_forwards_jacobian_flags(monkeypatch):
    from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli

    calls = []
    monkeypatch.setattr(
        sys, "argv",
        ["simulate-mujoco", "--headless", "--jacobian", "--jacobian-damping", "0.1"],
    )
    monkeypatch.setattr(sim_cli, "MuJoCoSimulator", lambda *a, **kw: object())
    monkeypatch.setattr(sim_cli, "run_with_controller", lambda *a, **kw: calls.append(kw) or 0)
    with pytest.raises(SystemExit) as result:
        sim_cli.main()
    assert result.value.code == 0
    assert calls[0]["use_jacobian"] is True
    assert calls[0]["jacobian_damping"] == pytest.approx(0.1)
