"""Reach/height are radial tool coordinates, not an arbitrary world X/Z plane."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("placo")
from xbox_soarm_teleop.kinematics.constrained import RobotKinematics
from xbox_soarm_teleop.kinematics.cylindrical import CylindricalKinematics

URDF = str(Path(__file__).resolve().parents[1] / "assets/so101.urdf")
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@pytest.mark.parametrize("direct", [(0, 13, 0), (35, -20, 80), (-45, 40, -60)])
def test_reachable_radius_height_matches_full_model(direct):
    reference = RobotKinematics(URDF, joint_names=NAMES)
    solver = CylindricalKinematics(URDF)
    pan, flex, roll = direct
    target = reference.forward_kinematics(np.array([pan, 30.0, -45.0, flex, roll]))[:3, 3]
    result = solver.inverse_cylindrical(
        np.array([28.0, -43.0]),
        np.hypot(*target[:2]),
        target[2],
        pan_deg=pan,
        wrist_flex_deg=flex,
        wrist_roll_deg=roll,
    )
    assert result is not None
    assert solver.last_ik_result.accepted
    achieved = reference.forward_kinematics(np.array([pan, *result, flex, roll]))[:3, 3]
    np.testing.assert_allclose(
        [np.hypot(*achieved[:2]), achieved[2]], [np.hypot(*target[:2]), target[2]], atol=1e-5
    )


def test_unreachable_target_rejects_and_restores_full_model():
    solver = CylindricalKinematics(URDF)
    q = np.array([30.0, -45.0])
    solver.forward_kinematics(q)
    before = {name: solver.robot.get_joint(name) for name in solver.robot.joint_names()}
    result = solver.inverse_cylindrical(q, 10.0, 10.0, pan_deg=20, wrist_flex_deg=10, wrist_roll_deg=40)
    assert result is None
    assert not solver.last_ik_result.accepted
    for name, value in before.items():
        assert solver.robot.get_joint(name) == pytest.approx(value)


def test_crane_home_geometry_uses_commanded_wrist():
    from xbox_soarm_teleop.processors.crane import CraneProcessor

    processor = CraneProcessor(urdf_path=URDF)
    assert processor._planar_ik.last_ik_result.accepted
    command = processor._current_command()
    reference = RobotKinematics(URDF, joint_names=NAMES)
    pose = reference.forward_kinematics(np.array([command.goals_deg[name] for name in NAMES]))
    assert np.hypot(*pose[:2, 3]) == pytest.approx(processor._reach_m, abs=1e-5)
    assert pose[2, 3] == pytest.approx(processor._height_m, abs=1e-5)


def test_crane_moves_reach_and_roll_with_consistent_tool_geometry():
    from xbox_soarm_teleop.processors.crane import CraneProcessor
    from xbox_soarm_teleop.teleoperators.xbox import XboxState

    processor = CraneProcessor(urdf_path=URDF)
    before = processor._current_command().goals_deg
    command = processor(XboxState(left_bumper=True, left_stick_y=-0.05, right_stick_x=0.05))
    assert processor._planar_ik.last_ik_result.accepted
    assert command.goals_deg["wrist_roll"] != before["wrist_roll"]
    reference = RobotKinematics(URDF, joint_names=NAMES)
    pose = reference.forward_kinematics(np.array([command.goals_deg[name] for name in NAMES]))
    assert np.hypot(*pose[:2, 3]) == pytest.approx(processor._reach_m, abs=1e-5)
    assert pose[2, 3] == pytest.approx(processor._height_m, abs=1e-5)
