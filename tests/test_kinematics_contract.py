"""Physical-model contracts; expected failures identify unresolved validation gates."""

from pathlib import Path

import numpy as np
import pytest

from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES

pytest.importorskip("placo")


@pytest.fixture
def model():
    from lerobot.model.kinematics import RobotKinematics

    return RobotKinematics(
        str(Path(__file__).resolve().parents[1] / "assets" / "so101.urdf"),
        joint_names=IK_JOINT_NAMES,
    )


def test_tool_point_depends_on_wrist_roll(model):
    """The offset tool point traces a circle when wrist roll changes."""
    joints = np.array([10.0, 30.0, -45.0, 20.0])
    model.robot.set_joint("wrist_roll", 0.0)
    initial = model.forward_kinematics(joints).copy()
    model.robot.set_joint("wrist_roll", np.pi / 2)
    rotated = model.forward_kinematics(joints).copy()
    displacement = np.linalg.norm(rotated[:3, 3] - initial[:3, 3])
    # Independent geometric prediction from the URDF fixed tool offset.
    expected = np.sqrt(2) * np.hypot(0.0079, 0.000218121)
    assert displacement == pytest.approx(expected, abs=1e-8)


@pytest.mark.xfail(
    strict=True,
    reason="Installed LeRobot joint_names filters output but does not lock excluded joints",
)
def test_reduced_ik_preserves_uncommanded_wrist_roll(model):
    """A reduced-chain command must not rely on discarded joint movement."""
    joints = np.array([10.0, 30.0, -45.0, 20.0])
    model.robot.set_joint("wrist_roll", 0.0)
    target = model.forward_kinematics(joints).copy()
    target[0, 3] += 0.01
    model.inverse_kinematics(joints, target, orientation_weight=0.0)
    assert model.robot.get_joint("wrist_roll") == pytest.approx(0.0, abs=1e-10)
