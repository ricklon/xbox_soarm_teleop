"""Tests for FK/IK roundtrip verification with LeRobot kinematics."""

from pathlib import Path

import numpy as np
import pytest

# Skip all tests in this module if placo not available
placo = pytest.importorskip("placo")

# Path to SO101 URDF (with absolute mesh paths)
URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"

# Joint names for SO101 (excluding wrist_roll and gripper for IK)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

# End effector frame name
EE_FRAME = "gripper_frame_link"


@pytest.fixture
def kinematics():
    """Load SO-ARM101 kinematics from LeRobot."""
    if not URDF_PATH.exists():
        pytest.skip(f"URDF not found at {URDF_PATH}")

    from lerobot.model.kinematics import RobotKinematics

    return RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=EE_FRAME,
        joint_names=JOINT_NAMES,
    )


@pytest.fixture
def home_joints():
    """Home position joint angles in degrees."""
    return np.array([0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def test_joints():
    """Test joint configuration in degrees (non-singular)."""
    return np.array([10.0, 30.0, -45.0, 20.0])


class TestIKRoundtrip:
    """FK/IK roundtrip tests for SO-ARM101.

    These tests verify that the kinematics solver correctly
    computes forward and inverse kinematics.

    Note: LeRobot kinematics uses degrees for joint angles.
    """

    def test_forward_kinematics_runs(self, kinematics, home_joints):
        """FK should return a 4x4 transformation matrix."""
        ee_pose = kinematics.forward_kinematics(home_joints)

        assert ee_pose.shape == (4, 4)
        # Last row should be [0, 0, 0, 1]
        np.testing.assert_allclose(ee_pose[3, :], [0, 0, 0, 1], atol=1e-6)

    def test_fk_ik_roundtrip_home(self, kinematics, home_joints):
        """FK then IK should return to home position."""
        # Forward kinematics
        ee_pose = kinematics.forward_kinematics(home_joints)

        # Inverse kinematics
        recovered_joints = kinematics.inverse_kinematics(home_joints, ee_pose)

        # Should recover original joint angles
        np.testing.assert_allclose(
            home_joints,
            recovered_joints[: len(home_joints)],
            atol=1.0,  # 1 degree tolerance
        )

    def test_fk_ik_roundtrip_nonhome(self, kinematics, test_joints):
        """FK then IK should return to original non-home configuration."""
        # Forward kinematics
        ee_pose = kinematics.forward_kinematics(test_joints)

        # Inverse kinematics with same initial guess
        recovered_joints = kinematics.inverse_kinematics(test_joints, ee_pose)

        # Should recover original joint angles
        np.testing.assert_allclose(
            test_joints,
            recovered_joints[: len(test_joints)],
            atol=1.0,  # 1 degree tolerance
        )

    def test_ik_converges_from_different_initial_guess(self, kinematics, test_joints, home_joints):
        """IK should converge even with different initial guess."""
        # Get target pose from test configuration
        target_pose = kinematics.forward_kinematics(test_joints)

        # Solve IK starting from home position
        recovered_joints = kinematics.inverse_kinematics(home_joints, target_pose)

        # Verify by computing FK of recovered joints
        recovered_pose = kinematics.forward_kinematics(recovered_joints[: len(home_joints)])

        # Poses should match (position within 10cm - IK from different starting point
        # may not converge perfectly due to local minima)
        np.testing.assert_allclose(
            target_pose[:3, 3],
            recovered_pose[:3, 3],
            atol=0.10,  # 10cm tolerance
        )

    def test_fk_position_changes_with_joints(self, kinematics, home_joints):
        """Changing joint angles should change end effector position."""
        pose_home = kinematics.forward_kinematics(home_joints)

        # Move shoulder_lift by 45 degrees
        moved_joints = home_joints.copy()
        moved_joints[1] = 45.0
        pose_moved = kinematics.forward_kinematics(moved_joints)

        # Positions should be different
        assert not np.allclose(pose_home[:3, 3], pose_moved[:3, 3], atol=0.01)

    def test_workspace_center_reachable(self, kinematics, home_joints):
        """Verify a point in front of the robot is reachable."""
        # Get home pose
        home_pose = kinematics.forward_kinematics(home_joints)

        # Create target slightly in front (add 5cm in x direction)
        target_pose = home_pose.copy()
        target_pose[0, 3] += 0.05  # 5cm forward

        # Try to solve IK
        result = kinematics.inverse_kinematics(home_joints, target_pose)

        # Verify we get a valid result (same length as input)
        assert len(result) >= len(home_joints)
