"""Test IK with updated wrist_flex limits (72° instead of 95°).

Verifies that:
1. IK respects the hardware limit of +72° for wrist_flex
2. FK/IK roundtrip works within calibrated ranges
3. Workspace is still reachable with corrected limits
4. Joint limits from config are properly respected
"""

from pathlib import Path

import numpy as np
import pytest

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    IK_JOINT_NAMES,
    JOINT_LIMITS_DEG,
)

# Skip if placo not available
placo = pytest.importorskip("placo")

URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"
EE_FRAME = "gripper_frame_link"


@pytest.fixture
def kinematics():
    """Load kinematics with current URDF."""
    if not URDF_PATH.exists():
        pytest.skip(f"URDF not found at {URDF_PATH}")

    from lerobot.model.kinematics import RobotKinematics

    return RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name=EE_FRAME,
        joint_names=IK_JOINT_NAMES,
    )


@pytest.fixture
def home_joints():
    """Home position in degrees (4 IK joints)."""
    return np.array([0.0, 0.0, 0.0, 0.0])


class TestWristFlexLimits:
    """Verify wrist_flex respects hardware limit of +72°."""

    def test_wrist_flex_limit_in_config(self):
        """Config should have updated wrist_flex limit (~72°)."""
        lower, upper = JOINT_LIMITS_DEG["wrist_flex"]
        # Updated limit should be ~72° (was 95°)
        assert upper <= 75.0, f"wrist_flex upper limit {upper}° > 75° (should be ~72°)"
        assert upper >= 70.0, f"wrist_flex upper limit {upper}° < 70° (should be ~72°)"
        print(f"✓ wrist_flex limit: [{lower:.1f}°, {upper:.1f}°]")

    def test_wrist_flex_72_reachable(self, kinematics, home_joints):
        """FK/IK roundtrip at wrist_flex = 72° should work."""
        # Set wrist_flex to 72° (at hardware limit)
        test_joints = home_joints.copy()
        test_joints[3] = 72.0  # wrist_flex is 4th IK joint

        # Forward kinematics
        ee_pose = kinematics.forward_kinematics(test_joints)

        # Inverse kinematics
        recovered = kinematics.inverse_kinematics(test_joints, ee_pose)

        # Should recover close to 72°
        np.testing.assert_allclose(
            test_joints,
            recovered[:4],
            atol=2.0,  # 2 degree tolerance at limit
        )
        print(f"✓ wrist_flex = 72°: recovered {recovered[3]:.1f}°")

    def test_wrist_flex_negative_works(self, kinematics, home_joints):
        """wrist_flex at -90° should work (no limit change on negative side)."""
        test_joints = home_joints.copy()
        test_joints[3] = -90.0

        ee_pose = kinematics.forward_kinematics(test_joints)
        recovered = kinematics.inverse_kinematics(test_joints, ee_pose)

        np.testing.assert_allclose(
            test_joints,
            recovered[:4],
            atol=1.0,
        )
        print(f"✓ wrist_flex = -90°: recovered {recovered[3]:.1f}°")

    def test_wrist_flex_workspace_reachable(self, kinematics, home_joints):
        """Verify wrist can still reach typical workspace positions."""
        # Test positions within typical workspace
        test_configs = [
            np.array([0.0, 45.0, -45.0, 45.0]),  # Extended with wrist up
            np.array([0.0, 30.0, -30.0, 0.0]),  # Neutral
            np.array([45.0, 20.0, -20.0, 30.0]),  # Rotated with wrist angle
        ]

        for i, joints in enumerate(test_configs):
            ee_pose = kinematics.forward_kinematics(joints)
            recovered = kinematics.inverse_kinematics(joints, ee_pose)

            # Position should match within 5cm
            recovered_pose = kinematics.forward_kinematics(recovered[:4])
            np.testing.assert_allclose(
                ee_pose[:3, 3],
                recovered_pose[:3, 3],
                atol=0.05,  # 5cm tolerance
                err_msg=f"Config {i} failed",
            )
        print(f"✓ {len(test_configs)} workspace configurations reachable")


class TestAllJointLimits:
    """Verify all joints respect their calibrated limits."""

    def test_joint_limits_match_ik_joints(self):
        """All IK joints should have limits defined."""
        for name in IK_JOINT_NAMES:
            assert name in JOINT_LIMITS_DEG, f"Missing limits for {name}"
            lower, upper = JOINT_LIMITS_DEG[name]
            assert lower < upper, f"Invalid limits for {name}: [{lower}, {upper}]"
        print(f"✓ All {len(IK_JOINT_NAMES)} IK joints have limits")

    def test_home_position_within_limits(self):
        """Home position should be within joint limits."""
        for i, name in enumerate(IK_JOINT_NAMES):
            home_deg = HOME_POSITION_DEG[name]
            lower, upper = JOINT_LIMITS_DEG[name]
            assert lower <= home_deg <= upper, (
                f"Home position for {name} ({home_deg}°) outside limits [{lower}, {upper}]"
            )
        print("✓ Home position within all joint limits")


class TestFKIKConsistency:
    """General FK/IK consistency tests with updated limits."""

    def test_fk_ik_roundtrip_at_limits(self, kinematics):
        """Test FK/IK at joint limit boundaries."""
        # Test at lower limits
        lower_joints = np.array(
            [
                JOINT_LIMITS_DEG["shoulder_pan"][0],
                JOINT_LIMITS_DEG["shoulder_lift"][0],
                JOINT_LIMITS_DEG["elbow_flex"][0],
                JOINT_LIMITS_DEG["wrist_flex"][0],
            ]
        )

        ee_pose = kinematics.forward_kinematics(lower_joints)
        recovered = kinematics.inverse_kinematics(lower_joints, ee_pose)

        np.testing.assert_allclose(
            lower_joints,
            recovered[:4],
            atol=2.0,
        )
        print("✓ FK/IK roundtrip at lower limits")

    def test_fk_ik_roundtrip_at_upper_limits(self, kinematics):
        """Test FK/IK at upper limits (including wrist 72°)."""
        upper_joints = np.array(
            [
                JOINT_LIMITS_DEG["shoulder_pan"][1],
                JOINT_LIMITS_DEG["shoulder_lift"][1],
                JOINT_LIMITS_DEG["elbow_flex"][1],
                JOINT_LIMITS_DEG["wrist_flex"][1],  # Should be ~72°
            ]
        )

        ee_pose = kinematics.forward_kinematics(upper_joints)
        recovered = kinematics.inverse_kinematics(upper_joints, ee_pose)

        np.testing.assert_allclose(
            upper_joints,
            recovered[:4],
            atol=2.0,
        )
        print(f"✓ FK/IK roundtrip at upper limits (wrist_flex ~{upper_joints[3]:.0f}°)")

    def test_ik_does_not_exceed_wrist_limit(self, kinematics, home_joints):
        """IK solutions should not exceed wrist_flex hardware limit."""
        # Try to reach a high position that might require extreme wrist angle
        home_pose = kinematics.forward_kinematics(home_joints)

        # Create target requiring wrist extension
        target_pose = home_pose.copy()
        target_pose[2, 3] += 0.10  # 10cm up

        # Solve IK
        result = kinematics.inverse_kinematics(home_joints, target_pose)
        wrist_flex_result = result[3]

        # Should respect hardware limit (~72°)
        wrist_limit = JOINT_LIMITS_DEG["wrist_flex"][1]
        assert wrist_flex_result <= wrist_limit + 2.0, (
            f"IK solution wrist_flex {wrist_flex_result:.1f}° exceeds limit {wrist_limit:.1f}°"
        )
        print(f"✓ IK respects wrist_flex limit: {wrist_flex_result:.1f}° <= {wrist_limit:.1f}°")
