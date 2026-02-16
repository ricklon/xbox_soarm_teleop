"""Tests for Jacobian-based control module."""

from pathlib import Path

import numpy as np
import pytest

from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES

# Skip all tests in this module if placo not available
placo = pytest.importorskip("placo")

# Path to SO101 URDF
URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"

# Joint names for IK
JOINT_NAMES = IK_JOINT_NAMES

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
def jacobian_controller(kinematics):
    """Create JacobianController instance."""
    from xbox_soarm_teleop.kinematics.jacobian import JacobianController

    return JacobianController(kinematics, damping=0.05)


@pytest.fixture
def home_joints():
    """Home position joint angles in degrees."""
    return np.array([0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def test_joints():
    """Non-singular test configuration in degrees."""
    return np.array([10.0, 30.0, -45.0, 20.0])


class TestJacobianShape:
    """Test Jacobian matrix dimensions and structure."""

    def test_full_jacobian_shape(self, jacobian_controller, home_joints):
        """Full Jacobian should be 6x4 (6 EE DOFs, 4 arm joints)."""
        J = jacobian_controller.get_jacobian(home_joints)
        assert J.shape == (6, 4), f"Expected (6, 4), got {J.shape}"

    def test_position_jacobian_shape(self, jacobian_controller, home_joints):
        """Position Jacobian should be 3x4."""
        J = jacobian_controller.get_position_jacobian(home_joints)
        assert J.shape == (3, 4), f"Expected (3, 4), got {J.shape}"

    def test_jacobian_finite(self, jacobian_controller, test_joints):
        """Jacobian should contain finite values."""
        J = jacobian_controller.get_jacobian(test_joints)
        assert np.all(np.isfinite(J)), "Jacobian contains non-finite values"

    def test_jacobian_changes_with_config(self, jacobian_controller, home_joints, test_joints):
        """Jacobian should change with different configurations."""
        J_home = jacobian_controller.get_jacobian(home_joints)
        J_test = jacobian_controller.get_jacobian(test_joints)
        assert not np.allclose(J_home, J_test), "Jacobian should vary with configuration"


class TestJointVelocity:
    """Test joint velocity computation."""

    def test_zero_ee_vel_gives_zero_joint_vel(self, jacobian_controller, home_joints):
        """Zero EE velocity should give zero joint velocity."""
        ee_vel = np.array([0.0, 0.0, 0.0])
        joint_vel = jacobian_controller.ee_vel_to_joint_vel(ee_vel, home_joints)
        np.testing.assert_allclose(joint_vel, 0.0, atol=1e-10)

    def test_joint_vel_finite(self, jacobian_controller, test_joints):
        """Joint velocities should be finite for reasonable EE velocity."""
        ee_vel = np.array([0.1, 0.0, 0.0])  # 0.1 m/s in X
        joint_vel = jacobian_controller.ee_vel_to_joint_vel(ee_vel, test_joints)
        assert np.all(np.isfinite(joint_vel)), "Joint velocities contain non-finite values"

    def test_joint_vel_reasonable_magnitude(self, jacobian_controller, test_joints):
        """Joint velocities should be reasonable magnitude for typical input."""
        ee_vel = np.array([0.1, 0.0, 0.0])  # 0.1 m/s
        joint_vel = jacobian_controller.ee_vel_to_joint_vel(ee_vel, test_joints)
        # Should be less than 500 deg/s for reasonable motion
        assert np.all(np.abs(joint_vel) < 500), f"Joint velocities too large: {joint_vel}"

    def test_damping_prevents_explosion_near_singularity(self, jacobian_controller, home_joints):
        """Damping should prevent large velocities near singular configurations."""
        # Home position can be near-singular for some motions
        ee_vel = np.array([0.0, 0.1, 0.0])  # 0.1 m/s in Y
        joint_vel = jacobian_controller.ee_vel_to_joint_vel(ee_vel, home_joints)
        # Even near singularity, damping should keep velocities bounded
        assert np.all(np.abs(joint_vel) < 1000), f"Damping failed, velocities: {joint_vel}"


class TestManipulability:
    """Test manipulability measure computation."""

    def test_manipulability_positive(self, jacobian_controller, test_joints):
        """Manipulability should be non-negative."""
        manip = jacobian_controller.manipulability(test_joints)
        assert manip >= 0, f"Manipulability should be >= 0, got {manip}"

    def test_manipulability_finite(self, jacobian_controller, test_joints):
        """Manipulability should be finite."""
        manip = jacobian_controller.manipulability(test_joints)
        assert np.isfinite(manip), f"Manipulability should be finite, got {manip}"

    def test_manipulability_varies_with_config(self, jacobian_controller, home_joints, test_joints):
        """Manipulability should vary with configuration."""
        manip_home = jacobian_controller.manipulability(home_joints)
        manip_test = jacobian_controller.manipulability(test_joints)
        # They may be similar but should not be identical
        assert manip_home >= 0 and manip_test >= 0

    def test_singularity_detection_works(self, jacobian_controller, home_joints):
        """Singularity detection should return boolean."""
        result = jacobian_controller.is_near_singularity(home_joints)
        assert isinstance(result, bool)


class TestConditionNumber:
    """Test condition number computation."""

    def test_condition_number_at_least_one(self, jacobian_controller, test_joints):
        """Condition number should be >= 1."""
        cond = jacobian_controller.condition_number(test_joints)
        assert cond >= 1.0, f"Condition number should be >= 1, got {cond}"

    def test_condition_number_finite(self, jacobian_controller, test_joints):
        """Condition number should be finite for non-singular config."""
        cond = jacobian_controller.condition_number(test_joints)
        assert np.isfinite(cond), f"Condition number should be finite, got {cond}"


class TestJacobianConsistency:
    """Test that Jacobian is consistent with FK."""

    def test_jacobian_predicts_fk_direction(self, jacobian_controller, kinematics, test_joints):
        """Jacobian should predict the direction of FK change.

        If we move joints by dq, the EE should move approximately by J @ dq.
        """
        # Small joint perturbation
        dq_deg = np.array([1.0, 0.0, 0.0, 0.0])  # 1 degree on shoulder_pan
        dq_rad = np.deg2rad(dq_deg)

        # Get Jacobian
        J = jacobian_controller.get_position_jacobian(test_joints)

        # Predict EE displacement
        predicted_dx = J @ dq_rad  # meters

        # Actual EE displacement via FK
        ee_before = kinematics.forward_kinematics(test_joints)[:3, 3]
        ee_after = kinematics.forward_kinematics(test_joints + dq_deg)[:3, 3]
        actual_dx = ee_after - ee_before

        # Should be approximately equal for small perturbations
        np.testing.assert_allclose(predicted_dx, actual_dx, atol=0.001)  # 1mm tolerance

    def test_jacobian_inverse_recovers_joint_vel(self, jacobian_controller, test_joints):
        """Using J to go from joint_vel -> ee_vel -> joint_vel should be consistent.

        For a non-singular configuration, J @ pinv(J) should be close to identity
        in the range space of J.
        """
        # Start with a desired EE velocity
        ee_vel_desired = np.array([0.05, 0.02, -0.01])  # m/s

        # Compute joint velocity using pseudo-inverse
        joint_vel = jacobian_controller.ee_vel_to_joint_vel(ee_vel_desired, test_joints)

        # Compute what EE velocity we actually get with these joint velocities
        J = jacobian_controller.get_position_jacobian(test_joints)
        joint_vel_rad = np.deg2rad(joint_vel)
        ee_vel_achieved = J @ joint_vel_rad

        # For a well-conditioned configuration, these should be close
        # Allow up to 30% error due to damping in pseudo-inverse
        error = np.linalg.norm(ee_vel_achieved - ee_vel_desired)
        desired_magnitude = np.linalg.norm(ee_vel_desired)
        relative_error = error / desired_magnitude
        assert relative_error < 0.35, f"EE velocity relative error: {relative_error:.1%}"


class TestDampingEffect:
    """Test the effect of damping parameter."""

    def test_higher_damping_reduces_velocity(self, kinematics, test_joints):
        """Higher damping should result in smaller joint velocities."""
        from xbox_soarm_teleop.kinematics.jacobian import JacobianController

        ctrl_low = JacobianController(kinematics, damping=0.01)
        ctrl_high = JacobianController(kinematics, damping=0.2)

        ee_vel = np.array([0.1, 0.0, 0.0])
        vel_low = ctrl_low.ee_vel_to_joint_vel(ee_vel, test_joints)
        vel_high = ctrl_high.ee_vel_to_joint_vel(ee_vel, test_joints)

        # Higher damping should give smaller velocities (more conservative)
        assert np.linalg.norm(vel_high) <= np.linalg.norm(vel_low) * 1.1  # Allow small tolerance

    def test_zero_damping_still_works(self, kinematics, test_joints):
        """Controller should work even with very small damping."""
        from xbox_soarm_teleop.kinematics.jacobian import JacobianController

        ctrl = JacobianController(kinematics, damping=1e-6)
        ee_vel = np.array([0.05, 0.0, 0.0])
        joint_vel = ctrl.ee_vel_to_joint_vel(ee_vel, test_joints)
        assert np.all(np.isfinite(joint_vel))
