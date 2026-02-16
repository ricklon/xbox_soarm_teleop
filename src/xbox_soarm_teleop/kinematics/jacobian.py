"""Jacobian-based resolved-rate control for SO-ARM.

This module provides an alternative to IK-based control using
the Jacobian pseudo-inverse for computing joint velocities from
end effector velocities.

Key advantages:
- Single matrix multiply instead of iterative IK solve
- Explicit singularity detection via manipulability measure
- Natural velocity limiting at joint level
- Potentially faster computation
"""

import numpy as np

from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES


class JacobianController:
    """Jacobian-based resolved-rate control for SO-ARM.

    Uses the Jacobian pseudo-inverse to convert end effector velocities
    to joint velocities, with damped least squares for singularity robustness.

    Attributes:
        damping: Damping factor for pseudo-inverse (default 0.05).
        manipulability_threshold: Below this, consider near-singular (default 0.001).
    """

    def __init__(
        self,
        kinematics,
        damping: float = 0.05,
        manipulability_threshold: float = 0.001,
    ):
        """Initialize Jacobian controller.

        Args:
            kinematics: RobotKinematics instance from LeRobot.
            damping: Damping factor for damped least squares.
            manipulability_threshold: Threshold for singularity detection.
        """
        self.kin = kinematics
        self.damping = damping
        self.manipulability_threshold = manipulability_threshold
        self.joint_names = IK_JOINT_NAMES

        # Placo uses a floating base (6 DOF) + arm joints
        # The Jacobian from frame_jacobian is 6xN where N = 6 (base) + arm joints
        # For SO-ARM101 with 4 IK joints: N = 10, we want columns 6:10
        self._joint_start_idx = 6
        self._n_joints = len(self.joint_names)

    def _update_robot_state(self, joint_pos_deg: np.ndarray) -> None:
        """Update the internal Placo robot state.

        Args:
            joint_pos_deg: Joint positions in degrees.
        """
        joint_pos_rad = np.deg2rad(joint_pos_deg)
        for i, name in enumerate(self.joint_names):
            self.kin.robot.set_joint(name, joint_pos_rad[i])
        self.kin.robot.update_kinematics()

    def get_jacobian(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Get the 6xN Jacobian for the current configuration.

        The Jacobian maps joint velocities to end effector velocities:
            ee_vel = J @ joint_vel

        Returns the arm-only columns (excluding floating base).

        Args:
            joint_pos_deg: Current joint positions in degrees (4 joints).

        Returns:
            6x4 Jacobian matrix (6 EE DOFs x 4 arm joints).
        """
        self._update_robot_state(joint_pos_deg)

        # Get full Jacobian from Placo using local_world_aligned frame
        # This gives the Jacobian with linear velocities expressed in world frame
        # but computed at the frame origin (consistent with FK differentiation)
        J_full = self.kin.robot.frame_jacobian("gripper_frame_link", "local_world_aligned")

        # Extract arm joint columns (skip floating base columns 0-5)
        J_arm = J_full[:, self._joint_start_idx : self._joint_start_idx + self._n_joints]

        return J_arm

    def get_position_jacobian(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Get the 3x4 position-only Jacobian.

        Args:
            joint_pos_deg: Current joint positions in degrees (4 joints).

        Returns:
            3x4 Jacobian matrix (3 position DOFs x 4 arm joints).
        """
        J = self.get_jacobian(joint_pos_deg)
        return J[:3, :]  # Position rows only (linear velocity)

    def get_position_pitch_jacobian(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Get the 4x4 Jacobian for position + pitch control.

        This extracts position (vx, vy, vz) and pitch (wy) rows.
        Pitch is rotation around the Y axis (tilting gripper up/down).

        With 4 DOFs and 4 joints, this is a square system - ideal match.

        Args:
            joint_pos_deg: Current joint positions in degrees (4 joints).

        Returns:
            4x4 Jacobian matrix (3 position + 1 pitch DOFs x 4 arm joints).
        """
        J = self.get_jacobian(joint_pos_deg)
        # Rows 0,1,2 = linear velocity (vx, vy, vz)
        # Row 4 = angular velocity around Y (wy = pitch)
        return J[[0, 1, 2, 4], :]

    def ee_vel_to_joint_vel(
        self,
        ee_velocity: np.ndarray,
        joint_pos_deg: np.ndarray,
        mode: str = "position",
        position_only: bool | None = None,  # Deprecated, use mode instead
    ) -> np.ndarray:
        """Convert end effector velocity to joint velocities.

        Uses damped least squares pseudo-inverse:
            joint_vel = J^T @ (J @ J^T + lambda^2 * I)^-1 @ ee_vel

        Args:
            ee_velocity: End effector velocity. Format depends on mode:
                - "position": [vx, vy, vz] (3 elements)
                - "position_pitch": [vx, vy, vz, pitch_vel] (4 elements)
                - "full": [vx, vy, vz, wx, wy, wz] (6 elements)
            joint_pos_deg: Current joint positions in degrees.
            mode: Control mode - "position", "position_pitch", or "full".
            position_only: Deprecated. Use mode="position" or mode="full".

        Returns:
            Joint velocities in degrees/second.
        """
        # Backward compatibility for position_only parameter
        if position_only is not None:
            mode = "position" if position_only else "full"

        if mode == "position":
            J = self.get_position_jacobian(joint_pos_deg)
            ee_vel = ee_velocity[:3]
        elif mode == "position_pitch":
            J = self.get_position_pitch_jacobian(joint_pos_deg)
            ee_vel = ee_velocity[:4]  # [vx, vy, vz, pitch_vel]
        else:  # full
            J = self.get_jacobian(joint_pos_deg)
            ee_vel = ee_velocity

        # Damped least squares pseudo-inverse
        # joint_vel = J^T @ (J @ J^T + lambda^2 * I)^-1 @ ee_vel
        JJT = J @ J.T
        damped = JJT + self.damping**2 * np.eye(JJT.shape[0])
        joint_vel_rad = J.T @ np.linalg.solve(damped, ee_vel)

        return np.rad2deg(joint_vel_rad)

    def manipulability(self, joint_pos_deg: np.ndarray, position_only: bool = True) -> float:
        """Compute Yoshikawa manipulability measure.

        manipulability = sqrt(det(J @ J^T))

        Higher values indicate better conditioning (farther from singularity).

        Args:
            joint_pos_deg: Current joint positions in degrees.
            position_only: If True, use position-only Jacobian.

        Returns:
            Manipulability measure (scalar, >= 0).
        """
        if position_only:
            J = self.get_position_jacobian(joint_pos_deg)
        else:
            J = self.get_jacobian(joint_pos_deg)

        JJT = J @ J.T
        det = np.linalg.det(JJT)
        return float(np.sqrt(max(det, 0.0)))

    def is_near_singularity(self, joint_pos_deg: np.ndarray, position_only: bool = True) -> bool:
        """Check if configuration is near a singularity.

        Args:
            joint_pos_deg: Current joint positions in degrees.
            position_only: If True, use position-only Jacobian.

        Returns:
            True if manipulability is below threshold.
        """
        return self.manipulability(joint_pos_deg, position_only) < self.manipulability_threshold

    def condition_number(self, joint_pos_deg: np.ndarray, position_only: bool = True) -> float:
        """Compute condition number of the Jacobian.

        Higher values indicate worse conditioning (closer to singularity).

        Args:
            joint_pos_deg: Current joint positions in degrees.
            position_only: If True, use position-only Jacobian.

        Returns:
            Condition number (>= 1, inf at singularity).
        """
        if position_only:
            J = self.get_position_jacobian(joint_pos_deg)
        else:
            J = self.get_jacobian(joint_pos_deg)

        return float(np.linalg.cond(J))
