"""Reduced-joint kinematics with explicit constraints on uncommanded joints."""

import logging
from dataclasses import dataclass

import numpy as np
from lerobot.model.kinematics import RobotKinematics as LeRobotKinematics

from xbox_soarm_teleop.config.joints import JOINT_LIMITS_RAD, parse_joint_limits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IKResult:
    """Acceptance evidence for the last solve; residuals are model-space only."""

    accepted: bool
    reason: str
    iterations: int
    position_error_m: float
    orientation_error_rad: float


class RobotKinematics(LeRobotKinematics):
    """Keep joints outside the requested chain fixed during IK.

    LeRobot's ``joint_names`` selects inputs/outputs, not solver degrees of
    freedom. Mask excluded joints so a solution cannot depend on motion that
    the caller will discard. Their current model positions remain adjustable
    for feedback and directly controlled joints.
    """

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ) -> None:
        super().__init__(urdf_path, target_frame_name, joint_names)
        self.solver.enable_joint_limits(True)
        limits = parse_joint_limits(urdf_path, self.joint_names)
        self._lower = np.array(
            [
                max(
                    limits.get(name, (-np.inf, np.inf))[0],
                    JOINT_LIMITS_RAD.get(name, (-np.inf, np.inf))[0],
                )
                for name in self.joint_names
            ]
        )
        self._upper = np.array(
            [
                min(
                    limits.get(name, (-np.inf, np.inf))[1],
                    JOINT_LIMITS_RAD.get(name, (-np.inf, np.inf))[1],
                )
                for name in self.joint_names
            ]
        )
        self.last_ik_result: IKResult | None = None
        for name in set(self.robot.joint_names()) - set(self.joint_names):
            self.solver.mask_dof(name)

    def forward_kinematics(
        self, joint_pos_deg: np.ndarray, *, wrist_roll_deg: float | None = None
    ) -> np.ndarray:
        """Compute a pose, optionally updating the separately controlled roll.

        Args:
            joint_pos_deg: Selected chain angles in degrees.
            wrist_roll_deg: Known roll angle for a chain excluding wrist roll.
                If omitted, retain the existing model angle.

        Returns:
            World-to-tool transform for the complete model state.
        """
        if wrist_roll_deg is not None:
            if "wrist_roll" in self.joint_names:
                raise ValueError("Pass wrist roll in the joint vector when it belongs to the chain")
            if not np.isfinite(wrist_roll_deg):
                raise ValueError("Wrist roll must be finite")
            self.robot.set_joint("wrist_roll", float(np.deg2rad(wrist_roll_deg)))
        return super().forward_kinematics(joint_pos_deg)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """Return a validated solution, or hold the seed if solving fails.

        Run at most 30 solver iterations. Enabled position and orientation tasks
        must reach 10 micrometers and 0.03 radians respectively in the model.
        Nonconvergence is not proof of global unreachability. Inspect
        ``last_ik_result`` to distinguish acceptance from a held command.
        Runtime velocity limiting remains the caller's responsibility.

        Raises:
            ValueError: Inputs are malformed, nonfinite, or the seed is outside
                the intersection of URDF and configured joint limits.
        """
        previous = self.last_ik_result
        self.last_ik_result = None
        seed = np.asarray(current_joint_pos, dtype=float)
        target = np.asarray(desired_ee_pose, dtype=float)
        count = len(self.joint_names)
        if seed.ndim != 1 or len(seed) < count or not np.all(np.isfinite(seed)):
            raise ValueError("IK seed must contain finite angles for every selected joint")
        if (
            target.shape != (4, 4)
            or not np.all(np.isfinite(target))
            or not np.allclose(target[3], [0, 0, 0, 1])
            or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(target[:3, :3]), 1.0, atol=1e-6)
        ):
            raise ValueError("IK target must be a finite rigid transform")
        if (
            not np.isfinite(position_weight)
            or not np.isfinite(orientation_weight)
            or position_weight < 0
            or orientation_weight < 0
            or position_weight + orientation_weight == 0
        ):
            raise ValueError("IK weights must be finite, nonnegative, and enable at least one task")
        if not self._within_limits(seed[:count]):
            raise ValueError("IK seed is outside configured/model joint limits")

        candidate = seed.copy()
        pose = self.forward_kinematics(candidate).copy()
        snapshot = {name: self.robot.get_joint(name) for name in self.robot.joint_names()}
        position_error = float("inf")
        orientation_error = float("inf")
        reason = "not_converged"
        attempts = 0
        for iterations in range(31):
            position_error = float(np.linalg.norm(target[:3, 3] - pose[:3, 3]))
            cosine = (np.trace(target[:3, :3].T @ pose[:3, :3]) - 1) / 2
            orientation_error = float(np.arccos(np.clip(cosine, -1, 1)))
            if (position_weight == 0 or position_error <= 1e-5) and (
                orientation_weight == 0 or orientation_error <= 0.03
            ):
                self.last_ik_result = IKResult(
                    True, "accepted", iterations, position_error, orientation_error
                )
                return candidate.copy()
            if iterations == 30:
                break
            attempts += 1
            try:
                result = super().inverse_kinematics(
                    candidate, target, position_weight, orientation_weight
                )
                if result is None:
                    reason = "invalid_output"
                    break
                result = np.asarray(result, dtype=float)
                if result.shape != seed.shape or not np.all(np.isfinite(result)):
                    reason = "invalid_output"
                    break
                if not self._within_limits(result[:count]):
                    reason = "joint_limit"
                    break
                candidate = result.copy()
                # Preserve extra caller channels, e.g. gripper, outside the IK chain.
                candidate[count:] = seed[count:]
                pose = self.forward_kinematics(candidate).copy()
                if not np.all(np.isfinite(pose)):
                    reason = "invalid_output"
                    break
            except RuntimeError:
                reason = "solver_error"
                break

        for name, angle in snapshot.items():
            self.robot.set_joint(name, angle)
        self.robot.update_kinematics()
        self.last_ik_result = IKResult(False, reason, attempts, position_error, orientation_error)
        if previous is None or previous.accepted or previous.reason != reason:
            logger.warning(
                "IK held selected joints: %s (position residual %.6f m)", reason, position_error
            )
        return seed.copy()

    def _within_limits(self, joints: np.ndarray) -> bool:
        radians = np.deg2rad(joints)
        return bool(np.all(radians >= self._lower - 1e-8) and np.all(radians <= self._upper + 1e-8))
