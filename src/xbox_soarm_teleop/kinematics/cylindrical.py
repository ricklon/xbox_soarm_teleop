"""Bounded reach/height IK with explicit direct-joint geometry."""

import numpy as np

from xbox_soarm_teleop.config.joints import JOINT_LIMITS_RAD, parse_joint_limits
from xbox_soarm_teleop.kinematics.constrained import IKResult, RobotKinematics


class CylindricalKinematics(RobotKinematics):
    """Solve only shoulder lift and elbow for radius and world height.

    Pan and wrist joints are fixed at their commanded angles. Tool azimuth and
    orientation are consequences of the full model, not extra IK constraints.
    """

    def __init__(self, urdf_path: str) -> None:
        super().__init__(urdf_path, joint_names=["shoulder_lift", "elbow_flex"])
        self._direct_limits = parse_joint_limits(
            urdf_path, ["shoulder_pan", "wrist_flex", "wrist_roll"]
        )

    def inverse_cylindrical(
        self,
        current_joint_pos: np.ndarray,
        reach_m: float,
        height_m: float,
        *,
        pan_deg: float,
        wrist_flex_deg: float,
        wrist_roll_deg: float,
    ) -> np.ndarray | None:
        """Solve a radial tool target, returning None on rejection.

        Positions are meters in the URDF world frame; joint values are degrees.
        The solve has at most 40 damped least-squares iterations, each with
        a two-column numerical Jacobian and up to eight backtracking trials. Acceptance requires a radial/height residual <= 10 um.
        Restore every model joint on exit, whether accepted or rejected.

        Raises:
            ValueError: Inputs are nonfinite or outside modeled joint limits.
        """
        self.last_ik_result = None
        seed = np.asarray(current_joint_pos, dtype=float)
        target = np.array([reach_m, height_m], dtype=float)
        if (
            seed.shape != (2,)
            or not np.all(np.isfinite(seed))
            or not np.all(np.isfinite(target))
            or reach_m < 0
            or not self._within_limits(seed)
        ):
            raise ValueError("Invalid cylindrical seed or target")
        direct = dict(shoulder_pan=pan_deg, wrist_flex=wrist_flex_deg, wrist_roll=wrist_roll_deg)
        for name, angle in direct.items():
            lo, hi = self._direct_limits[name]
            config_lo, config_hi = JOINT_LIMITS_RAD[name]
            if (
                not np.isfinite(angle)
                or not max(lo, config_lo) - 1e-8 <= np.deg2rad(angle) <= min(hi, config_hi) + 1e-8
            ):
                raise ValueError(f"Invalid direct joint angle: {name}")
        snapshot = {name: self.robot.get_joint(name) for name in self.robot.joint_names()}
        try:
            for name, angle in direct.items():
                self.robot.set_joint(name, float(np.deg2rad(angle)))

            def residual(joints: np.ndarray) -> np.ndarray:
                pose = self.forward_kinematics(np.rad2deg(joints))
                x, y, z = pose[:3, 3]
                return np.array([np.hypot(x, y), z]) - target

            joints = np.clip(np.deg2rad(seed), self._lower, self._upper)
            error = residual(joints)
            iterations = 0
            for iterations in range(41):
                if np.linalg.norm(error) <= 1e-5 or iterations == 40:
                    break
                epsilon = 1e-5
                jacobian = np.column_stack(
                    [
                        (residual(joints + np.eye(2)[axis] * epsilon) - error) / epsilon
                        for axis in range(2)
                    ]
                )
                step = -jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-6 * np.eye(2), error)
                peak = float(np.max(np.abs(step)))
                if peak > 0.25:
                    step *= 0.25 / peak
                for trial in range(8):
                    candidate = np.clip(joints + step * (0.5**trial), self._lower, self._upper)
                    candidate_error = residual(candidate)
                    if np.linalg.norm(candidate_error) < np.linalg.norm(error):
                        joints, error = candidate, candidate_error
                        break
                else:
                    break
            error_norm = float(np.linalg.norm(error))
            accepted = bool(
                np.all(np.isfinite(joints))
                and error_norm <= 1e-5
                and self._within_limits(np.rad2deg(joints))
            )
            self.last_ik_result = IKResult(
                accepted, "accepted" if accepted else "not_converged", iterations, error_norm, 0.0
            )
            return np.rad2deg(joints) if accepted else None
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            self.last_ik_result = IKResult(False, "solver_error", 0, float("inf"), 0.0)
            return None
        finally:
            for name, angle in snapshot.items():
                self.robot.set_joint(name, angle)
            self.robot.update_kinematics()
