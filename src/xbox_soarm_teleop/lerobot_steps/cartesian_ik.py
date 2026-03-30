"""LeRobot processor step: EE delta -> joint action via IK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotActionProcessorStep

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    IK_JOINT_NAMES,
    IK_JOINT_VEL_LIMITS_ARRAY,
    JOINT_LIMITS_DEG,
    JOINT_NAMES_WITH_GRIPPER,
)
from xbox_soarm_teleop.config.workspace import load_workspace_limits
from xbox_soarm_teleop.control.cartesian import make_cartesian_state, step_cartesian_home
from xbox_soarm_teleop.control.pose import euler_to_rotation_matrix
from xbox_soarm_teleop.control.safety import apply_strict_safety, clip_workspace
from xbox_soarm_teleop.control.units import deg_to_normalized, normalized_to_deg
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta, apply_axis_mapping


@ProcessorStepRegistry.register("soarm_cartesian_ik")
@dataclass
class SoArmCartesianIKProcessor(RobotActionProcessorStep):
    """Convert EE delta actions into joint position actions using IK."""

    urdf_path: str
    dt: float = 1.0 / 30.0
    swap_xy: bool = False
    strict_safety: bool = True
    max_linear_speed: float = 0.02
    max_angular_speed: float = 0.25
    allow_orientation: bool = False
    orientation_weight: float = 0.1
    position_weight: float = 1.0
    ik_vel_scale: float = 1.0
    gripper_rate: float = 2.0
    seed_from_observation: bool = True

    _kinematics: Any = field(init=False, repr=False)
    _workspace_limits: dict[str, tuple[float, float]] = field(init=False, repr=False)
    _ik_joint_vel_limits: np.ndarray = field(init=False, repr=False)

    _ik_joint_pos_deg: np.ndarray | None = field(init=False, default=None, repr=False)
    _wrist_roll_deg: float = field(init=False, default=0.0, repr=False)
    _target_pitch: float = field(init=False, default=0.0, repr=False)
    _target_yaw: float = field(init=False, default=0.0, repr=False)
    _gripper_pos: float = field(init=False, default=0.0, repr=False)
    _homing_active: bool = field(init=False, default=False, repr=False)

    last_target_pose: np.ndarray | None = field(init=False, default=None)
    last_obs_pose: np.ndarray | None = field(init=False, default=None)
    last_delta: EEDelta | None = field(init=False, default=None)
    last_flags: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        from lerobot.model.kinematics import RobotKinematics

        self._kinematics = RobotKinematics(
            urdf_path=self.urdf_path,
            target_frame_name="gripper_frame_link",
            joint_names=IK_JOINT_NAMES,
        )
        base_limits, strict_limits = load_workspace_limits()
        self._workspace_limits = strict_limits if self.strict_safety else base_limits
        self._ik_joint_vel_limits = IK_JOINT_VEL_LIMITS_ARRAY * max(self.ik_vel_scale, 0.1)

    def get_config(self) -> dict[str, Any]:
        return {
            "urdf_path": self.urdf_path,
            "dt": self.dt,
            "swap_xy": self.swap_xy,
            "strict_safety": self.strict_safety,
            "max_linear_speed": self.max_linear_speed,
            "max_angular_speed": self.max_angular_speed,
            "allow_orientation": self.allow_orientation,
            "orientation_weight": self.orientation_weight,
            "position_weight": self.position_weight,
            "ik_vel_scale": self.ik_vel_scale,
            "gripper_rate": self.gripper_rate,
            "seed_from_observation": self.seed_from_observation,
        }

    def reset(self) -> None:
        self._ik_joint_pos_deg = None
        self._wrist_roll_deg = float(HOME_POSITION_DEG["wrist_roll"])
        self._target_pitch = 0.0
        self._target_yaw = 0.0
        self._gripper_pos = 0.0
        self._homing_active = False

    def start_homing(self) -> None:
        """Begin a rate-limited return to the processor home pose."""
        self._homing_active = True

    def _seed_from_observation(self, obs: dict[str, Any]) -> None:
        ik_joints = np.array(
            [
                normalized_to_deg(float(obs.get(f"{name}.pos", 0.0)), name)
                for name in IK_JOINT_NAMES
            ],
            dtype=float,
        )
        self._ik_joint_pos_deg = ik_joints
        self._wrist_roll_deg = normalized_to_deg(float(obs.get("wrist_roll.pos", 0.0)), "wrist_roll")
        gripper_norm = float(obs.get("gripper.pos", self._gripper_pos * 100.0))
        self._gripper_pos = float(np.clip(gripper_norm / 100.0, 0.0, 1.0))

    def _ensure_state(self, obs: dict[str, Any]) -> None:
        if self.seed_from_observation and obs:
            self._seed_from_observation(obs)
            return
        if self._ik_joint_pos_deg is None:
            self._ik_joint_pos_deg = np.array(
                [HOME_POSITION_DEG[name] for name in IK_JOINT_NAMES], dtype=float
            )
            self._wrist_roll_deg = float(HOME_POSITION_DEG["wrist_roll"])
            self._gripper_pos = 0.0

    def action(self, action: RobotAction) -> RobotAction:
        obs = self.transition.get("observation") or {}
        self._ensure_state(obs)

        ik_joint_pos_deg = self._ik_joint_pos_deg.copy()
        ee_pose = self._kinematics.forward_kinematics(ik_joint_pos_deg)
        self.last_obs_pose = ee_pose.copy()

        delta = EEDelta(
            dx=float(action.get("dx", 0.0)),
            dy=float(action.get("dy", 0.0)),
            dz=float(action.get("dz", 0.0)),
            droll=float(action.get("droll", 0.0)),
            dpitch=float(action.get("dpitch", 0.0)),
            dyaw=float(action.get("dyaw", 0.0)),
            gripper=float(action.get("gripper", self._gripper_pos)),
        )
        delta = apply_axis_mapping(delta, swap_xy=self.swap_xy)

        flags = {
            "ws_clip_x": 0,
            "ws_clip_y": 0,
            "ws_clip_z": 0,
            "speed_clip": 0,
            "orient_clip": 0,
            "joint_clip": 0,
            "reject": 0,
        }

        if self._homing_active:
            home_state = make_cartesian_state(
                self._kinematics,
                ik_joint_pos_deg,
                wrist_roll_deg=self._wrist_roll_deg,
                target_pitch=self._target_pitch,
                target_yaw=self._target_yaw,
                gripper_pos=self._gripper_pos,
            )
            done = step_cartesian_home(
                home_state,
                self._kinematics,
                np.array([HOME_POSITION_DEG[name] for name in IK_JOINT_NAMES], dtype=float),
                home_wrist_roll_deg=float(HOME_POSITION_DEG["wrist_roll"]),
                home_gripper_pos=0.0,
                ik_joint_max_step_deg=self._ik_joint_vel_limits,
                wrist_roll_vel_deg_s=90.0,
                gripper_rate=self.gripper_rate,
                dt=self.dt,
            )
            ik_joint_pos_deg = home_state.ik_joint_pos_deg.copy()
            self._wrist_roll_deg = float(home_state.wrist_roll_deg)
            self._target_pitch = float(home_state.target_pitch)
            self._target_yaw = float(home_state.target_yaw)
            self._gripper_pos = float(home_state.gripper_pos)
            target_pose = home_state.last_target_pose.copy()
            delta = EEDelta(gripper=self._gripper_pos)
            self._homing_active = not done
        else:
            if self.strict_safety:
                delta, safety_flags = apply_strict_safety(
                    delta,
                    max_linear_speed=self.max_linear_speed,
                    max_angular_speed=self.max_angular_speed,
                    allow_orientation=self.allow_orientation,
                )
                flags["speed_clip"] = safety_flags["speed_clip"]
                flags["orient_clip"] = safety_flags["orient_clip"]

            # Rate-limit gripper toward target
            gripper_target = float(np.clip(delta.gripper, 0.0, 1.0))
            gripper_diff = gripper_target - self._gripper_pos
            max_delta = max(self.gripper_rate, 0.0) * self.dt
            if abs(gripper_diff) > max_delta:
                self._gripper_pos += max_delta if gripper_diff > 0 else -max_delta
            else:
                self._gripper_pos = gripper_target

        if not self._homing_active and not delta.is_zero_motion():
            target_pos = ee_pose[:3, 3].copy()
            target_pos[0] += delta.dx * self.dt
            target_pos[1] += delta.dy * self.dt
            target_pos[2] += delta.dz * self.dt

            target_pos, ws_flags = clip_workspace(target_pos, self._workspace_limits)
            flags.update(ws_flags)

            pitch_limit = np.deg2rad(25.0 if self.strict_safety else 90.0)
            yaw_limit = np.deg2rad(45.0 if self.strict_safety else 180.0)
            if abs(delta.dpitch) > 0.001:
                self._target_pitch = float(
                    np.clip(self._target_pitch + delta.dpitch * self.dt, -pitch_limit, pitch_limit)
                )
            if abs(delta.dyaw) > 0.001:
                self._target_yaw = float(
                    np.clip(self._target_yaw + delta.dyaw * self.dt, -yaw_limit, yaw_limit)
                )

            target_pose = ee_pose.copy()
            target_pose[:3, 3] = target_pos
            has_orientation = abs(self._target_pitch) > 0.01 or abs(self._target_yaw) > 0.01
            if has_orientation:
                target_pose[:3, :3] = euler_to_rotation_matrix(
                    0.0, self._target_pitch, self._target_yaw
                )
                orientation_weight = self.orientation_weight
            else:
                orientation_weight = 0.0

            new_joints = self._kinematics.inverse_kinematics(
                ik_joint_pos_deg,
                target_pose,
                position_weight=self.position_weight,
                orientation_weight=orientation_weight,
            )
            ik_result = new_joints[: len(IK_JOINT_NAMES)]
            max_delta = self._ik_joint_vel_limits * self.dt
            joint_delta = np.clip(ik_result - ik_joint_pos_deg, -max_delta, max_delta)
            ik_joint_pos_deg = ik_joint_pos_deg + joint_delta
            for idx, name in enumerate(IK_JOINT_NAMES):
                lo, hi = JOINT_LIMITS_DEG[name]
                clipped = float(np.clip(ik_joint_pos_deg[idx], lo, hi))
                if clipped != ik_joint_pos_deg[idx]:
                    flags["joint_clip"] = 1
                ik_joint_pos_deg[idx] = clipped
        elif not self._homing_active:
            target_pose = ee_pose.copy()

        if not self._homing_active and abs(delta.droll) > 0.001:
            roll_delta_deg = np.rad2deg(delta.droll * self.dt)
            self._wrist_roll_deg += roll_delta_deg
            wr_lo, wr_hi = JOINT_LIMITS_DEG["wrist_roll"]
            self._wrist_roll_deg = float(np.clip(self._wrist_roll_deg, wr_lo, wr_hi))

        self._ik_joint_pos_deg = ik_joint_pos_deg
        self.last_target_pose = target_pose.copy()
        self.last_delta = delta
        self.last_flags = flags

        full_joint_pos_deg = np.array(
            [
                ik_joint_pos_deg[0],
                ik_joint_pos_deg[1],
                ik_joint_pos_deg[2],
                ik_joint_pos_deg[3],
                self._wrist_roll_deg,
            ],
            dtype=float,
        )
        action_out: RobotAction = {}
        for idx, name in enumerate(JOINT_NAMES_WITH_GRIPPER[:-1]):
            action_out[f"{name}.pos"] = deg_to_normalized(full_joint_pos_deg[idx], name)
        action_out["gripper.pos"] = self._gripper_pos * 100.0
        return action_out

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Remove delta keys if present
        for key in ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]:
            features[PipelineFeatureType.ACTION].pop(key, None)
        for name in JOINT_NAMES_WITH_GRIPPER:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features
