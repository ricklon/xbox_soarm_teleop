"""Configuration for Xbox SO-ARM teleoperation."""

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    HOME_POSITION_RAW,
    IK_JOINT_NAMES,
    IK_JOINT_VEL_LIMITS_ARRAY,
    IK_JOINT_VEL_LIMITS_DEG,
    JOINT_LIMITS_DEG,
    JOINT_LIMITS_RAD,
    JOINT_NAMES,
    JOINT_NAMES_WITH_GRIPPER,
    MOTOR_IDS,
    RAW_CENTER,
    RAW_PER_DEGREE,
    SWEEP_TEST_POSITIONS,
    deg_to_raw,
    limits_rad_to_deg,
    parse_joint_limits,
    raw_to_deg,
)
from xbox_soarm_teleop.config.xbox_config import XboxConfig

__all__ = [
    "HOME_POSITION_DEG",
    "HOME_POSITION_RAW",
    "IK_JOINT_NAMES",
    "IK_JOINT_VEL_LIMITS_ARRAY",
    "IK_JOINT_VEL_LIMITS_DEG",
    "JOINT_LIMITS_DEG",
    "JOINT_LIMITS_RAD",
    "JOINT_NAMES",
    "JOINT_NAMES_WITH_GRIPPER",
    "MOTOR_IDS",
    "RAW_CENTER",
    "RAW_PER_DEGREE",
    "SWEEP_TEST_POSITIONS",
    "XboxConfig",
    "deg_to_raw",
    "limits_rad_to_deg",
    "parse_joint_limits",
    "raw_to_deg",
]
