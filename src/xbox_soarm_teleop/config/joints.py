"""Shared joint configuration for SO-ARM teleoperation.

Single source of truth for joint names, limits, home positions,
velocity limits, and servo conversion utilities.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np

# ---------------------------------------------------------------------------
# Joint name lists (order matters - matches URDF and robot)
# ---------------------------------------------------------------------------

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

JOINT_NAMES_WITH_GRIPPER = JOINT_NAMES + ["gripper"]

# IK joint names (include base, exclude wrist_roll/gripper)
IK_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

# ---------------------------------------------------------------------------
# Motor IDs (Feetech STS3215 bus IDs)
# ---------------------------------------------------------------------------

MOTOR_IDS: dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

# ---------------------------------------------------------------------------
# Servo conversion constants (STS3215: 4096 steps = 360 degrees)
# ---------------------------------------------------------------------------

RAW_PER_DEGREE: float = 4096.0 / 360.0  # ~11.378
RAW_CENTER: int = 2048


def raw_to_deg(raw: int) -> float:
    """Convert raw servo position (0-4095) to degrees (-180 to +180)."""
    return (raw - RAW_CENTER) / RAW_PER_DEGREE


def deg_to_raw(deg: float) -> int:
    """Convert degrees to raw servo position (0-4095)."""
    return int(RAW_CENTER + deg * RAW_PER_DEGREE)


# ---------------------------------------------------------------------------
# Joint limits (pre-computed from URDF, verified)
# ---------------------------------------------------------------------------

JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.256),  # -95° to +72° (hardware verified, hard mechanical limit)
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.0349066, 2.21657),  # -2° to +127° (measured on hardware)
}

JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    name: (np.rad2deg(low), np.rad2deg(high)) for name, (low, high) in JOINT_LIMITS_RAD.items()
}

# ---------------------------------------------------------------------------
# Home position (folded / parked)
# ---------------------------------------------------------------------------

# Raw servo values for the parked position
HOME_POSITION_RAW: dict[str, int] = {
    "shoulder_pan": 2032,
    "shoulder_lift": 958,
    "elbow_flex": 3139,
    "wrist_flex": 2838,
    "wrist_roll": 2078,
    "gripper": 2042,
}

# Same position in degrees (derived from raw values)
HOME_POSITION_DEG: dict[str, float] = {name: raw_to_deg(raw) for name, raw in HOME_POSITION_RAW.items()}

# ---------------------------------------------------------------------------
# Per-joint test positions for ROM sweep (collision avoidance)
# ---------------------------------------------------------------------------
# When sweeping a joint through its full ROM, holding all other joints at
# home can cause collisions (e.g., shoulder_lift sweep drives arm into table).
# This dict maps the joint being tested to the other-joint positions that
# provide clearance.  Joints not listed here use HOME_POSITION_DEG (default).

SWEEP_TEST_POSITIONS: dict[str, dict[str, float]] = {
    "shoulder_lift": {
        "shoulder_lift": 99.0,  # start extended forward
        "elbow_flex": -88.0,  # arm straight
        "wrist_flex": 13.0,  # wrist neutral
    },
    "wrist_flex": {
        "elbow_flex": 15.0,  # straighten elbow for clearance
    },
    "gripper": {
        "elbow_flex": 0.0,  # extend elbow to clear table
    },
}

# ---------------------------------------------------------------------------
# IK joint velocity limits (deg/s)
# ---------------------------------------------------------------------------

IK_JOINT_VEL_LIMITS_DEG: dict[str, float] = {
    "shoulder_pan": 120.0,
    "shoulder_lift": 90.0,
    "elbow_flex": 90.0,
    "wrist_flex": 90.0,
}

# Numpy array form matching IK_JOINT_NAMES order
IK_JOINT_VEL_LIMITS_ARRAY: np.ndarray = np.array(
    [IK_JOINT_VEL_LIMITS_DEG[name] for name in IK_JOINT_NAMES]
)

# ---------------------------------------------------------------------------
# URDF parsing utilities
# ---------------------------------------------------------------------------


def parse_joint_limits(urdf_path: str | Path, joint_names: list[str]) -> dict[str, tuple[float, float]]:
    """Parse joint limits from a URDF file.

    Args:
        urdf_path: Path to the URDF file.
        joint_names: Joint names to extract limits for.

    Returns:
        Dict mapping joint name to (lower, upper) limits in radians.
    """
    limits: dict[str, tuple[float, float]] = {}
    root = ElementTree.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in joint_names:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        if lower is None or upper is None:
            continue
        limits[name] = (float(lower), float(upper))
    return limits


def limits_rad_to_deg(
    limits: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Convert joint limits from radians to degrees.

    Args:
        limits: Dict mapping joint name to (lower, upper) in radians.

    Returns:
        Dict mapping joint name to (lower, upper) in degrees.
    """
    return {name: (np.rad2deg(low), np.rad2deg(high)) for name, (low, high) in limits.items()}
