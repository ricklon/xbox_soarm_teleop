"""Versioned dataset schemas for recorded teleoperation episodes.

One schema per recordable control mode. ``build_schema_metadata`` is the single
source of truth (names, units, descriptions) and is written to
``meta/schema.json`` next to each LeRobot dataset; ``build_dataset_features``
derives the LeRobot feature spec from it. See ``docs/schemas.md``.

Bump ``DATASET_SCHEMA_VERSION`` (semver) whenever a recorded field is added,
removed, reordered, or changes meaning or units.
"""

from __future__ import annotations

import copy

DATASET_SCHEMA_VERSION = "1.0.0"

RECORDABLE_MODES = ("joint", "crane", "cartesian")

_MODE_DESCRIPTIONS = {
    "joint": "Direct per-joint velocity control; actions are commanded joint positions.",
    "crane": (
        "Cylindrical crane control (pan, reach, height, wrist); actions are the commanded "
        "joint positions produced by the crane processor."
    ),
    "cartesian": (
        "End-effector Cartesian control through IK; actions are commanded joint positions, "
        "plus the operator's EE velocity command, the IK target, and safety flags."
    ),
}

_JOINT_UNIT = (
    "LeRobot normalized position: arm joints -100..100 (RANGE_M100_100), "
    "gripper 0..100 (RANGE_0_100, 0 = open)"
)
_POSE_NAMES = [f"m{row}{col}" for row in range(4) for col in range(4)]
_POSE_UNIT = "4x4 homogeneous transform, row-major; translation in m, robot base frame"

_EE_DELTA_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
_SAFETY_FLAG_NAMES = [
    "ws_clip_x",
    "ws_clip_y",
    "ws_clip_z",
    "speed_clip",
    "orient_clip",
    "joint_clip",
    "reject",
]


def _joint_features(joint_names: list[str]) -> dict:
    names = [f"{name}.pos" for name in joint_names]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
            "unit": _JOINT_UNIT,
            "description": "Measured joint positions read from the follower arm.",
        },
        "action": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": list(names),
            "unit": _JOINT_UNIT,
            "description": "Joint position targets sent to the follower arm.",
        },
    }


def _cartesian_features() -> dict:
    return {
        "action.ee_delta": {
            "dtype": "float32",
            "shape": (len(_EE_DELTA_NAMES),),
            "names": list(_EE_DELTA_NAMES),
            "unit": "m/s (dx, dy, dz); rad/s (droll, dpitch, dyaw); gripper 0..1 (1 = closed)",
            "description": "Operator end-effector velocity command after mapping and safety.",
        },
        "action.ee_target": {
            "dtype": "float32",
            "shape": (16,),
            "names": list(_POSE_NAMES),
            "unit": _POSE_UNIT,
            "description": "IK target pose for this step (after workspace clipping).",
        },
        "observation.ee_pose": {
            "dtype": "float32",
            "shape": (16,),
            "names": list(_POSE_NAMES),
            "unit": _POSE_UNIT,
            "description": "End-effector pose from forward kinematics of the IK joint state.",
        },
        "safety.flags": {
            "dtype": "float32",
            "shape": (len(_SAFETY_FLAG_NAMES),),
            "names": list(_SAFETY_FLAG_NAMES),
            "unit": "0/1 flag",
            "description": (
                "Set when the step was clipped to the workspace (per axis), speed- or "
                "orientation-limited, joint-clipped, or rejected by strict safety."
            ),
        },
    }


def build_schema_metadata(mode: str, joint_names: list[str]) -> dict:
    """Return the full, documented schema for a recordable mode.

    Args:
        mode: One of ``RECORDABLE_MODES``.
        joint_names: Joint names in recorded order (including the gripper).

    Returns:
        ``{"schema_version", "mode", "description", "features"}`` where each feature
        has ``dtype``, ``shape``, ``names``, ``unit`` and ``description``.

    Raises:
        ValueError: ``mode`` is not recordable.
    """
    if mode not in RECORDABLE_MODES:
        raise ValueError(
            f"Unknown recording mode '{mode}'. Recordable modes: {', '.join(RECORDABLE_MODES)}"
        )
    features = _joint_features(list(joint_names))
    if mode == "cartesian":
        features.update(_cartesian_features())
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "mode": mode,
        "description": _MODE_DESCRIPTIONS[mode],
        "features": features,
    }


def build_dataset_features(mode: str, joint_count: int, joint_names: list[str] | None = None) -> dict:
    """Return the LeRobotDataset feature spec (``dtype``, ``shape``, ``names`` only).

    Args:
        mode: One of ``RECORDABLE_MODES``.
        joint_count: Number of recorded joints (including the gripper).
        joint_names: Joint names; defaults to ``joint_0..joint_{n-1}``.
    """
    if joint_names is None:
        joint_names = [f"joint_{i}" for i in range(joint_count)]
    if len(joint_names) != joint_count:
        raise ValueError("joint_names must have joint_count entries")
    meta = build_schema_metadata(mode=mode, joint_names=list(joint_names))
    return {
        key: {field: copy.deepcopy(spec[field]) for field in ("dtype", "shape", "names")}
        for key, spec in meta["features"].items()
    }
