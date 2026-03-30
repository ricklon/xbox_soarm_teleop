"""Dataset feature schemas for recording."""

from __future__ import annotations


def build_dataset_features(mode: str, joint_count: int) -> dict:
    """Build LeRobotDataset feature specs for a given control mode."""
    features: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (joint_count,),
        },
        "action": {
            "dtype": "float32",
            "shape": (joint_count,),
        },
    }

    if mode == "cartesian":
        features["action.ee_delta"] = {"dtype": "float32", "shape": (7,)}
        features["action.ee_target"] = {"dtype": "float32", "shape": (16,)}
        features["observation.ee_pose"] = {"dtype": "float32", "shape": (16,)}
        features["safety.flags"] = {"dtype": "float32", "shape": (7,)}

    return features


def build_schema_metadata(mode: str, joint_names: list[str]) -> dict:
    """Create a schema metadata payload for recorded datasets."""
    features = build_dataset_features(mode=mode, joint_count=len(joint_names))
    if "observation.state" in features:
        features["observation.state"]["names"] = [f"{m}.pos" for m in joint_names]
    if "action" in features:
        features["action"]["names"] = [f"{m}.pos" for m in joint_names]
    if mode == "cartesian":
        features["action.ee_delta"]["names"] = [
            "dx",
            "dy",
            "dz",
            "droll",
            "dpitch",
            "dyaw",
            "gripper",
        ]
        features["safety.flags"]["names"] = [
            "ws_clip_x",
            "ws_clip_y",
            "ws_clip_z",
            "speed_clip",
            "orient_clip",
            "joint_clip",
            "reject",
        ]

    return {
        "schema_version": "2026-03-26",
        "mode": mode,
        "features": features,
    }
