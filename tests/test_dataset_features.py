"""Tests for dataset feature schemas."""

from xbox_soarm_teleop.recording.features import build_dataset_features, build_schema_metadata


def test_joint_features_minimum_keys():
    features = build_dataset_features(mode="joint", joint_count=6)
    assert "observation.state" in features
    assert "action" in features


def test_cartesian_features_include_ee_fields():
    features = build_dataset_features(mode="cartesian", joint_count=6)
    assert "action.ee_delta" in features
    assert "action.ee_target" in features
    assert "observation.ee_pose" in features
    assert "safety.flags" in features


def test_safety_flags_shape():
    features = build_dataset_features(mode="cartesian", joint_count=6)
    assert features["safety.flags"]["shape"] == (7,)


def test_schema_metadata_contains_mode_and_features():
    meta = build_schema_metadata(mode="joint", joint_names=["a", "b"])
    assert meta["mode"] == "joint"
    assert "features" in meta
