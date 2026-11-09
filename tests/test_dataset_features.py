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


# ── Versioned, documented schemas (#20) ──────────────────────────────────────

import inspect  # noqa: E402

import pytest  # noqa: E402

from xbox_soarm_teleop.config.joints import JOINT_NAMES_WITH_GRIPPER  # noqa: E402
from xbox_soarm_teleop.recording.features import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    RECORDABLE_MODES,
)


def test_schema_version_is_semver():
    parts = DATASET_SCHEMA_VERSION.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)
    meta = build_schema_metadata(mode="joint", joint_names=JOINT_NAMES_WITH_GRIPPER)
    assert meta["schema_version"] == DATASET_SCHEMA_VERSION


def test_every_recordable_mode_is_explicit_and_unknown_modes_fail():
    assert set(RECORDABLE_MODES) == {"joint", "crane", "cartesian"}
    for mode in RECORDABLE_MODES:
        build_schema_metadata(mode=mode, joint_names=JOINT_NAMES_WITH_GRIPPER)
    with pytest.raises(ValueError, match="mode"):
        build_schema_metadata(mode="telepathy", joint_names=JOINT_NAMES_WITH_GRIPPER)


@pytest.mark.parametrize("mode", ["joint", "crane", "cartesian"])
def test_every_feature_has_names_units_and_description(mode):
    meta = build_schema_metadata(mode=mode, joint_names=JOINT_NAMES_WITH_GRIPPER)
    assert meta["mode"] == mode and meta["description"]
    for key, spec in meta["features"].items():
        assert len(spec["names"]) == spec["shape"][0], key
        assert spec["unit"], key
        assert spec["description"], key


def test_joint_values_document_lerobot_normalized_units():
    meta = build_schema_metadata(mode="joint", joint_names=JOINT_NAMES_WITH_GRIPPER)
    unit = meta["features"]["observation.state"]["unit"]
    assert "-100" in unit and "0..100" in unit


def test_lerobot_features_only_contain_lerobot_keys():
    features = build_dataset_features(mode="cartesian", joint_count=len(JOINT_NAMES_WITH_GRIPPER))
    for spec in features.values():
        assert set(spec) <= {"dtype", "shape", "names"}


def test_record_xbox_uses_the_shared_schema_names():
    from xbox_soarm_teleop.cli import record_xbox

    source = inspect.getsource(record_xbox)
    assert '"ws_clip_x"' not in source  # names come from recording.features only
