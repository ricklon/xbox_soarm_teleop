"""Offline evidence reports must distinguish calibration formats and provenance."""

import hashlib
import json

import pytest

from xbox_soarm_teleop.cli.audit_calibration import audit_calibration


def test_reports_modern_endpoints_without_inventing_model_zero(tmp_path):
    path = tmp_path / "arm.json"
    data = {
        "wrist_flex": {
            "id": 4,
            "drive_mode": 1,
            "homing_offset": 123,
            "range_min": 900,
            "range_max": 3100,
        }
    }
    path.write_text(json.dumps(data))
    report = audit_calibration(path)
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["schema"] == "lerobot_motor_calibration"
    joint = report["joints"]["wrist_flex"]
    assert joint["raw_at_normalized_minus_100"] == 3100
    assert joint["raw_at_normalized_zero"] == 2000
    assert joint["raw_at_normalized_plus_100"] == 900
    assert report["model_mapping_verified"] is False
    assert "raw_at_model_zero" not in joint


def test_legacy_file_is_preserved_not_silently_reinterpreted(tmp_path):
    path = tmp_path / "legacy.json"
    data = dict(
        motor_names=["wrist_roll"],
        homing_offset=[3094],
        drive_mode=[1],
        start_pos=[3065],
        end_pos=[-2070],
        calib_mode=["DEGREE"],
    )
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    report = audit_calibration(path)
    assert report["schema"] == "legacy_array_calibration"
    assert report["joints"]["wrist_roll"]["end_pos"] == -2070
    assert "raw_at_normalized_zero" not in report["joints"]["wrist_roll"]
    assert report["model_mapping_verified"] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"motor_names": ["wrist_flex"]},
        {
            "wrist_flex": {
                "id": 4,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 1000,
                "range_max": 1000,
            }
        },
    ],
)
def test_rejects_unknown_or_invalid_calibration(tmp_path, data):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        audit_calibration(path)


@pytest.mark.parametrize("drive_mode, offset, zero", [(0, -2019, 2019), (1, 2192, 2192)])
def test_legacy_nominal_zero_uses_software_offset_convention(tmp_path, drive_mode, offset, zero):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            dict(
                motor_names=["joint"],
                homing_offset=[offset],
                drive_mode=[drive_mode],
                start_pos=[100],
                end_pos=[200],
                calib_mode=["DEGREE"],
            )
        )
    )
    joint = audit_calibration(path)["joints"]["joint"]
    assert joint["encoder_at_legacy_nominal_zero"] == zero
    assert joint["legacy_nominal_sign"] * zero + offset == 0


def test_gripper_uses_zero_to_100_not_arm_range(tmp_path):
    path = tmp_path / "gripper.json"
    path.write_text(
        json.dumps({"gripper": dict(id=6, drive_mode=1, homing_offset=0, range_min=100, range_max=900)})
    )
    joint = audit_calibration(path)["joints"]["gripper"]
    assert joint["raw_at_normalized_zero"] == 900
    assert joint["raw_at_normalized_100"] == 100
    assert "raw_at_normalized_minus_100" not in joint
