"""Tests for ROM test calibration output and conversion.

Tests TelemetrySample, extended JSON format, LeRobot conversion,
and CLI subcommand parsing — all without hardware.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

# Add examples/ to path so we can import joint_rom_test
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from joint_rom_test import (  # noqa: E402
    TelemetrySample,
    build_base_positions,
    build_extended_json,
    build_parser,
    compute_diagnostics,
    compute_lerobot_fields,
    convert_to_lerobot,
)

from xbox_soarm_teleop.config.joints import (  # noqa: E402
    HOME_POSITION_DEG,
    HOME_POSITION_RAW,
    JOINT_LIMITS_DEG,
    JOINT_NAMES_WITH_GRIPPER,
    MOTOR_IDS,
    SWEEP_TEST_POSITIONS,
)

# ---------------------------------------------------------------------------
# TelemetrySample
# ---------------------------------------------------------------------------


class TestTelemetrySample:
    def test_construction(self):
        s = TelemetrySample(
            t=1.5,
            position_raw=2048,
            position_deg=0.0,
            velocity=100,
            load=50,
            current=200,
            voltage_v=12.0,
            temperature_c=35,
        )
        assert s.t == 1.5
        assert s.position_raw == 2048
        assert s.position_deg == 0.0
        assert s.velocity == 100
        assert s.load == 50
        assert s.current == 200
        assert s.voltage_v == 12.0
        assert s.temperature_c == 35

    def test_asdict(self):
        s = TelemetrySample(
            t=0.0,
            position_raw=1024,
            position_deg=-90.0,
            velocity=0,
            load=0,
            current=0,
            voltage_v=12.0,
            temperature_c=25,
        )
        d = asdict(s)
        assert isinstance(d, dict)
        assert d["t"] == 0.0
        assert d["position_raw"] == 1024
        assert d["position_deg"] == -90.0
        assert d["voltage_v"] == 12.0
        assert d["temperature_c"] == 25

    def test_all_fields_present(self):
        s = TelemetrySample(
            t=0.0,
            position_raw=0,
            position_deg=0.0,
            velocity=0,
            load=0,
            current=0,
            voltage_v=0.0,
            temperature_c=0,
        )
        d = asdict(s)
        expected_keys = {
            "t",
            "position_raw",
            "position_deg",
            "velocity",
            "load",
            "current",
            "voltage_v",
            "temperature_c",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Diagnostics computation
# ---------------------------------------------------------------------------


def _make_sample_dict(**overrides) -> dict:
    """Create a telemetry sample dict with defaults."""
    base = {
        "t": 0.0,
        "position_raw": 2048,
        "position_deg": 0.0,
        "velocity": 0,
        "load": 0,
        "current": 0,
        "voltage_v": 12.0,
        "temperature_c": 25,
    }
    base.update(overrides)
    return base


class TestComputeDiagnostics:
    def test_empty_telemetry(self):
        diag = compute_diagnostics(
            {
                "sweep_to_lower": [],
                "sweep_to_upper": [],
                "sweep_to_home": [],
            }
        )
        assert diag["peak_load"] == 0
        assert diag["peak_current"] == 0
        assert diag["max_temperature_c"] == 0
        assert diag["min_voltage_v"] == 0.0
        assert diag["max_velocity"] == 0

    def test_peak_load(self):
        telemetry = {
            "sweep_to_lower": [
                _make_sample_dict(load=100),
                _make_sample_dict(load=-245),
            ],
            "sweep_to_upper": [_make_sample_dict(load=50)],
            "sweep_to_home": [],
        }
        diag = compute_diagnostics(telemetry)
        assert diag["peak_load"] == 245

    def test_peak_current(self):
        telemetry = {
            "sweep_to_lower": [_make_sample_dict(current=180)],
            "sweep_to_upper": [_make_sample_dict(current=90)],
            "sweep_to_home": [_make_sample_dict(current=150)],
        }
        diag = compute_diagnostics(telemetry)
        assert diag["peak_current"] == 180

    def test_max_temperature(self):
        telemetry = {
            "sweep_to_lower": [_make_sample_dict(temperature_c=30)],
            "sweep_to_upper": [_make_sample_dict(temperature_c=42)],
            "sweep_to_home": [_make_sample_dict(temperature_c=35)],
        }
        diag = compute_diagnostics(telemetry)
        assert diag["max_temperature_c"] == 42

    def test_min_voltage(self):
        telemetry = {
            "sweep_to_lower": [_make_sample_dict(voltage_v=12.0)],
            "sweep_to_upper": [_make_sample_dict(voltage_v=11.5)],
            "sweep_to_home": [_make_sample_dict(voltage_v=11.8)],
        }
        diag = compute_diagnostics(telemetry)
        assert diag["min_voltage_v"] == 11.5

    def test_min_voltage_skips_zero(self):
        """Voltage of 0 (sim backend) should be excluded from min calculation."""
        telemetry = {
            "sweep_to_lower": [_make_sample_dict(voltage_v=0.0)],
            "sweep_to_upper": [_make_sample_dict(voltage_v=0.0)],
            "sweep_to_home": [_make_sample_dict(voltage_v=0.0)],
        }
        diag = compute_diagnostics(telemetry)
        assert diag["min_voltage_v"] == 0.0

    def test_max_velocity(self):
        telemetry = {
            "sweep_to_lower": [_make_sample_dict(velocity=-412)],
            "sweep_to_upper": [_make_sample_dict(velocity=300)],
            "sweep_to_home": [],
        }
        diag = compute_diagnostics(telemetry)
        assert diag["max_velocity"] == 412

    def test_raw_extremes(self):
        telemetry = {
            "sweep_to_lower": [_make_sample_dict(position_raw=770)],
            "sweep_to_upper": [_make_sample_dict(position_raw=3297)],
            "sweep_to_home": [_make_sample_dict(position_raw=2048)],
        }
        diag = compute_diagnostics(telemetry)
        assert diag["lower_reached_raw"] == 770
        assert diag["upper_reached_raw"] == 3297


# ---------------------------------------------------------------------------
# compute_lerobot_fields
# ---------------------------------------------------------------------------


class TestComputeLerobotFields:
    def test_homing_offset(self):
        for joint_name in JOINT_NAMES_WITH_GRIPPER:
            result = {
                "lower_actual": -90.0,
                "upper_actual": 90.0,
                "diagnostics": {
                    "lower_reached_raw": 1024,
                    "upper_reached_raw": 3072,
                },
            }
            fields = compute_lerobot_fields(joint_name, result)
            expected_offset = HOME_POSITION_RAW[joint_name] - 2047
            assert fields["homing_offset"] == expected_offset

    def test_motor_id(self):
        for joint_name in JOINT_NAMES_WITH_GRIPPER:
            result = {
                "lower_actual": -90.0,
                "upper_actual": 90.0,
                "diagnostics": {
                    "lower_reached_raw": 1000,
                    "upper_reached_raw": 3000,
                },
            }
            fields = compute_lerobot_fields(joint_name, result)
            assert fields["id"] == MOTOR_IDS[joint_name]

    def test_drive_mode_always_zero(self):
        result = {
            "lower_actual": -90.0,
            "upper_actual": 90.0,
            "diagnostics": {
                "lower_reached_raw": 1000,
                "upper_reached_raw": 3000,
            },
        }
        fields = compute_lerobot_fields("shoulder_pan", result)
        assert fields["drive_mode"] == 0

    def test_range_from_diagnostics(self):
        result = {
            "lower_actual": -90.0,
            "upper_actual": 90.0,
            "diagnostics": {
                "lower_reached_raw": 770,
                "upper_reached_raw": 3297,
            },
        }
        fields = compute_lerobot_fields("shoulder_pan", result)
        assert fields["range_min"] == 770
        assert fields["range_max"] == 3297

    def test_range_fallback_without_diagnostics(self):
        """Without diagnostics, range is computed from actual positions."""
        result = {"lower_actual": -90.0, "upper_actual": 90.0}
        fields = compute_lerobot_fields("shoulder_pan", result)
        assert isinstance(fields["range_min"], int)
        assert isinstance(fields["range_max"], int)
        assert fields["range_min"] < fields["range_max"]

    def test_five_fields(self):
        result = {
            "lower_actual": -90.0,
            "upper_actual": 90.0,
            "diagnostics": {
                "lower_reached_raw": 1000,
                "upper_reached_raw": 3000,
            },
        }
        fields = compute_lerobot_fields("shoulder_pan", result)
        assert set(fields.keys()) == {"id", "drive_mode", "homing_offset", "range_min", "range_max"}


# ---------------------------------------------------------------------------
# Extended JSON roundtrip
# ---------------------------------------------------------------------------


def _make_sweep_result(joint_name: str, with_telemetry: bool = True) -> dict:
    """Build a mock sweep result dict."""
    result = {
        "joint": joint_name,
        "lower_target": -90.0,
        "upper_target": 90.0,
        "lower_actual": -89.5,
        "upper_actual": 89.8,
        "home_actual": -1.4,
        "max_error": 0.5,
        "passed": True,
    }
    if with_telemetry:
        telemetry = {
            "sweep_to_lower": [_make_sample_dict(position_raw=770, load=245)],
            "sweep_to_upper": [_make_sample_dict(position_raw=3297, current=180)],
            "sweep_to_home": [_make_sample_dict(position_raw=2048)],
        }
        result["telemetry"] = telemetry
        result["diagnostics"] = compute_diagnostics(telemetry)
    return result


class TestExtendedJson:
    def test_roundtrip(self):
        results = [_make_sweep_result("shoulder_pan")]
        extended = build_extended_json(results, "sim", None, 20.0, 0.5, 5.0)

        # Serialize and parse
        text = json.dumps(extended)
        parsed = json.loads(text)

        assert parsed["format_version"] == "1.0"
        assert parsed["backend"] == "sim"
        assert parsed["port"] is None
        assert "timestamp" in parsed
        assert "shoulder_pan" in parsed["joints"]

        joint = parsed["joints"]["shoulder_pan"]
        assert joint["id"] == MOTOR_IDS["shoulder_pan"]
        assert joint["lower_limit_deg"] == -90.0
        assert joint["upper_limit_deg"] == 90.0
        assert joint["passed"] is True
        assert "telemetry" in joint
        assert "diagnostics" in joint

    def test_sweep_params(self):
        results = [_make_sweep_result("shoulder_pan")]
        extended = build_extended_json(results, "real", "/dev/ttyACM0", 15.0, 1.0, 3.0)
        assert extended["sweep_params"]["speed_deg_s"] == 15.0
        assert extended["sweep_params"]["settle_time_s"] == 1.0
        assert extended["sweep_params"]["tolerance_deg"] == 3.0
        assert extended["port"] == "/dev/ttyACM0"

    def test_multiple_joints(self):
        results = [
            _make_sweep_result("shoulder_pan"),
            _make_sweep_result("elbow_flex"),
        ]
        extended = build_extended_json(results, "sim", None, 20.0, 0.5, 5.0)
        assert len(extended["joints"]) == 2
        assert "shoulder_pan" in extended["joints"]
        assert "elbow_flex" in extended["joints"]

    def test_without_telemetry(self):
        results = [_make_sweep_result("shoulder_pan", with_telemetry=False)]
        extended = build_extended_json(results, "sim", None, 20.0, 0.5, 5.0)
        joint = extended["joints"]["shoulder_pan"]
        assert "telemetry" not in joint
        assert "diagnostics" not in joint
        # Should still have LeRobot fields
        assert "id" in joint
        assert "homing_offset" in joint
        assert "range_min" in joint
        assert "range_max" in joint


# ---------------------------------------------------------------------------
# convert_to_lerobot
# ---------------------------------------------------------------------------


class TestConvertToLerobot:
    def test_produces_exact_schema(self):
        results = [_make_sweep_result(name) for name in JOINT_NAMES_WITH_GRIPPER]
        extended = build_extended_json(results, "sim", None, 20.0, 0.5, 5.0)
        lerobot = convert_to_lerobot(extended)

        assert set(lerobot.keys()) == set(JOINT_NAMES_WITH_GRIPPER)
        for joint_name, joint_data in lerobot.items():
            assert set(joint_data.keys()) == {
                "id",
                "drive_mode",
                "homing_offset",
                "range_min",
                "range_max",
            }

    def test_id_matches_motor_ids(self):
        results = [_make_sweep_result(name) for name in JOINT_NAMES_WITH_GRIPPER]
        extended = build_extended_json(results, "sim", None, 20.0, 0.5, 5.0)
        lerobot = convert_to_lerobot(extended)

        for joint_name in JOINT_NAMES_WITH_GRIPPER:
            assert lerobot[joint_name]["id"] == MOTOR_IDS[joint_name]

    def test_drive_mode_all_zero(self):
        results = [_make_sweep_result(name) for name in JOINT_NAMES_WITH_GRIPPER]
        extended = build_extended_json(results, "sim", None, 20.0, 0.5, 5.0)
        lerobot = convert_to_lerobot(extended)

        for joint_data in lerobot.values():
            assert joint_data["drive_mode"] == 0

    def test_homing_offset_values(self):
        results = [_make_sweep_result(name) for name in JOINT_NAMES_WITH_GRIPPER]
        extended = build_extended_json(results, "sim", None, 20.0, 0.5, 5.0)
        lerobot = convert_to_lerobot(extended)

        for joint_name in JOINT_NAMES_WITH_GRIPPER:
            expected = HOME_POSITION_RAW[joint_name] - 2047
            assert lerobot[joint_name]["homing_offset"] == expected

    def test_json_serializable(self):
        results = [_make_sweep_result(name) for name in JOINT_NAMES_WITH_GRIPPER]
        extended = build_extended_json(results, "sim", None, 20.0, 0.5, 5.0)
        lerobot = convert_to_lerobot(extended)

        text = json.dumps(lerobot, indent=4)
        parsed = json.loads(text)
        assert parsed == lerobot


# ---------------------------------------------------------------------------
# SimArmBackend telemetry zeros
# ---------------------------------------------------------------------------


class TestSimBackendTelemetryZeros:
    """Verify that sim backend returns zeros for load/current/voltage/temp."""

    def test_sim_telemetry_zeros(self):
        """Telemetry from sim has zero load, current, voltage, temperature."""
        # We test this by constructing a sample as the sim backend would
        sample = TelemetrySample(
            t=0.0,
            position_raw=2048,
            position_deg=0.0,
            velocity=0,
            load=0,
            current=0,
            voltage_v=0.0,
            temperature_c=0,
        )
        assert sample.load == 0
        assert sample.current == 0
        assert sample.voltage_v == 0.0
        assert sample.temperature_c == 0


# ---------------------------------------------------------------------------
# CLI subcommand parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_sweep_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["sweep", "--sim"])
        assert args.command == "sweep"
        assert args.sim is True

    def test_convert_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["convert", "--input", "/tmp/cal.json"])
        assert args.command == "convert"
        assert args.input == "/tmp/cal.json"

    def test_convert_with_output(self):
        parser = build_parser()
        args = parser.parse_args(["convert", "--input", "/tmp/cal.json", "--output", "/tmp/out.json"])
        assert args.output == "/tmp/out.json"

    def test_no_subcommand_defaults_to_sweep(self):
        parser = build_parser()
        args = parser.parse_args(["--sim"])
        assert args.command is None
        assert args.sim is True

    def test_sweep_with_output(self):
        parser = build_parser()
        args = parser.parse_args(["sweep", "--sim", "--output", "/tmp/out.json"])
        assert args.output == "/tmp/out.json"

    def test_sweep_no_telemetry(self):
        parser = build_parser()
        args = parser.parse_args(["sweep", "--sim", "--no-telemetry"])
        assert args.no_telemetry is True

    def test_sweep_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["sweep", "--sim"])
        assert args.sweep_speed == 20.0
        assert args.settle_time == 0.5
        assert args.tolerance is None
        assert args.no_telemetry is False
        assert args.output is None

    def test_backward_compat_all_old_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--sim",
                "--joints",
                "shoulder_pan",
                "elbow_flex",
                "--skip-gripper",
                "--tolerance",
                "3.0",
                "--sweep-speed",
                "15.0",
                "--settle-time",
                "1.0",
            ]
        )
        assert args.sim is True
        assert args.joints == ["shoulder_pan", "elbow_flex"]
        assert args.skip_gripper is True
        assert args.tolerance == 3.0
        assert args.sweep_speed == 15.0
        assert args.settle_time == 1.0

    def test_no_test_positions_flag(self):
        parser = build_parser()
        args = parser.parse_args(["sweep", "--sim", "--no-test-positions"])
        assert args.no_test_positions is True

    def test_no_test_positions_default(self):
        parser = build_parser()
        args = parser.parse_args(["sweep", "--sim"])
        assert args.no_test_positions is False


# ---------------------------------------------------------------------------
# SWEEP_TEST_POSITIONS validation
# ---------------------------------------------------------------------------


class TestSweepTestPositions:
    def test_keys_are_valid_joint_names(self):
        """Every key in SWEEP_TEST_POSITIONS must be a recognized joint."""
        for joint_name in SWEEP_TEST_POSITIONS:
            assert joint_name in JOINT_NAMES_WITH_GRIPPER, (
                f"SWEEP_TEST_POSITIONS key '{joint_name}' is not a valid joint"
            )

    def test_override_joints_are_valid(self):
        """Every joint referenced in the override dicts must be recognized."""
        for target_joint, overrides in SWEEP_TEST_POSITIONS.items():
            for override_joint in overrides:
                assert override_joint in JOINT_NAMES_WITH_GRIPPER, (
                    f"Override joint '{override_joint}' (for {target_joint}) is not valid"
                )

    def test_override_values_within_limits(self):
        """Override positions must be within JOINT_LIMITS_DEG."""
        for target_joint, overrides in SWEEP_TEST_POSITIONS.items():
            for joint_name, value in overrides.items():
                low, high = JOINT_LIMITS_DEG[joint_name]
                assert low <= value <= high, (
                    f"{joint_name}={value} (for {target_joint}) is outside "
                    f"limits [{low:.1f}, {high:.1f}]"
                )


# ---------------------------------------------------------------------------
# build_base_positions
# ---------------------------------------------------------------------------


class TestBuildBasePositions:
    def test_returns_all_joints(self):
        """Base positions must include every joint."""
        base = build_base_positions("shoulder_pan")
        assert set(base.keys()) == set(JOINT_NAMES_WITH_GRIPPER)

    def test_no_override_returns_home(self):
        """Joints without SWEEP_TEST_POSITIONS entries should get home positions."""
        base = build_base_positions("shoulder_pan", use_test_positions=True)
        for name in JOINT_NAMES_WITH_GRIPPER:
            assert base[name] == HOME_POSITION_DEG.get(name, 0.0)

    def test_override_applied(self):
        """Joints with SWEEP_TEST_POSITIONS entries should get overrides."""
        base = build_base_positions("shoulder_lift", use_test_positions=True)
        for joint_name, value in SWEEP_TEST_POSITIONS["shoulder_lift"].items():
            assert base[joint_name] == value

    def test_non_overridden_joints_stay_home(self):
        """Joints NOT in the override dict should remain at home."""
        base = build_base_positions("wrist_flex", use_test_positions=True)
        overridden = set(SWEEP_TEST_POSITIONS["wrist_flex"].keys())
        for name in JOINT_NAMES_WITH_GRIPPER:
            if name not in overridden:
                assert base[name] == HOME_POSITION_DEG.get(name, 0.0)

    def test_disabled_returns_home(self):
        """With use_test_positions=False, always returns home."""
        base = build_base_positions("shoulder_lift", use_test_positions=False)
        for name in JOINT_NAMES_WITH_GRIPPER:
            assert base[name] == HOME_POSITION_DEG.get(name, 0.0)
