"""Inspect calibration evidence offline; never connect to or configure a robot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def audit_calibration(path: str | Path) -> dict:
    """Report calibration format and encoder references without assuming URDF zero.

    Args:
        path: Existing calibration JSON, read without modification.

    Returns:
        JSON-compatible evidence report with a source fingerprint.

    Raises:
        ValueError: The calibration format or values are invalid.
    """
    source = Path(path)
    raw = source.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not data:
        raise ValueError("Expected a nonempty calibration object")
    report = {
        "source": str(source.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "model_mapping_verified": False,
        "joints": {},
    }
    if "motor_names" in data:
        names = data["motor_names"]
        fields = ("homing_offset", "drive_mode", "start_pos", "end_pos", "calib_mode")
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
            or any(
                not isinstance(data.get(key), list) or len(data[key]) != len(names) for key in fields
            )
        ):
            raise ValueError("Legacy calibration arrays must match unique motor_names")
        report["schema"] = "legacy_array_calibration"
        report["notes"] = [
            "Legacy DEGREE applies direction, then software homing offset, then degree scale.",
            "Legacy nominal zero is not established as URDF zero.",
            "Do not reinterpret start_pos/end_pos as modern calibrated range endpoints.",
        ]
        for index, name in enumerate(names):
            joint = {key: data[key][index] for key in fields}
            if (
                not all(_number(joint[key]) for key in fields[:4])
                or joint["drive_mode"] not in (0, 1)
                or joint["calib_mode"] not in ("DEGREE", "LINEAR")
            ):
                raise ValueError(f"Invalid legacy calibration for {name}")
            if joint["calib_mode"] == "DEGREE":
                sign = -1 if joint["drive_mode"] else 1
                joint["encoder_at_legacy_nominal_zero"] = -sign * joint["homing_offset"]
                joint["legacy_nominal_sign"] = sign
            report["joints"][name] = joint
    else:
        report["schema"] = "lerobot_motor_calibration"
        report["notes"] = [
            "Arm rows assume RANGE_M100_100 and apply_drive_mode=True, as in SOFollower.",
            "Encoder coordinates already include the active hardware homing offset.",
            "Normalized zero is the calibrated range midpoint, not a verified URDF zero.",
            "Model mapping requires an independently established encoder zero, sign, and scale.",
        ]
        fields = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
        for name, entry in data.items():
            if (
                not isinstance(entry, dict)
                or any(key not in entry for key in fields)
                or not all(_number(entry[key]) for key in fields)
                or entry["drive_mode"] not in (0, 1)
                or entry["range_min"] >= entry["range_max"]
            ):
                raise ValueError(f"Invalid modern motor calibration for {name}")
            joint = {key: entry[key] for key in fields}
            lower, upper = entry["range_min"], entry["range_max"]
            if entry["drive_mode"]:
                lower, upper = upper, lower
            if name == "gripper":
                joint.update(raw_at_normalized_zero=lower, raw_at_normalized_100=upper)
            else:
                joint.update(
                    raw_at_normalized_minus_100=lower,
                    raw_at_normalized_zero=(lower + upper) / 2,
                    raw_at_normalized_plus_100=upper,
                )
            report["joints"][name] = joint
    return report


def main() -> int:
    """Print an offline calibration report as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Calibration JSON to inspect")
    args = parser.parse_args()
    try:
        report = audit_calibration(args.input)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
