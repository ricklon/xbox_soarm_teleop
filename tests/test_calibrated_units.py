"""Calibration conversion contracts, using synthetic data and no hardware."""

import numpy as np
import pytest

from xbox_soarm_teleop.control.calibrated_units import CalibratedJointMapping


def mapping(**changes):
    values = dict(
        range_min=900,
        range_max=3100,
        drive_mode=0,
        raw_at_model_zero=2000.0,
        model_sign=1,
        ticks_per_turn=4096,
    )
    return CalibratedJointMapping(**(values | changes))


def test_asymmetric_range_does_not_define_model_zero():
    joint = mapping(raw_at_model_zero=1800)
    assert joint.normalized_to_degrees(0) == pytest.approx(200 * 360 / 4096)
    assert joint.degrees_to_normalized(0) == pytest.approx(-200 / 1100 * 100)


@pytest.mark.parametrize("drive_mode", [0, 1])
@pytest.mark.parametrize("model_sign", [-1, 1])
def test_direction_and_roundtrip(drive_mode, model_sign):
    joint = mapping(drive_mode=drive_mode, model_sign=model_sign)
    for command in [-100, -75, 0, 35, 100]:
        angle = joint.normalized_to_degrees(command)
        assert joint.degrees_to_normalized(angle) == pytest.approx(command)
    assert np.sign(joint.normalized_to_degrees(100)) == model_sign * (-1 if drive_mode else 1)


@pytest.mark.parametrize("drive_mode", [0, 1])
def test_matches_installed_lerobot_bus_without_connecting(drive_mode):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    # Construct only the conversion state: no serial port or connection.
    bus = FeetechMotorsBus.__new__(FeetechMotorsBus)
    bus._id_to_name_dict = {4: "wrist_flex"}
    bus.motors = {"wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100)}
    bus.calibration = {"wrist_flex": MotorCalibration(4, drive_mode, 123, 900, 3100)}
    joint = mapping(drive_mode=drive_mode, raw_at_model_zero=1800, model_sign=-1)
    for raw in [900, 1500, 1800, 2000, 2500, 3100]:
        observed = bus._normalize({4: raw})[4]
        expected = -(raw - 1800) * 360 / 4096
        assert joint.normalized_to_degrees(observed) == pytest.approx(expected)
        command = joint.degrees_to_normalized(expected)
        # LeRobot truncates the inverse result to an encoder tick.
        assert abs(bus._unnormalize({4: command})[4] - raw) <= 1


@pytest.mark.parametrize(
    "field, value",
    [
        ("range_min", 3100),
        ("range_max", 800),
        ("drive_mode", 2),
        ("model_sign", 0),
        ("ticks_per_turn", 0),
        ("raw_at_model_zero", float("nan")),
        ("ticks_per_turn", float("inf")),
    ],
)
def test_invalid_mapping_rejected(field, value):
    with pytest.raises(ValueError):
        mapping(**{field: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -101, 101])
def test_bad_normalized_value_is_not_silently_clipped(value):
    with pytest.raises(ValueError):
        mapping().normalized_to_degrees(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -180, 180])
def test_invalid_or_out_of_calibration_command_rejected(value):
    with pytest.raises(ValueError):
        mapping().degrees_to_normalized(value)
