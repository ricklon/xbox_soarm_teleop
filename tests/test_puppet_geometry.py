"""Puppet solves with the wrist command from the same input sample."""

from types import SimpleNamespace

import numpy as np
import pytest

from xbox_soarm_teleop.config.joints import HOME_POSITION_DEG, JOINT_LIMITS_DEG
from xbox_soarm_teleop.processors.puppet import PuppetProcessor
from xbox_soarm_teleop.teleoperators.xbox import XboxState


def make_processor(monkeypatch, accepted=True):
    monkeypatch.setattr(
        "xbox_soarm_teleop.processors.puppet.JoyConIMU",
        lambda **kwargs: SimpleNamespace(available=True, read=lambda: (0.1, 0.2)),
    )
    processor = PuppetProcessor()
    calls = []

    def solve(seed, reach, height, **kwargs):
        calls.append(kwargs)
        return np.asarray(seed).copy() if accepted else None

    processor._planar_ik = SimpleNamespace(
        inverse_cylindrical=solve, last_ik_result=SimpleNamespace(accepted=accepted)
    )
    return processor, calls


def test_imu_sample_is_included_before_solve(monkeypatch):
    processor, calls = make_processor(monkeypatch)
    before = processor._current_command().goals_deg
    result = processor(XboxState(left_bumper=True))
    assert len(calls) == 1
    expected = np.clip(
        HOME_POSITION_DEG["wrist_flex"] + np.rad2deg(0.1), *JOINT_LIMITS_DEG["wrist_flex"]
    )
    assert calls[0]["wrist_flex_deg"] == pytest.approx(expected)
    assert calls[0]["wrist_roll_deg"] == pytest.approx(
        HOME_POSITION_DEG["wrist_roll"] + np.rad2deg(0.2)
    )
    assert abs(result.goals_deg["wrist_roll"] - before["wrist_roll"]) <= 90 * processor.loop_dt + 1e-12


def test_rejected_puppet_target_holds_all_arm_joints(monkeypatch):
    processor, calls = make_processor(monkeypatch, accepted=False)
    before = processor._current_command().goals_deg
    result = processor(XboxState(left_bumper=True, left_stick_x=1, left_stick_y=-1))
    for name in before:
        if name != "gripper":
            assert result.goals_deg[name] == before[name]
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["crane", "puppet"])
def test_deadman_gates_home_and_gripper(monkeypatch, mode):
    from xbox_soarm_teleop.processors.crane import CraneProcessor

    if mode == "crane":
        processor = CraneProcessor()
        processor._homing_active = True
    else:
        processor, _ = make_processor(monkeypatch)
    processor._pan_deg = 30.0
    before = processor._current_command().goals_deg
    result = processor(XboxState(left_bumper=False, a_button_pressed=True, right_trigger=1.0))
    assert result.goals_deg == before
