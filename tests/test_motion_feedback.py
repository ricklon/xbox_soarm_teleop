"""User-facing motion feedback: why the arm stopped, and an honest IK-log verdict."""

import numpy as np
import pytest

from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES, JOINT_LIMITS_DEG
from xbox_soarm_teleop.control.motion_feedback import TrackingStats, describe_motion_limit
from xbox_soarm_teleop.kinematics.constrained import IKResult
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta

BOUNDS = {"x": (0.05, 0.5), "y": (None, None), "z": (0.05, 0.45)}
MID_JOINTS = np.zeros(4)
MID_POS = np.array([0.3, 0.0, 0.2])


def _describe(**overrides):
    kwargs = dict(
        homing=False,
        delta=EEDelta(),
        target_pos=MID_POS,
        workspace_bounds=BOUNDS,
        ik_joints_deg=MID_JOINTS,
        ik_result=None,
    )
    kwargs.update(overrides)
    return describe_motion_limit(**kwargs)


class TestDescribeMotionLimit:
    def test_free_motion_has_no_message(self):
        assert _describe(delta=EEDelta(dx=0.05)) == ""

    def test_idle_has_no_message_even_at_a_bound(self):
        assert _describe(target_pos=np.array([0.3, 0.0, 0.05])) == ""

    def test_homing_is_reported(self):
        assert _describe(homing=True) == "homing"

    def test_floor(self):
        msg = _describe(delta=EEDelta(dz=-0.1), target_pos=np.array([0.3, 0.0, 0.05]))
        assert msg == "at floor"

    def test_moving_away_from_floor_is_not_blocked(self):
        assert _describe(delta=EEDelta(dz=0.1), target_pos=np.array([0.3, 0.0, 0.05])) == ""

    @pytest.mark.parametrize(
        ("delta", "pos", "expected"),
        [
            (EEDelta(dz=0.1), [0.3, 0.0, 0.45], "at ceiling"),
            (EEDelta(dx=0.1), [0.5, 0.0, 0.2], "at front limit"),
            (EEDelta(dx=-0.1), [0.05, 0.0, 0.2], "at back limit"),
        ],
    )
    def test_other_workspace_bounds(self, delta, pos, expected):
        assert _describe(delta=delta, target_pos=np.array(pos)) == expected

    def test_side_limit(self):
        bounds = dict(BOUNDS, y=(-0.3, 0.3))
        msg = _describe(
            delta=EEDelta(dy=0.1), target_pos=np.array([0.3, 0.3, 0.2]), workspace_bounds=bounds
        )
        assert msg == "at side limit"

    def test_joint_limit_names_the_joint(self):
        joints = MID_JOINTS.copy()
        joints[IK_JOINT_NAMES.index("wrist_flex")] = JOINT_LIMITS_DEG["wrist_flex"][1]
        assert _describe(delta=EEDelta(dx=0.05), ik_joints_deg=joints) == "joint limit: wrist_flex"

    def test_joint_at_limit_is_not_reported_while_motion_still_progresses(self):
        joints = MID_JOINTS.copy()
        joints[IK_JOINT_NAMES.index("wrist_flex")] = JOINT_LIMITS_DEG["wrist_flex"][1]
        assert _describe(delta=EEDelta(dy=0.05), ik_joints_deg=joints, progress=0.95) == ""
        assert (
            _describe(delta=EEDelta(dy=0.05), ik_joints_deg=joints, progress=0.1)
            == "joint limit: wrist_flex"
        )

    def test_ik_rejection_is_out_of_reach(self):
        result = IKResult(False, "not_converged", 30, 0.002, 0.0)
        assert _describe(delta=EEDelta(dx=0.05), ik_result=result) == "out of reach"

    def test_ik_joint_limit_rejection(self):
        result = IKResult(False, "joint_limit", 3, 0.01, 0.0)
        assert _describe(delta=EEDelta(dx=0.05), ik_result=result) == "joint limit"

    def test_accepted_ik_is_not_reported(self):
        result = IKResult(True, "accepted", 2, 0.0, 0.0)
        assert _describe(delta=EEDelta(dx=0.05), ik_result=result) == ""

    def test_workspace_message_takes_priority_over_ik(self):
        result = IKResult(False, "not_converged", 30, 0.002, 0.0)
        msg = _describe(
            delta=EEDelta(dz=-0.1), target_pos=np.array([0.3, 0.0, 0.05]), ik_result=result
        )
        assert msg == "at floor"


class TestTrackingStats:
    def test_homing_samples_do_not_affect_verdict(self):
        """Regression: a home move logged 103 mm 'error' and the session printed FAIL."""
        stats = TrackingStats(max_err_mm=30.0, mean_err_mm=10.0)
        for _ in range(100):
            stats.add(0.0005, phase="moving")
        for err in np.linspace(0.1, 0.0, 40):
            stats.add(float(err), phase="homing")
        assert stats.passed
        assert stats.count("homing") == 40
        assert stats.max_mm() == pytest.approx(0.5)

    def test_real_tracking_error_still_fails(self):
        stats = TrackingStats(max_err_mm=30.0, mean_err_mm=10.0)
        stats.add(0.05, phase="moving")
        assert not stats.passed

    def test_limit_stops_are_counted(self):
        stats = TrackingStats(max_err_mm=30.0, mean_err_mm=10.0)
        stats.add(0.0, phase="moving", limit="at floor")
        stats.add(0.0, phase="moving", limit="at floor")
        stats.add(0.0, phase="moving", limit="out of reach")
        assert stats.limit_counts == {"at floor": 2, "out of reach": 1}

    def test_no_tracked_samples_passes_with_note(self):
        stats = TrackingStats(max_err_mm=30.0, mean_err_mm=10.0)
        stats.add(0.1, phase="homing")
        assert stats.passed
        assert "no tracked samples" in "\n".join(stats.summary_lines())

    def test_summary_mentions_excluded_homing(self):
        stats = TrackingStats(max_err_mm=30.0, mean_err_mm=10.0)
        stats.add(0.001, phase="moving")
        stats.add(0.1, phase="homing")
        text = "\n".join(stats.summary_lines())
        assert "homing" in text and "excluded" in text
        assert text.splitlines()[-1] == "PASS"


def test_commanded_progress():
    from xbox_soarm_teleop.control.motion_feedback import commanded_progress

    d = EEDelta(dx=0.1)
    assert commanded_progress([0, 0, 0], [0.002, 0, 0], d, 0.02) == pytest.approx(1.0)
    assert commanded_progress([0, 0, 0], [0.0, 0.001, 0], d, 0.02) == pytest.approx(0.0)
    assert commanded_progress([0, 0, 0], [0.001, 0, 0], EEDelta(), 0.02) is None
