"""Tests for joint configuration correctness."""

from pathlib import Path

import numpy as np
import pytest

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    HOME_POSITION_RAW,
    IK_JOINT_NAMES,
    IK_JOINT_VEL_LIMITS_ARRAY,
    IK_JOINT_VEL_LIMITS_DEG,
    JOINT_LIMITS_DEG,
    JOINT_LIMITS_RAD,
    JOINT_NAMES_WITH_GRIPPER,
    MOTOR_IDS,
    RAW_CENTER,
    RAW_PER_DEGREE,
    deg_to_raw,
    limits_rad_to_deg,
    parse_joint_limits,
    raw_to_deg,
)

URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"


# ---------------------------------------------------------------------------
# All joints have limits and home positions
# ---------------------------------------------------------------------------


class TestConfigCompleteness:
    def test_all_joints_have_limits(self):
        for name in JOINT_NAMES_WITH_GRIPPER:
            assert name in JOINT_LIMITS_DEG, f"Missing limits for {name}"
            assert name in JOINT_LIMITS_RAD, f"Missing rad limits for {name}"

    def test_all_joints_have_home_position(self):
        for name in JOINT_NAMES_WITH_GRIPPER:
            assert name in HOME_POSITION_DEG, f"Missing home deg for {name}"
            assert name in HOME_POSITION_RAW, f"Missing home raw for {name}"

    def test_all_joints_have_motor_ids(self):
        for name in JOINT_NAMES_WITH_GRIPPER:
            assert name in MOTOR_IDS, f"Missing motor ID for {name}"

    def test_ik_joints_have_vel_limits(self):
        for name in IK_JOINT_NAMES:
            assert name in IK_JOINT_VEL_LIMITS_DEG, f"Missing vel limit for {name}"

    def test_vel_limits_array_matches_dict(self):
        expected = np.array([IK_JOINT_VEL_LIMITS_DEG[n] for n in IK_JOINT_NAMES])
        np.testing.assert_array_equal(IK_JOINT_VEL_LIMITS_ARRAY, expected)

    def test_vel_limits_array_length(self):
        assert len(IK_JOINT_VEL_LIMITS_ARRAY) == len(IK_JOINT_NAMES)


# ---------------------------------------------------------------------------
# Home positions are within limits
# ---------------------------------------------------------------------------


class TestHomeWithinLimits:
    def test_home_within_limits_deg(self):
        for name in JOINT_NAMES_WITH_GRIPPER:
            home = HOME_POSITION_DEG[name]
            lower, upper = JOINT_LIMITS_DEG[name]
            assert lower <= home <= upper, (
                f"{name} home={home:.1f} outside limits [{lower:.1f}, {upper:.1f}]"
            )


# ---------------------------------------------------------------------------
# Servo conversion roundtrip
# ---------------------------------------------------------------------------


class TestServoConversion:
    @pytest.mark.parametrize("raw", [0, 1024, 2048, 3072, 4095])
    def test_raw_to_deg_roundtrip(self, raw: int):
        deg = raw_to_deg(raw)
        recovered = deg_to_raw(deg)
        # Allow ±1 raw step due to int truncation
        assert abs(recovered - raw) <= 1, (
            f"raw={raw} -> deg={deg:.2f} -> raw={recovered}"
        )

    @pytest.mark.parametrize("deg", [-180.0, -90.0, 0.0, 45.0, 90.0, 180.0])
    def test_deg_to_raw_roundtrip(self, deg: float):
        raw = deg_to_raw(deg)
        recovered = raw_to_deg(raw)
        assert abs(recovered - deg) < 0.1, (
            f"deg={deg} -> raw={raw} -> deg={recovered:.2f}"
        )

    def test_center_is_zero_deg(self):
        assert raw_to_deg(RAW_CENTER) == 0.0

    def test_raw_per_degree_positive(self):
        assert RAW_PER_DEGREE > 0


# ---------------------------------------------------------------------------
# Limits consistency: rad <-> deg
# ---------------------------------------------------------------------------


class TestLimitsConsistency:
    def test_rad_to_deg_conversion(self):
        for name in JOINT_NAMES_WITH_GRIPPER:
            rad_low, rad_high = JOINT_LIMITS_RAD[name]
            deg_low, deg_high = JOINT_LIMITS_DEG[name]
            np.testing.assert_allclose(
                deg_low, np.rad2deg(rad_low), atol=0.01,
                err_msg=f"{name} lower limit mismatch",
            )
            np.testing.assert_allclose(
                deg_high, np.rad2deg(rad_high), atol=0.01,
                err_msg=f"{name} upper limit mismatch",
            )

    def test_lower_less_than_upper(self):
        for name in JOINT_NAMES_WITH_GRIPPER:
            lower, upper = JOINT_LIMITS_DEG[name]
            assert lower < upper, f"{name}: lower={lower} >= upper={upper}"


# ---------------------------------------------------------------------------
# Pre-computed limits match URDF parse (if URDF present)
# ---------------------------------------------------------------------------


class TestURDFMatch:
    @pytest.mark.skipif(
        not URDF_PATH.exists(),
        reason="URDF file not found",
    )
    def test_precomputed_limits_match_urdf(self):
        # Some limits are intentionally tighter than URDF based on hardware measurement:
        #   gripper lower: -2° vs URDF -10° (hardware floor)
        #   wrist_flex upper: 72° vs URDF 95° (hard mechanical stop)
        INTENTIONAL_DIVERGENCE = {"gripper", "wrist_flex"}
        urdf_limits = parse_joint_limits(URDF_PATH, JOINT_NAMES_WITH_GRIPPER)
        for name, (urdf_low, urdf_high) in urdf_limits.items():
            if name in INTENTIONAL_DIVERGENCE:
                continue
            precomputed_low, precomputed_high = JOINT_LIMITS_RAD[name]
            np.testing.assert_allclose(
                precomputed_low, urdf_low, atol=1e-4,
                err_msg=f"{name} lower limit doesn't match URDF",
            )
            np.testing.assert_allclose(
                precomputed_high, urdf_high, atol=1e-4,
                err_msg=f"{name} upper limit doesn't match URDF",
            )


# ---------------------------------------------------------------------------
# limits_rad_to_deg utility
# ---------------------------------------------------------------------------


class TestLimitsRadToDeg:
    def test_basic_conversion(self):
        limits = {"test": (-np.pi, np.pi)}
        result = limits_rad_to_deg(limits)
        np.testing.assert_allclose(result["test"][0], -180.0, atol=0.01)
        np.testing.assert_allclose(result["test"][1], 180.0, atol=0.01)

    def test_empty_dict(self):
        assert limits_rad_to_deg({}) == {}
