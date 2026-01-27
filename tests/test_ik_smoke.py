"""Optional IK smoke test (opt-in via IK_SMOKE=1)."""

from __future__ import annotations

import os

import pytest

from xbox_soarm_teleop.cli import ik_smoke


def test_ik_smoke_opt_in():
    if os.environ.get("IK_SMOKE") != "1":
        pytest.skip("Set IK_SMOKE=1 to enable the IK smoke test.")

    # Skip if placo isn't available
    pytest.importorskip("placo")

    # Fast but meaningful: short run, modest thresholds
    exit_code = ik_smoke.run_smoke_test(
        duration_s=3.0,
        step_hz=30.0,
        max_pos_err_m=0.03,  # 30 mm
        mean_pos_err_m=0.01,  # 10 mm
        verbose=False,
    )
    assert exit_code == 0
