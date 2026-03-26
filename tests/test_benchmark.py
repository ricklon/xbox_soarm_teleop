"""Tests for LoopTimer benchmark instrumentation."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

from xbox_soarm_teleop.diagnostics.benchmark import LoopTimer


def _fill_timer(n: int = 10, mode: str = "crane") -> LoopTimer:
    timer = LoopTimer(mode=mode, rss_every=5)
    for i in range(n):
        timer.start_frame()
        time.sleep(0.001)  # 1 ms simulated work
        timer.record(i, controller_ms=0.5, ik_ms=1.0, servo_ms=0.3)
    return timer


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_frame_count_zero_initially():
    timer = LoopTimer(mode="joint")
    assert timer.frame_count == 0


def test_frame_count_increments():
    timer = _fill_timer(7)
    assert timer.frame_count == 7


def test_record_stores_all_fields():
    timer = LoopTimer(mode="joint", rss_every=0)
    timer.start_frame()
    timer.record(0, controller_ms=0.5, ik_ms=1.2, servo_ms=0.8)
    row = timer._rows[0]
    assert row["mode"] == "joint"
    assert float(row["controller_read_ms"]) == pytest.approx(0.5, abs=0.001)
    assert float(row["ik_solve_ms"]) == pytest.approx(1.2, abs=0.001)
    assert float(row["servo_write_ms"]) == pytest.approx(0.8, abs=0.001)
    assert float(row["loop_total_ms"]) >= 0.0


def test_loop_total_reflects_elapsed():
    timer = LoopTimer(mode="crane", rss_every=0)
    timer.start_frame()
    time.sleep(0.005)
    timer.record(0, controller_ms=1.0, ik_ms=0.0, servo_ms=0.5)
    total = float(timer._rows[0]["loop_total_ms"])
    # Should be at least 5ms (sleep) and not more than 50ms (CI slack)
    assert total >= 4.0
    assert total < 100.0


def test_rss_sampled_at_correct_interval():
    timer = LoopTimer(mode="crane", rss_every=3)
    for i in range(9):
        timer.start_frame()
        timer.record(i, controller_ms=0.0, ik_ms=0.0, servo_ms=0.0)
    # Frames 0, 3, 6 should have rss populated
    for i, row in enumerate(timer._rows):
        if i % 3 == 0:
            assert row["rss_mb"] != "", f"frame {i} should have rss"
        else:
            assert row["rss_mb"] == "", f"frame {i} should not have rss"


def test_rss_disabled_when_zero():
    timer = LoopTimer(mode="joint", rss_every=0)
    for i in range(5):
        timer.start_frame()
        timer.record(i, 0.0, 0.0, 0.0)
    for row in timer._rows:
        assert row["rss_mb"] == ""


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def test_write_csv_creates_file(tmp_path: Path):
    timer = _fill_timer(3)
    out = tmp_path / "bench.csv"
    timer.write_csv(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_write_csv_header(tmp_path: Path):
    timer = _fill_timer(3)
    out = tmp_path / "bench.csv"
    timer.write_csv(out)
    with open(out) as f:
        reader = csv.DictReader(f)
        assert set(LoopTimer._HEADER).issubset(set(reader.fieldnames or []))


def test_write_csv_row_count(tmp_path: Path):
    n = 8
    timer = _fill_timer(n)
    out = tmp_path / "bench.csv"
    timer.write_csv(out)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n


def test_write_csv_returns_path(tmp_path: Path):
    timer = _fill_timer(2)
    out = tmp_path / "bench.csv"
    result = timer.write_csv(out)
    assert result == out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_no_data():
    timer = LoopTimer(mode="crane")
    s = timer.summary()
    assert "No benchmark data" in s


def test_summary_contains_key_fields():
    timer = _fill_timer(20)
    s = timer.summary()
    assert "frames" in s
    assert "Hz" in s
    assert "controller_read" in s
    assert "ik_solve" in s
    assert "servo_write" in s


# ---------------------------------------------------------------------------
# default_path
# ---------------------------------------------------------------------------


def test_default_path_is_csv():
    p = LoopTimer.default_path("test_bench")
    assert p.suffix == ".csv"
    assert "test_bench" in p.name
