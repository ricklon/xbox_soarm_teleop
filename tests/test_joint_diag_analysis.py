"""Tests for joint diagnostic CSV analysis."""

from pathlib import Path

from xbox_soarm_teleop.diagnostics.joint_diag_analysis import analyze_joint_diagnostic_csv


def test_analyze_joint_diagnostic_csv_basic(tmp_path: Path):
    csv_path = tmp_path / "diag.csv"
    csv_path.write_text(
        "\n".join(
            [
                "t_s,selected_joint,cmd_vel_deg_s,selected_goal_deg,selected_pos_deg",
                "0.0,shoulder_pan,10,0,0",
                "0.1,shoulder_pan,10,1,0.8",
                "0.2,shoulder_pan,10,2,1.7",
                "0.0,wrist_roll,0,5,5",
                "0.1,wrist_roll,0,5,5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = analyze_joint_diagnostic_csv(csv_path, cmd_threshold_deg_s=5.0)

    assert summary.total_rows == 5
    by_joint = {j.joint: j for j in summary.per_joint}

    shoulder = by_joint["shoulder_pan"]
    assert shoulder.samples == 3
    assert shoulder.active_cmd_samples == 3
    assert shoulder.cmd_abs_peak_deg_s == 10.0
    assert shoulder.measured_abs_mean_deg_s > 0.0
    assert shoulder.tracking_err_mean_deg > 0.0
    assert shoulder.observed_span_deg == 1.7

    wrist = by_joint["wrist_roll"]
    assert wrist.samples == 2
    assert wrist.active_cmd_samples == 0
    assert wrist.measured_abs_peak_deg_s == 0.0
    assert wrist.tracking_err_mean_deg == 0.0


def test_analyze_joint_diagnostic_csv_ignores_unknown_joint(tmp_path: Path):
    csv_path = tmp_path / "diag_unknown.csv"
    csv_path.write_text(
        "\n".join(
            [
                "t_s,selected_joint,cmd_vel_deg_s,selected_goal_deg,selected_pos_deg",
                "0.0,unknown_joint,20,0,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = analyze_joint_diagnostic_csv(csv_path)
    assert summary.total_rows == 1
    assert summary.per_joint == []
