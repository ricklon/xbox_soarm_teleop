"""Compare agent trial conditions side by side (from agent_trial.py's JSONL).

    uv run python scripts/agent_report.py twin_logs/agent_ab.jsonl

One row per model, harness and variant (no_judge, judge, judge+required).
Runs where the judge could not answer (check_errors) or a provider refused the
agent model or the pointer (HTTP 402, 429, 5xx) are counted as invalid and left
out of the rates:
how often the object really ended in the box, how often the agent stopped by
itself, how often its RESULT claim was true, how many success checks it made
and how many of those were right, median time, and median and total cost
(agent model + pointer + judge).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# The same pattern as agent_trial.PROVIDER_ERROR: a provider refused a request.
PROVIDER_ERROR = re.compile(r"HTTP (402|429|5\d\d)")


def _invalid(row: dict) -> bool:
    """A judge that could not answer, or a provider that refused the agent model
    or the pointer: the run tests neither the agent nor the judge."""
    return bool(
        row.get("check_errors")
        or row.get("provider_errors")
        or PROVIDER_ERROR.search(str(row.get("agent_error") or ""))
    )


def _rate(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(bool(v) for v in values) / len(values), 2) if values else None


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("model"), row.get("harness", "opencode"), row.get("variant", "no_judge"))
        groups.setdefault(key, []).append(row)
    table = []
    for (model, harness, variant), everything in sorted(
        groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        # A judged run whose judge could not answer tests nothing about the judge.
        group = [row for row in everything if not _invalid(row)]
        invalid = len(everything) - len(group)
        if not group:
            table.append(
                {
                    "model": model,
                    "harness": harness,
                    "variant": variant,
                    "runs": 0,
                    "invalid_runs": invalid,
                }
            )
            continue
        checks = [check for row in group for check in row.get("success_checks") or []]
        costs = [row.get("total_cost_usd") for row in group if row.get("total_cost_usd") is not None]
        seconds = [row.get("seconds") for row in group if row.get("seconds") is not None]
        table.append(
            {
                "model": model,
                "harness": harness,
                "variant": variant,
                "runs": len(group),
                "invalid_runs": invalid,
                "in_box": _rate(row.get("in_goal") for row in group),
                "stopped_cleanly": _rate(row.get("stopped_cleanly") for row in group),
                "claim_correct": _rate(row.get("claim_correct") for row in group),
                "checks_per_run": round(len(checks) / len(group), 2),
                "checks_right": _rate(check.get("correct") for check in checks),
                "median_s": round(statistics.median(seconds), 1) if seconds else None,
                "median_cost_usd": round(statistics.median(costs), 4) if costs else None,
                "total_cost_usd": round(sum(costs), 4) if costs else None,
            }
        )
    return table


def print_table(table: list[dict]) -> None:
    columns = [
        ("model", "model"),
        ("variant", "variant"),
        ("runs", "runs"),
        ("invalid_runs", "invalid"),
        ("in_box", "in box"),
        ("stopped_cleanly", "stopped itself"),
        ("claim_correct", "claim right"),
        ("checks_per_run", "checks/run"),
        ("checks_right", "checks right"),
        ("median_s", "median s"),
        ("median_cost_usd", "median $"),
        ("total_cost_usd", "total $"),
    ]
    print("| " + " | ".join(title for _, title in columns) + " |")
    print("|" + "---|" * len(columns))
    for row in table:
        print(
            "| "
            + " | ".join("-" if row.get(key) is None else str(row[key]) for key, _ in columns)
            + " |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help="agent_trial.py JSONL output")
    args = parser.parse_args(argv)
    rows = []
    for name in args.files:
        for line in Path(name).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("no runs", file=sys.stderr)
        return 1
    print_table(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
