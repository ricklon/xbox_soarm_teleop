"""soarm-challenge: check, prove and report on a challenge file.

    soarm-challenge check challenges/red_swap_tricky.yaml
    soarm-challenge prove challenges/red_swap_tricky.yaml --aim-runs 3
    soarm-challenge report challenges/red_swap_tricky.yaml twin_logs/my_runs.jsonl -o findings.md

Agents are run on a challenge with ``scripts/agent_trial.py --challenge FILE``
(or the agent panel on a hub arm the challenge is loaded on); ``report`` turns
those runs into a findings report to fill in. See ``docs/challenges.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from importlib import resources
from pathlib import Path

# A run a provider refused, or could not reach, tested nothing (scripts/agent_report.py).
PROVIDER_ERROR = re.compile(r"HTTP (402|429|5\d\d)|cannot reach ")


def _load(path: str):
    from xbox_soarm_teleop.twin.challenges import ChallengeError, load_challenge

    try:
        return load_challenge(Path(path))
    except (ChallengeError, OSError) as exc:
        print(f"Cannot set this challenge up: {exc}")
        return None


def _describe(challenge) -> list[str]:
    summary = challenge.summary()
    lines = [f"Challenge: {summary['name']} (scene {summary['scene']}, names: {summary['names']})"]
    for request in summary["requests"]:
        lines.append(f"  Request: {request}")
    for name, where in summary["objects"].items():
        turned = f", turned {where['turn_deg']:g}°" if where["turn_deg"] else ""
        goes = summary["expect"].get(name)
        lines.append(
            f"  {name} at ({where['x']}, {where['y']}){turned} -> {goes or 'stays out of the bins'}"
        )
    if not summary["possible"]:
        lines.append(f"  Impossible on purpose. Asked: {summary['asked'] or 'not given'}")
    return lines


def cmd_check(args) -> int:
    from xbox_soarm_teleop.twin.challenges import check

    challenge = _load(args.file)
    if challenge is None:
        return 1
    print("\n".join(_describe(challenge)))
    warnings = check(challenge)
    if warnings:
        print("Notes (allowed, but they change what is being tested):")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("No notes: it keeps the standard desk's layout rules.")
    return 0


def _proof_text(proof: dict) -> str:
    lines = [f"**Verdict: {proof['verdict'].replace('_', ' ')}.** {proof['why'].capitalize()}."]
    true_run = proof.get("true_run")
    if true_run:
        lines.append(
            f"- From true positions: {'done' if true_run['done'] else 'not done'}; ended {true_run['in_bin']}."
        )
        for step in true_run["steps"]:
            if not step["pick"].get("ok"):
                lines.append(f"  - picking {step['object']} failed: {step['pick'].get('detail')}")
    aims = proof.get("aim_runs") or []
    if aims:
        done = sum(run["done"] for run in aims)
        lines.append(
            f"- With an agent's aim error (1.5 cm far, 0.5 cm close up): done {done} of {len(aims)}."
        )
    for hidden in proof.get("hidden") or []:
        lines.append(f"- Camera: {hidden}.")
    return "\n".join(lines)


def cmd_prove(args) -> int:
    from xbox_soarm_teleop.twin.challenges import prove

    challenge = _load(args.file)
    if challenge is None:
        return 1
    print("\n".join(_describe(challenge)))
    print("Proving with the arm's own skills...", flush=True)
    proof = prove(challenge, aim_runs=args.aim_runs, seed=args.seed)
    print(_proof_text(proof))
    if args.json:
        Path(args.json).write_text(json.dumps(proof, indent=2, default=str))
        print(f"Saved: {args.json}")
    return 0


def _invalid(row: dict) -> bool:
    return bool(
        row.get("check_errors")
        or row.get("provider_errors")
        or PROVIDER_ERROR.search(str(row.get("agent_error") or ""))
    )


def _cell(value) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def build_report(
    challenge, rows: list[dict], proof: dict | None = None, date: str | None = None
) -> str:
    """The findings template filled with the challenge, its proof and its runs."""
    summary = challenge.summary()
    mine = [row for row in rows if row.get("task") == challenge.name]
    valid = [row for row in mine if not _invalid(row)]
    left = len(mine) - len(valid)
    run_rows = []
    for index, row in enumerate(valid, start=1):
        steps = (row.get("agent_usage") or {}).get("steps", "")
        cost = row.get("total_cost_usd")
        run_rows.append(
            "| "
            + " | ".join(
                _cell(v)
                for v in (
                    index,
                    row.get("model", ""),
                    "sees" if row.get("agent_sees") else "blind",
                    row.get("request", ""),
                    "yes" if row.get("done") else "no",
                    f"{row.get('right', '')} / {row.get('wrong', '')} / {row.get('unsorted', '')}",
                    row.get("claim") or row.get("stop_reason") or "",
                    {True: "yes", False: "no"}.get(row.get("claim_correct"), "-"),
                    steps,
                    round(float(row.get("seconds") or 0)),
                    f"${float(cost):.3f}" if cost is not None else "",
                )
            )
            + " |"
        )
    done = sum(bool(row.get("done")) for row in valid)
    false_claims = sum(row.get("claim") == "done" and not row.get("done") for row in valid)
    runs_summary = (
        f"Done in **{done} of {len(valid)}** runs; claimed done when it was not in {false_claims}."
        if valid
        else "No runs yet: run `scripts/agent_trial.py --challenge <file>` or use the agent panel."
    )
    object_rows = [
        f"| {name} | ({where['x']}, {where['y']}) | {where['turn_deg']:g}° | "
        f"{summary['expect'].get(name) or 'out of the bins'} |"
        for name, where in summary["objects"].items()
    ]
    template = (
        resources.files("xbox_soarm_teleop.twin").joinpath("templates/challenge_report.md").read_text()
    )
    return template.format(
        name=summary["name"],
        date=date or time.strftime("%Y-%m-%d"),
        request_lines="\n".join(f"> {request}" for request in summary["requests"]),
        object_rows="\n".join(object_rows),
        names=summary["names"],
        possible="no: the agent should notice it cannot be done"
        + (f" (asked: {summary['asked']})" if summary["asked"] else "")
        if not summary["possible"]
        else "no, it is meant to be possible",
        notes=f"- **Notes:** {summary['notes']}\n" if summary["notes"] else "",
        proof=_proof_text(proof)
        if proof
        else "(Run `soarm-challenge prove <file>` and paste its verdict here.)",
        runs_summary=runs_summary,
        run_rows="\n".join(run_rows),
        left_out=(
            f"{left} run{'s' if left != 1 else ''} left out: the model or judge service refused or "
            "could not be reached, so it tested nothing."
            if left
            else ""
        ),
    )


def cmd_report(args) -> int:
    challenge = _load(args.file)
    if challenge is None:
        return 1
    rows = [json.loads(line) for line in Path(args.runs).read_text().splitlines() if line.strip()]
    proof = json.loads(Path(args.proof).read_text()) if args.proof else None
    text = build_report(challenge, rows, proof)
    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {args.output}: fill in the sections in brackets.")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soarm-challenge", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check", help="Load a challenge and list what it changes from the standard desk"
    )
    check.add_argument("file")
    check.set_defaults(run=cmd_check)
    prove = sub.add_parser(
        "prove", help="Can the arm do it? Its own skills try, from true positions and with aim error"
    )
    prove.add_argument("file")
    prove.add_argument("--aim-runs", type=int, default=3, help="Tries with an agent's aim error")
    prove.add_argument("--seed", type=int, default=0)
    prove.add_argument("--json", default=None, help="Save the proof here (for report --proof)")
    prove.set_defaults(run=cmd_prove)
    report = sub.add_parser(
        "report", help="A findings report from agent runs (agent_trial.py --challenge)"
    )
    report.add_argument("file")
    report.add_argument("runs", help="The runs' JSONL file (agent_trial.py --out)")
    report.add_argument("--proof", default=None, help="A proof saved with prove --json")
    report.add_argument("-o", "--output", default=None, help="Write the report here (Markdown)")
    report.set_defaults(run=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
