"""Is Clef a good success judge? Graded by the twin's own state.

Generates situations an agent has to recognize without ``in_goal`` or
``attached`` (the cube or the bean bag on the table, in the goal box, just
outside it, held over the table or over the box, tipped onto an edge on the
table or in the box), photographs each from four views, and asks Clef the
judge's questions (``twin.judge``) with 1, 2 and 4 pictures. Its answers are
scored against the truth: accuracy, Brier score, log loss and expected
calibration error per question, a confusion table for "where", and
calibration bins. A judge that always answers the base rate is the baseline.

    uv run python scripts/judge_benchmark.py                       # clef and clef-flash, 42 situations
    uv run python scripts/judge_benchmark.py --model clef --situations 70 --images twin_logs/judge_imgs

Needs CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID (environment or .env);
they are never printed. Results go to --out as JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

from xbox_soarm_teleop.twin import build_twin  # noqa: E402
from xbox_soarm_teleop.twin.cameras import WRIST_SIZE_LARGE, CameraRig, encode  # noqa: E402
from xbox_soarm_teleop.twin.judge import (  # noqa: E402
    NOUL_IDS,
    WHERE_OPTIONS,
    ClefJudge,
    JudgeError,
    calibration_bins,
    noul_metrics,
    truth_labels,
    where_metrics,
)
from xbox_soarm_teleop.twin.layouts import GOALS, LAYOUTS, object_description  # noqa: E402
from xbox_soarm_teleop.twin.scenarios import (  # noqa: E402
    grasp_and_lift,
    place_in_goal,
    place_near_goal,
    place_on_table,
    place_tipped,
    place_with_distractors,
    random_table_spot,
    reset,
)

W, H = 640, 480


def size(view: str) -> tuple[int, int]:
    """640x480, but the wrist camera is 16:9 like the real one (640x360)."""
    return WRIST_SIZE_LARGE if view == "wrist" else (W, H)


VIEWS = ("front", "overhead", "side", "wrist")
VIEW_SETS = {"1": ("front",), "2": ("front", "overhead"), "4": VIEWS}
KINDS = (
    "on_table",
    "in_goal",
    "near_goal",
    "held_over_table",
    "held_over_goal",
    "tipped_on_table",
    "tipped_in_goal",
)
OBJECTS = ("pick_cube", "bean_bag")
# --distractors KIND: the red cube among these; the first is the one moved about.
DISTRACTOR_SETS = {
    "color": ["blue_cube", "green_cube"],
    "shape": ["red_bar"],
    "size": ["red_cube_small"],
    "clutter": ["red_cube_small", "red_bar", "blue_cube", "yellow_cube"],
}
# Situations where the words must pick out the right object.
DISTRACTOR_KINDS = (
    "target_on_table",
    "target_in_goal",
    "distractor_in_goal",
    "both_in_goal",
    "held_target",
    "held_distractor",
)


def make_distractor_situation(twin, kind: str, others: list[str], rng: np.random.Generator) -> bool:
    """The red cube among ``others`` as ``kind``; False if it could not be done."""
    reset(twin)
    target, first = "pick_cube", others[0]
    place_with_distractors(twin, target, others, rng)
    if kind == "target_in_goal":
        place_in_goal(twin, target, *rng.uniform(-0.01, 0.01, size=2), park_others=False)
    elif kind == "distractor_in_goal":
        place_in_goal(twin, first, *rng.uniform(-0.005, 0.005, size=2), park_others=False)
    elif kind == "both_in_goal":
        # Side by side across the box; a long object lies along it.
        place_in_goal(twin, target, dy=-0.02, park_others=False)
        place_in_goal(twin, first, dy=0.024, yaw=math.pi / 2, park_others=False)
    elif kind in ("held_target", "held_distractor"):
        name = target if kind == "held_target" else first
        lift = [rng.uniform(0.26, 0.34), rng.uniform(-0.06, 0.04), rng.uniform(0.21, 0.25)]
        return grasp_and_lift(twin, name, lift)
    elif kind != "target_on_table":
        raise ValueError(kind)
    return True


def make_situation(twin, kind: str, name: str, rng: np.random.Generator) -> bool:
    """Set the twin up as ``kind`` with object ``name``; False if it could not be done."""
    reset(twin)
    yaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
    goal = GOALS["pick_place_basic"]
    if kind == "on_table":
        place_on_table(twin, name, *random_table_spot(rng), yaw=yaw)
    elif kind == "in_goal":
        place_in_goal(twin, name, *rng.uniform(-0.012, 0.012, size=2), yaw=yaw)
    elif kind == "near_goal":
        place_near_goal(twin, name, rng, yaw=yaw)
    elif kind in ("held_over_table", "held_over_goal"):
        place_on_table(
            twin, name, *random_table_spot(rng, clearance=0.10), yaw=float(rng.uniform(-0.3, 0.3))
        )
        if kind == "held_over_table":
            target = [rng.uniform(0.24, 0.34), rng.uniform(-0.12, 0.02), rng.uniform(0.20, 0.26)]
        else:
            cx, cy, _ = goal.center_xyz_m
            target = [
                cx + rng.uniform(-0.01, 0.01),
                cy + rng.uniform(-0.01, 0.01),
                rng.uniform(0.21, 0.25),
            ]
        return grasp_and_lift(twin, name, target)
    elif kind == "tipped_on_table":
        place_tipped(twin, name, *random_table_spot(rng), tilt=float(rng.uniform(0.5, 0.9)), yaw=yaw)
    elif kind == "tipped_in_goal":
        cx, cy, floor = goal.center_xyz_m
        place_tipped(twin, name, cx, cy, tilt=float(rng.uniform(0.5, 0.8)), yaw=yaw, rest_z=floor)
    else:
        raise ValueError(kind)
    return True


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("verdict"):
            groups.setdefault((row["model"], row["images"]), []).append(row)
    table = []
    for (model, images), group in sorted(groups.items()):
        entry = {"model": model, "images": images, "n": len(group)}
        for q in NOUL_IDS:
            entry[q] = noul_metrics([r["verdict"][q] for r in group], [r["truth"][q] for r in group])
        entry["where"] = where_metrics(
            [r["verdict"]["where"] for r in group], [r["truth"]["where"] for r in group]
        )
        entry["calibration"] = calibration_bins(
            [r["verdict"][q] for r in group for q in NOUL_IDS],
            [r["truth"][q] for r in group for q in NOUL_IDS],
        )
        entry["input_tokens"] = int(sum(r["verdict"].get("input_tokens") or 0 for r in group))
        costs = [r["verdict"].get("cost_usd") for r in group]
        entry["cost_usd"] = None if any(c is None for c in costs) else float(sum(costs))
        entry["model_s"] = round(sum(r["seconds"] for r in group), 1)
        entry["median_s"] = float(np.median([r["seconds"] for r in group]))
        table.append(entry)
    return table


def baseline(situations: list[dict]) -> dict:
    """Always answer the base rate: what the judge must beat."""
    out = {}
    for q in NOUL_IDS:
        truth = [s["truth"][q] for s in situations]
        rate = float(np.mean(truth))
        out[q] = noul_metrics([rate] * len(truth), truth)
    counts = {
        o: sum(s["truth"]["where"] == o for s in situations) / len(situations) for o in WHERE_OPTIONS
    }
    out["where"] = where_metrics([counts] * len(situations), [s["truth"]["where"] for s in situations])
    return out


def print_tables(table: list[dict], base: dict) -> None:
    print(
        "| model | images | "
        + " | ".join(f"{q} acc / Brier / ECE" for q in NOUL_IDS)
        + " | where acc / p(true) |"
    )
    print("|" + "---|" * (3 + len(NOUL_IDS)))

    def cells(entry):
        parts = []
        for q in NOUL_IDS:
            m = entry[q]
            parts.append(f"{m['accuracy']:.2f} / {m['brier']:.3f} / {m['ece']:.3f}")
        w = entry["where"]
        parts.append(f"{w['accuracy']:.2f} / {w['p_true_mean']:.2f}")
        return " | ".join(parts)

    print(f"| base rate | - | {cells(base)} |")
    for entry in table:
        print(f"| {entry['model']} | {entry['images']} | {cells(entry)} |")
    print()
    for entry in table:
        print(
            f"{entry['model']} with {entry['images']} image(s): {entry['n']} situations, "
            f"{entry['input_tokens']} input tokens"
            + ("" if entry.get("cost_usd") is None else f" (${entry['cost_usd']:.4f})")
            + f", median {entry['median_s']:.2f} s"
        )
        confusion = entry["where"]["confusion"]
        print(
            "  where, actual -> chosen: "
            + "; ".join(
                f"{a}: " + ", ".join(f"{c} {n}" for c, n in row.items() if n)
                for a, row in confusion.items()
                if any(row.values())
            )
        )
        print(
            "  calibration (predicted -> observed, n): "
            + ", ".join(
                f"{b['predicted']:.2f}->{b['observed']:.2f} ({b['n']})" for b in entry["calibration"]
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", action="append", choices=("clef", "clef-flash"), help="Default: both"
    )
    parser.add_argument("--situations", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-sets", default="1,2,4", help="Numbers of pictures to try (1, 2, 4)")
    parser.add_argument("--out", default="twin_logs/judge_benchmark.json")
    parser.add_argument("--images", default=None, help="Save each picture here (PNG)")
    parser.add_argument(
        "--distractors",
        choices=sorted(DISTRACTOR_SETS),
        default=None,
        help="Ask about the red cube among these objects, in situations where the words matter",
    )
    args = parser.parse_args(argv)
    models = args.model or ["clef", "clef-flash"]
    sets = [s.strip() for s in args.image_sets.split(",") if s.strip()]
    judges = {model: ClefJudge(model=model) for model in models}

    twin = build_twin(scene="pick_place_basic", with_controller=False)
    twin.tick()
    rig = CameraRig(twin.sim.model, twin.sim.data)
    image_dir = Path(args.images) if args.images else None
    if image_dir:
        image_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    run_started = time.monotonic()
    rng = np.random.default_rng(args.seed)
    situations: list[dict] = []
    rows: list[dict] = []
    for index in range(args.situations):
        others: list[str] = []
        layout = None
        if args.distractors:
            kind = DISTRACTOR_KINDS[index % len(DISTRACTOR_KINDS)]
            name, others = "pick_cube", DISTRACTOR_SETS[args.distractors]
            layout = LAYOUTS[f"distractors_{args.distractors}"]
            ready = make_distractor_situation(twin, kind, others, rng)
        else:
            kind = KINDS[index % len(KINDS)]
            name = OBJECTS[(index // len(KINDS)) % len(OBJECTS)]
            ready = make_situation(twin, kind, name, rng)
        if not ready:
            print(f"situation {index}: could not set up {kind} with {name}; skipped", flush=True)
            continue
        labels = truth_labels(twin, name)
        pictures = {view: encode(rig.render(view, *size(view), twin.sim.data), "png") for view in VIEWS}
        if image_dir:
            for view, png in pictures.items():
                (image_dir / f"{index:03d}-{kind}-{name}-{view}.png").write_bytes(png)
        situation = {
            "index": index,
            "kind": kind,
            "object": name,
            "distractors": others,
            "truth": labels,
        }
        situations.append(situation)
        description = object_description(name, layout)
        other_words = [object_description(other, layout) for other in others]
        brief = []
        for model, judge in judges.items():
            for key in sets:
                views = VIEW_SETS[key]
                started = time.monotonic()
                try:
                    verdict = judge.judge(
                        [pictures[v] for v in views], description, views=list(views), others=other_words
                    )
                    error = None
                except JudgeError as exc:
                    verdict, error = None, str(exc)
                    print(f"  {model}/{key}: {error}", file=sys.stderr, flush=True)
                rows.append(
                    {
                        **situation,
                        "model": model,
                        "images": key,
                        "verdict": verdict,
                        "error": error,
                        "seconds": round(time.monotonic() - started, 3),
                    }
                )
                if verdict:
                    brief.append(f"{model}/{key} in_goal={verdict['in_goal']:.2f}")
        print(f"{index:3d} {kind:16s} {name:9s} truth={labels} | " + ", ".join(brief), flush=True)

    rig.close()
    table = summarize(rows)
    base = baseline(situations)
    wall_s = time.monotonic() - run_started
    model_s = sum(r["seconds"] for r in rows)
    timing = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_s": round(wall_s, 1),
        "model_s": round(model_s, 1),  # waiting on Clef
        "sim_s": round(wall_s - model_s, 1),  # setting up situations and drawing them
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "timing": timing,
                "situations": situations,
                "baseline": base,
                "summary": table,
                "rows": rows,
            },
            indent=1,
        )
    )
    print()
    print_tables(table, base)
    print(
        f"\nRun: {timing['wall_s']:.0f} s ({timing['model_s']:.0f} s waiting on Clef, "
        f"{timing['sim_s']:.0f} s setting up and drawing)"
    )
    errors = sum(1 for r in rows if r["error"])
    if errors:
        print(f"\n{errors} judge calls failed (see --out)")
    print(f"\nResults: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
