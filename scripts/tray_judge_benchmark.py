"""How well does Clef say which tray an object is in, on the sorting desk?

    uv run python scripts/tray_judge_benchmark.py --repeats 2

Sets up known situations in the sort scene: each cube in each tray (the
right one and the wrong ones), on the desk, and held in the gripper, with the
other cubes shuffled into trays or left on the desk as look-alikes. Asks Clef
"where is the <cube>?" with the cell's trays as options, from overhead, from
the front, and from both, and scores every answer against the simulator:
accuracy, mean probability on the right answer, and how often the confident
answers (0.9 or more) were right. Needs CLOUDFLARE_API_TOKEN and
CLOUDFLARE_ACCOUNT_ID (environment or .env).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

VIEW_SETS = (("overhead",), ("front",), ("overhead", "front"))
CARRY_XYZ = (0.24, 0.0, 0.20)  # where a held cube is shown (the jaw center)


def render(twin, view: str) -> bytes:
    future = twin.render(view, width=640, height=480, fmt="png")
    twin.tick()
    return future.result(timeout=10)


def settle(twin, ticks: int = 60) -> None:
    for _ in range(ticks):
        twin.tick()


def desk_spot(rng, twin, taken) -> tuple[float, float]:
    """A random spot in front of the arm, clear of the trays and of other objects."""
    from xbox_soarm_teleop.twin.skills import _in_a_bin

    while True:
        r, a = rng.uniform(0.21, 0.32), math.radians(rng.uniform(-35, 35))
        x, y = r * math.cos(a), r * math.sin(a)
        if _in_a_bin(twin, (x, y, 0.0)):
            continue
        if all(math.dist((x, y), other) >= 0.08 for other in taken):
            return x, y


def situations(rng, twin, objects, bins, repeats):
    """(target, where, others' places) for every case, ``repeats`` times each."""
    cases = []
    for _ in range(repeats):
        for target in objects:
            for where in [*bins, "on_table", "in_gripper"]:
                others = {
                    other: str(rng.choice([*bins, "on_table", "on_table"]))
                    for other in objects
                    if other != target
                }
                cases.append((target, where, others))
    return cases


def arrange(twin, rng, target, where, others) -> bool:
    """Put the scene in the case's arrangement; False if the grasp for a held case failed."""
    from xbox_soarm_teleop.twin import scenarios

    scenarios.reset(twin)
    settle(twin, 20)
    taken = []
    used_bins = set()
    for name, place in [*others.items(), (target, where)]:
        if place == "in_gripper":
            continue
        if place == "on_table":
            x, y = desk_spot(rng, twin, taken)
            scenarios.place_start(twin, name, x, y)
            taken.append((x, y))
        else:
            dx = dy = 0.0
            if place in used_bins:  # a second object in a bin goes to its other side
                dx, dy = 0.025, 0.025
            scenarios.place_in_bin(twin, name, place, dx=dx, dy=dy)
            used_bins.add(place)
    settle(twin)
    if where == "in_gripper":
        x, y = desk_spot(rng, twin, taken)
        scenarios.place_start(twin, target, x, y)
        settle(twin, 30)
        if not scenarios.grasp_and_lift(twin, target, CARRY_XYZ):
            return False
        settle(twin, 30)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=2, help="Each case this many times")
    parser.add_argument("--models", default="clef", help="clef, clef-flash or both (comma)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="twin_logs/tray_judge_benchmark.json")
    args = parser.parse_args(argv)

    from xbox_soarm_teleop.twin import build_twin
    from xbox_soarm_teleop.twin.judge import ClefJudge, tray_metrics, tray_options, tray_truth
    from xbox_soarm_teleop.twin.layouts import object_description

    rng = np.random.default_rng(args.seed)
    twin = build_twin(scene="sort_colors", with_controller=False)
    twin.tick()
    scene = twin.observe()["scene"]
    bins = [(b["name"], b["label"]) for b in scene["bins"]]
    options = tray_options(bins)
    objects = list(scene["objects"])
    judges = {model: ClefJudge(model=model) for model in args.models.split(",")}
    records = []
    started = time.monotonic()
    for target, where, others in situations(rng, twin, objects, [b for b, _ in bins], args.repeats):
        if not arrange(twin, rng, target, where, others):
            print(f"skip: could not grasp {target}")
            continue
        truth = tray_truth(twin, target)
        pictures = {view: render(twin, view) for view in ("overhead", "front")}
        others_words = [object_description(o) for o in objects if o != target]
        for model, judge in judges.items():
            for views in VIEW_SETS:
                answer = judge.judge_trays(
                    [pictures[v] for v in views],
                    object_description(target),
                    bins,
                    views=list(views),
                    others=others_words,
                )
                records.append(
                    {
                        "model": model,
                        "views": "+".join(views),
                        "target": target,
                        "case": where,
                        "truth": truth,
                        "others": others,
                        "answer": answer["where"],
                        "cost_usd": answer["cost_usd"],
                    }
                )
        print(f"{target:11s} {where:10s} truth {truth}", flush=True)
    twin.close()

    summary = {}
    for model in judges:
        for views in VIEW_SETS:
            key = f"{model} {'+'.join(views)}"
            rows = [r for r in records if r["model"] == model and r["views"] == "+".join(views)]
            summary[key] = tray_metrics(
                [r["answer"] for r in rows], [r["truth"] for r in rows], options
            )
            summary[key]["cost_usd"] = round(sum(r["cost_usd"] or 0 for r in rows), 5)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "options": options,
                "summary": summary,
                "records": records,
                "seconds": round(time.monotonic() - started, 1),
            },
            indent=1,
        )
    )
    print("| judge, views | n | accuracy | p on truth | confident right | cost $ |")
    print("|---|---|---|---|---|---|")
    for key, m in summary.items():
        sure = m["confident"]
        print(
            f"| {key} | {m['n']} | {m['accuracy']:.2f} | {m['p_true_mean']:.2f} | "
            f"{sure['right']}/{sure['n']} | {m['cost_usd']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
