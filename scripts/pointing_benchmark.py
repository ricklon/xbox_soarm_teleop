"""How well do pointing models find objects? Graded by the twin's ground truth.

For each trial an object (the red cube or the blue bean bag) is put at a
random spot on the table, turned at random. Every camera view is drawn at
640x480 and each pointer is asked "point to the red cube". Its pixel is turned
into xyz on the plane the object's center rests at (``twin.projection``) and
compared with the true center.

Two references come with every run:

- ``perfect``: the centroid of the object's own pixels (segmentation). The
  best any pointer could do with this method; what is left is the method's
  error (a pixel on the visible surface is not the center).
- a backend that always answers the middle of the image, for scale.

    uv run python scripts/pointing_benchmark.py --pointer molmo --pointer gemini
    uv run python scripts/pointing_benchmark.py --pointer ollama:gemma4:12b --trials 10

Backends (``twin.pointing``): ``oracle``, ``gemini[:model]`` (GEMINI_API_KEY),
``molmo[:url]`` (start ``scripts/molmo_server.py`` first), ``ollama[:model]``,
``openrouter[:model]`` (OPENROUTER_API_KEY; e.g. google/gemma-4-31b-it:free).
A backend that cannot be reached is reported as skipped. Keys are never printed.
Results go to ``--out`` as JSON (one row per trial, view and backend) and a
summary table is printed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

from xbox_soarm_teleop.twin import build_twin  # noqa: E402
from xbox_soarm_teleop.twin.cameras import (  # noqa: E402
    WRIST_SIZE_LARGE,
    CameraRig,
    encode,
    point_camera,
)
from xbox_soarm_teleop.twin.layouts import GOALS, LAYOUTS, object_description  # noqa: E402
from xbox_soarm_teleop.twin.pointing import (  # noqa: E402
    FixedPointer,
    Point,
    PointerError,
    PointRequest,
    add_usage,
    make_pointer,
    mask_secrets,
)
from xbox_soarm_teleop.twin.projection import camera_geometry, pixel_to_xyz  # noqa: E402
from xbox_soarm_teleop.twin.scenarios import place_with_distractors, random_table_spot  # noqa: E402

W, H = 640, 480


def size(view: str) -> tuple[int, int]:
    """640x480, but the wrist camera is 16:9 like the real one (640x360)."""
    return WRIST_SIZE_LARGE if view == "wrist" else (W, H)


VIEWS = ("overhead", "front", "side", "wrist")
OBJECTS = ("pick_cube", "bean_bag")
# --distractors KIND: the red cube among these (twin.layouts' distractor scenes).
DISTRACTOR_SETS = {
    "color": ["blue_cube", "green_cube"],
    "shape": ["red_bar"],
    "size": ["red_cube_small"],
    "clutter": ["red_cube_small", "red_bar", "blue_cube", "yellow_cube"],
}
# Where objects are put: within the arm's reach, clear of the goal box. The
# table ends at y = -0.10: below that the object is drawn beside it, at table
# height. Pointing is graded the same (the plane is right), and the published
# numbers used these ranges; twin.scenarios.random_table_spot keeps to the table.
X_RANGE = (0.22, 0.40)
Y_RANGE = (-0.18, 0.18)


def random_spot(rng: np.random.Generator) -> tuple[float, float, float]:
    """A spot (x, y, yaw) on the table away from the goal box."""
    goal = GOALS["pick_place_basic"]
    while True:
        x, y = rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)
        if abs(x - goal.center_xyz_m[0]) < 0.08 and abs(y - goal.center_xyz_m[1]) < 0.08:
            continue
        return float(x), float(y), float(rng.uniform(-math.pi / 2, math.pi / 2))


def place(twin, name: str, x: float, y: float, yaw: float) -> np.ndarray:
    """Put one object on the table at (x, y) turned by ``yaw``; park the others."""
    grasp = twin.grasp
    grasp.park([other for other in grasp.objects if other != name])
    z = twin.table_top_z + float(grasp.size(name)[2]) / 2.0
    grasp.place(name, np.array([x, y, z]))
    obj = grasp.objects[name]
    twin.sim.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7] = [
        math.cos(yaw / 2),
        0.0,
        0.0,
        math.sin(yaw / 2),
    ]
    mujoco.mj_forward(twin.sim.model, twin.sim.data)
    return grasp.position(name)


def run_action(twin, request: dict) -> bool:
    future = twin.submit(request)
    twin.tick()
    status = future.result(timeout=0)
    for _ in range(1500):
        status = twin.wait(status.id, 0)
        if status.state.terminal:
            break
        twin.tick()
    return status.state.value == "SUCCEEDED"


def aim_wrist(twin, xyz: np.ndarray) -> bool:
    """Bring the wrist camera where a grasp starts: 6 cm above the object,
    jaws turned side to side, so the object is between the jaw tips."""
    above = [float(xyz[0]), float(xyz[1]), float(xyz[2]) + 0.06]
    return run_action(twin, {"type": "move_ee_to", "params": {"xyz_m": above}}) and run_action(
        twin, {"type": "set_wrist_roll", "params": {"angle_rad": -1.5}}
    )


def segment(seg_renderer, twin, view: str) -> np.ndarray:
    """Which geom each pixel shows (MuJoCo segmentation: object id, object type)."""
    camera = mujoco.MjvCamera()
    point_camera(camera, twin.sim.model, view)
    seg_renderer.update_scene(twin.sim.data, camera=camera)
    return seg_renderer.render()


def object_mask(seg: np.ndarray, twin, name: str) -> np.ndarray:
    body = twin.grasp.objects[name].body_id
    geoms = np.where(twin.sim.model.geom_bodyid == body)[0]
    return (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], geoms)


def what_is_at(seg: np.ndarray, twin, point: Point | None) -> str | None:
    """The object the pointed pixel shows ("background" for anything else)."""
    if point is None:
        return None
    row, col = int(point.v), int(point.u)
    if not (0 <= row < seg.shape[0] and 0 <= col < seg.shape[1]):
        return "background"
    geom_id, kind = int(seg[row, col, 0]), int(seg[row, col, 1])
    if kind != mujoco.mjtObj.mjOBJ_GEOM or geom_id < 0:
        return "background"
    body = int(twin.sim.model.geom_bodyid[geom_id])
    for name, obj in twin.grasp.objects.items():
        if obj.body_id == body:
            return name
    return "background"


def usage_totals(rows: list[dict]) -> dict:
    """Tokens and cost over rows (cost None if any call's cost is unknown)."""
    total = None
    for row in rows:
        total = add_usage(total, row.get("usage"))
    if total is None:
        return {"input_tokens": None, "output_tokens": None, "cost_usd": None}
    return {key: total[key] for key in ("input_tokens", "output_tokens", "cost_usd")}


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["backend"], row["view"], row.get("condition", "alone")), []).append(row)
    table = []
    for (backend, view, condition), group in sorted(groups.items()):
        visible = [r for r in group if r["visible"]]
        found = [r for r in visible if r["error_cm"] is not None]
        errors = sorted(r["error_cm"] for r in found)
        pixels = [r["pixel_error"] for r in found if r["pixel_error"] is not None]
        seconds = [r["seconds"] for r in group if r["seconds"] is not None]
        table.append(
            {
                "backend": backend,
                "view": view,
                "condition": condition,
                "trials": len(group),
                "visible": len(visible),
                "found": len(found),
                "errors": sum(1 for r in group if r.get("error")),
                "median_cm": round(statistics.median(errors), 2) if errors else None,
                "mean_cm": round(statistics.fmean(errors), 2) if errors else None,
                "p90_cm": round(float(np.percentile(errors, 90)), 2) if errors else None,
                "within_2cm": round(sum(e <= 2.0 for e in errors) / len(visible), 2)
                if visible
                else None,
                "median_px": round(statistics.median(pixels), 1) if pixels else None,
                "median_s": round(statistics.median(seconds), 2) if seconds else None,
                "model_s": round(sum(seconds), 1),
                **usage_totals(group),
                # What the pointed pixel shows: the target, another object, or neither.
                "on_target": round(sum(r["hit"] == r["object"] for r in found) / len(found), 2)
                if found
                else None,
                "wrong_object": round(
                    sum(r["hit"] not in (r["object"], "background", None) for r in found) / len(found),
                    2,
                )
                if found
                else None,
            }
        )
    return table


def print_table(table: list[dict]) -> None:
    header = (
        "| backend | view | condition | visible | found | API errors | on target | wrong object "
        "| median cm | mean cm | p90 cm | <=2 cm | median px | s/call |"
    )
    print(header)
    print("|" + "---|" * 14)
    for row in table:
        cells = [
            row["backend"],
            row["view"],
            row["condition"],
            f"{row['visible']}/{row['trials']}",
            row["found"],
            row["errors"],
            row["on_target"],
            row["wrong_object"],
            row["median_cm"],
            row["mean_cm"],
            row["p90_cm"],
            row["within_2cm"],
            row["median_px"],
            row["median_s"],
        ]
        print("| " + " | ".join("-" if c is None else str(c) for c in cells) + " |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pointer", action="append", default=[], help="Backend spec (repeatable)")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", action="append", choices=VIEWS, help="Views (default: all)")
    parser.add_argument("--out", default="twin_logs/pointing_benchmark.json")
    parser.add_argument("--images", default=None, help="Save each picture here (PNG)")
    parser.add_argument(
        "--distractors",
        choices=sorted(DISTRACTOR_SETS),
        default=None,
        help="Point at the red cube alone and then among these distractors, at the same spot",
    )
    args = parser.parse_args(argv)
    # Wrist last: aiming it moves the arm, which may nudge the object.
    views = tuple(sorted(set(args.view or VIEWS), key=VIEWS.index))

    pointers = [
        FixedPointer(answer=lambda r: Point(r.width / 2, r.height / 2, raw="center"), name="center")
    ]
    for spec in args.pointer:
        pointer = make_pointer(spec)
        pointer.name = spec  # two models of one backend stay apart in the results
        pointers.append(pointer)
    twin = build_twin(scene="pick_place_basic", with_controller=False)
    twin.tick()
    rig = CameraRig(twin.sim.model, twin.sim.data)
    segs: dict[tuple[int, int], mujoco.Renderer] = {}

    def seg_for(view: str) -> mujoco.Renderer:
        w, h = size(view)
        if (w, h) not in segs:
            segs[(w, h)] = mujoco.Renderer(twin.sim.model, h, w)
            segs[(w, h)].enable_segmentation_rendering()
        return segs[(w, h)]

    images = Path(args.images) if args.images else None
    if images:
        images.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    run_started = time.monotonic()
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    skipped: dict[str, str] = {}
    disturbed: list[int] = []
    for trial in range(args.trials):
        if args.distractors:
            name = "pick_cube"
            layout = LAYOUTS[f"distractors_{args.distractors}"]
            x, y = random_table_spot(rng)  # on the table, unlike random_spot's older range
            yaw = float(rng.uniform(-math.pi / 2, math.pi / 2))
            conditions = [("alone", []), ("distractors", DISTRACTOR_SETS[args.distractors])]
        else:
            name = OBJECTS[trial % len(OBJECTS)]
            layout = None
            x, y, yaw = random_spot(rng)
            conditions = [("alone", None)]
        for condition, others in conditions:
            run_action(twin, {"type": "home"})
            if others is None:
                truth = place(twin, name, x, y, yaw)
            else:
                truth = place_with_distractors(
                    twin, name, others, rng, target_xy=(x, y), target_yaw=yaw
                )[name]
            description = object_description(name, layout)  # the same words alone and among others
            plane_z = twin.table_top_z + float(twin.grasp.half_extents_world(name)[2])
            for view in views:
                if view == "wrist":
                    before = {n: twin.grasp.position(n).copy() for n in twin.grasp.positions()}
                    aimed = aim_wrist(twin, truth)
                    truth = twin.grasp.position(name)
                    moved = max(
                        float(np.linalg.norm(twin.grasp.position(n) - xyz)) for n, xyz in before.items()
                    )
                    if not aimed or moved > 0.003:
                        disturbed.append(trial)  # the arm nudged something, or could not get there
                        continue
                mujoco.mj_forward(twin.sim.model, twin.sim.data)
                width, height = size(view)
                geometry = camera_geometry(twin.sim.model, twin.sim.data, view, width, height)
                pixels = rig.render(view, width, height, twin.sim.data)
                png = encode(pixels, "png")
                if images:
                    (images / f"{trial:02d}-{condition}-{name}-{view}.png").write_bytes(png)
                seg = segment(seg_for(view), twin, view)
                mask = object_mask(seg, twin, name)
                visible = bool(mask.sum() >= 20)
                truth_uv = geometry.project(truth)
                perfect = None
                if visible:
                    rows_, cols_ = np.nonzero(mask)
                    perfect = Point(cols_.mean() + 0.5, rows_.mean() + 0.5, raw="segmentation")
                request = PointRequest(png, "image/png", width, height, description, truth_uv=truth_uv)

                answers: list[tuple[str, Point | None, float | None, str | None, dict | None]] = [
                    ("perfect", perfect, 0.0, None, None)
                ]
                for pointer in pointers:
                    if pointer.name in skipped:
                        continue
                    started = time.monotonic()
                    pointer.last_usage = None  # a failed call must not report the last one's
                    try:
                        point = pointer.point(request)
                        answers.append(
                            (pointer.name, point, time.monotonic() - started, None, pointer.last_usage)
                        )
                    except PointerError as exc:
                        reason = mask_secrets(str(exc))
                        if trial == 0 and view == views[0] and condition == conditions[0][0]:
                            skipped[pointer.name] = reason
                            print(f"skipping {pointer.name}: {reason}", file=sys.stderr)
                            continue
                        answers.append(
                            (pointer.name, None, time.monotonic() - started, reason, pointer.last_usage)
                        )
                for backend, point, seconds, error, usage in answers:
                    xyz = pixel_to_xyz(geometry, point.u, point.v, plane_z) if point else None
                    rows.append(
                        {
                            "trial": trial,
                            "object": name,
                            "condition": condition,
                            "description": description,
                            "view": view,
                            "backend": backend,
                            "visible": visible,
                            "target_px": int(mask.sum()),
                            "hit": what_is_at(seg, twin, point),
                            "truth_xyz_m": [round(float(v), 4) for v in truth],
                            "truth_uv": [round(c, 1) for c in truth_uv] if truth_uv else None,
                            "pixel": [round(point.u, 1), round(point.v, 1)] if point else None,
                            "pixel_error": round(
                                float(np.hypot(point.u - perfect.u, point.v - perfect.v)), 1
                            )
                            if point and perfect
                            else None,
                            "xyz_m": [round(float(v), 4) for v in xyz] if xyz is not None else None,
                            "error_cm": round(float(np.linalg.norm(xyz - truth)) * 100, 2)
                            if xyz is not None
                            else None,
                            "seconds": round(seconds, 3) if seconds is not None else None,
                            "usage": usage,
                            "error": error,
                            "raw": (point.raw if point else (error or ""))[:200],
                        }
                    )
        done = [r for r in rows if r["trial"] == trial and r["backend"] not in ("perfect", "center")]
        brief = ", ".join(
            f"{r['backend']}/{r['view']}/{r['condition']}={r['error_cm']}({r['hit']})" for r in done
        )
        print(f"trial {trial:2d} {name:9s} at ({x:.3f}, {y:.3f}): {brief}", flush=True)

    rig.close()
    for renderer in segs.values():
        renderer.close()
    table = summarize(rows)
    wall_s = time.monotonic() - run_started
    model_s = sum(r["seconds"] or 0.0 for r in rows if r["backend"] not in ("perfect", "center"))
    by_backend = {}
    for backend in dict.fromkeys(r["backend"] for r in rows):
        mine = [r for r in rows if r["backend"] == backend]
        by_backend[backend] = {
            "calls": len(mine),
            "model_s": round(sum(r["seconds"] or 0.0 for r in mine), 1),
            **usage_totals(mine),
        }
    timing = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_s": round(wall_s, 1),
        "model_s": round(model_s, 1),  # waiting on pointing models
        "sim_s": round(wall_s - model_s, 1),  # placing, moving the arm, drawing, scoring
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "trials": args.trials,
                "seed": args.seed,
                "size": [W, H],
                "timing": timing,
                "by_backend": by_backend,
                "skipped": skipped,
                "wrist_trials_dropped": disturbed,
                "summary": table,
                "rows": rows,
            },
            indent=1,
        )
    )
    print()
    print_table(table)
    print(
        f"\nRun: {timing['wall_s']:.0f} s ({timing['model_s']:.0f} s waiting on models, "
        f"{timing['sim_s']:.0f} s simulating and drawing)"
    )
    for backend, use in by_backend.items():
        if backend in ("perfect", "center"):
            continue
        if all(r.get("usage") is None for r in rows if r["backend"] == backend):
            print(f"  {backend}: {use['calls']} calls, {use['model_s']:.0f} s (not a model)")
            continue
        cost = "unknown" if use["cost_usd"] is None else f"${use['cost_usd']:.4f}"
        tokens = (
            "-"
            if use["input_tokens"] is None
            else f"{use['input_tokens']} in / {use['output_tokens']} out"
        )
        print(
            f"  {backend}: {use['calls']} calls, {use['model_s']:.0f} s, tokens {tokens}, cost {cost}"
        )
    if disturbed:
        print(f"\nwrist view dropped for trials {disturbed}: the arm moved the object getting there")
    for name, reason in skipped.items():
        print(f"\nskipped {name}: {reason}")
    print(f"\nResults: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
