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
from pathlib import Path

import mujoco
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

from xbox_soarm_teleop.twin import build_twin  # noqa: E402
from xbox_soarm_teleop.twin.cameras import CameraRig, encode, point_camera  # noqa: E402
from xbox_soarm_teleop.twin.layouts import GOALS, object_description  # noqa: E402
from xbox_soarm_teleop.twin.pointing import (  # noqa: E402
    FixedPointer,
    Point,
    PointerError,
    PointRequest,
    make_pointer,
    mask_secrets,
)
from xbox_soarm_teleop.twin.projection import camera_geometry, pixel_to_xyz  # noqa: E402

W, H = 640, 480
VIEWS = ("overhead", "front", "side", "wrist")
OBJECTS = ("pick_cube", "bean_bag")
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
    """Put the wrist camera behind and above the object, looking at it."""
    up = [float(xyz[0]) - 0.10, float(xyz[1]), 0.26]  # over it first, clear of the object
    target = [float(xyz[0]) - 0.10, float(xyz[1]), 0.22]
    return run_action(twin, {"type": "move_ee_to", "params": {"xyz_m": up}}) and run_action(
        twin, {"type": "move_ee_to", "params": {"xyz_m": target}}
    )


def object_mask(seg_renderer, twin, view: str, name: str) -> np.ndarray:
    camera = mujoco.MjvCamera()
    point_camera(camera, twin.sim.model, view)
    seg_renderer.update_scene(twin.sim.data, camera=camera)
    seg = seg_renderer.render()
    body = twin.grasp.objects[name].body_id
    geoms = np.where(twin.sim.model.geom_bodyid == body)[0]
    return (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], geoms)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["backend"], row["view"]), []).append(row)
    table = []
    for (backend, view), group in sorted(groups.items()):
        visible = [r for r in group if r["visible"]]
        found = [r for r in visible if r["error_cm"] is not None]
        errors = sorted(r["error_cm"] for r in found)
        pixels = [r["pixel_error"] for r in found if r["pixel_error"] is not None]
        seconds = [r["seconds"] for r in group if r["seconds"] is not None]
        table.append(
            {
                "backend": backend,
                "view": view,
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
            }
        )
    return table


def print_table(table: list[dict]) -> None:
    header = (
        "| backend | view | visible | found | API errors | median cm | mean cm | p90 cm "
        "| <=2 cm | median px | s/call |"
    )
    print(header)
    print("|" + "---|" * 11)
    for row in table:
        cells = [
            row["backend"],
            row["view"],
            f"{row['visible']}/{row['trials']}",
            row["found"],
            row["errors"],
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
    args = parser.parse_args(argv)
    views = tuple(args.view or VIEWS)

    pointers = [FixedPointer(u=W / 2, v=H / 2, name="center")]
    for spec in args.pointer:
        pointer = make_pointer(spec)
        pointer.name = spec  # two models of one backend stay apart in the results
        pointers.append(pointer)
    twin = build_twin(scene="pick_place_basic", with_controller=False)
    twin.tick()
    rig = CameraRig(twin.sim.model, twin.sim.data)
    seg = mujoco.Renderer(twin.sim.model, H, W)
    seg.enable_segmentation_rendering()
    images = Path(args.images) if args.images else None
    if images:
        images.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    skipped: dict[str, str] = {}
    disturbed: list[int] = []
    for trial in range(args.trials):
        name = OBJECTS[trial % len(OBJECTS)]
        x, y, yaw = random_spot(rng)
        run_action(twin, {"type": "home"})
        truth = place(twin, name, x, y, yaw)
        description = object_description(name)
        plane_z = twin.table_top_z + float(twin.grasp.half_extents_world(name)[2])
        for view in views:
            if view == "wrist":
                before = truth.copy()
                aimed = aim_wrist(twin, truth)
                truth = twin.grasp.position(name)
                if not aimed or np.linalg.norm(truth - before) > 0.003:
                    disturbed.append(trial)  # the arm nudged it, or could not get there
                    continue
            mujoco.mj_forward(twin.sim.model, twin.sim.data)
            geometry = camera_geometry(twin.sim.model, twin.sim.data, view, W, H)
            pixels = rig.render(view, W, H, twin.sim.data)
            png = encode(pixels, "png")
            if images:
                (images / f"{trial:02d}-{name}-{view}.png").write_bytes(png)
            mask = object_mask(seg, twin, view, name)
            visible = bool(mask.sum() >= 20)
            truth_uv = geometry.project(truth)
            perfect = None
            if visible:
                rows_, cols_ = np.nonzero(mask)
                perfect = Point(cols_.mean() + 0.5, rows_.mean() + 0.5, raw="segmentation")
            request = PointRequest(png, "image/png", W, H, description, truth_uv=truth_uv)

            answers: list[tuple[str, Point | None, float | None, str | None]] = [
                ("perfect", perfect, 0.0, None)
            ]
            for pointer in pointers:
                if pointer.name in skipped:
                    continue
                started = time.monotonic()
                try:
                    point = pointer.point(request)
                    answers.append((pointer.name, point, time.monotonic() - started, None))
                except PointerError as exc:
                    reason = mask_secrets(str(exc))
                    if trial == 0 and view == views[0]:
                        skipped[pointer.name] = reason
                        print(f"skipping {pointer.name}: {reason}", file=sys.stderr)
                        continue
                    answers.append((pointer.name, None, time.monotonic() - started, reason))
            for backend, point, seconds, error in answers:
                xyz = pixel_to_xyz(geometry, point.u, point.v, plane_z) if point else None
                rows.append(
                    {
                        "trial": trial,
                        "object": name,
                        "view": view,
                        "backend": backend,
                        "visible": visible,
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
                        "error": error,
                        "raw": (point.raw if point else (error or ""))[:200],
                    }
                )
        done = [r for r in rows if r["trial"] == trial and r["backend"] != "perfect"]
        brief = ", ".join(
            f"{r['backend']}/{r['view']}={r['error_cm']}" for r in done if r["backend"] != "center"
        )
        print(f"trial {trial:2d} {name:9s} at ({x:.3f}, {y:.3f}): {brief}", flush=True)

    rig.close()
    seg.close()
    table = summarize(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "trials": args.trials,
                "seed": args.seed,
                "size": [W, H],
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
    if disturbed:
        print(f"\nwrist view dropped for trials {disturbed}: the arm moved the object getting there")
    for name, reason in skipped.items():
        print(f"\nskipped {name}: {reason}")
    print(f"\nResults: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
