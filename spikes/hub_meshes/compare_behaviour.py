"""Does the arm behave the same with simplified meshes? (spike)

Runs the same scripted moves on a one-arm hub built with the original meshes
and with simplified ones, and prints where the cube ends up:

- a pick and place (assisted grasp, which does not use the meshes), and
- a push with the closed gripper (contact, which uses each part's convex hull).

Usage:
    uv run python spikes/hub_meshes/compare_behaviour.py --cell-mm 0 --cell-mm 2
"""

from __future__ import annotations

import argparse
import logging
import warnings

import numpy as np
from bench import make_hub


def run(hub, room, request: dict, ticks: int = 600) -> str:
    future = room.service.submit(request)
    for _ in range(ticks):
        hub.tick()
        if future.done():
            status = future.result()
            if status.state.value not in {"EXECUTING", "PENDING"}:
                break
    status = room.service.wait(future.result(timeout=0).id, 0)
    for _ in range(ticks):
        if status.state.value not in {"EXECUTING", "PENDING"}:
            break
        hub.tick()
        status = room.service.wait(status.id, 0)
    return f"{status.state.value}{'/' + status.reason.value if status.reason else ''}"


def cube(room) -> np.ndarray:
    return np.array(room.service.observe()["scene"]["objects"]["pick_cube"]["xyz_m"])


def move(xyz) -> dict:
    return {"type": "move_ee_to", "params": {"xyz_m": [float(v) for v in xyz]}}


def grip(position: float) -> dict:
    return {"type": "set_gripper", "params": {"position": position}}


def scenario(cell_mm: float) -> dict:
    hub, _ = make_hub(1, cell_mm)
    room = next(iter(hub.rooms.values()))
    hub.tick()
    start = cube(room)
    outcomes = []
    # Pick and place into the goal box.
    for request in (
        move(start + [0, 0, 0.08]), move(start + [0, 0, 0.01]), grip(1.0),
        move(start + [0, 0, 0.10]), move([0.29, 0.13, 0.24]), move([0.29, 0.13, 0.17]), grip(0.0),
        move([0.29, 0.13, 0.26]),
    ):  # fmt: skip
        outcomes.append(run(hub, room, request))
    for _ in range(100):
        hub.tick()
    placed = cube(room)
    # Push: put the cube back, close the gripper beside it and sweep through it.
    room.service.reset_scene()
    for _ in range(100):
        hub.tick()
    for request in (
        grip(1.0), move(start + [-0.02, -0.09, 0.06]), move(start + [-0.02, -0.09, 0.012]),
        move(start + [-0.02, 0.06, 0.012]), move(start + [-0.02, 0.06, 0.10]),
    ):  # fmt: skip
        outcomes.append(run(hub, room, request))
    for _ in range(100):
        hub.tick()
    pushed = cube(room)
    hub.close()
    return {"start": start, "placed": placed, "pushed": pushed, "outcomes": outcomes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cell-mm", type=float, action="append")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)
    results = {cell: scenario(cell) for cell in (args.cell_mm or [0.0, 2.0])}
    base = next(iter(results.values()))
    for cell, r in results.items():
        label = "full" if cell == 0 else f"{cell:g}mm"
        failed = [o for o in r["outcomes"] if not o.startswith("SUCCEEDED")]
        print(
            f"{label:5s} placed {np.round(r['placed'], 4)}  pushed {np.round(r['pushed'], 4)}  "
            f"pushed by {1000 * np.linalg.norm(r['pushed'] - r['start']):.1f} mm  "
            f"not succeeded: {failed or 'none'}"
        )
        if r is not base:
            print(
                f"      differs from full: placed {1000 * np.linalg.norm(r['placed'] - base['placed']):.2f} mm, "
                f"pushed {1000 * np.linalg.norm(r['pushed'] - base['pushed']):.2f} mm"
            )


if __name__ == "__main__":
    main()
