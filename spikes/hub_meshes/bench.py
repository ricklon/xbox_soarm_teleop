"""How many camera pictures a second can the hub draw, with full or simplified meshes? (spike)

Builds a hub whose model uses meshes simplified on an N mm grid (0 = the
original meshes), then has scripted "students" ask for camera pictures the
way the web page does, and counts what comes back.

Nothing in the hub is changed: the simplified model is swapped in only while
this script builds its hub. The IK still reads the original URDF.

Usage:
    uv run python spikes/hub_meshes/bench.py --cell-mm 0 --cell-mm 2 --seats 30
    uv run python spikes/hub_meshes/bench.py --cell-mm 2 --seats 10 --save shots/
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mujoco_web"))

from export_scene import simplify_mesh  # noqa: E402

from xbox_soarm_teleop.cli import simulate_mujoco as sm  # noqa: E402
from xbox_soarm_teleop.sim.world import apply_world  # noqa: E402
from xbox_soarm_teleop.twin.hub import Hub  # noqa: E402

VIEWS = ("overhead", "front", "side", "wrist")
PAGE_PAUSE_S = 0.130  # the student page waits this long between pictures


def build_model(scene: str, world: str, cell_mm: float, mesh_dir: Path):
    """Compile the hub's model, reading meshes simplified to ``cell_mm``."""
    spec = sm._robot_spec(str(sm.URDF_PATH))
    has_floor = apply_world(spec, world)
    sm._add_pick_place_scene(spec, add_floor=not has_floor)
    if cell_mm > 0:
        for mesh in spec.meshes:
            simplify_mesh(sm.URDF_PATH.parent / mesh.file, mesh_dir / mesh.file, cell_mm / 1000)
        spec.meshdir = str(mesh_dir)
    return spec.compile()


def make_hub(seats: int, cell_mm: float) -> tuple[Hub, float]:
    """A hub whose shared model uses the given mesh detail; also the compile time."""
    with tempfile.TemporaryDirectory() as tmp:
        started = time.perf_counter()
        model = build_model("pick_place_basic", sm.DEFAULT_WORLD, cell_mm, Path(tmp))
        compile_s = time.perf_counter() - started
    original = sm.load_model_with_cameras
    sm.load_model_with_cameras = lambda *args, **kwargs: model
    try:
        hub = Hub(seats=seats)
    finally:
        sm.load_model_with_cameras = original
    return hub, compile_s


def measure(hub: Hub, *, pause_s: float, seconds: float, save: Path | None, label: str) -> dict:
    """Have every room's "student" request pictures for ``seconds``."""
    stop = threading.Event()
    counts = [0] * len(hub.rooms)
    waits: list[float] = []

    def student(index: int, room) -> None:
        view = VIEWS[index % len(VIEWS)]
        while not stop.is_set():
            asked = time.perf_counter()
            try:
                image = room.service.render(view, width=480, height=360, client_id="web").result(10)
            except Exception:  # noqa: BLE001 - a busy or closing hub; the page retries too
                continue
            waits.append(time.perf_counter() - asked)
            counts[index] += 1
            if save is not None and counts[index] == 3 and index < len(VIEWS):
                save.mkdir(parents=True, exist_ok=True)
                (save / f"{label}-{view}.jpg").write_bytes(image)
            if pause_s:
                stop.wait(pause_s)

    threads = [
        threading.Thread(target=student, args=(i, room), daemon=True)
        for i, room in enumerate(hub.rooms.values())
    ]
    for thread in threads:
        thread.start()
    time.sleep(2.0)  # renderers are made on first use
    counts[:] = [0] * len(counts)
    waits.clear()
    ticks, started = hub.ticks, time.perf_counter()
    time.sleep(seconds)
    elapsed = time.perf_counter() - started
    total, loop_hz = sum(counts), (hub.ticks - ticks) / elapsed
    ordered = sorted(waits) or [0.0]
    stop.set()
    for thread in threads:
        thread.join(timeout=12)
    return {
        "total_fps": total / elapsed,
        "per_student_fps": total / elapsed / len(counts),
        "slowest_student_fps": min(counts) / elapsed,
        "median_wait_ms": 1000 * ordered[len(ordered) // 2],
        "loop_hz": loop_hz,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cell-mm", type=float, action="append", help="0 = original meshes")
    parser.add_argument("--seats", type=int, action="append")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--save", type=Path, help="Directory for one sample picture per view")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

    print("meshes  triangles  compile  seats  mode       total/s  each/s  slowest/s  wait ms  loop Hz")
    for cell_mm in args.cell_mm or [0.0, 2.0]:
        for seats in args.seats or [10, 30]:
            hub, compile_s = make_hub(seats, cell_mm)
            hub.start()
            label = "full" if cell_mm == 0 else f"{cell_mm:g}mm"
            try:
                for mode, pause in (("page", PAGE_PAUSE_S), ("flat out", 0.0)):
                    r = measure(hub, pause_s=pause, seconds=args.seconds, save=args.save, label=label)
                    print(
                        f"{label:6s}  {hub.parts.model.nmeshface:9d}  {compile_s:6.2f}s  {seats:5d}  "
                        f"{mode:9s}  {r['total_fps']:7.1f}  {r['per_student_fps']:6.2f}  "
                        f"{r['slowest_student_fps']:9.2f}  {r['median_wait_ms']:7.0f}  {r['loop_hz']:7.1f}",
                        flush=True,
                    )
            finally:
                hub.close()


if __name__ == "__main__":
    main()
