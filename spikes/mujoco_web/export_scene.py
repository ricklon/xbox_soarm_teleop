"""Write the twin's scene as files a browser can load (spike).

The twin builds its MuJoCo model in Python from the robot URDF plus a world
and a scene. This writes the same model as MJCF XML and copies the meshes it
names, with a manifest, so the page next to this script can compile it with
MuJoCo's WebAssembly build.

Usage:
    uv run python spikes/mujoco_web/export_scene.py
    uv run python spikes/mujoco_web/export_scene.py --scene stack
    uv run python spikes/mujoco_web/export_scene.py --cell-mm 2 --out spikes/mujoco_web/scene-2mm

``--cell-mm`` writes simplified meshes: corners that fall in the same cube of
that size are merged into one. It reports how far each part's contact shell
(its convex hull) moved.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from xbox_soarm_teleop.cli import simulate_mujoco as sm
from xbox_soarm_teleop.sim.world import apply_world

HERE = Path(__file__).resolve().parent


STL_RECORD = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attr", "<u2")])


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary STL as welded (points, faces)."""
    raw = path.read_bytes()
    count = int(np.frombuffer(raw, "<u4", 1, 80)[0])
    corners = np.frombuffer(raw, STL_RECORD, count, 84)["corners"].reshape(-1, 3)
    points, index = np.unique(corners, axis=0, return_inverse=True)
    return points.astype(np.float32), index.reshape(-1, 3).astype(np.int32)


def write_stl(path: Path, points: np.ndarray, faces: np.ndarray) -> None:
    """Write (points, faces) as a binary STL."""
    records = np.zeros(len(faces), STL_RECORD)
    corners = points[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    records["normal"] = normal / np.where(length > 0, length, 1)
    records["corners"] = corners
    with path.open("wb") as file:
        file.write(b"simplified".ljust(80, b" "))
        file.write(np.uint32(len(faces)).tobytes())
        file.write(records.tobytes())


def hulls(model) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each mesh's contact shell as MuJoCo compiled it: (points, outward planes).

    Points are returned in the mesh file's own frame, so shells from two
    compiles can be compared.
    """
    import mujoco

    shells = {}
    for i in range(model.nmesh):
        start = model.mesh_graphadr[i]
        if start < 0:
            continue
        graph = model.mesh_graph[start:]
        nvert, nface = int(graph[0]), int(graph[1])
        ids = graph[2 + nvert : 2 + 2 * nvert]
        face_start = 2 + 2 * nvert + nvert + 3 * nface
        faces = graph[face_start : face_start + 3 * nface].reshape(-1, 3)
        vert0 = model.mesh_vertadr[i]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.mesh_quat[i])
        every = model.mesh_vert[vert0 : vert0 + model.mesh_vertnum[i]].astype(np.float64)
        every = every @ rotation.reshape(3, 3).T + model.mesh_pos[i]
        corners = every[faces]
        side1, side2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        normal = np.cross(side1, side2)
        # Sliver faces have no reliable direction; their neighbours bound the same region.
        sharpness = np.linalg.norm(side1, axis=1) * np.linalg.norm(side2, axis=1)
        keep = np.linalg.norm(normal, axis=1) > 0.02 * np.maximum(sharpness, 1e-30)
        normal = normal[keep] / np.linalg.norm(normal[keep], axis=1, keepdims=True)
        offset = -np.einsum("ij,ij->i", normal, corners[keep][:, 0])
        inside = every[ids].mean(axis=0)
        flip = (normal @ inside + offset) > 0
        normal[flip], offset[flip] = -normal[flip], -offset[flip]
        shells[model.mesh(i).name] = (every[ids], np.column_stack([normal, offset]))
    return shells


def shell_shift_mm(before, after) -> float:
    """Largest distance (mm) by which one contact shell sticks out of the other."""

    def outside(points: np.ndarray, planes: np.ndarray) -> float:
        return float(np.max(points @ planes[:, :3].T + planes[:, 3]))

    return 1000 * max(outside(before[0], after[1]), outside(after[0], before[1]), 0.0)


def simplify_mesh(source: Path, target: Path, cell_m: float) -> dict:
    """Write ``source`` with corners merged on a grid of ``cell_m``; report the counts.

    Vertex clustering: it works on any mesh, including these CAD exports with
    loose internal pieces, where edge-collapse simplifiers stalled and pushed
    corners outward.
    """
    points, faces = read_stl(source)
    cells = np.floor(points / cell_m).astype(np.int64)
    _, group = np.unique(cells, axis=0, return_inverse=True)
    group = group.ravel()
    merged = np.zeros((group.max() + 1, 3))
    np.add.at(merged, group, points)
    merged /= np.bincount(group)[:, None]
    new_faces = group[faces]
    flat = (new_faces[:, 0] == new_faces[:, 1]) | (new_faces[:, 1] == new_faces[:, 2])
    new_faces = new_faces[~(flat | (new_faces[:, 0] == new_faces[:, 2]))]
    _, first = np.unique(np.sort(new_faces, axis=1), axis=0, return_index=True)
    new_faces = new_faces[np.sort(first)]
    write_stl(target, merged.astype(np.float32), new_faces)
    return {"faces_before": len(faces), "faces": len(new_faces), "shell_shift_mm": 0.0}


def export(scene: str, world: str, out: Path, cell_mm: float = 0.0) -> dict:
    """Write ``scene.xml``, its meshes and ``manifest.json`` into ``out``."""
    spec = sm._robot_spec(str(sm.URDF_PATH))
    has_floor = apply_world(spec, world)
    if scene == "stack":
        sm._add_stack_scene(spec, add_floor=not has_floor)
    elif scene == "pick_place_basic":
        sm._add_pick_place_scene(spec, add_floor=not has_floor)
    model = spec.compile()
    xml = spec.to_xml()

    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.xml").write_text(xml, encoding="utf-8")
    meshes = sorted(set(re.findall(r'file="([^"]+)"', xml)))
    parts = {}
    for name in meshes:
        if cell_mm > 0:
            parts[name] = simplify_mesh(sm.URDF_PATH.parent / name, out / name, cell_mm / 1000)
        else:
            shutil.copyfile(sm.URDF_PATH.parent / name, out / name)
    if cell_mm > 0:  # count what the simplified files compile to
        import mujoco

        before = hulls(model)
        model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
        after = hulls(model)
        for name in meshes:
            key = Path(name).stem
            parts[name]["shell_shift_mm"] = round(shell_shift_mm(before[key], after[key]), 3)
    manifest = {
        "scene": scene,
        "world": world,
        "cell_mm": cell_mm,
        "parts": parts,
        "xml": "scene.xml",
        "files": meshes,
        "bytes": sum((out / name).stat().st_size for name in meshes),
        "mesh_faces": int(model.nmeshface),
        "timestep": float(model.opt.timestep),
        "arm_joints": list(sm.JOINT_NAMES),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="pick_place_basic", choices=["pick_place_basic", "stack"])
    parser.add_argument("--world", default=sm.DEFAULT_WORLD)
    parser.add_argument("--out", type=Path, default=HERE / "scene")
    parser.add_argument("--cell-mm", type=float, default=0.0, help="Merge corners on a grid this fine")
    args = parser.parse_args()
    manifest = export(args.scene, args.world, args.out, args.cell_mm)
    size = manifest["bytes"] / 1e6
    print(
        f"Wrote {args.out} ({len(manifest['files'])} meshes, {size:.1f} MB, "
        f"{manifest['mesh_faces']} triangles)"
    )
    if manifest["parts"]:
        worst = max(manifest["parts"].items(), key=lambda item: item[1]["shell_shift_mm"])
        print(f"Largest contact-shell change: {worst[1]['shell_shift_mm']} mm ({worst[0]})")


if __name__ == "__main__":
    main()
