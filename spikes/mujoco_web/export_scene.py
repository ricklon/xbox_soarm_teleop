"""Write the twin's scene as files a browser can load (spike).

The twin builds its MuJoCo model in Python from the robot URDF plus a world
and a scene. This writes the same model as MJCF XML and copies the meshes it
names, with a manifest, so the page next to this script can compile it with
MuJoCo's WebAssembly build.

Usage:
    uv run python spikes/mujoco_web/export_scene.py
    uv run python spikes/mujoco_web/export_scene.py --scene stack
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from xbox_soarm_teleop.cli import simulate_mujoco as sm
from xbox_soarm_teleop.sim.world import apply_world

HERE = Path(__file__).resolve().parent


def export(scene: str, world: str, out: Path) -> dict:
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
    for name in meshes:
        shutil.copyfile(sm.URDF_PATH.parent / name, out / name)
    manifest = {
        "scene": scene,
        "world": world,
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
    args = parser.parse_args()
    manifest = export(args.scene, args.world, args.out)
    size = manifest["bytes"] / 1e6
    print(
        f"Wrote {args.out} ({len(manifest['files'])} meshes, {size:.1f} MB, "
        f"{manifest['mesh_faces']} triangles)"
    )


if __name__ == "__main__":
    main()
