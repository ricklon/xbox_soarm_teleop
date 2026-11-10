"""Record a MuJoCo compatibility snapshot for the SO-ARM101 simulator scenes.

Loads the portable URDF into each scene (default, stack, pick/place), records
tool positions at fixed joint configurations, steps 100 times, and resets.
Running it under two MuJoCo versions and diffing the JSON shows whether the
simulator behaves the same.

Usage:
    uv run python scripts/mujoco_compat_probe.py docs/evidence/mujoco_<ver>_compatibility.json
"""

import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

import mujoco
import numpy as np

from xbox_soarm_teleop.cli.simulate_mujoco import URDF_PATH, MuJoCoSimulator

DEPENDENCIES = ("mujoco", "numpy", "placo", "lerobot", "cmeel-urdfdom", "cmeel-tinyxml2")
PROBE_JOINTS_DEG = ([0, 0, 0, 0, 0, 0], [10, 30, -45, 20, 35, 0], [-20, 10, -30, 10, -60, 0])
STEP_JOINTS_DEG = np.array([10, 30, -45, 20, 35, 0.0])


def build_report() -> dict:
    """Run the probe in the installed MuJoCo and return the snapshot."""
    report = {
        "python": platform.python_version(),
        "mujoco": mujoco.mj_versionString(),
        "dependencies": {name: metadata.version(name) for name in DEPENDENCIES},
        "urdf_sha256": hashlib.sha256(URDF_PATH.read_bytes()).hexdigest(),
        "scenes": {},
    }
    for scene in ("default", "stack", "pick_place_basic"):
        sim = MuJoCoSimulator(str(URDF_PATH), scene=scene)
        row = {
            "nq": sim.model.nq,
            "nv": sim.model.nv,
            "nbody": sim.model.nbody,
            "dynamic_step": sim.physics_scene_active,
            "tool_positions": [],
        }
        for q in PROBE_JOINTS_DEG:
            sim.set_joint_targets(np.array(q, dtype=float))
            mujoco.mj_forward(sim.model, sim.data)
            row["tool_positions"].append(sim.get_ee_position().tolist())
        start = sim.data.time
        for _ in range(100):
            sim.set_joint_targets(STEP_JOINTS_DEG)
            sim.step()
        row["stepped_time"] = sim.data.time - start
        row["qpos_after_100_steps"] = sim.data.qpos.tolist()
        row["finite"] = bool(np.all(np.isfinite(sim.data.qpos)) and np.all(np.isfinite(sim.data.qvel)))
        if scene == "stack":
            sim.reset_stack_scene()
        elif scene == "pick_place_basic":
            sim.reset_pick_place_scene()
        else:
            sim.go_home()
        row["qpos_after_reset"] = sim.data.qpos.tolist()
        report["scenes"][scene] = row
    return report


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    report = build_report()
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
    print(report["mujoco"], "report saved to", sys.argv[1])


if __name__ == "__main__":
    main()
