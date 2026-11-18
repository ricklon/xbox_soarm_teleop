"""A held object sits between the jaws as drawn: neither jaw sinks into it,
neither stands off it, and it is on the pads, not beside them.

Measured on the drawn jaw meshes (not their collision hulls), after the arm's
own ``pick``, for shapes that grip differently: the cube, a thin slab (the
jaws nearly shut), a tall tower (reaching up the fingers, where the moving jaw
curves in), a wide block, a bar, a small cube and the soft bean bag.
"""

import logging
import threading
import warnings

import mujoco
import numpy as np
import pytest

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
from xbox_soarm_teleop.twin import build_parts, build_twin
from xbox_soarm_teleop.twin.grasp import JAW_PAD_HALF_WIDTH_M

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

SPOT = (float(sim_cli.PICK_PLACE_CUBE_START[0]), float(sim_cli.PICK_PLACE_CUBE_START[1]))
SHAPES = [
    "pick_cube",
    "pink_slab",
    "teal_tower",
    "brown_block",
    "red_bar",
    "red_cube_small",
    "bean_bag",
]
# How far a drawn jaw may sink into, or stand off, a held object (m).
OVERLAP_M = 0.001
STAND_OFF_M = 0.0015
# A soft object is pulled to the middle of the pads; a rigid one stays where
# the jaws met it, which for a short object is low on the tilted pads.
SOFT_CENTRE_M = 0.001


@pytest.fixture(scope="module")
def parts():
    return build_parts(scene="pick_place_basic")


def put_out(twin, name):
    """Only ``name`` on the table, at the cube's usual spot, square to the table."""
    grasp = twin.grasp
    grasp.park([other for other in grasp.objects if other != name])
    z = sim_cli.PICK_PLACE_TABLE_HEIGHT + grasp.size(name)[2] / 2
    grasp.place(name, np.array([SPOT[0], SPOT[1], z]))
    mujoco.mj_forward(twin.sim.model, twin.sim.data)
    for _ in range(20):
        twin.tick()


def pick(twin, name) -> dict:
    """The arm's own pick skill, with the twin ticking beside it."""
    alive = threading.Event()
    alive.set()

    def loop():
        while alive.is_set():
            twin.tick()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        return twin.pick(name, client_id="test")
    finally:
        alive.clear()
        thread.join(timeout=10)


def _mesh_points(model, data, body):
    first, count = model.body_geomadr[body], model.body_geomnum[body]
    points = []
    for geom in range(first, first + count):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        start, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[start : start + n]
        points.append(data.geom_xpos[geom] + verts @ data.geom_xmat[geom].reshape(3, 3).T)
    return np.vstack(points)


def nearest_m(twin, name, body) -> float:
    """Signed distance (m) from a body's drawn mesh to the object's box: below 0 inside."""
    model, data, grasp = twin.sim.model, twin.sim.data, twin.grasp
    mujoco.mj_kinematics(model, data)
    local = (_mesh_points(model, data, body) - grasp.position(name)) @ grasp._rotation(name)
    q = np.abs(local) - grasp.size(name) / 2
    return float((np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)).min())


def across_pads_m(twin, name) -> float:
    origin, rotation = twin.grasp._gripper_frame()
    return float((rotation.T @ (twin.grasp.position(name) - origin))[1])


@pytest.mark.parametrize("name", SHAPES)
def test_a_held_object_sits_between_the_drawn_jaws(parts, name):
    twin = build_twin(parts=parts, with_controller=False)
    try:
        twin.tick()
        put_out(twin, name)
        result = pick(twin, name)
        assert result["ok"] is True, result
        grasp = twin.grasp
        assert grasp.attached == name
        for jaw, body in (("fixed", grasp._gripper_body), ("moving", grasp._jaw_body)):
            gap = nearest_m(twin, name, body)
            assert gap >= -OVERLAP_M, f"the {jaw} jaw sinks {-gap * 1e3:.1f} mm into {name}"
            assert gap <= STAND_OFF_M, f"the {jaw} jaw stands {gap * 1e3:.1f} mm off {name}"
        off = abs(across_pads_m(twin, name))
        limit = SOFT_CENTRE_M if grasp.objects[name].soft else JAW_PAD_HALF_WIDTH_M
        assert off <= limit, f"{name} is {off * 1e3:.1f} mm off the pads' middle"
    finally:
        twin.close()
