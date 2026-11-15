"""Put a twin's objects into known situations, for perception benchmarks.

Each function sets up one situation on a twin driven by hand (``tick`` on
the calling thread): an object on the table, in the goal box, just outside
it, tipped onto an edge, or held in the gripper. Placements set the pose
directly and do not step physics, so a tipped object stays tipped while it
is photographed; :func:`grasp_and_lift` runs the real grasp.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
from xbox_soarm_teleop.twin.grasp import GraspSettings
from xbox_soarm_teleop.twin.layouts import BINS, GOALS

# The table top (m, base frame), and how far inside its edges an object's
# center stays: half the largest object's diagonal footprint.
_TABLE_C = sim_cli.PICK_PLACE_TABLE_CENTER
_TABLE_H = sim_cli.PICK_PLACE_TABLE_HALF_EXTENTS
EDGE_MARGIN_M = 0.045
# Where objects are put on the table: within the arm's reach, and on the table.
TABLE_X = (0.22, 0.40)
TABLE_Y = (
    float(_TABLE_C[1] - _TABLE_H[1]) + EDGE_MARGIN_M,
    float(_TABLE_C[1] + _TABLE_H[1]) - EDGE_MARGIN_M,
)
# The goal box's outer half-size (inner half plus a wall), and the floor's top.
GOAL_OUTER_HALF = float(
    sim_cli.PICK_PLACE_GOAL_INNER_HALF_EXTENTS[0] + 2 * sim_cli.PICK_PLACE_GOAL_WALL_THICKNESS
)


def _quat_yaw_tilt(yaw: float, tilt: float = 0.0) -> np.ndarray:
    """Turn by ``yaw`` about z after tilting by ``tilt`` about the object's x axis."""
    q_yaw = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    q_tilt = np.array([math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q_yaw, q_tilt)
    return out


def reset(twin) -> None:
    """Home the arm and restore the scene (as the reset button does)."""
    future = twin.reset_scene(client_id="scenario")
    twin.tick()
    future.result(timeout=0)


def _set_pose(twin, name: str, xy, rest_z: float, yaw: float, tilt: float = 0.0) -> np.ndarray:
    """Rest one object on ``rest_z`` at ``xy``, turned as asked; touch nothing else."""
    grasp = twin.grasp
    obj = grasp.objects[name]
    data = twin.sim.data
    data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7] = _quat_yaw_tilt(yaw, tilt)
    mujoco.mj_forward(twin.sim.model, data)
    z = rest_z + float(grasp.half_extents_world(name)[2])
    data.qpos[obj.qpos_adr : obj.qpos_adr + 3] = [float(xy[0]), float(xy[1]), z]
    data.qvel[obj.qvel_adr : obj.qvel_adr + 6] = 0.0
    mujoco.mj_forward(twin.sim.model, data)
    return grasp.position(name)


def _pose(
    twin, name: str, xy, rest_z: float, yaw: float, tilt: float = 0.0, park_others: bool = True
) -> np.ndarray:
    """Like :func:`_set_pose`; by default the other objects are parked first."""
    if park_others:
        twin.grasp.park([other for other in twin.grasp.objects if other != name])
    elif name in twin.grasp.parked:
        twin.grasp.park([other for other in twin.grasp.parked if other != name])
    return _set_pose(twin, name, xy, rest_z, yaw, tilt)


def _goal():
    return GOALS["pick_place_basic"]


def random_table_spot(rng: np.random.Generator, clearance: float = 0.08) -> tuple[float, float]:
    """A spot on the table within reach, ``clearance`` away from the goal box's center."""
    goal = _goal()
    while True:
        x, y = rng.uniform(*TABLE_X), rng.uniform(*TABLE_Y)
        if abs(x - goal.center_xyz_m[0]) < clearance and abs(y - goal.center_xyz_m[1]) < clearance:
            continue
        return float(x), float(y)


# How far a random start keeps an object's center from the goal box's center,
# on at least one axis: the 4 cm cube then clears the box's outer walls by 3 cm.
START_CLEARANCE_M = 0.10


def start_spot(seed: int, run: int) -> tuple[float, float]:
    """Run ``run``'s random start for an agent trial: the same for every variant
    and model given the same seed, on the table, within reach, clear of the box."""
    rng = np.random.default_rng([seed, run])
    gx, gy, _ = _goal().center_xyz_m
    while True:
        x, y = float(rng.uniform(*TABLE_X)), float(rng.uniform(*TABLE_Y))
        if max(abs(x - gx), abs(y - gy)) >= START_CLEARANCE_M:
            return x, y


def parse_start(text: str) -> tuple[float, float]:
    """``"X,Y"`` (m, base frame) to a start spot."""
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"start must be X,Y in meters, not {text!r}")
    return float(parts[0]), float(parts[1])


def place_start(twin, name: str, x: float, y: float) -> np.ndarray:
    """Start ``name`` flat on the table at (x, y), leaving every other object as it is."""
    return _pose(twin, name, (x, y), twin.table_top_z, 0.0, park_others=False)


def place_on_table(twin, name: str, x: float, y: float, yaw: float = 0.0) -> np.ndarray:
    """Lay the object flat on the table at (x, y)."""
    return _pose(twin, name, (x, y), twin.table_top_z, yaw)


def place_in_bin(
    twin, name: str, bin_name: str, dx: float = 0.0, dy: float = 0.0, yaw: float = 0.0
) -> np.ndarray:
    """Lay the object flat on a bin's floor, (dx, dy) from its center, leaving
    every other object as it is."""
    box = next(box for box in BINS[twin.sim.scene] if box.name == bin_name)
    cx, cy, floor = box.center_xyz_m
    return _pose(twin, name, (cx + dx, cy + dy), floor, yaw, park_others=False)


def place_in_goal(
    twin,
    name: str,
    dx: float = 0.0,
    dy: float = 0.0,
    yaw: float = 0.0,
    park_others: bool = True,
) -> np.ndarray:
    """Lay the object flat on the goal box floor, (dx, dy) from its center.

    ``park_others=False`` leaves the rest of the scene as it is.
    """
    cx, cy, floor = _goal().center_xyz_m
    return _pose(twin, name, (cx + dx, cy + dy), floor, yaw, park_others=park_others)


def _footprints_clear(twin, a: str, b: str, gap: float) -> bool:
    pa, pb = twin.grasp.position(a), twin.grasp.position(b)
    ea, eb = twin.grasp.half_extents_world(a), twin.grasp.half_extents_world(b)
    gap_x = abs(pa[0] - pb[0]) - ea[0] - eb[0]
    gap_y = abs(pa[1] - pb[1]) - ea[1] - eb[1]
    return max(gap_x, gap_y) >= gap


def place_with_distractors(
    twin,
    target: str,
    distractors,
    rng: np.random.Generator,
    *,
    gap: float = 0.02,
    target_xy=None,
    target_yaw: float | None = None,
    max_tries: int = 100,
    max_layouts: int = 200,
) -> dict[str, np.ndarray]:
    """Put the target and its distractors flat on the table at random, apart.

    Every other object is parked. Footprints (as turned) keep ``gap`` between
    them, all on the table and clear of the goal box. ``target_xy`` fixes the
    target's spot, and ``target_yaw`` its turn (for paired trials with and
    without distractors).

    Raises:
        RuntimeError: no arrangement fits after ``max_layouts`` tries (each
            object gets ``max_tries`` spots before the arrangement starts over).
    """
    names = [target, *distractors]
    twin.grasp.park([other for other in twin.grasp.objects if other not in names])
    for _ in range(max_layouts):
        placed: list[str] = []
        for name in names:
            for _ in range(max_tries):
                if name == target and target_xy is not None:
                    xy = target_xy
                else:
                    xy = random_table_spot(rng)
                yaw = float(rng.uniform(-math.pi / 2, math.pi / 2))
                if name == target and target_yaw is not None:
                    yaw = target_yaw
                _set_pose(twin, name, xy, twin.table_top_z, yaw)
                if all(_footprints_clear(twin, name, other, gap) for other in placed):
                    placed.append(name)
                    break
            else:
                break  # early ones took the room: start the arrangement over
        if len(placed) == len(names):
            return {name: twin.grasp.position(name) for name in names}
    raise RuntimeError(f"no room for {', '.join(names)} after {max_layouts} arrangements")


def place_near_goal(twin, name: str, rng: np.random.Generator, yaw: float | None = None) -> np.ndarray:
    """Lay the object on the table just outside one of the goal box's walls (5-30 mm gap).

    The hard negative for "is it in the box?". Not on the robot's left of the
    box (+y): the table ends there.
    """
    yaw = float(rng.uniform(-math.pi / 4, math.pi / 4)) if yaw is None else yaw
    cx, cy, _ = _goal().center_xyz_m
    # Lay it down first to learn its footprint as turned.
    _pose(twin, name, (cx, cy), twin.table_top_z, yaw)
    half = twin.grasp.half_extents_world(name)
    gap = float(rng.uniform(0.005, 0.03))
    along = float(rng.uniform(-0.03, 0.03))
    side = rng.choice(["-x", "+x", "-y"])
    if side == "-x":
        xy = (cx - GOAL_OUTER_HALF - half[0] - gap, cy + along)
    elif side == "+x":
        xy = (cx + GOAL_OUTER_HALF + half[0] + gap, cy + along)
    else:
        xy = (cx + along, cy - GOAL_OUTER_HALF - half[1] - gap)
    # Keep it on the table (the box is near the table's edges); it stays clear of the walls.
    x_min = float(_TABLE_C[0] - _TABLE_H[0]) + float(half[0])
    y_max = float(_TABLE_C[1] + _TABLE_H[1]) - float(half[1])
    xy = (max(xy[0], x_min), min(xy[1], y_max))
    return _pose(twin, name, xy, twin.table_top_z, yaw)


def place_tipped(
    twin,
    name: str,
    x: float,
    y: float,
    tilt: float,
    yaw: float = 0.0,
    rest_z: float | None = None,
) -> np.ndarray:
    """Rest the object on an edge, tilted by ``tilt`` (rad), on the table or ``rest_z``."""
    return _pose(twin, name, (x, y), twin.table_top_z if rest_z is None else rest_z, yaw, tilt)


def run(twin, request: dict, max_ticks: int = 1500) -> bool:
    """Submit one action and tick until it ends; True if it succeeded."""
    future = twin.submit(request, client_id="scenario")
    twin.tick()
    status = future.result(timeout=0)
    for _ in range(max_ticks):
        status = twin.wait(status.id, 0)
        if status.state.terminal:
            break
        twin.tick()
    return status.state.value == "SUCCEEDED"


def grasp_and_lift(twin, name: str, lift_to) -> bool:
    """Pick the object up the way the agent instructions say, then carry it to ``lift_to``.

    Uses the true position (this sets up a situation; it is not a test of
    finding the object). True if it is held at the end.
    """
    settings = GraspSettings()
    twin.tick()  # the snapshot read below must list an object just placed

    def obj() -> dict:
        return twin.observe_truth()["scene"]["objects"][name]

    x, y, z = obj()["xyz_m"]
    run(twin, {"type": "move_ee_to", "params": {"xyz_m": [x, y, z + 0.06]}})
    run(twin, {"type": "set_wrist_roll", "params": {"angle_rad": -1.5}})
    width = obj()["grip_width_m"]
    run(
        twin,
        {"type": "set_gripper", "params": {"position": settings.gripper_for_opening(width + 0.05)}},
    )
    for _ in range(2):
        jaw, at = twin.observe_truth()["jaw_center_xyz_m"], obj()["xyz_m"]
        run(twin, {"type": "move_ee_by", "params": {"dxyz_m": [at[0] - jaw[0], at[1] - jaw[1], 0.0]}})
    lowest = twin.workspace_bounds["z"][0] + 0.015
    drop = max(obj()["xyz_m"][2], lowest) - twin.observe_truth()["jaw_center_xyz_m"][2]
    run(twin, {"type": "move_ee_by", "params": {"dxyz_m": [0.0, 0.0, drop]}})
    run(twin, {"type": "set_gripper", "params": {"position": settings.gripper_for_opening(width)}})
    if twin.grasp.attached != name:
        return False
    run(twin, {"type": "move_ee_by", "params": {"dxyz_m": [0.0, 0.0, 0.06]}})
    run(twin, {"type": "move_ee_to", "params": {"xyz_m": [float(v) for v in lift_to]}})
    return twin.grasp.attached == name
