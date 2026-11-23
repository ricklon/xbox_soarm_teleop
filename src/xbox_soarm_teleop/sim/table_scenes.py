"""Table scenes: a work surface, the bins on it, and the cameras that watch it.

A table scene is a recipe for a work cell. Every table scene has the same
objects (a red cube, a bean bag and five look-alikes; see ``simulate_mujoco``);
what differs is the cell: the table (its size and height), the fixed
furniture on it (one goal box, or several bins), and, optionally, cameras
aimed at it. Each recipe compiles into its own MuJoCo model, so a twin (or a
hub running many twins) uses one recipe at a time. Which objects are out and
where they start is a *layout* (``twin.layouts``); what goes where is a
*task* (``twin.tasks``).

To make a new cell: add a ``TableScene`` to ``TABLE_SCENES``, then its
layouts and tasks, then prove each task can be done (``tests/test_twin_sort.py``
shows how: a scripted run with the true positions must finish it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TABLE_HEIGHT_M = 0.12  # top of the single-box scene's raised table (m, base frame)
# The raised table's reach: x from 0.15 to 0.47 m, y from -0.10 to 0.18 m.
TABLE_CENTER_XY_M = (0.31, 0.04)
TABLE_HALF_XY_M = (0.16, 0.14)

GOAL_GREEN = (0.10, 0.65, 0.35, 0.85)
BIN_FLOOR_M = 0.004  # half the thickness of a bin's floor plate


@dataclass(frozen=True)
class Table:
    """The work surface: a box standing on the floor, top at ``top_z``.

    Args:
        center_xy_m: Middle of the table (m, base frame).
        half_xy_m: Half its length (x) and width (y).
        top_z: Height of its top. A desk the arm is clamped to has its top at
            the arm's base, a few millimetres up.
        rgba: Colour.
    """

    center_xy_m: tuple[float, float]
    half_xy_m: tuple[float, float]
    top_z: float
    rgba: tuple[float, float, float, float] = (0.60, 0.60, 0.60, 1.0)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """(x, y) lies on the table, at least ``margin`` in from every edge."""
        cx, cy = self.center_xy_m
        hx, hy = self.half_xy_m
        return abs(x - cx) <= hx - margin and abs(y - cy) <= hy - margin


@dataclass(frozen=True)
class Bin:
    """An open box on the table: a floor plate and four walls.

    Args:
        name: What observations and agents call it ("red_bin").
        label: How it is described in words ("red bin").
        center_xy_m: Middle of the bin (m, base frame).
        inner_half_m: Half its inside length and width, and half its wall height.
        rgba: Colour of its walls and floor.
        wall_m: Wall thickness.
        geom_prefix: Prefix of its MuJoCo geom names (default: ``name``).
        movable: A box the arm can push about (a free body, with things in
            it riding along), not furniture fixed to the table. Movable boxes
            go in ``TableScene.boxes``; ``center_xy_m`` is then unused (a
            layout says where each starts).
        mass_kg: A movable box's mass.
    """

    name: str
    label: str
    center_xy_m: tuple[float, float]
    inner_half_m: tuple[float, float, float] = (0.045, 0.045, 0.03)
    rgba: tuple[float, float, float, float] = GOAL_GREEN
    wall_m: float = 0.005
    geom_prefix: str | None = None
    movable: bool = False
    mass_kg: float = 0.08

    def base_center_m(self, table_top_z: float) -> tuple[float, float, float]:
        """Center of the floor plate (it sits 1 mm above the table)."""
        return (*self.center_xy_m, table_top_z + 0.001)

    def floor_top_m(self, table_top_z: float) -> float:
        """Height of the top of its floor plate: objects in it rest here."""
        return table_top_z + 0.001 + BIN_FLOOR_M

    def parts(self) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        """Each geom as (name, center offset from ``base_center_m``, half-size)."""
        prefix = self.geom_prefix or self.name
        hx, hy, hz = self.inner_half_m
        w = self.wall_m
        return [
            (f"{prefix}_base", (0.0, 0.0, 0.0), (hx, hy, BIN_FLOOR_M)),
            (f"{prefix}_wall_left", (-(hx + w), 0.0, hz), (w, hy, hz)),
            (f"{prefix}_wall_right", (hx + w, 0.0, hz), (w, hy, hz)),
            (f"{prefix}_wall_back", (0.0, -(hy + w), hz), (hx + w, w, hz)),
            (f"{prefix}_wall_front", (0.0, hy + w, hz), (hx + w, w, hz)),
        ]


@dataclass(frozen=True)
class TableScene:
    """A compiled work cell: the table, its bins and, optionally, its own cameras.

    ``views`` aims the named cameras (``front``, ``overhead``, ``side``) as
    lookat, distance, azimuth and elevation, like the shared presets in
    ``twin.cameras``. A scene without them uses those presets.

    ``boxes`` are movable boxes: free bodies, parked out of sight unless a
    layout puts them on the table, like the scene's other objects.

    ``game`` is what students choose on a hub that lets them switch
    (``pick_and_place``, ``sorting``).
    """

    name: str
    table: Table
    bins: tuple[Bin, ...]
    views: dict[str, dict] | None = None
    stat_center: tuple[float, float, float] = (0.2245, 0.04, 0.0779)
    stat_extent: float = 0.6128
    boxes: tuple[Bin, ...] = ()
    game: str = "pick_and_place"


RAISED_TABLE = Table(TABLE_CENTER_XY_M, TABLE_HALF_XY_M, TABLE_HEIGHT_M)

# The original pick-and-place box. Its geoms keep their old names ("goal_base",
# ...), so pictures and benchmarks of the single-box scene are unchanged.
GOAL_BOX = Bin(
    name="goal_box",
    label="green goal box",
    center_xy_m=(0.29, 0.13),
    geom_prefix="goal",
)

# The sorting desk: a board at the arm's base, as on a real desk it is clamped
# to, covering the ring the arm grasps in (about 17 to 36 cm out). Items are
# picked in front of the arm and placed to its sides, so the arm never passes
# over items still to be sorted and nothing stands between the cameras and
# them. Trays: 10 cm inside (a cube carried to the side lands turned with the
# gripper), 2 cm walls (cameras see over them), 27 cm out, inside the enforced
# workspace (x >= 0.05 m, |y| <= 0.30 m for the tool point).
DESK = Table(center_xy_m=(0.17, 0.0), half_xy_m=(0.27, 0.42), top_z=0.002, rgba=(0.62, 0.57, 0.5, 1.0))
TRAY_RADIUS_M = 0.27
_TRAY = (0.05, 0.05, 0.01)


def _tray(name: str, label: str, angle_deg: float, rgba) -> Bin:
    a = math.radians(angle_deg)
    xy = (round(TRAY_RADIUS_M * math.cos(a), 4), round(TRAY_RADIUS_M * math.sin(a), 4))
    return Bin(name, label, xy, _TRAY, rgba)


RED_BIN = _tray("red_bin", "red bin", 45.0, (0.80, 0.20, 0.18, 0.9))
GREEN_BIN = _tray("green_bin", "green bin", 72.0, GOAL_GREEN)
BLUE_BIN = _tray("blue_bin", "blue bin", -70.0, (0.20, 0.35, 0.85, 0.9))

# A box that slides when pushed: 7 cm inside (a cube fits with room), low
# walls like the trays', light enough to push with what is in it.
ORANGE_BOX = Bin(
    "orange_box", "orange box", (0.0, 0.0), (0.035, 0.035, 0.012), (0.95, 0.55, 0.12, 1.0), movable=True
)

DESK_VIEWS = {
    # Straight down over the whole ring; picture up is forward (+x).
    "overhead": {"lookat": (0.19, 0.0, 0.02), "distance": 0.78, "azimuth": 0.0, "elevation": -89.0},
    # From beyond the pick area, looking back at the arm.
    "front": {"lookat": (0.20, 0.0, 0.02), "distance": 0.75, "azimuth": 180.0, "elevation": -35.0},
    # From the arm's left.
    "side": {"lookat": (0.20, 0.0, 0.02), "distance": 0.80, "azimuth": 90.0, "elevation": -30.0},
}

TABLE_SCENES: dict[str, TableScene] = {
    scene.name: scene
    for scene in (
        TableScene("pick_place_basic", RAISED_TABLE, (GOAL_BOX,)),
        TableScene(
            "sort_bins",
            DESK,
            (RED_BIN, BLUE_BIN, GREEN_BIN),
            views=DESK_VIEWS,
            stat_center=(0.17, 0.0, 0.1),
            stat_extent=0.7,
            boxes=(ORANGE_BOX,),
            game="sorting",
        ),
    )
}


def is_table_scene(scene: str) -> bool:
    """A scene with a table (and objects on it), as opposed to ``default`` or ``stack``."""
    return scene in TABLE_SCENES


# A model with every table scene's furniture (``simulate_mujoco._add_classroom_scene``),
# for a hub whose students each choose their own game. Each scene's table and
# bins ride on one mocap body, a "cell": a room poses its own at the origin
# and parks the rest out of sight, far above and away.
CLASSROOM = "classroom"
CELL_PARK_XYZ = (0.0, 40.0, 20.0)  # the first cell's parking spot; the rest 5 m apart


def cell_body_name(scene: str) -> str:
    """The mocap body carrying ``scene``'s furniture in the classroom model."""
    return f"cell_{scene}"


def cell_camera_name(view: str, scene: str) -> str:
    """The classroom model's camera for a view of ``scene`` (``front``, ...)."""
    return f"{view}_{scene}"


def cell_park_xyz(index: int) -> tuple[float, float, float]:
    """Where the ``index``-th cell waits while another is in use."""
    x, y, z = CELL_PARK_XYZ
    return (x + 5.0 * index, y, z)
