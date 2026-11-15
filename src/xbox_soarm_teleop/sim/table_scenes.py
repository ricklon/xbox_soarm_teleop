"""Table scenes: the shared table, plus the bins objects go into.

A table scene is a recipe. Every table scene has the same table and the same
objects (a red cube, a bean bag and five look-alikes; see ``simulate_mujoco``);
what differs is the fixed furniture on the table: one goal box, or several
bins. Each recipe compiles into its own MuJoCo model, so a twin (or a hub
running many twins) uses one recipe at a time. Which objects are out, where
they start and which bin each belongs in is a *layout* (``twin.layouts``),
and a scene can have many.

To make a new scene: add a ``TableScene`` to ``TABLE_SCENES`` with its bins,
then add its layouts in ``twin/layouts.py``, then prove each layout can be
done (``tests/test_twin_sort.py`` shows how: a scripted run with the true
positions must finish it).
"""

from __future__ import annotations

from dataclasses import dataclass

TABLE_HEIGHT_M = 0.12  # top of the table (m, base frame)
# The table's reach: x from 0.15 to 0.47 m, y from -0.10 to 0.18 m.
TABLE_CENTER_XY_M = (0.31, 0.04)
TABLE_HALF_XY_M = (0.16, 0.14)

GOAL_GREEN = (0.10, 0.65, 0.35, 0.85)


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
    """

    name: str
    label: str
    center_xy_m: tuple[float, float]
    inner_half_m: tuple[float, float, float] = (0.045, 0.045, 0.03)
    rgba: tuple[float, float, float, float] = GOAL_GREEN
    wall_m: float = 0.005
    geom_prefix: str | None = None

    @property
    def base_center_m(self) -> tuple[float, float, float]:
        """Center of the floor plate (it sits 1 mm above the table)."""
        return (*self.center_xy_m, TABLE_HEIGHT_M + 0.001)

    @property
    def floor_top_m(self) -> float:
        """Height of the top of its floor plate: objects in it rest here."""
        return TABLE_HEIGHT_M + 0.001 + 0.004

    def parts(self) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        """Each geom as (name, center offset from ``base_center_m``, half-size)."""
        prefix = self.geom_prefix or self.name
        hx, hy, hz = self.inner_half_m
        w = self.wall_m
        return [
            (f"{prefix}_base", (0.0, 0.0, 0.0), (hx, hy, 0.004)),
            (f"{prefix}_wall_left", (-(hx + w), 0.0, hz), (w, hy, hz)),
            (f"{prefix}_wall_right", (hx + w, 0.0, hz), (w, hy, hz)),
            (f"{prefix}_wall_back", (0.0, -(hy + w), hz), (hx + w, w, hz)),
            (f"{prefix}_wall_front", (0.0, hy + w, hz), (hx + w, w, hz)),
        ]


@dataclass(frozen=True)
class TableScene:
    """A compiled scene: the table and its bins."""

    name: str
    bins: tuple[Bin, ...]


# The original pick-and-place box. Its geoms keep their old names ("goal_base",
# ...), so pictures and benchmarks of the single-box scene are unchanged.
GOAL_BOX = Bin(
    name="goal_box",
    label="green goal box",
    center_xy_m=(0.29, 0.13),
    geom_prefix="goal",
)

# Three bins in a column along the far edge of the table, each its own
# direction from the base: reaching into one never swings the forearm over
# another (in a row leading away from the base, lowering into the far bin
# knocked a cube out of the near one). 7 cm inside, 1 cm apart.
# Low trays (2 cm walls): they stand between the front camera and the table,
# and 6 cm walls hid the cubes behind them from locate.
_SORT_BIN = (0.035, 0.035, 0.01)
RED_BIN = Bin("red_bin", "red bin", (0.41, 0.13), _SORT_BIN, (0.80, 0.20, 0.18, 0.85))
BLUE_BIN = Bin("blue_bin", "blue bin", (0.41, 0.04), _SORT_BIN, (0.20, 0.35, 0.85, 0.85))
GREEN_BIN = Bin("green_bin", "green bin", (0.41, -0.05), _SORT_BIN, GOAL_GREEN)

TABLE_SCENES: dict[str, TableScene] = {
    scene.name: scene
    for scene in (
        TableScene("pick_place_basic", (GOAL_BOX,)),
        TableScene("sort_bins", (RED_BIN, BLUE_BIN, GREEN_BIN)),
    )
}


def is_table_scene(scene: str) -> bool:
    """A scene with the table (and objects on it), as opposed to ``default`` or ``stack``."""
    return scene in TABLE_SCENES
