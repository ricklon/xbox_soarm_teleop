"""Scene layouts: which objects are on the table, switchable while a twin runs.

A compiled model holds every object its layouts use, so changing layout never
recompiles anything (the hub shares one model between all its rooms). A
layout names the objects that are on the table and where they start; the
twin keeps the rest parked out of sight and out of reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli


@dataclass(frozen=True)
class Layout:
    """One arrangement of a model scene's objects.

    Args:
        name: What agents and the command line call it.
        model_scene: The compiled scene (``MuJoCoSimulator.scene``) it arranges.
        difficulty: ``"easy"`` or ``"hard"``.
        description: One sentence for an agent choosing a scene.
        objects: Start position (m, base frame) of each object on the table.
    """

    name: str
    model_scene: str
    difficulty: str
    description: str
    objects: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "difficulty": self.difficulty,
            "description": self.description,
            "objects": list(self.objects),
        }


_START_XY = tuple(float(v) for v in sim_cli.PICK_PLACE_CUBE_START[:2])
_TABLE = float(sim_cli.PICK_PLACE_TABLE_HEIGHT)

LAYOUTS: dict[str, Layout] = {
    layout.name: layout
    for layout in (
        Layout(
            name="pick_place_easy",
            model_scene="pick_place_basic",
            difficulty="easy",
            description=(
                "A soft blue bean bag and a green goal box. The bean bag gives way "
                "to the gripper, so it is not knocked about when the jaws touch it."
            ),
            objects={
                sim_cli.BEAN_BAG_NAME: (*_START_XY, _TABLE + float(sim_cli.BEAN_BAG_HALF_EXTENTS[2]))
            },
        ),
        Layout(
            name="pick_place_basic",
            model_scene="pick_place_basic",
            difficulty="hard",
            description=(
                "A rigid red cube and a green goal box. The gripper pushes the cube "
                "if it touches it, so the approach has to keep clear until it closes."
            ),
            objects={"pick_cube": tuple(float(v) for v in sim_cli.PICK_PLACE_CUBE_START)},
        ),
    )
}


@dataclass(frozen=True)
class GoalBox:
    """An open box objects are placed in (m, base frame)."""

    name: str
    center_xyz_m: tuple[float, float, float]  # middle of its floor's top face
    inner_size_m: tuple[float, float, float]  # inside length, width and wall height

    def summary(self) -> dict:
        return {
            "name": self.name,
            "center_xyz_m": [round(v, 4) for v in self.center_xyz_m],
            "inner_size_m": [round(v, 4) for v in self.inner_size_m],
        }

    def holds(self, center, half_extents_world) -> bool:
        """An object with this center and half-extents (along x, y, z) lies inside."""
        cx, cy, floor = self.center_xyz_m
        lx, ly, height = self.inner_size_m
        hx, hy, hz = (float(v) for v in half_extents_world)
        x, y, z = (float(v) for v in center)
        inside = abs(x - cx) <= lx / 2 - hx + 1e-3 and abs(y - cy) <= ly / 2 - hy + 1e-3
        resting = floor + hz - 0.01 <= z <= floor + hz + 0.01  # on the floor, not on a wall
        return inside and resting


_GOAL_FLOOR = float(sim_cli.PICK_PLACE_GOAL_CENTER[2]) + 0.004  # top of the box's floor plate
GOALS: dict[str, GoalBox] = {
    "pick_place_basic": GoalBox(
        name="goal_box",
        center_xyz_m=(
            float(sim_cli.PICK_PLACE_GOAL_CENTER[0]),
            float(sim_cli.PICK_PLACE_GOAL_CENTER[1]),
            round(_GOAL_FLOOR, 4),
        ),
        inner_size_m=tuple(float(2 * v) for v in sim_cli.PICK_PLACE_GOAL_INNER_HALF_EXTENTS),
    )
}


def model_scene(name: str) -> str:
    """The compiled scene a scene or layout name runs on."""
    layout = LAYOUTS.get(name)
    return layout.model_scene if layout is not None else name


def layouts_for(scene: str) -> list[Layout]:
    """Layouts of a compiled scene, easiest first (empty if it has only one arrangement)."""
    return [layout for layout in LAYOUTS.values() if layout.model_scene == scene]
