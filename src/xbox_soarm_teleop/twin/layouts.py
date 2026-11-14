"""Scene layouts: which objects are on the table, switchable while a twin runs.

A compiled model holds every object its layouts use, so changing layout never
recompiles anything (the hub shares one model between all its rooms). A
layout names the objects that are on the table and where they start; the
twin keeps the rest parked out of sight and out of reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
from xbox_soarm_teleop.sim.table_scenes import TABLE_SCENES, Bin


@dataclass(frozen=True)
class Layout:
    """One arrangement of a model scene's objects.

    Args:
        name: What agents and the command line call it.
        model_scene: The compiled scene (``MuJoCoSimulator.scene``) it arranges.
        difficulty: ``"easy"`` or ``"hard"``.
        description: One sentence for an agent choosing a scene.
        objects: Start position (m, base frame) of each object on the table.
        target: The object a task in this scene is about, if there is one.
        listed: Offered by ``list_scenes`` (the workshop's scene picker). Hidden
            scenes still load by name.
        descriptions: How objects are described in this scene, where the
            usual words (``OBJECT_DESCRIPTIONS``) would be ambiguous.
    """

    name: str
    model_scene: str
    difficulty: str
    description: str
    objects: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    target: str | None = None
    listed: bool = True
    descriptions: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "difficulty": self.difficulty,
            "description": self.description,
            "objects": list(self.objects),
            "target": self.target,
        }


_START_XY = tuple(float(v) for v in sim_cli.PICK_PLACE_CUBE_START[:2])
_TABLE = float(sim_cli.PICK_PLACE_TABLE_HEIGHT)
_HALF = {name: half for name, half, _, _ in sim_cli.DISTRACTOR_OBJECTS}
_CUBE = tuple(float(v) for v in sim_cli.PICK_PLACE_CUBE_START)


def _on_table(name: str, x: float, y: float) -> tuple[float, float, float]:
    return (x, y, _TABLE + _HALF[name][2])


def _on_table_cube(x: float, y: float) -> tuple[float, float, float]:
    return (x, y, _CUBE[2])


def _distractors(name: str, description: str, others: dict, descriptions=None) -> "Layout":
    """A hidden scene: the red cube at its usual start, among ``others``."""
    return Layout(
        name=name,
        model_scene="pick_place_basic",
        difficulty="hard",
        description=description,
        objects={"pick_cube": _CUBE, **{n: _on_table(n, *xy) for n, xy in others.items()}},
        target="pick_cube",
        listed=False,
        descriptions=descriptions or {},
    )


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
            target="pick_cube",
        ),
        _distractors(
            "distractors_color",
            "The red cube beside a blue cube and a green cube: find it by colour.",
            # Clear of the arm's path from home, which brushes a 4 cm cube at (0.25, 0.04).
            {"blue_cube": (0.42, 0.04), "green_cube": (0.25, -0.07)},
        ),
        _distractors(
            "distractors_shape",
            "The red cube beside a red bar: find it by shape.",
            {"red_bar": (0.25, 0.04)},
        ),
        _distractors(
            "distractors_size",
            "The large red cube beside a small red cube: find it by size.",
            {"red_cube_small": (0.25, -0.06)},
            {"pick_cube": "large red cube"},
        ),
        _distractors(
            "distractors_clutter",
            "The large red cube among a small red cube, a red bar, a blue cube and a yellow cube.",
            {
                "red_cube_small": (0.25, -0.06),
                "red_bar": (0.23, 0.05),
                "blue_cube": (0.42, 0.04),
                "yellow_cube": (0.43, -0.07),
            },
            {"pick_cube": "large red cube"},
        ),
        # The sort table (sim/table_scenes.py): what goes where is up to the
        # request (twin/tasks.py), not the layout. Starts are near the base,
        # well behind the bins and clear of the arm's path from home.
        Layout(
            name="sort_colors",
            model_scene="sort_bins",
            difficulty="easy",
            description="A red, a blue and a green cube, and a red, a blue and a green bin.",
            objects={
                "pick_cube": _on_table_cube(0.22, -0.06),
                "blue_cube": _on_table("blue_cube", 0.28, -0.08),
                "green_cube": _on_table("green_cube", 0.21, 0.01),
            },
        ),
        Layout(
            name="sort_colors_decoy",
            model_scene="sort_bins",
            difficulty="hard",
            description=(
                "A red, a blue, a green and a yellow cube, and a red, a blue and a green "
                "bin. There is no yellow bin."
            ),
            objects={
                "pick_cube": _on_table_cube(0.22, -0.06),
                "blue_cube": _on_table("blue_cube", 0.28, -0.08),
                "green_cube": _on_table("green_cube", 0.21, 0.01),
                "yellow_cube": _on_table("yellow_cube", 0.30, -0.02),
            },
        ),
    )
}
DISTRACTOR_LAYOUTS = tuple(name for name in LAYOUTS if name.startswith("distractors_"))


@dataclass(frozen=True)
class GoalBox:
    """An open box objects are placed in (m, base frame): the goal box, or a bin."""

    name: str
    center_xyz_m: tuple[float, float, float]  # middle of its floor's top face
    inner_size_m: tuple[float, float, float]  # inside length, width and wall height
    label: str = ""

    def summary(self) -> dict:
        summary = {
            "name": self.name,
            "center_xyz_m": [round(v, 4) for v in self.center_xyz_m],
            "inner_size_m": [round(v, 4) for v in self.inner_size_m],
        }
        if self.label:
            summary["label"] = self.label
        return summary

    def holds(self, center, half_extents_world) -> bool:
        """An object with this center and half-extents (along x, y, z) lies inside."""
        cx, cy, floor = self.center_xyz_m
        lx, ly, height = self.inner_size_m
        hx, hy, hz = (float(v) for v in half_extents_world)
        x, y, z = (float(v) for v in center)
        inside = abs(x - cx) <= lx / 2 - hx + 1e-3 and abs(y - cy) <= ly / 2 - hy + 1e-3
        resting = floor + hz - 0.01 <= z <= floor + hz + 0.01  # on the floor, not on a wall
        return inside and resting


def _box(bin_: Bin) -> GoalBox:
    return GoalBox(
        name=bin_.name,
        center_xyz_m=(*(float(v) for v in bin_.center_xy_m), round(bin_.floor_top_m, 4)),
        inner_size_m=tuple(float(2 * v) for v in bin_.inner_half_m),
        label=bin_.label,
    )


# Every table scene's bins; a scene with exactly one is a goal-box scene, whose
# box is also in GOALS (and in observe as scene.goal, and objects' in_goal).
BINS: dict[str, tuple[GoalBox, ...]] = {
    name: tuple(_box(bin_) for bin_ in scene.bins) for name, scene in TABLE_SCENES.items()
}
GOALS: dict[str, GoalBox] = {
    name: replace(boxes[0], label="") for name, boxes in BINS.items() if len(boxes) == 1
}


# What each object looks like, in words a pointing model understands. Molmo2-ER
# takes the category literally: shown the bean bag, it found a "blue bean bag"
# in 1 of 9 pictures and a "blue object" in 9 of 9 (it is the only blue thing).
OBJECT_DESCRIPTIONS: dict[str, str] = {
    "pick_cube": "red cube",
    sim_cli.BEAN_BAG_NAME: "blue object",
    "red_cube_small": "small red cube",
    "blue_cube": "blue cube",
    "red_bar": "red bar",
    "green_cube": "green cube",
    "yellow_cube": "yellow cube",
}


def object_description(name: str, layout: "Layout | None" = None) -> str:
    """An object's description for a pointing model ("red cube"), as ``layout`` words it."""
    if layout is not None and name in layout.descriptions:
        return layout.descriptions[name]
    return OBJECT_DESCRIPTIONS.get(name, name.replace("_", " "))


def model_scene(name: str) -> str:
    """The compiled scene a scene or layout name runs on."""
    layout = LAYOUTS.get(name)
    return layout.model_scene if layout is not None else name


def layouts_for(scene: str, *, include_hidden: bool = False) -> list[Layout]:
    """Layouts of a compiled scene, easiest first (empty if it has only one arrangement).

    Hidden layouts (the distractor scenes) are left out unless asked for.
    """
    return [
        layout
        for layout in LAYOUTS.values()
        if layout.model_scene == scene and (layout.listed or include_hidden)
    ]
