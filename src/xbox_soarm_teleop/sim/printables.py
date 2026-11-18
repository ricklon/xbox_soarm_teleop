"""Printable models of the twin's objects and trays, sized from the simulator.

Every size comes from the constants the simulation itself uses
(``cli/simulate_mujoco.py``, ``sim/table_scenes.py``), so a printed cube is the
cube the arm practised on. ``scripts/export_printables.py`` builds each part
in SolidCraft (``solidcraft-cli``) from :func:`solidcraft_script` and writes a
STEP (for CAD), an STL (for a slicer) and the SolidCraft design.

Units are millimetres. Each part sits on z = 0, centred on x and y.

One difference from the simulation, for printing: a tray's walls stand on its
floor plate's underside. In the simulation they start at the middle of the
floor plate, 4 mm above the table, a gap no camera sees and nothing falls into.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
from xbox_soarm_teleop.sim.table_scenes import (
    BIN_FLOOR_M,
    BLUE_BIN,
    GOAL_BOX,
    GREEN_BIN,
    ORANGE_BOX,
    RED_BIN,
    Bin,
)
from xbox_soarm_teleop.twin.naming import APPEARANCE

PLA_DENSITY_G_PER_CM3 = 1.24
CUBE_RGBA = (0.85, 0.25, 0.20, 1.0)  # simulate_mujoco: the red cube


@dataclass(frozen=True)
class Part:
    """A printable part (mm).

    Args:
        name: File name, the name a person uses (``red_cube``).
        outer_mm: Outside length, width and height.
        pocket_mm: For an open box: inside length, width and depth (from the
            top). None for a solid block.
        rgba: Its colour in the simulation (0..1).
        sim_mass_g: Its mass in the simulation (0: fixed furniture).
        note: What it is for.
    """

    name: str
    outer_mm: tuple[float, float, float]
    pocket_mm: tuple[float, float, float] | None
    rgba: tuple[float, ...]
    sim_mass_g: float
    note: str = ""

    def volume_mm3(self) -> float:
        x, y, z = self.outer_mm
        volume = x * y * z
        if self.pocket_mm is not None:
            px, py, pz = self.pocket_mm
            volume -= px * py * pz
        return volume

    def solid_pla_g(self) -> float:
        """Its mass printed solid in PLA (g): infill brings it down toward ``sim_mass_g``."""
        return self.volume_mm3() / 1000.0 * PLA_DENSITY_G_PER_CM3

    def colour_hex(self) -> str:
        return "#" + "".join(f"{round(255 * c):02X}" for c in self.rgba[:3])


def _mm(values) -> tuple[float, ...]:
    return tuple(round(float(v) * 1000.0, 3) for v in values)


def _person_name(name: str) -> str:
    colour, shape = APPEARANCE.get(name, (None, name))
    return f"{colour}_{shape}" if colour else name


def _block(name: str, half_m, rgba, mass_kg: float, note: str) -> Part:
    return Part(
        _person_name(name), _mm(2.0 * np.asarray(half_m)), None, tuple(rgba), mass_kg * 1000.0, note
    )


def _open_box(box: Bin, note: str) -> Part:
    hx, hy, hz = box.inner_half_m
    w = box.wall_m
    outer = _mm((2 * (hx + w), 2 * (hy + w), BIN_FLOOR_M + 2 * hz))
    pocket = _mm((2 * hx, 2 * hy, 2 * hz - BIN_FLOOR_M))
    mass = box.mass_kg * 1000.0 if box.movable else 0.0
    return Part(box.name, outer, pocket, tuple(box.rgba[:3]) + (1.0,), mass, note)


_NOTES = {
    "pick_cube": "The red cube every scene starts from.",
    "red_cube_small": "Beside the red cube: 'the small one'.",
    "red_bar": "A red bar: grip it across its 3 cm side.",
    "pink_slab": "Thin: the jaws close nearly shut on it.",
    "teal_tower": "Tall and narrow: the arm's own pick knocks it over about half the time.",
    "brown_block": "Wide: the jaws open nearly as far as they approach.",
}


def parts() -> list[Part]:
    """Every rigid object a scene can put out, the movable box, the trays and the goal box.

    The bean bag is left out: it is soft (``assets/printables/README.md`` says how to sew one).
    """
    blocks = [
        _block(
            "pick_cube",
            (sim_cli.PICK_PLACE_CUBE_HALF_EXTENT,) * 3,
            CUBE_RGBA,
            sim_cli.PICK_PLACE_CUBE_MASS,
            _NOTES["pick_cube"],
        )
    ]
    blocks += [
        _block(name, half, rgba, mass, _NOTES.get(name, "A cube for sorting by colour."))
        for name, half, rgba, mass in sim_cli.DISTRACTOR_OBJECTS
    ]
    boxes = [
        _open_box(ORANGE_BOX, "Pushed about by the arm; what is in it rides along."),
        _open_box(RED_BIN, "Sorting desk tray, fixed to the desk."),
        _open_box(BLUE_BIN, "Sorting desk tray, fixed to the desk."),
        _open_box(GREEN_BIN, "Sorting desk tray, fixed to the desk."),
        _open_box(GOAL_BOX, "The pick-and-place table's goal box, fixed to the table."),
    ]
    return blocks + boxes


def solidcraft_script(part: Part, at_mm: tuple[float, float] = (0.0, 0.0)) -> dict:
    """A ``solidcraft-cli run`` script that builds the part as one named,
    coloured body, centred on ``at_mm`` (x, y)."""
    x, y, z = part.outer_mm
    ax, ay = at_mm
    commands = [
        {
            "command": "solid.box",
            "params": {
                "length": f"{x:g}",
                "width": f"{y:g}",
                "height": f"{z:g}",
                "center": [ax, ay, z / 2.0],
                "operation": "new_body",
                "body_name": part.name,
            },
        }
    ]
    if part.pocket_mm is not None:
        px, py, depth = part.pocket_mm
        overshoot = 1.0  # past the top, so the cut leaves no sliver of a face there
        commands.append(
            {
                "command": "solid.box",
                "params": {
                    "length": f"{px:g}",
                    "width": f"{py:g}",
                    "height": f"{depth + overshoot:g}",
                    "center": [ax, ay, z - depth + (depth + overshoot) / 2.0],
                    "operation": "cut",
                    "targets": [part.name],
                },
            }
        )
    commands.append(
        {"command": "appearance.assign", "params": {"bodies": [part.name], "color": part.colour_hex()}}
    )
    return {"commands": commands}


def preview_script(gap_mm: float = 20.0) -> dict:
    """Every part side by side (objects in front, boxes behind), for a picture."""
    commands: list[dict] = []
    rows = [
        [p for p in parts() if p.pocket_mm is None],
        [p for p in parts() if p.pocket_mm is not None],
    ]
    y = 0.0
    for row in rows:
        depth = max(p.outer_mm[1] for p in row)
        x = 0.0
        for part in row:
            commands += solidcraft_script(part, (x + part.outer_mm[0] / 2.0, y + depth / 2.0))[
                "commands"
            ]
            x += part.outer_mm[0] + gap_mm
        y += depth + gap_mm
    return {"commands": commands}


def read_stl(data: bytes) -> np.ndarray:
    """Triangles (m x 3 x 3) from a binary STL."""
    (count,) = struct.unpack_from("<I", data, 80)
    rows = np.frombuffer(data, dtype=np.uint8, count=50 * count, offset=84).reshape(count, 50)
    corners = rows[:, 12:48].copy().view("<f4").reshape(count, 3, 3)
    return corners.astype(float)
