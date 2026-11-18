"""The printable files match the simulation's objects.

assets/printables/ is built by scripts/export_printables.py with SolidCraft.
These checks need no SolidCraft: they read the files and compare them with
the sizes the simulation uses, so a resized object fails here until the files
are rebuilt.
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from xbox_soarm_teleop.sim.printables import parts, read_stl, solidcraft_script

FOLDER = Path(__file__).resolve().parents[1] / "assets" / "printables"
PARTS = {part.name: part for part in parts()}


def test_every_object_a_challenge_can_use_has_a_part():
    from xbox_soarm_teleop.twin.challenges import PERSON_NAMES

    rigid = {person for internal, person in PERSON_NAMES.items() if internal != "bean_bag"}
    assert rigid <= set(PARTS)
    assert {"red_bin", "blue_bin", "green_bin", "goal_box"} <= set(PARTS)


@pytest.mark.parametrize("name", sorted(PARTS))
def test_the_stl_is_a_closed_solid_of_the_right_size(name):
    part = PARTS[name]
    tris = read_stl((FOLDER / "stl" / f"{name}.stl").read_bytes())
    corners = tris.reshape(-1, 3)
    size = corners.max(axis=0) - corners.min(axis=0)
    np.testing.assert_allclose(size, part.outer_mm, atol=1e-3)
    assert corners[:, 2].min() == pytest.approx(0.0, abs=1e-3)  # on its base
    np.testing.assert_allclose(corners[:, :2].max(axis=0) + corners[:, :2].min(axis=0), 0.0, atol=1e-3)
    volume = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0
    assert volume == pytest.approx(part.volume_mm3(), rel=1e-4)  # outward normals, and the pocket
    keys = np.round(tris, 4)
    edges = Counter()
    for tri in keys:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges[tuple(sorted((tuple(tri[a]), tuple(tri[b]))))] += 1
    # Watertight: every edge is shared by exactly two triangles.
    assert set(edges.values()) == {2}, Counter(edges.values())


@pytest.mark.parametrize("name", sorted(PARTS))
def test_the_step_is_the_part_in_millimetres(name):
    part = PARTS[name]
    text = (FOLDER / "step" / f"{name}.step").read_text()
    assert text.startswith("ISO-10303-21;")
    assert "SI_UNIT(.MILLI.,.METRE.)" in text
    assert text.count("MANIFOLD_SOLID_BREP") == 1
    # The solid's corners: the points its VERTEX_POINTs name (planes are anchored
    # at points that may lie outside it, and curves on faces have 2D points).
    point_at = {
        ident: [float(v) for v in xyz.split(",")]
        for ident, xyz in re.findall(r"#(\d+)=CARTESIAN_POINT\('[^']*',\(([^)]*)\)\)", text)
    }
    points = np.array([point_at[ref] for ref in re.findall(r"VERTEX_POINT\('[^']*',#(\d+)\)", text)])
    assert len(points) == (16 if part.pocket_mm else 8)
    np.testing.assert_allclose(points.max(axis=0) - points.min(axis=0), part.outer_mm, atol=1e-3)
    faces = text.count("ADVANCED_FACE")
    assert faces == (11 if part.pocket_mm else 6)


def test_the_script_cuts_a_pocket_only_for_an_open_box():
    for part in PARTS.values():
        ops = [c["params"].get("operation") for c in solidcraft_script(part)["commands"]]
        assert ops.count("cut") == (1 if part.pocket_mm else 0)


def test_the_guide_lists_every_part():
    guide = (FOLDER / "README.md").read_text()
    for name, part in PARTS.items():
        size = " x ".join(f"{v:g}" for v in part.outer_mm)
        assert f"| `{name}` | {size} |" in guide, name
