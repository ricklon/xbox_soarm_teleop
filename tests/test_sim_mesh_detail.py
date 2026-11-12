"""Simplified meshes: fewer triangles, the same robot."""

import logging
import warnings

import numpy as np
import pytest

from xbox_soarm_teleop.sim.mesh_detail import read_stl, simplify, write_simplified, write_stl

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)


def grid_sheet(n: int = 40, size: float = 0.04):
    """A flat n x n sheet of small triangles."""
    xs = np.linspace(0, size, n + 1)
    points = np.array([(x, y, 0.0) for y in xs for x in xs], dtype=np.float32)
    faces = []
    for row in range(n):
        for col in range(n):
            a = row * (n + 1) + col
            faces += [(a, a + 1, a + n + 2), (a, a + n + 2, a + n + 1)]
    return points, np.array(faces, dtype=np.int32)


def test_simplify_merges_corners_and_stays_within_a_cell():
    points, faces = grid_sheet()
    new_points, new_faces = simplify(points, faces, cell_m=0.004)
    assert 0 < len(new_faces) < len(faces) / 4
    assert new_faces.max() < len(new_points)
    assert np.all(new_points.min(axis=0) >= points.min(axis=0) - 1e-9)
    assert np.all(new_points.max(axis=0) <= points.max(axis=0) + 1e-9)
    assert np.all(points.max(axis=0) - new_points.max(axis=0) <= 0.004)
    corners = new_faces[:, [0, 1, 2]]
    assert not np.any((corners[:, 0] == corners[:, 1]) | (corners[:, 1] == corners[:, 2]))


def test_stl_round_trip(tmp_path):
    points, faces = grid_sheet(4)
    write_stl(tmp_path / "sheet.stl", points, faces)
    read_points, read_faces = read_stl(tmp_path / "sheet.stl")
    assert len(read_faces) == len(faces)
    np.testing.assert_allclose(
        np.sort(read_points[read_faces].reshape(-1, 9), axis=0),
        np.sort(points[faces].reshape(-1, 9), axis=0),
    )


def test_write_simplified_rejects_a_silly_cell(tmp_path):
    points, faces = grid_sheet(4)
    write_stl(tmp_path / "sheet.stl", points, faces)
    with pytest.raises(ValueError):
        write_simplified(tmp_path / "sheet.stl", tmp_path / "out.stl", cell_m=0.0)


def test_simplified_model_is_the_same_robot_with_fewer_triangles():
    from xbox_soarm_teleop.cli.simulate_mujoco import URDF_PATH, load_model_with_cameras

    full = load_model_with_cameras(str(URDF_PATH), scene="pick_place_basic")
    light = load_model_with_cameras(str(URDF_PATH), scene="pick_place_basic", mesh_cell_mm=2.0)
    assert light.nmeshface < full.nmeshface / 8
    for count in ("nq", "nv", "nbody", "ngeom", "nmesh", "ncam", "njnt"):
        assert getattr(light, count) == getattr(full, count), count
    np.testing.assert_allclose(light.body_pos, full.body_pos)
    np.testing.assert_allclose(light.jnt_range, full.jnt_range)
