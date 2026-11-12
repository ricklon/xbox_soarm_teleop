"""Simplified copies of the robot's meshes, for drawing.

The SO-ARM101 meshes are CAD exports with about 320,000 triangles. Drawing
them is what limits how many camera pictures a hub can serve, and none of
that detail is needed to recognise the arm. Simplifying merges every corner
that falls in the same small cube into one (vertex clustering). It works on
any mesh, including these, which have loose internal pieces that made
edge-collapse simplifiers stall and push corners outward.

Corners move by less than the cube's size, so each part keeps its outline.
The simulator can keep the original meshes for contact while a camera rig
draws the simplified ones (``CameraRig(state_model=...)``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_STL_RECORD = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attr", "<u2")])


def read_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary STL as (points, faces), with shared corners merged."""
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path} is not a binary STL")
    count = int(np.frombuffer(raw, "<u4", 1, 80)[0])
    if len(raw) < 84 + count * _STL_RECORD.itemsize:
        raise ValueError(f"{path} is not a binary STL")
    corners = np.frombuffer(raw, _STL_RECORD, count, 84)["corners"].reshape(-1, 3)
    points, index = np.unique(corners, axis=0, return_inverse=True)
    return points.astype(np.float32), index.reshape(-1, 3).astype(np.int32)


def write_stl(path: str | Path, points: np.ndarray, faces: np.ndarray) -> None:
    """Write (points, faces) as a binary STL."""
    records = np.zeros(len(faces), _STL_RECORD)
    corners = np.asarray(points, dtype=np.float32)[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    records["normal"] = normal / np.where(length > 0, length, 1)
    records["corners"] = corners
    with Path(path).open("wb") as file:
        file.write(b"xbox_soarm_teleop simplified mesh".ljust(80, b" "))
        file.write(np.uint32(len(faces)).tobytes())
        file.write(records.tobytes())


def simplify(points: np.ndarray, faces: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge corners that share a cube of side ``cell_m``; drop collapsed faces.

    Returns:
        (points, faces): each merged corner sits at the mean of the corners it
        replaces, so it stays inside the original mesh's bounding box.
    """
    if not cell_m > 0:
        raise ValueError("cell_m must be greater than 0")
    points = np.asarray(points, dtype=np.float64)
    _, group = np.unique(np.floor(points / cell_m).astype(np.int64), axis=0, return_inverse=True)
    group = group.ravel()
    merged = np.zeros((int(group.max()) + 1, 3))
    np.add.at(merged, group, points)
    merged /= np.bincount(group)[:, None]
    new_faces = group[np.asarray(faces)]
    collapsed = (
        (new_faces[:, 0] == new_faces[:, 1])
        | (new_faces[:, 1] == new_faces[:, 2])
        | (new_faces[:, 0] == new_faces[:, 2])
    )
    new_faces = new_faces[~collapsed]
    _, first = np.unique(np.sort(new_faces, axis=1), axis=0, return_index=True)
    return merged.astype(np.float32), new_faces[np.sort(first)].astype(np.int32)


def write_simplified(source: str | Path, target: str | Path, cell_m: float) -> tuple[int, int]:
    """Write a simplified copy of an STL; returns (faces before, faces after)."""
    if not cell_m > 0:
        raise ValueError("cell_m must be greater than 0")
    points, faces = read_stl(source)
    new_points, new_faces = simplify(points, faces, cell_m)
    write_stl(target, new_points, new_faces)
    return len(faces), len(new_faces)
