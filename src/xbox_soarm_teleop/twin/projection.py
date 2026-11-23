"""From a pixel in a twin camera image to a point on the table, and back.

A pointing model answers "where is the red cube?" with a pixel. The camera
that took the picture is known exactly (its pose and field of view), so the
pixel fixes a ray in space; where that ray meets the plane the object rests
on is its position. On the real arm the same holds for a calibrated camera.

The math follows MuJoCo's own cameras (``mjv_updateCamera``): a free camera
sits ``distance`` back from ``lookat`` along ``azimuth`` / ``elevation`` and
uses the model's global ``fovy``; a camera in the model (``wrist``) uses its
body pose and its own ``fovy``. ``fovy`` is the vertical field of view; the
horizontal one follows from the image's aspect ratio. Pixels count from the
image's top-left corner; ``(u + 0.5, v + 0.5)`` is a pixel's center.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from xbox_soarm_teleop.twin.cameras import FIXED_VIEWS, VIEWS, view_camera_id


@dataclass(frozen=True)
class CameraGeometry:
    """Where a camera is and how it sees, for one image size.

    Attributes:
        view: The view's name.
        position: Camera center (m, base frame).
        right: Unit vector toward the image's right edge.
        up: Unit vector toward the image's top edge.
        forward: Unit vector the camera looks along.
        fovy_deg: Vertical field of view (degrees).
        width: Image width (pixels).
        height: Image height (pixels).
    """

    view: str
    position: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    fovy_deg: float
    width: int
    height: int

    @property
    def _half_height(self) -> float:
        return math.tan(math.radians(self.fovy_deg) / 2.0)

    @property
    def _half_width(self) -> float:
        return self._half_height * self.width / self.height

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """The ray through image point (u, v): its origin and unit direction.

        ``u`` runs right and ``v`` down from the top-left corner, in pixels;
        a pixel's center is at ``(column + 0.5, row + 0.5)``.
        """
        x = (2.0 * u / self.width - 1.0) * self._half_width
        y = (1.0 - 2.0 * v / self.height) * self._half_height
        direction = self.forward + x * self.right + y * self.up
        return self.position.copy(), direction / np.linalg.norm(direction)

    def project(self, xyz) -> tuple[float, float] | None:
        """Image point (u, v) of a world point; None if it is behind the camera."""
        offset = np.asarray(xyz, dtype=float) - self.position
        depth = float(offset @ self.forward)
        if depth <= 1e-9:
            return None
        x = float(offset @ self.right) / depth / self._half_width
        y = float(offset @ self.up) / depth / self._half_height
        return (x + 1.0) * self.width / 2.0, (1.0 - y) * self.height / 2.0

    def to_dict(self) -> dict:
        return {
            "view": self.view,
            "position_m": [round(float(v), 5) for v in self.position],
            "forward": [round(float(v), 5) for v in self.forward],
            "up": [round(float(v), 5) for v in self.up],
            "fovy_deg": round(self.fovy_deg, 3),
            "size": [self.width, self.height],
        }


def camera_geometry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    view: str,
    width: int,
    height: int,
    *,
    scene: str | None = None,
) -> CameraGeometry:
    """The geometry of a named view (``twin.cameras``) as it is drawn now.

    ``data`` must have current kinematics (``mj_kinematics`` or a step ran)
    for the views that move with the arm. ``scene`` is the table scene
    ``data`` is in, for a model with several (``twin.cameras.view_camera_id``).
    """
    if view not in VIEWS:
        raise ValueError(f"Unknown view '{view}'. Known: {', '.join(VIEWS)}")
    own = view_camera_id(model, view, scene)
    if view in FIXED_VIEWS and own < 0:  # a shared preset, not the scene's own camera
        preset = FIXED_VIEWS[view]
        az = math.radians(preset["azimuth"])
        el = math.radians(preset["elevation"])
        forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])
        position = np.asarray(preset["lookat"], dtype=float) - preset["distance"] * forward
        fovy = float(model.vis.global_.fovy)
    else:
        cam = own
        frame = data.cam_xmat[cam].reshape(3, 3)
        position = data.cam_xpos[cam].copy()
        up = frame[:, 1].copy()
        forward = -frame[:, 2]
        fovy = float(model.cam_fovy[cam])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return CameraGeometry(
        view=view,
        position=position,
        right=right,
        up=up / np.linalg.norm(up),
        forward=forward / np.linalg.norm(forward),
        fovy_deg=fovy,
        width=int(width),
        height=int(height),
    )


def ray_to_plane(origin: np.ndarray, direction: np.ndarray, z: float) -> np.ndarray | None:
    """Where a ray meets the horizontal plane at height ``z``; None if it never does."""
    if abs(direction[2]) < 1e-9:
        return None
    distance = (z - origin[2]) / direction[2]
    if distance <= 0:
        return None
    return origin + distance * direction


def pixel_to_xyz(geometry: CameraGeometry, u: float, v: float, z: float) -> np.ndarray | None:
    """The point at height ``z`` (m) that image point (u, v) shows; None if none."""
    origin, direction = geometry.ray(u, v)
    return ray_to_plane(origin, direction, z)
