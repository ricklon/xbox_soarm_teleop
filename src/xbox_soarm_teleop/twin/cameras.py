"""Offscreen camera views of the twin for agents.

Views:

- ``wrist``: on the gripper, looking past the fixed jaw toward the tool point;
  moves with the gripper.
- ``overhead``: straight down, like a map: robot base at the bottom, forward
  (+x) up the image, the robot's left (+y) on the image's left.
- ``front``: from in front of the table looking back at the robot, so the
  robot's left (+y) appears on the image's right.
- ``side``: from the robot's left (+y side), forward (+x) to the image's left.

Rendering needs the OpenGL context, so it must run on the loop thread
(``TwinService.render`` queues requests there). On headless Linux set
``MUJOCO_GL=egl`` before importing MuJoCo.
"""

from __future__ import annotations

import io

import mujoco
import numpy as np
from PIL import Image

_LOOKAT = (0.25, 0.0, 0.14)
FIXED_VIEWS = {
    "overhead": {"lookat": _LOOKAT, "distance": 0.65, "azimuth": 0.0, "elevation": -89.0},
    "front": {"lookat": _LOOKAT, "distance": 0.75, "azimuth": 180.0, "elevation": -30.0},
    "side": {"lookat": _LOOKAT, "distance": 0.75, "azimuth": 90.0, "elevation": -20.0},
}
VIEWS = ("wrist", *FIXED_VIEWS)
DEFAULT_SIZE = (320, 240)
MAX_SIZE = (640, 480)  # MuJoCo's default offscreen buffer
FORMATS = {"jpeg": "JPEG", "png": "PNG"}


class CameraRig:
    """Renders named views of a model (create and use on the loop thread)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def render(self, view: str, width: int, height: int) -> np.ndarray:
        """Return an RGB image (height x width x 3, uint8)."""
        if view not in VIEWS:
            raise ValueError(f"Unknown view '{view}'. Known: {', '.join(VIEWS)}")
        if not (16 <= width <= MAX_SIZE[0] and 16 <= height <= MAX_SIZE[1]):
            raise ValueError(f"Image size must be within 16x16 .. {MAX_SIZE[0]}x{MAX_SIZE[1]}")
        renderer = self._renderers.get((width, height))
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height, width)
            self._renderers[(width, height)] = renderer
        if view == "wrist":
            renderer.update_scene(self.data, camera="wrist")
        else:
            preset = FIXED_VIEWS[view]
            camera = mujoco.MjvCamera()
            camera.lookat[:] = preset["lookat"]
            camera.distance = preset["distance"]
            camera.azimuth = preset["azimuth"]
            camera.elevation = preset["elevation"]
            renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()


def encode(pixels: np.ndarray, fmt: str = "jpeg", quality: int = 85) -> bytes:
    """Encode an RGB image as JPEG or PNG bytes."""
    if fmt not in FORMATS:
        raise ValueError(f"Unknown image format '{fmt}'. Known: {', '.join(FORMATS)}")
    buffer = io.BytesIO()
    options = {"quality": quality} if fmt == "jpeg" else {}
    Image.fromarray(pixels).save(buffer, FORMATS[fmt], **options)
    return buffer.getvalue()
