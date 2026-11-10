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
# What the viewer window can show: the same views, plus the mouse-driven camera.
WINDOW_VIEWS = ("free", *VIEWS)
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
        camera = mujoco.MjvCamera()
        point_camera(camera, self.model, view)
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()


def point_camera(camera: mujoco.MjvCamera, model: mujoco.MjModel, view: str) -> None:
    """Point a camera (offscreen or the viewer window's) at a named view.

    ``free`` hands the camera back to the mouse and leaves it where it is.
    """
    if view not in WINDOW_VIEWS:
        raise ValueError(f"Unknown view '{view}'. Known: {', '.join(WINDOW_VIEWS)}")
    if view == "wrist":
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        return
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    if view != "free":
        preset = FIXED_VIEWS[view]
        camera.lookat[:] = preset["lookat"]
        camera.distance = preset["distance"]
        camera.azimuth = preset["azimuth"]
        camera.elevation = preset["elevation"]


def next_view(current: str, dpad_x: float, previous_dpad_x: float) -> str:
    """The window view after a D-pad reading: right = next, left = previous.

    Only a fresh press switches, so holding the D-pad does not spin through
    the views.
    """
    if abs(dpad_x) < 0.5 or abs(previous_dpad_x) >= 0.5:
        return current
    step = 1 if dpad_x > 0 else -1
    return WINDOW_VIEWS[(WINDOW_VIEWS.index(current) + step) % len(WINDOW_VIEWS)]


def encode(pixels: np.ndarray, fmt: str = "jpeg", quality: int = 85) -> bytes:
    """Encode an RGB image as JPEG or PNG bytes."""
    if fmt not in FORMATS:
        raise ValueError(f"Unknown image format '{fmt}'. Known: {', '.join(FORMATS)}")
    buffer = io.BytesIO()
    options = {"quality": quality} if fmt == "jpeg" else {}
    Image.fromarray(pixels).save(buffer, FORMATS[fmt], **options)
    return buffer.getvalue()
