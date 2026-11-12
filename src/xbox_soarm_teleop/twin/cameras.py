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
    """Renders named views of a model (create and use on the loop thread).

    Args:
        model: The model to draw.
        data: The state drawn when ``render`` is not given one.
        reflections: Draw floor reflections.
        shadows: Draw shadows. Reflections and shadows each cost about 5 ms
            of a 14 ms frame, whatever the image size, so a rig that serves
            many viewers turns them off.
        state_model: The model the simulators run, when it is not the one
            drawn. It must have the same joints, bodies and geoms as
            ``model``, which may differ in mesh detail
            (``load_model_with_cameras(mesh_cell_mm=...)``). The rig then
            copies each state's joint positions into its own and works out
            where everything is, so contact can use the original meshes
            while pictures use lighter ones.

    Drawing has two steps so that a host can split them between threads:
    :meth:`aim` reads the simulator state (no OpenGL; call it where the state
    is safe to read), and :meth:`capture` draws (call it on the thread that
    created the renderer with :meth:`renderer`).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        reflections: bool = True,
        shadows: bool = True,
        state_model: mujoco.MjModel | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.reflections = reflections
        self.shadows = shadows
        self._mirror: mujoco.MjData | None = None
        if state_model is not None and state_model is not model:
            for count in ("nq", "nbody", "ngeom", "ncam", "nlight", "nmocap"):
                if getattr(state_model, count) != getattr(model, count):
                    raise ValueError(
                        f"state_model and model differ in {count}: "
                        f"{getattr(state_model, count)} != {getattr(model, count)}"
                    )
            self._mirror = mujoco.MjData(model)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def has_renderer(self, width: int, height: int) -> bool:
        return (width, height) in self._renderers

    def renderer(self, width: int, height: int) -> mujoco.Renderer:
        """The renderer for an image size, created on first use (OpenGL)."""
        if not (16 <= width <= MAX_SIZE[0] and 16 <= height <= MAX_SIZE[1]):
            raise ValueError(f"Image size must be within 16x16 .. {MAX_SIZE[0]}x{MAX_SIZE[1]}")
        renderer = self._renderers.get((width, height))
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height, width)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(self.reflections)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(self.shadows)
            self._renderers[(width, height)] = renderer
        return renderer

    def aim(
        self, view: str, width: int, height: int, data: mujoco.MjData | None = None
    ) -> mujoco.Renderer:
        """Load a view of ``data`` into the renderer for this size (no OpenGL)."""
        if view not in VIEWS:
            raise ValueError(f"Unknown view '{view}'. Known: {', '.join(VIEWS)}")
        renderer = self.renderer(width, height)
        camera = mujoco.MjvCamera()
        point_camera(camera, self.model, view)
        renderer.update_scene(self._drawable(self.data if data is None else data), camera=camera)
        return renderer

    def _drawable(self, data: mujoco.MjData) -> mujoco.MjData:
        """``data`` itself, or its pose copied into this rig's own model."""
        if self._mirror is None:
            return data
        mirror = self._mirror
        mirror.qpos[:] = data.qpos
        mirror.mocap_pos[:] = data.mocap_pos
        mirror.mocap_quat[:] = data.mocap_quat
        mirror.time = data.time
        mujoco.mj_kinematics(self.model, mirror)
        mujoco.mj_camlight(self.model, mirror)
        return mirror

    @staticmethod
    def capture(renderer: mujoco.Renderer) -> np.ndarray:
        """Draw what :meth:`aim` loaded; returns an RGB image."""
        return renderer.render()

    def render(
        self, view: str, width: int, height: int, data: mujoco.MjData | None = None
    ) -> np.ndarray:
        """Return an RGB image (height x width x 3, uint8).

        Args:
            view: One of ``VIEWS``.
            width: Image width in pixels.
            height: Image height in pixels.
            data: State to draw; any ``MjData`` of this rig's model (or of its
                ``state_model``), so one rig can serve several simulators
                that share the model.
        """
        return self.capture(self.aim(view, width, height, data))

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
