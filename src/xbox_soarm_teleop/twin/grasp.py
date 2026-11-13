"""Assisted grasp: closing the gripper to fit a nearby object attaches it.

Contact grasping with a kinematically posed arm is unreliable, so the
workshop twin uses an assisted grasp: when the jaws close down to the width of
an object whose center is within a radius of the tool point, the object is
carried rigidly with the tool until the jaws open wider than it again.

How far the jaws close matters, as it does on the real arm:

- open wider than the object: nothing is held;
- closed to its width, or a little past (``squeeze_m``): held;
- closed further on a rigid object: it is squeezed out and falls.

A rigid object must also sit square between the jaws (``align_tol_deg``).
A soft object (one the arm does not collide with, like the bean bag) takes
any closure and any angle.

Only the act of closing grabs: jaws already closed past an object's width
push it without picking it up, and re-arm once they open wider than it.

Objects a scene layout leaves out are parked: held far above the scene, where
no camera looks and nothing can reach them, and left out of every report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

# Parked objects are held here (m), one metre apart, clear of every camera.
PARK_XYZ = np.array([0.0, 0.0, 50.0])
_UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class GraspSettings:
    """Numbers the grasp runs on. PLACEHOLDERS until measured on the real arm.

    The jaw opening is taken from the twin's model, whose gripper travel
    (-10 to 100 degrees) is not the real arm's (-2 to 127 degrees measured),
    so the same gripper value does not yet mean the same opening on both.

    Args:
        jaw_gap_closed_m: Opening between the jaws at the tool point with the
            gripper at 1.0. From the model.
        jaw_gap_per_unit_m: Opening gained per unit the gripper opens. From
            the model, fitted over 0.7..1.0. Measure: opening at two readings.
        touch_margin_m: The jaws count as touching this far before the
            object's width. Measure: reading at first touch.
        release_margin_m: Opening this far past the width lets go.
        rigid_squeeze_m: How far past its width the jaws may close on a rigid
            object before it pops out. Measure: reading at slip, TPU jaw on.
        rigid_align_tol_deg: How far off square the jaws may be to a rigid
            object's faces. Measure: largest wrist-roll error that still holds.
        radius_m: Maximum tool-to-object-center distance for attaching.
    """

    jaw_gap_closed_m: float = 0.0024
    jaw_gap_per_unit_m: float = 0.147
    touch_margin_m: float = 0.003
    release_margin_m: float = 0.005
    rigid_squeeze_m: float = 0.006
    rigid_align_tol_deg: float = 25.0
    radius_m: float = 0.045

    def jaw_opening_m(self, gripper: float) -> float:
        """Opening between the jaws (m) at a gripper position (0 open .. 1 closed)."""
        return self.jaw_gap_closed_m + (1.0 - float(gripper)) * self.jaw_gap_per_unit_m

    def gripper_for_opening(self, opening_m: float) -> float:
        """Gripper position that leaves this opening, kept within 0..1."""
        position = 1.0 - (float(opening_m) - self.jaw_gap_closed_m) / self.jaw_gap_per_unit_m
        return float(np.clip(position, 0.0, 1.0))


@dataclass
class _FreeObject:
    body_id: int
    qpos_adr: int
    qvel_adr: int
    size_m: np.ndarray  # full extents along the object's own axes
    soft: bool


class AssistedGrasp:
    """Attach the nearest free object when the jaws close to fit it.

    Args:
        model: MuJoCo model; every body with a free joint is graspable. Its
            size is read from its first geom, and it is soft if the robot
            does not collide with that geom.
        data: MuJoCo data.
        settings: The numbers to run on (:class:`GraspSettings`).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        settings: GraspSettings | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.settings = settings or GraspSettings()
        self.radius_m = self.settings.radius_m
        self._gripper_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
        self._jaw_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1_link")
        self.objects: dict[str, _FreeObject] = {}
        for body_id in range(model.nbody):
            jnt = model.body_jntadr[body_id]
            if jnt >= 0 and model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"
                geom = int(model.body_geomadr[body_id])
                self.objects[name] = _FreeObject(
                    body_id,
                    int(model.jnt_qposadr[jnt]),
                    int(model.jnt_dofadr[jnt]),
                    size_m=2.0 * model.geom_size[geom].copy(),
                    soft=not (int(model.geom_contype[geom]) & 1),
                )
        self.parked: tuple[str, ...] = ()
        self.attached: str | None = None
        self._armed = True
        self._offset = np.zeros(3)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._held_width = 0.0

    def position(self, name: str) -> np.ndarray:
        """Current center of an object (m, world frame)."""
        obj = self.objects[name]
        return self.data.qpos[obj.qpos_adr : obj.qpos_adr + 3].copy()

    def positions(self) -> dict[str, np.ndarray]:
        """Centers of the objects in the scene (parked ones are left out)."""
        return {name: self.position(name) for name in self.objects if name not in self.parked}

    def place(self, name: str, xyz: np.ndarray) -> None:
        """Put an object upright and at rest with its center at ``xyz``."""
        obj = self.objects[name]
        self.data.qpos[obj.qpos_adr : obj.qpos_adr + 3] = xyz
        self.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7] = _UPRIGHT
        self.data.qvel[obj.qvel_adr : obj.qvel_adr + 6] = 0.0

    def park(self, names) -> None:
        """Take these objects out of the scene; all others are in it."""
        names = set(names)
        self.parked = tuple(name for name in self.objects if name in names)
        if self.attached in self.parked:
            self.attached = None
        self.hold_parked()

    def hold_parked(self) -> None:
        """Keep parked objects in place (call before each physics step)."""
        for index, name in enumerate(self.parked):
            self.place(name, PARK_XYZ + np.array([float(index), 0.0, 0.0]))

    def size(self, name: str) -> np.ndarray:
        """Full extents of an object along its own axes (m)."""
        return self.objects[name].size_m.copy()

    def closing_axis(self) -> np.ndarray | None:
        """Direction the jaws close along (world frame), or None without a gripper."""
        if self._gripper_body < 0 or self._jaw_body < 0:
            return None
        hinge = self.data.xmat[self._jaw_body].reshape(3, 3)[:, 2]
        along = -self.data.xmat[self._gripper_body].reshape(3, 3)[:, 2]
        axis = np.cross(hinge, along)
        norm = float(np.linalg.norm(axis))
        return axis / norm if norm > 1e-9 else None

    def fit(self, name: str) -> tuple[float, float]:
        """How an object sits between the jaws now: (width in m, degrees off square).

        The width is the object's extent along whichever of its axes the jaws
        most nearly close along.
        """
        obj = self.objects[name]
        axis = self.closing_axis()
        if axis is None:
            return float(obj.size_m.min()), 0.0
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7])
        along = np.abs(rotation.reshape(3, 3).T @ axis)
        face = int(np.argmax(along))
        off_square = math.degrees(math.acos(min(1.0, float(along[face]))))
        return float(obj.size_m[face]), off_square

    def hold_limit(self) -> float | None:
        """Gripper position at which jaws driven shut stop on a held rigid object.

        A person closing by button or trigger cannot pick a closure to the
        millimetre, so their jaws stop here; None when nothing rigid is held.
        """
        if self.attached is None or self.objects[self.attached].soft:
            return None
        cfg = self.settings
        return cfg.gripper_for_opening(self._held_width - cfg.rigid_squeeze_m / 2.0)

    def update(self, tool_pos: np.ndarray, gripper: float) -> list[str]:
        """Attach, squeeze out or release based on the jaws; return event messages."""
        events: list[str] = []
        cfg = self.settings
        opening = cfg.jaw_opening_m(gripper)
        if self.attached is not None:
            obj = self.objects[self.attached]
            width = self._held_width
            if opening > width + cfg.release_margin_m:
                events.append(f"{self.attached} released")
            elif not obj.soft and opening < width - cfg.rigid_squeeze_m:
                events.append(f"{self.attached} squeezed out: the gripper closed too far")
            else:
                return events
            self.data.qvel[obj.qvel_adr : obj.qvel_adr + 6] = 0.0
            self.attached = None
            self._armed = False
            return events

        distances = {
            name: float(np.linalg.norm(xyz - tool_pos)) for name, xyz in self.positions().items()
        }
        nearest = min(distances, key=distances.get) if distances else None
        if nearest is None or distances[nearest] > self.radius_m:
            self._armed = True  # nothing between the jaws: the next close can grab
            return events
        obj = self.objects[nearest]
        width, off_square = self.fit(nearest)
        if opening > width + cfg.touch_margin_m:
            self._armed = True
            return events
        if not self._armed:
            return events
        self._armed = False
        if not obj.soft and off_square > cfg.rigid_align_tol_deg:
            events.append(f"{nearest} slipped: the jaws are {off_square:.0f} deg off square to it")
            return events
        self.attached = nearest
        self._held_width = width
        self._offset = self.position(nearest) - tool_pos
        self._quat = self.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7].copy()
        events.append(f"{nearest} attached")
        return events

    def carry(self, tool_pos: np.ndarray) -> None:
        """Move an attached object with the tool (call before each physics step)."""
        if self.attached is None:
            return
        obj = self.objects[self.attached]
        self.data.qpos[obj.qpos_adr : obj.qpos_adr + 3] = tool_pos + self._offset
        self.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7] = self._quat
        self.data.qvel[obj.qvel_adr : obj.qvel_adr + 6] = 0.0

    def reset(self) -> None:
        self.attached = None
        self._armed = True
