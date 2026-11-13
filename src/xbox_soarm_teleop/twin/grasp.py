"""Assisted grasp: closing the gripper near a free object attaches it.

Contact grasping with a kinematically posed arm is unreliable, so the
workshop twin uses an assisted grasp: when the gripper closes past a threshold
with an object's center within a radius of the tool point, the object is
carried rigidly with the tool until the gripper opens again.

Only the act of closing grabs: a gripper that is already closed can push
objects without picking them up, and it re-arms once it opens past the
threshold again.

Objects a scene layout leaves out are parked: held far above the scene, where
no camera looks and nothing can reach them, and left out of every report.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Parked objects are held here (m), one metre apart, clear of every camera.
PARK_XYZ = np.array([0.0, 0.0, 50.0])
_UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass
class _FreeObject:
    body_id: int
    qpos_adr: int
    qvel_adr: int


class AssistedGrasp:
    """Attach the nearest free object when the gripper closes on it.

    Args:
        model: MuJoCo model; every body with a free joint is graspable.
        data: MuJoCo data.
        close_threshold: Gripper position (0 open .. 1 closed) that attaches.
        release_threshold: Gripper position at or below which it releases.
        radius_m: Maximum tool-to-object-center distance for attaching.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        close_threshold: float = 0.75,
        release_threshold: float = 0.35,
        radius_m: float = 0.045,
    ) -> None:
        self.model = model
        self.data = data
        self.close_threshold = close_threshold
        self.release_threshold = release_threshold
        self.radius_m = radius_m
        self.objects: dict[str, _FreeObject] = {}
        for body_id in range(model.nbody):
            jnt = model.body_jntadr[body_id]
            if jnt >= 0 and model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body{body_id}"
                self.objects[name] = _FreeObject(
                    body_id, int(model.jnt_qposadr[jnt]), int(model.jnt_dofadr[jnt])
                )
        self.parked: tuple[str, ...] = ()
        self.attached: str | None = None
        self._armed = True
        self._offset = np.zeros(3)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])

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

    def update(self, tool_pos: np.ndarray, gripper: float) -> list[str]:
        """Attach or release based on the gripper; return event messages."""
        events: list[str] = []
        closing = self._armed and gripper >= self.close_threshold
        if gripper < self.close_threshold:
            self._armed = True
        elif closing:
            self._armed = False
        if self.attached is None and closing:
            distances = {
                name: float(np.linalg.norm(xyz - tool_pos)) for name, xyz in self.positions().items()
            }
            nearest = min(distances, key=distances.get) if distances else None
            if nearest is not None and distances[nearest] <= self.radius_m:
                obj = self.objects[nearest]
                self.attached = nearest
                self._offset = self.position(nearest) - tool_pos
                self._quat = self.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7].copy()
                events.append(f"{nearest} attached")
        elif self.attached is not None and gripper <= self.release_threshold:
            obj = self.objects[self.attached]
            self.data.qvel[obj.qvel_adr : obj.qvel_adr + 6] = 0.0
            events.append(f"{self.attached} released")
            self.attached = None
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
