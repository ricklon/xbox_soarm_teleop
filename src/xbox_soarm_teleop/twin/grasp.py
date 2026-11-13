"""Assisted grasp: closing the gripper to fit a nearby object attaches it.

Contact grasping with a kinematically posed arm is unreliable, so the
workshop twin uses an assisted grasp: when the jaws close down to the width of
an object that lies between their pads, the object is carried rigidly with the
gripper (turning with the wrist) until the jaws open wider than it again.

The tool point the arm is steered by is the URDF's ``gripper_frame_link``: the
inner face of the fixed jaw, 6 mm from its tip, as on the real arm. It is not
the middle of the jaws. :meth:`AssistedGrasp.jaw_center` is, and an object is
only grabbed if it lies between the pads (see the ``JAW_*`` constants); a
refused grasp says how far to move to center the jaws on it.

How far the jaws close matters, as it does on the real arm:

- open wider than the object: nothing is held;
- closed to its width, or a little past (``squeeze_m``): held;
- closed further on a rigid object: it is squeezed out and falls.

A rigid object must be gripped by two of its side faces, square between the
jaws (``align_tol_deg``). Its top and bottom do not count: one of them rests
on the table or box floor, where the lower jaw cannot reach, as on the real
arm. At wrist roll 0 the twin's jaws close nearly top to bottom, so a rigid
object needs the wrist turned about 90 degrees; a refused grasp says which
roll would square the jaws. A soft object (one the arm does not collide with,
like the bean bag) takes any closure and any angle.

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

# Jaw pads in the gripper_link frame (m), measured from the SO-ARM101 meshes
# (geometry, not placeholders). The jaws close along +x from the fixed jaw's
# inner face; the fingers run along -z; y is across the finger.
JAW_FIXED_FACE_X = -0.008
JAW_PAD_TIP_Z = -0.104
JAW_PAD_TOP_Z = -0.065
# Where an object's center goes: toward the tips, so a 4 cm object clears the palm.
JAW_GRIP_Z = -0.092
JAW_PAD_HALF_WIDTH_M = 0.009
# How much of an object a pad must cover, across and along the finger (m).
JAW_MIN_OVERLAP_M = 0.012


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
            object's sides; closing jaws turn a slightly skewed cube square.
            Measure: largest skew that still holds.
        radius_m: How near (jaw center to object center) an object must be
            for a close to try it; whether it is between the pads decides.
    """

    jaw_gap_closed_m: float = 0.0024
    jaw_gap_per_unit_m: float = 0.147
    touch_margin_m: float = 0.003
    release_margin_m: float = 0.005
    rigid_squeeze_m: float = 0.006
    rigid_align_tol_deg: float = 30.0
    radius_m: float = 0.06

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
        self._roll_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_roll")
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
        self._offset = np.zeros(3)  # held object's center, gripper frame
        self._rel_rotation = np.eye(3)  # held object's turn, gripper frame
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

    def half_extents_world(self, name: str) -> np.ndarray:
        """Half the object's extent along the world x, y and z axes (m), as it is turned now."""
        return np.abs(self._rotation(name)) @ (self.objects[name].size_m / 2.0)

    def closing_axis(self) -> np.ndarray | None:
        """Direction the jaws close along (world frame), or None without a gripper."""
        if self._gripper_body < 0 or self._jaw_body < 0:
            return None
        hinge = self.data.xmat[self._jaw_body].reshape(3, 3)[:, 2]
        along = -self.data.xmat[self._gripper_body].reshape(3, 3)[:, 2]
        axis = np.cross(hinge, along)
        norm = float(np.linalg.norm(axis))
        return axis / norm if norm > 1e-9 else None

    def _rotation(self, name: str) -> np.ndarray:
        obj = self.objects[name]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7])
        return rotation.reshape(3, 3)

    def grip_faces(self, name: str) -> list[int]:
        """The object's axes whose faces the jaws may grip.

        All three for a soft object; for a rigid one, the two axes that lie
        most nearly level (its top and bottom face the surface it rests on).
        """
        if self.objects[name].soft:
            return [0, 1, 2]
        upright = int(np.argmax(np.abs(self._rotation(name)[2, :])))
        return [axis for axis in range(3) if axis != upright]

    def fit(self, name: str) -> tuple[float, float]:
        """How an object sits between the jaws now: (width in m, degrees off square).

        The width is the object's extent along whichever of its grip axes
        (:meth:`grip_faces`) the jaws most nearly close along.
        """
        obj = self.objects[name]
        axis = self.closing_axis()
        faces = self.grip_faces(name)
        if axis is None:
            return float(min(obj.size_m[i] for i in faces)), 0.0
        along = np.abs(self._rotation(name).T @ axis)
        face = max(faces, key=lambda i: along[i])
        off_square = math.degrees(math.acos(min(1.0, float(along[face]))))
        return float(obj.size_m[face]), off_square

    def squaring_roll(self, name: str) -> float | None:
        """Wrist roll (rad) that would square the jaws to the object, or None if unknown.

        Turns the closing axis about the wrist-roll axis onto the grip face it
        can square up to best, by the smallest turn within the joint's range
        (the jaws are the same either way round, so half a turn is a second choice).
        """
        axis = self.closing_axis()
        if axis is None or self._roll_joint < 0:
            return None
        joint = self._roll_joint
        roll_axis = self.data.xaxis[joint] / np.linalg.norm(self.data.xaxis[joint])
        current = float(self.data.qpos[self.model.jnt_qposadr[joint]])
        low, high = (float(v) for v in self.model.jnt_range[joint])
        rotation = self._rotation(name)
        axis = axis - roll_axis * float(axis @ roll_axis)  # the part the roll can turn
        if np.linalg.norm(axis) < 1e-6:
            return None
        axis = axis / np.linalg.norm(axis)
        # The face the turned jaws can square up to best is the one whose
        # direction lies closest to the plane the wrist turns them in.
        face = min(self.grip_faces(name), key=lambda i: abs(float(rotation[:, i] @ roll_axis)))
        normal = rotation[:, face]
        target = normal - roll_axis * float(normal @ roll_axis)
        if np.linalg.norm(target) < 1e-6:
            return None
        target = target / np.linalg.norm(target)
        best = None
        for direction in (target, -target):  # the jaws are the same either way round
            turn = math.atan2(float(np.cross(axis, direction) @ roll_axis), float(axis @ direction))
            for roll in (current + turn, current + turn + 2 * math.pi, current + turn - 2 * math.pi):
                if low <= roll <= high and (best is None or abs(roll - current) < abs(best - current)):
                    best = roll
        return best

    def best_square_deg(self, name: str) -> float | None:
        """Least the jaws can be off square to the object by turning the wrist alone."""
        if self._roll_joint < 0:
            return None
        roll_axis = self.data.xaxis[self._roll_joint] / np.linalg.norm(
            self.data.xaxis[self._roll_joint]
        )
        rotation = self._rotation(name)
        lean = min(abs(float(rotation[:, i] @ roll_axis)) for i in self.grip_faces(name))
        return math.degrees(math.asin(min(1.0, lean)))

    def hold_limit(self) -> float | None:
        """Gripper position at which jaws driven shut stop on a held rigid object.

        A person closing by button or trigger cannot pick a closure to the
        millimetre, so their jaws stop here; None when nothing rigid is held.
        """
        if self.attached is None or self.objects[self.attached].soft:
            return None
        cfg = self.settings
        return cfg.gripper_for_opening(self._held_width - cfg.rigid_squeeze_m / 2.0)

    def _gripper_frame(self) -> tuple[np.ndarray, np.ndarray]:
        body = self._gripper_body
        return self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy()

    def jaw_center(self, opening_m: float) -> np.ndarray | None:
        """Grip point (world, m) with the jaws this far open, or None.

        Midway between the pads and 1.2 cm short of the tips: where an
        object's center should be when the jaws close on it.
        """
        if self._gripper_body < 0:
            return None
        origin, rotation = self._gripper_frame()
        local = np.array([JAW_FIXED_FACE_X + opening_m / 2.0, 0.0, JAW_GRIP_Z])
        return origin + rotation @ local

    def jaw_miss(self, name: str) -> np.ndarray | None:
        """How far (world, m) to move the tool to center the jaws on an object.

        None when it already lies between the pads (or there is no gripper).
        """
        if self._gripper_body < 0:
            return None
        origin, rotation = self._gripper_frame()
        center = rotation.T @ (self.position(name) - origin)
        half = np.abs(rotation.T @ self._rotation(name)) @ (self.objects[name].size_m / 2.0)
        across = abs(center[1]) <= half[1] + JAW_PAD_HALF_WIDTH_M - JAW_MIN_OVERLAP_M
        overlap = min(center[2] + half[2], JAW_PAD_TOP_Z) - max(center[2] - half[2], JAW_PAD_TIP_Z)
        along = overlap >= min(JAW_MIN_OVERLAP_M, 2.0 * half[2])
        inside = JAW_FIXED_FACE_X - 0.003 <= center[0] <= JAW_FIXED_FACE_X + 2.0 * half[0] + 0.02
        if across and along and inside:
            return None
        width, _ = self.fit(name)
        target = self.jaw_center(width)
        return self.position(name) - target

    def update(self, tool_pos: np.ndarray, gripper: float) -> list[str]:
        """Attach, squeeze out or release based on the jaws; return event messages.

        ``tool_pos`` is the tool point; the jaws are read from the model.
        """
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

        middle = self.jaw_center(opening)
        if middle is None:
            middle = tool_pos
        distances = {
            name: float(np.linalg.norm(xyz - middle)) for name, xyz in self.positions().items()
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
            hint = ""
            roll = self.squaring_roll(nearest)
            if roll is not None:
                hint = f"; open, then try set_wrist_roll(angle_rad={roll:.2f})"
                best = self.best_square_deg(nearest)
                if best is not None and best > cfg.rigid_align_tol_deg:
                    hint += (
                        f" (even then about {best:.0f} deg off: it is turned too far"
                        " from the way the arm faces it)"
                    )
            events.append(
                f"{nearest} slipped: the jaws are {off_square:.0f} deg off square to its sides{hint}"
            )
            return events
        miss = self.jaw_miss(nearest)
        if miss is not None:
            dx, dy, dz = (round(float(v), 3) + 0.0 for v in miss)
            events.append(
                f"{nearest} missed: it is not between the jaw pads; open, then"
                f" move_by(dx={dx}, dy={dy}, dz={dz}) to center the jaws on it"
            )
            return events
        self.attached = nearest
        self._held_width = width
        origin, rotation = self._gripper_frame()
        local = rotation.T @ (self.position(nearest) - origin)
        local[0] = JAW_FIXED_FACE_X + width / 2.0  # the jaws center it as they close
        self._offset = local
        self._rel_rotation = rotation.T @ self._rotation(nearest)
        self.carry()
        events.append(f"{nearest} attached")
        return events

    def carry(self, tool_pos: np.ndarray | None = None) -> None:
        """Move an attached object with the gripper (call before each physics step).

        It keeps its place and turn relative to the gripper, so it turns with
        the wrist. ``tool_pos`` is unused, kept for callers.
        """
        if self.attached is None:
            return
        obj = self.objects[self.attached]
        origin, rotation = self._gripper_frame()
        self.data.qpos[obj.qpos_adr : obj.qpos_adr + 3] = origin + rotation @ self._offset
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rotation @ self._rel_rotation).reshape(-1))
        self.data.qpos[obj.qpos_adr + 3 : obj.qpos_adr + 7] = quat
        self.data.qvel[obj.qvel_adr : obj.qvel_adr + 6] = 0.0

    def reset(self) -> None:
        self.attached = None
        self._armed = True
