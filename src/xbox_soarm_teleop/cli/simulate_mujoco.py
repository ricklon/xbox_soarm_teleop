#!/usr/bin/env python3
"""MuJoCo simulation with teleoperation via Xbox, Joy-Con, or keyboard.

This example runs SO-ARM teleoperation with MuJoCo physics simulation
and real-time 3D visualization.

Usage:
    uv run python examples/simulate_mujoco.py
    uv run python examples/simulate_mujoco.py --controller keyboard --mode joint
    uv run python examples/simulate_mujoco.py --controller joycon

Options:
    --controller xbox|joycon|keyboard   Input device (default: xbox)
    --mode cartesian|joint|crane        Control mode (default: crane)
    --no-controller                     Run without controller (demo mode)
    --challenge                         Run benchmark challenge mode with targets

Controls are printed at startup based on the active controller and mode.

Controls (Demo mode, --no-controller):
    - Automatic demo movement pattern
    - Ctrl+C or close window: Exit

Challenge mode:
    - Collect targets by moving gripper close to them
    - Starts with 1 target, difficulty increases
    - Tracks accuracy, smoothness, and time metrics
"""

import argparse
import csv
import json
import signal
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from xbox_soarm_teleop.config.joints import (
    IK_JOINT_NAMES,
    IK_JOINT_VEL_LIMITS_ARRAY,
    JOINT_LIMITS_DEG,
    JOINT_NAMES_WITH_GRIPPER,
    limits_rad_to_deg,
    parse_joint_limits,
)
from xbox_soarm_teleop.control.cartesian import (
    advance_cartesian_target,
    apply_ik_solution,
    full_joint_positions,
    make_cartesian_state,
    step_cartesian_home,
    step_gripper_toward,
    step_wrist_roll,
    sync_cartesian_state,
)
from xbox_soarm_teleop.control.routines import plane_offset
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta, apply_axis_mapping
from xbox_soarm_teleop.runtime import build_control_runtime, print_controls

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = PROJECT_ROOT / "assets" / "so101_abs.urdf"

# Joint names (order matters - matches URDF joint order)
JOINT_NAMES = JOINT_NAMES_WITH_GRIPPER
CAMERA_PRESET_NAMES = ["front_right", "left", "top", "back", "isometric"]
CAMERA_PRESETS = {
    "front_right": {
        "lookat": np.array([0.18, 0.0, 0.16], dtype=float),
        "distance": 0.70,
        "azimuth": 135.0,
        "elevation": -22.0,
    },
    "left": {
        "lookat": np.array([0.18, 0.0, 0.16], dtype=float),
        "distance": 0.65,
        "azimuth": 90.0,
        "elevation": -18.0,
    },
    "top": {
        "lookat": np.array([0.18, 0.0, 0.16], dtype=float),
        "distance": 0.60,
        "azimuth": 180.0,
        "elevation": -89.0,
    },
    "back": {
        "lookat": np.array([0.18, 0.0, 0.16], dtype=float),
        "distance": 0.70,
        "azimuth": 0.0,
        "elevation": -18.0,
    },
    "isometric": {
        "lookat": np.array([0.18, 0.0, 0.16], dtype=float),
        "distance": 0.75,
        "azimuth": 45.0,
        "elevation": -28.0,
    },
}
GRIPPER_FRAME_LOCAL_OFFSET = np.array([-0.0079, -0.000218121, -0.0981274], dtype=float)
GRIPPER_JAW_LATERAL_OFFSET = 0.018

# Control loop rate
CONTROL_RATE = 50  # Hz
LOOP_PERIOD = 1.0 / CONTROL_RATE
STACK_CUBE_COUNT = 3
STACK_CUBE_HALF_EXTENT = 0.02
STACK_CUBE_MASS = 0.05
STACK_BASE_CENTER = np.array([0.355, 0.0, STACK_CUBE_HALF_EXTENT], dtype=float)
STACK_LAYOUT_OFFSETS = [
    np.array([0.0, -0.028, 0.0], dtype=float),
    np.array([0.0, 0.028, 0.0], dtype=float),
    np.array([0.0, 0.0, 2.15 * STACK_CUBE_HALF_EXTENT], dtype=float),
]
STACK_COLORS = [
    (0.82, 0.14, 0.74, 1.0),
    (0.65, 0.65, 0.12, 1.0),
    (0.12, 0.12, 0.65, 1.0),
]
PICK_PLACE_TABLE_HEIGHT = 0.12
PICK_PLACE_TABLE_CENTER = np.array([0.31, 0.04, PICK_PLACE_TABLE_HEIGHT / 2.0], dtype=float)
PICK_PLACE_TABLE_HALF_EXTENTS = np.array([0.16, 0.14, PICK_PLACE_TABLE_HEIGHT / 2.0], dtype=float)
PICK_PLACE_CUBE_HALF_EXTENT = 0.02
PICK_PLACE_CUBE_MASS = 0.04
PICK_PLACE_CUBE_START = np.array([0.34, -0.03, PICK_PLACE_TABLE_HEIGHT + PICK_PLACE_CUBE_HALF_EXTENT], dtype=float)
PICK_PLACE_GOAL_CENTER = np.array([0.29, 0.13, PICK_PLACE_TABLE_HEIGHT + 0.001], dtype=float)
PICK_PLACE_GOAL_INNER_HALF_EXTENTS = np.array([0.045, 0.045, 0.03], dtype=float)
PICK_PLACE_GOAL_WALL_THICKNESS = 0.005


class StackChallenge:
    """Physical cube stack diagnostic scored by cube displacement."""

    def __init__(self, sim: "MuJoCoSimulator", move_threshold: float = 0.025):
        self.sim = sim
        self.move_threshold = move_threshold
        self.completed: set[str] = set()

    def start(self) -> None:
        print("\n=== STACK CHALLENGE ===", flush=True)
        print("Touch or push the physical cube stack until each cube moves.", flush=True)
        print(f"Success threshold: {self.move_threshold * 100:.1f}cm displacement from rest\n", flush=True)

    def update(self) -> list[str]:
        moved_now: list[str] = []
        for name, displacement in self.sim.get_stack_cube_displacements().items():
            if displacement >= self.move_threshold and name not in self.completed:
                self.completed.add(name)
                moved_now.append(name)
        return moved_now

    def print_summary(self) -> None:
        print("\n=== STACK SUMMARY ===", flush=True)
        print(f"Cubes moved: {len(self.completed)} / {STACK_CUBE_COUNT}", flush=True)
        if self.completed:
            print("Moved cubes:", ", ".join(sorted(self.completed)), flush=True)

    def status_text(self) -> str:
        moved = len(self.completed)
        return f"Stack moved: {moved}/{STACK_CUBE_COUNT}"

    def reset(self) -> None:
        self.completed.clear()


class PickPlaceTask:
    """Assisted pick-and-place scene with a physical cube and goal box."""

    def __init__(
        self,
        sim: "MuJoCoSimulator",
        grasp_close_threshold: float = 0.75,
        release_open_threshold: float = 0.35,
        grasp_radius: float = 0.045,
    ):
        self.sim = sim
        self.grasp_close_threshold = grasp_close_threshold
        self.release_open_threshold = release_open_threshold
        self.grasp_radius = grasp_radius
        self.attached = False
        self.completed = False

    def start(self) -> None:
        print("\n=== PICK AND PLACE ===", flush=True)
        print("Pick the cube from the table and place it inside the goal box.", flush=True)
        print("Assisted grasp: close gripper near cube to attach, open to release.\n", flush=True)

    def update(self, gripper_pos: float) -> list[str]:
        events: list[str] = []
        cube_pos = self.sim.get_pick_cube_position()
        if cube_pos is None:
            return events

        ee_pos = self.sim.get_ee_position()
        if not self.attached and gripper_pos >= self.grasp_close_threshold:
            if np.linalg.norm(cube_pos - ee_pos) <= self.grasp_radius:
                self.attached = True
                events.append("cube attached")

        if self.attached:
            carried_pos = ee_pos + np.array([0.0, 0.0, -0.015], dtype=float)
            carried_pos[2] = max(carried_pos[2], PICK_PLACE_TABLE_HEIGHT + PICK_PLACE_CUBE_HALF_EXTENT)
            self.sim.set_pick_cube_pose(carried_pos)
            if gripper_pos <= self.release_open_threshold:
                self.attached = False
                events.append("cube released")

        if not self.attached and not self.completed and self.sim.cube_in_pick_place_goal():
            self.completed = True
            events.append("goal reached")
        return events

    def print_summary(self) -> None:
        print("\n=== PICK AND PLACE SUMMARY ===", flush=True)
        print("Success" if self.completed else "Incomplete", flush=True)

    def status_text(self) -> str:
        if self.completed:
            return "Pick/place complete"
        if self.attached:
            return "Cube attached"
        return "Acquire cube"

    def reset(self) -> None:
        self.attached = False
        self.completed = False


def reset_active_scene(
    sim: "MuJoCoSimulator",
    *,
    control_mode,
    processor,
    mapper,
    cartesian_state,
    kinematics,
    stack_challenge: "StackChallenge | None",
    pick_place_task: "PickPlaceTask | None",
    trace_points: list[np.ndarray],
) -> bool:
    """Reset the active physical scene and associated runtime state.

    Returns:
        True if a scene reset was performed, False otherwise.
    """
    reset_performed = False
    if sim.has_stack_scene():
        print("\nResetting stack scene...", flush=True)
        sim.reset_stack_scene()
        if stack_challenge is not None:
            stack_challenge.reset()
        reset_performed = True
    if sim.has_pick_place_scene():
        print("\nResetting pick/place scene...", flush=True)
        sim.reset_pick_place_scene()
        if pick_place_task is not None:
            pick_place_task.reset()
        reset_performed = True

    if not reset_performed:
        return False

    trace_points.clear()
    if hasattr(mapper, "reset"):
        mapper.reset()

    if getattr(control_mode, "value", None) == "cartesian" and cartesian_state is not None and kinematics is not None:
        sync_cartesian_state(
            cartesian_state,
            kinematics,
            np.zeros(4, dtype=float),
            wrist_roll_deg=0.0,
            target_pitch=0.0,
            target_yaw=0.0,
            gripper_pos=0.0,
        )
    elif hasattr(processor, "reset"):
        processor.reset()

    return True


class MuJoCoSimulator:
    """MuJoCo-based SO-ARM101 simulator."""

    def __init__(self, urdf_path: str, scene: str = "default"):
        """Initialize MuJoCo simulator.

        Args:
            urdf_path: Path to robot URDF file.
            scene: Scene variant to augment on top of the robot model.
        """
        self.scene = scene
        self.model = load_model_with_cameras(urdf_path, scene=scene)
        self.data = mujoco.MjData(self.model)
        self.physics_scene_active = scene in {"stack", "pick_place_basic"}

        # Get joint indices
        self.joint_ids = {}
        self.joint_qvel_adrs = {}
        for name in JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                self.joint_ids[name] = jnt_id
                self.joint_qvel_adrs[name] = self.model.jnt_dofadr[jnt_id]

        # Target joint positions (radians)
        self.target_pos = np.zeros(len(JOINT_NAMES))
        self.stack_cube_body_ids: dict[str, int] = {}
        self.stack_initial_positions: dict[str, np.ndarray] = {}
        self.stack_spawn_positions: dict[str, np.ndarray] = {}
        self.pick_cube_body_id: int | None = None
        self.pick_cube_qpos_adr: int | None = None
        self.pick_cube_qvel_adr: int | None = None

        # Initialize to home position
        self.go_home()
        if scene == "stack":
            self._settle_stack_scene()
        elif scene == "pick_place_basic":
            self._bind_pick_place_scene()

    def go_home(self) -> None:
        """Reset to home position."""
        self.target_pos = np.zeros(len(JOINT_NAMES))
        # Set qpos directly
        for i, name in enumerate(JOINT_NAMES):
            if name in self.joint_ids:
                jnt_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] = 0.0
                qvel_adr = self.joint_qvel_adrs[name]
                if qvel_adr >= 0:
                    self.data.qvel[qvel_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_joint_targets(self, positions_deg: np.ndarray) -> None:
        """Set target joint positions.

        Args:
            positions_deg: Joint positions in degrees.
        """
        positions_rad = np.deg2rad(positions_deg)
        for i, name in enumerate(JOINT_NAMES):
            if i < len(positions_rad) and name in self.joint_ids:
                jnt_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] = positions_rad[i]
                qvel_adr = self.joint_qvel_adrs[name]
                if qvel_adr >= 0:
                    self.data.qvel[qvel_adr] = 0.0

    def set_gripper(self, position: float) -> None:
        """Set gripper position (0=open, 1=closed).

        Args:
            position: Gripper position 0-1.
        """
        if "gripper" in self.joint_ids:
            jnt_id = self.joint_ids["gripper"]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            # Map 0-1 to joint limits: open=1.74, closed=-0.17
            gripper_open = 1.74533
            gripper_closed = -0.174533
            self.data.qpos[qpos_adr] = gripper_open - position * (gripper_open - gripper_closed)
            qvel_adr = self.joint_qvel_adrs["gripper"]
            if qvel_adr >= 0:
                self.data.qvel[qvel_adr] = 0.0

    def get_joint_positions_deg(self) -> np.ndarray:
        """Get current joint positions in degrees."""
        positions = []
        for name in JOINT_NAMES:
            if name in self.joint_ids:
                jnt_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                positions.append(np.rad2deg(self.data.qpos[qpos_adr]))
        return np.array(positions)

    def get_ee_position(self) -> np.ndarray:
        """Get end effector position."""
        return self.get_gripper_touch_points()[0]

    def get_gripper_touch_points(self) -> list[np.ndarray]:
        """Return contact proxy points for the gripper center and both jaws."""
        gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
        moving_jaw_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1_link"
        )
        if gripper_body_id < 0:
            return [self.data.xpos[-1].copy()]

        gripper_origin = self.data.xpos[gripper_body_id].copy()
        gripper_rot = self.data.xmat[gripper_body_id].reshape(3, 3).copy()
        center_point = gripper_origin + gripper_rot @ GRIPPER_FRAME_LOCAL_OFFSET
        lateral_axis = gripper_rot[:, 1]

        touch_points = [
            center_point,
            center_point + lateral_axis * GRIPPER_JAW_LATERAL_OFFSET,
            center_point - lateral_axis * GRIPPER_JAW_LATERAL_OFFSET,
        ]
        if moving_jaw_body_id >= 0:
            touch_points.append(self.data.xpos[moving_jaw_body_id].copy())
        return touch_points

    def step(self) -> None:
        """Advance the simulator."""
        if self.physics_scene_active:
            mujoco.mj_step(self.model, self.data)
        else:
            mujoco.mj_forward(self.model, self.data)

    def has_stack_scene(self) -> bool:
        return self.scene == "stack"

    def has_pick_place_scene(self) -> bool:
        return self.scene == "pick_place_basic"

    def _settle_stack_scene(self, steps: int = 200) -> None:
        """Let the stack fall onto the plane and record resting positions."""
        self.stack_cube_body_ids = {}
        for idx in range(STACK_CUBE_COUNT):
            body_name = f"stack_cube_{idx}"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                self.stack_cube_body_ids[body_name] = body_id
        self.stack_spawn_positions = {
            name: self.data.xpos[body_id].copy() for name, body_id in self.stack_cube_body_ids.items()
        }
        for _ in range(steps):
            self.step()
        self.stack_initial_positions = {
            name: self.data.xpos[body_id].copy() for name, body_id in self.stack_cube_body_ids.items()
        }

    def get_stack_cube_displacements(self) -> dict[str, float]:
        """Return displacement-from-rest for each stack cube."""
        if not self.stack_cube_body_ids:
            return {}
        return {
            name: float(np.linalg.norm(self.data.xpos[body_id] - self.stack_initial_positions[name]))
            for name, body_id in self.stack_cube_body_ids.items()
        }

    def _bind_pick_place_scene(self) -> None:
        """Bind the free cube handles for the pick/place scene."""
        self.pick_cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")
        if self.pick_cube_body_id < 0:
            self.pick_cube_body_id = None
            return
        jnt_adr = self.model.body_jntadr[self.pick_cube_body_id]
        if jnt_adr < 0:
            self.pick_cube_body_id = None
            return
        self.pick_cube_qpos_adr = self.model.jnt_qposadr[jnt_adr]
        self.pick_cube_qvel_adr = self.model.jnt_dofadr[jnt_adr]

    def get_pick_cube_position(self) -> np.ndarray | None:
        """Return the current cube center position for the pick/place scene."""
        if self.pick_cube_body_id is None:
            return None
        return self.data.xpos[self.pick_cube_body_id].copy()

    def set_pick_cube_pose(self, position: np.ndarray) -> None:
        """Directly place the pick/place cube at a target position."""
        if self.pick_cube_qpos_adr is None or self.pick_cube_qvel_adr is None:
            return
        self.data.qpos[self.pick_cube_qpos_adr : self.pick_cube_qpos_adr + 3] = position
        self.data.qpos[self.pick_cube_qpos_adr + 3 : self.pick_cube_qpos_adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        self.data.qvel[self.pick_cube_qvel_adr : self.pick_cube_qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def cube_in_pick_place_goal(self) -> bool:
        """Return True when the cube lies inside the goal box interior."""
        cube_pos = self.get_pick_cube_position()
        if cube_pos is None:
            return False
        xy_ok = np.all(
            np.abs(cube_pos[:2] - PICK_PLACE_GOAL_CENTER[:2])
            <= (PICK_PLACE_GOAL_INNER_HALF_EXTENTS[:2] - PICK_PLACE_CUBE_HALF_EXTENT)
        )
        z_min = PICK_PLACE_TABLE_HEIGHT + PICK_PLACE_CUBE_HALF_EXTENT - 0.01
        z_max = PICK_PLACE_TABLE_HEIGHT + PICK_PLACE_GOAL_INNER_HALF_EXTENTS[2]
        z_ok = z_min <= cube_pos[2] <= z_max
        return bool(xy_ok and z_ok)

    def reset_stack_scene(self) -> None:
        """Restore stack cubes to their spawn positions and let them settle."""
        if not self.stack_cube_body_ids:
            return
        self.go_home()
        for name, body_id in self.stack_cube_body_ids.items():
            spawn_pos = self.stack_spawn_positions.get(name)
            if spawn_pos is None:
                continue
            jnt_adr = self.model.body_jntadr[body_id]
            if jnt_adr < 0:
                continue
            qpos_adr = self.model.jnt_qposadr[jnt_adr]
            qvel_adr = self.model.jnt_dofadr[jnt_adr]
            self.data.qpos[qpos_adr : qpos_adr + 3] = spawn_pos
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._settle_stack_scene()

    def reset_pick_place_scene(self) -> None:
        """Restore the pick/place cube to its start pose and home the arm."""
        self.go_home()
        self.set_pick_cube_pose(PICK_PLACE_CUBE_START.copy())


def build_stack_scene_model(urdf_path: str) -> mujoco.MjModel:
    """Compile a physical stack scene on top of the robot model."""
    base_model = mujoco.MjModel.from_xml_path(urdf_path)
    with tempfile.NamedTemporaryFile(
        suffix=".xml",
        dir=str(PROJECT_ROOT / "assets"),
        delete=False,
    ) as tmp:
        tmp_path = tmp.name
    try:
        mujoco.mj_saveLastXML(tmp_path, base_model)
        spec = mujoco.MjSpec.from_file(tmp_path)

        plane = spec.worldbody.add_geom()
        plane.name = "stack_ground"
        plane.type = mujoco.mjtGeom.mjGEOM_PLANE
        plane.size = [0.6, 0.6, 0.05]
        plane.pos = [0.2, 0.0, 0.0]
        plane.rgba = [0.45, 0.45, 0.45, 1.0]

        for idx, offset in enumerate(STACK_LAYOUT_OFFSETS):
            body = spec.worldbody.add_body()
            body.name = f"stack_cube_{idx}"
            body.pos = (STACK_BASE_CENTER + offset).tolist()
            body.add_freejoint()

            geom = body.add_geom()
            geom.name = f"stack_cube_{idx}_geom"
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = [STACK_CUBE_HALF_EXTENT, STACK_CUBE_HALF_EXTENT, STACK_CUBE_HALF_EXTENT]
            geom.mass = STACK_CUBE_MASS
            geom.rgba = STACK_COLORS[idx % len(STACK_COLORS)]
            geom.friction = [1.1, 0.02, 0.001]
        return spec.compile()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def build_pick_place_scene_model(urdf_path: str) -> mujoco.MjModel:
    """Compile a simple physical pick-and-place scene on top of the robot model."""
    base_model = mujoco.MjModel.from_xml_path(urdf_path)
    with tempfile.NamedTemporaryFile(
        suffix=".xml",
        dir=str(PROJECT_ROOT / "assets"),
        delete=False,
    ) as tmp:
        tmp_path = tmp.name
    try:
        mujoco.mj_saveLastXML(tmp_path, base_model)
        spec = mujoco.MjSpec.from_file(tmp_path)

        floor = spec.worldbody.add_geom()
        floor.name = "pick_place_floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.8, 0.8, 0.05]
        floor.pos = [0.2, 0.0, 0.0]
        floor.rgba = [0.45, 0.45, 0.45, 1.0]

        table = spec.worldbody.add_geom()
        table.name = "pick_place_table"
        table.type = mujoco.mjtGeom.mjGEOM_BOX
        table.size = PICK_PLACE_TABLE_HALF_EXTENTS.tolist()
        table.pos = PICK_PLACE_TABLE_CENTER.tolist()
        table.rgba = [0.60, 0.60, 0.60, 1.0]
        table.friction = [1.1, 0.02, 0.001]

        cube_body = spec.worldbody.add_body()
        cube_body.name = "pick_cube"
        cube_body.pos = PICK_PLACE_CUBE_START.tolist()
        cube_body.add_freejoint()
        cube_geom = cube_body.add_geom()
        cube_geom.name = "pick_cube_geom"
        cube_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        cube_geom.size = [PICK_PLACE_CUBE_HALF_EXTENT] * 3
        cube_geom.mass = PICK_PLACE_CUBE_MASS
        cube_geom.rgba = [0.85, 0.25, 0.20, 1.0]
        cube_geom.friction = [1.1, 0.02, 0.001]

        goal_parts = [
            ("goal_base", [0.0, 0.0, 0.0], [*PICK_PLACE_GOAL_INNER_HALF_EXTENTS[:2], 0.004]),
            (
                "goal_wall_left",
                [-(PICK_PLACE_GOAL_INNER_HALF_EXTENTS[0] + PICK_PLACE_GOAL_WALL_THICKNESS), 0.0, PICK_PLACE_GOAL_INNER_HALF_EXTENTS[2]],
                [PICK_PLACE_GOAL_WALL_THICKNESS, PICK_PLACE_GOAL_INNER_HALF_EXTENTS[1], PICK_PLACE_GOAL_INNER_HALF_EXTENTS[2]],
            ),
            (
                "goal_wall_right",
                [(PICK_PLACE_GOAL_INNER_HALF_EXTENTS[0] + PICK_PLACE_GOAL_WALL_THICKNESS), 0.0, PICK_PLACE_GOAL_INNER_HALF_EXTENTS[2]],
                [PICK_PLACE_GOAL_WALL_THICKNESS, PICK_PLACE_GOAL_INNER_HALF_EXTENTS[1], PICK_PLACE_GOAL_INNER_HALF_EXTENTS[2]],
            ),
            (
                "goal_wall_back",
                [0.0, -(PICK_PLACE_GOAL_INNER_HALF_EXTENTS[1] + PICK_PLACE_GOAL_WALL_THICKNESS), PICK_PLACE_GOAL_INNER_HALF_EXTENTS[2]],
                [PICK_PLACE_GOAL_INNER_HALF_EXTENTS[0] + PICK_PLACE_GOAL_WALL_THICKNESS, PICK_PLACE_GOAL_WALL_THICKNESS, PICK_PLACE_GOAL_INNER_HALF_EXTENTS[2]],
            ),
            (
                "goal_wall_front",
                [0.0, (PICK_PLACE_GOAL_INNER_HALF_EXTENTS[1] + PICK_PLACE_GOAL_WALL_THICKNESS), PICK_PLACE_GOAL_INNER_HALF_EXTENTS[2]],
                [PICK_PLACE_GOAL_INNER_HALF_EXTENTS[0] + PICK_PLACE_GOAL_WALL_THICKNESS, PICK_PLACE_GOAL_WALL_THICKNESS, PICK_PLACE_GOAL_INNER_HALF_EXTENTS[2]],
            ),
        ]
        for name, offset, size in goal_parts:
            geom = spec.worldbody.add_geom()
            geom.name = name
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.pos = (PICK_PLACE_GOAL_CENTER + np.array(offset, dtype=float)).tolist()
            geom.size = size
            geom.rgba = [0.10, 0.65, 0.35, 0.85]
            geom.contype = 1
            geom.conaffinity = 1
        return spec.compile()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def load_model_with_cameras(urdf_path: str, scene: str = "default") -> mujoco.MjModel:
    """Load the simulator model, optionally augmenting it with a physical scene."""
    if scene == "stack":
        return build_stack_scene_model(urdf_path)
    if scene == "pick_place_basic":
        return build_pick_place_scene_model(urdf_path)
    return mujoco.MjModel.from_xml_path(urdf_path)


def available_camera_presets() -> list[str]:
    """Return the available named viewer presets."""
    return [name for name in CAMERA_PRESET_NAMES if name in CAMERA_PRESETS]


def set_viewer_camera(viewer: mujoco.viewer.Handle, camera_name: str) -> bool:
    """Apply a named preset to the free camera."""
    preset = CAMERA_PRESETS.get(camera_name)
    if preset is None:
        return False
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = preset["lookat"]
    viewer.cam.distance = preset["distance"]
    viewer.cam.azimuth = preset["azimuth"]
    viewer.cam.elevation = preset["elevation"]
    return True


def set_viewer_free_camera(viewer: mujoco.viewer.Handle) -> None:
    """Return the viewer to the standard free camera."""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE


class ChallengeTarget:
    """A single target in the challenge mode."""

    def __init__(self, position: np.ndarray, target_id: int, label: str | None = None):
        self.position = position.copy()
        self.target_id = target_id
        self.label = label or f"target_{target_id}"
        self.spawn_time = time.monotonic()
        self.collected = False
        self.collect_time: float | None = None
        # Tracking for smoothness metrics
        self.approach_velocities: list[float] = []
        self.approach_jerks: list[float] = []
        self.path_length = 0.0
        self.direct_distance: float | None = None
        self.final_error: float | None = None

    def time_to_collect(self) -> float | None:
        if self.collect_time is None:
            return None
        return self.collect_time - self.spawn_time

    def path_efficiency(self) -> float | None:
        """Return path efficiency: direct_distance / path_length (1.0 = perfect, <1.0 = longer path)."""
        if self.direct_distance is None or self.path_length < 0.001:
            return None
        return self.direct_distance / self.path_length

    def mean_jerk(self) -> float | None:
        if not self.approach_jerks:
            return None
        return float(np.mean(np.abs(self.approach_jerks)))


class ChallengeManager:
    """Manages challenge mode with targets to collect."""

    # Target colors (RGBA) - cycle through for multiple targets
    TARGET_COLORS = [
        (1.0, 0.2, 0.2, 0.9),  # Red
        (0.2, 1.0, 0.2, 0.9),  # Green
        (0.2, 0.2, 1.0, 0.9),  # Blue
        (1.0, 1.0, 0.2, 0.9),  # Yellow
        (1.0, 0.2, 1.0, 0.9),  # Magenta
        (0.2, 1.0, 1.0, 0.9),  # Cyan
    ]

    @staticmethod
    def box_to_box_distance(
        center_a: np.ndarray,
        half_extents_a: np.ndarray,
        center_b: np.ndarray,
        half_extents_b: np.ndarray,
    ) -> float:
        """Return Euclidean distance between two axis-aligned boxes."""
        delta = np.abs(center_a - center_b) - (half_extents_a + half_extents_b)
        outside = np.maximum(delta, 0.0)
        return float(np.linalg.norm(outside))

    @staticmethod
    def boxes_overlap(
        center_a: np.ndarray,
        half_extents_a: np.ndarray,
        center_b: np.ndarray,
        half_extents_b: np.ndarray,
    ) -> bool:
        """Return True when two axis-aligned boxes overlap or touch."""
        return bool(np.all(np.abs(center_a - center_b) <= (half_extents_a + half_extents_b)))

    def __init__(
        self,
        kinematics,
        joint_limits_deg: dict[str, tuple[float, float]],
        collect_radius: float = 0.03,
        target_size: float = 0.015,
        initial_targets: int = 1,
        targets_per_level: int = 5,
        max_targets: int = 3,
        seed: int | None = None,
        workspace_margin: float = 0.05,
        initial_ee_position: np.ndarray | None = None,
        layout: str = "random",
    ):
        self.kinematics = kinematics
        self.joint_limits_deg = joint_limits_deg
        self.collect_radius = collect_radius
        self.target_size = target_size
        touch_half_extent = max(self.collect_radius * 0.5, self.target_size)
        self.gripper_touch_half_extents = np.full(3, touch_half_extent, dtype=float)
        self.initial_targets = initial_targets
        self.targets_per_level = targets_per_level
        self.max_targets = max_targets
        self.workspace_margin = workspace_margin
        self.initial_ee_position = initial_ee_position
        self.layout = layout

        self.rng = np.random.default_rng(seed)
        self.active_targets: list[ChallengeTarget] = []
        self.collected_targets: list[ChallengeTarget] = []
        self.diagnostic_targets: list[tuple[str, np.ndarray]] = []
        self.total_spawned = 0
        self.current_level = 1
        self.level_collected = 0

        # Pre-compute reachable workspace bounds
        self._compute_workspace_bounds()

        # Tracking for smoothness
        self.last_ee_pos: np.ndarray | None = None
        self.last_ee_vel: np.ndarray | None = None
        self.last_update_time: float | None = None

    def _compute_workspace_bounds(self) -> None:
        """Build list of reachable positions using a small grid around home."""
        # Use provided initial EE position if available, otherwise use FK
        if self.initial_ee_position is not None:
            self.home_pos = self.initial_ee_position.copy()
            print(
                f"  Challenge: Using actual EE position [{self.home_pos[0]:.3f}, "
                f"{self.home_pos[1]:.3f}, {self.home_pos[2]:.3f}]",
                flush=True,
            )
        else:
            home_joints = np.zeros(len(self.joint_limits_deg))
            home_pose = self.kinematics.forward_kinematics(home_joints)
            self.home_pos = home_pose[:3, 3].copy()
            print(
                f"  Challenge: Home position at [{self.home_pos[0]:.3f}, "
                f"{self.home_pos[1]:.3f}, {self.home_pos[2]:.3f}]",
                flush=True,
            )

        # Use a simple grid around home - small enough to be definitely reachable
        # These offsets are conservative and should always be reachable
        self.verified_positions: list[np.ndarray] = []

        # Grid: X (forward/back), Y (left/right), Z (up/down)
        # Keep offsets small to ensure reachability
        x_offsets = [-0.06, -0.03, 0.0, 0.03]  # Mostly forward (negative X)
        y_offsets = [-0.06, -0.03, 0.0, 0.03, 0.06]  # Left/right symmetric
        z_offsets = [-0.04, 0.0, 0.04, 0.08]  # Up/down

        for dx in x_offsets:
            for dy in y_offsets:
                for dz in z_offsets:
                    # Skip origin (too close to home)
                    if abs(dx) < 0.02 and abs(dy) < 0.02 and abs(dz) < 0.02:
                        continue

                    candidate = self.home_pos + np.array([dx, dy, dz])

                    # Must stay above table (min Z = 0.12m)
                    if candidate[2] < 0.12:
                        continue

                    self.verified_positions.append(candidate.copy())

        print(f"  Challenge: {len(self.verified_positions)} target positions available", flush=True)
        self._build_diagnostic_targets()

    def _nearest_verified_position(self, desired: np.ndarray, used_indices: set[int]) -> np.ndarray:
        """Pick the nearest verified target position that has not been used yet."""
        if not self.verified_positions:
            return desired.copy()

        best_idx = None
        best_dist = float("inf")
        for idx, pos in enumerate(self.verified_positions):
            if idx in used_indices:
                continue
            dist = float(np.linalg.norm(pos - desired))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is None:
            return desired.copy()

        used_indices.add(best_idx)
        return self.verified_positions[best_idx].copy()

    def _build_diagnostic_targets(self) -> None:
        """Build fixed targets around home for controller diagnostics."""
        used_indices: set[int] = set()
        desired_specs = [
            ("forward", np.array([-0.06, 0.00, 0.00])),
            ("back", np.array([0.03, 0.00, 0.00])),
            ("left", np.array([0.00, -0.06, 0.00])),
            ("right", np.array([0.00, 0.06, 0.00])),
            ("up", np.array([0.00, 0.00, 0.08])),
            ("down", np.array([0.00, 0.00, -0.04])),
        ]
        self.diagnostic_targets = [
            (label, self._nearest_verified_position(self.home_pos + offset, used_indices))
            for label, offset in desired_specs
        ]

    def _sample_reachable_position(self, avoid_positions: list[np.ndarray] | None = None) -> np.ndarray:
        """Sample a random position from verified reachable positions."""
        min_separation = self.collect_radius * 3

        # Shuffle verified positions
        candidates = self.verified_positions.copy()
        self.rng.shuffle(candidates)

        for pos in candidates:
            # Check separation from existing targets
            if avoid_positions:
                too_close = False
                for other in avoid_positions:
                    if np.linalg.norm(pos - other) < min_separation:
                        too_close = True
                        break
                if too_close:
                    continue
            return pos.copy()

        # Fallback: return home position with small offset
        offset = self.rng.uniform(-0.03, 0.03, size=3)
        offset[2] = abs(offset[2])  # Keep Z positive
        return self.home_pos + offset

    def spawn_targets(self, count: int) -> None:
        """Spawn new targets."""
        existing_positions = [t.position for t in self.active_targets]
        for _ in range(count):
            pos = self._sample_reachable_position(existing_positions)
            target = ChallengeTarget(pos, self.total_spawned)
            self.active_targets.append(target)
            existing_positions.append(pos)
            print(
                f"  Spawned target {self.total_spawned} at [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                flush=True,
            )
            self.total_spawned += 1

    def spawn_diagnostic_targets(self) -> None:
        """Spawn one fixed target for each main diagnostic direction."""
        self.active_targets.clear()
        for label, pos in self.diagnostic_targets:
            target = ChallengeTarget(pos, self.total_spawned, label=label)
            self.active_targets.append(target)
            print(
                f"  Spawned {label:>7} target {self.total_spawned} at "
                f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                flush=True,
            )
            self.total_spawned += 1

    def start(self) -> None:
        """Start the challenge with initial targets."""
        if self.layout == "diagnostic":
            self.spawn_diagnostic_targets()
        else:
            self.spawn_targets(self.initial_targets)
        print("\n=== CHALLENGE MODE ===", flush=True)
        print(
            f"Collect targets by touching the target box with the gripper zone "
            f"(target {self.target_size * 200:.1f}cm wide, touch box {self.gripper_touch_half_extents[0] * 200:.1f}cm)",
            flush=True,
        )
        if self.layout == "diagnostic":
            print("Diagnostic layout: fixed targets in forward/back/left/right/up/down\n", flush=True)
            print("Suggested camera views:", flush=True)
            print("  front_right  general piloting / gripper visibility", flush=True)
            print("  top          XY direction check", flush=True)
            print("  left         height / down target check", flush=True)
            print("  Use ] and [  cycle cameras, ESC for free camera\n", flush=True)
        else:
            print(f"Starting with {self.initial_targets} target(s)", flush=True)
            print(f"Difficulty increases every {self.targets_per_level} collections\n", flush=True)

    def update(self, ee_position: np.ndarray, dt: float) -> list[ChallengeTarget]:
        """Update challenge state, return newly collected targets."""
        return self.update_with_touch_points(ee_position, dt, touch_points=None)

    def update_with_touch_points(
        self,
        ee_position: np.ndarray,
        dt: float,
        touch_points: list[np.ndarray] | None,
    ) -> list[ChallengeTarget]:
        """Update challenge state with optional gripper touch proxies."""
        now = time.monotonic()
        collected_this_frame: list[ChallengeTarget] = []
        candidate_points = touch_points or [ee_position]

        # Compute velocity and jerk for smoothness tracking
        ee_vel = np.zeros(3)
        ee_jerk = 0.0
        if self.last_ee_pos is not None and dt > 0:
            ee_vel = (ee_position - self.last_ee_pos) / dt
            if self.last_ee_vel is not None:
                ee_accel = (ee_vel - self.last_ee_vel) / dt
                ee_jerk = float(np.linalg.norm(ee_accel))

        # Update path tracking for all active targets
        for target in self.active_targets:
            if target.direct_distance is None:
                target.direct_distance = float(np.linalg.norm(ee_position - target.position))

            if self.last_ee_pos is not None:
                target.path_length += float(np.linalg.norm(ee_position - self.last_ee_pos))

            target.approach_velocities.append(float(np.linalg.norm(ee_vel)))
            if ee_jerk > 0:
                target.approach_jerks.append(ee_jerk)

        # Check for collections
        for target in self.active_targets[:]:  # Copy list for safe removal
            target_half_extents = np.full(3, self.target_size, dtype=float)
            distances = [
                self.box_to_box_distance(
                    point,
                    self.gripper_touch_half_extents,
                    target.position,
                    target_half_extents,
                )
                for point in candidate_points
            ]
            box_error = min(distances)
            if any(
                self.boxes_overlap(
                    point,
                    self.gripper_touch_half_extents,
                    target.position,
                    target_half_extents,
                )
                for point in candidate_points
            ):
                target.collected = True
                target.collect_time = now
                target.final_error = box_error
                self.active_targets.remove(target)
                self.collected_targets.append(target)
                collected_this_frame.append(target)
                self.level_collected += 1

                # Print collection info
                ttc = target.time_to_collect()
                eff = target.path_efficiency()
                jerk = target.mean_jerk()
                print(f"\n  Target {target.target_id} ({target.label}) collected!", flush=True)
                eff_str = f"{eff:.0%}" if eff is not None else "N/A"
                err_str = f"{target.final_error * 1000:.1f}mm" if target.final_error else "N/A"
                ttc_str = f"{ttc:.1f}s" if ttc is not None else "N/A"
                print(f"    Time: {ttc_str} | Efficiency: {eff_str} | Error: {err_str}", flush=True)
                if jerk is not None:
                    print(f"    Mean jerk: {jerk:.1f} m/s³", flush=True)

        if self.layout == "diagnostic":
            self.last_ee_pos = ee_position.copy()
            self.last_ee_vel = ee_vel.copy()
            self.last_update_time = now
            return collected_this_frame

        # Level up check
        if self.level_collected >= self.targets_per_level:
            self.level_collected = 0
            if self.current_level < self.max_targets:
                self.current_level += 1
                print(f"\n  Level up! Now spawning {self.current_level} targets at a time", flush=True)

        # Spawn replacements
        targets_needed = self.current_level - len(self.active_targets)
        if targets_needed > 0:
            self.spawn_targets(targets_needed)

        self.last_ee_pos = ee_position.copy()
        self.last_ee_vel = ee_vel.copy()
        self.last_update_time = now

        return collected_this_frame

    def draw_targets(self, viewer) -> None:
        """Draw active targets in the viewer."""
        use_markers = hasattr(viewer, "add_marker")
        use_user_scn = hasattr(viewer, "user_scn")

        if not use_markers and not use_user_scn:
            return

        scene = viewer.user_scn if use_user_scn else None
        mat = np.eye(3).flatten()

        for i, target in enumerate(self.active_targets):
            color = self.TARGET_COLORS[target.target_id % len(self.TARGET_COLORS)]
            size = self.target_size

            if use_markers:
                viewer.add_marker(
                    pos=target.position,
                    size=[size, size, size],
                    rgba=color,
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                )
            elif scene is not None and scene.ngeom < scene.maxgeom:
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_BOX,
                    np.array([size, size, size]),
                    target.position,
                    mat,
                    np.array(color),
                )
                scene.ngeom += 1

    def print_summary(self) -> None:
        """Print end-of-session summary."""
        if not self.collected_targets:
            print("\n=== CHALLENGE SUMMARY ===", flush=True)
            print("No targets collected.", flush=True)
            return

        times = [t.time_to_collect() for t in self.collected_targets if t.time_to_collect() is not None]
        efficiencies = [
            t.path_efficiency() for t in self.collected_targets if t.path_efficiency() is not None
        ]
        errors = [t.final_error for t in self.collected_targets if t.final_error is not None]
        jerks = [t.mean_jerk() for t in self.collected_targets if t.mean_jerk() is not None]

        print("\n" + "=" * 50, flush=True)
        print("CHALLENGE SUMMARY", flush=True)
        print("=" * 50, flush=True)
        print(f"Targets collected: {len(self.collected_targets)}", flush=True)
        print(f"Final level: {self.current_level}", flush=True)

        if times:
            print("\nTime to collect:", flush=True)
            print(
                f"  Mean: {np.mean(times):.1f}s | Min: {np.min(times):.1f}s | Max: {np.max(times):.1f}s",
                flush=True,
            )

        if efficiencies:
            print("\nPath efficiency (direct/actual):", flush=True)
            print(
                f"  Mean: {np.mean(efficiencies):.0%} | Min: {np.min(efficiencies):.0%} | Max: {np.max(efficiencies):.0%}",
                flush=True,
            )

        if errors:
            errors_mm = [e * 1000 for e in errors]
            print("\nFinal position error:", flush=True)
            print(
                f"  Mean: {np.mean(errors_mm):.1f}mm | Min: {np.min(errors_mm):.1f}mm | Max: {np.max(errors_mm):.1f}mm",
                flush=True,
            )

        if jerks:
            print("\nMotion smoothness (mean jerk, lower=smoother):", flush=True)
            print(
                f"  Mean: {np.mean(jerks):.1f} m/s³ | Min: {np.min(jerks):.1f} | Max: {np.max(jerks):.1f}",
                flush=True,
            )

        # Overall score (lower is better)
        if times and efficiencies and errors:
            # Normalized score: time penalty + inefficiency penalty + error penalty
            time_score = np.mean(times) / 5.0  # 5s = 1.0
            eff_score = 1.0 - np.mean(efficiencies)  # 100% efficiency = 0.0
            err_score = np.mean(errors) * 100  # 1cm = 1.0
            total_score = time_score + eff_score + err_score
            print(f"\nOverall score (lower is better): {total_score:.2f}", flush=True)
        print("=" * 50 + "\n", flush=True)


def sample_workspace_points(
    kinematics,
    limits_deg: dict[str, tuple[float, float]],
    samples: int = 2000,
    seed: int | None = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    joint_ranges: list[tuple[float, float]] = []
    for name in IK_JOINT_NAMES:
        if name in limits_deg:
            joint_ranges.append(limits_deg[name])
        else:
            joint_ranges.append((-180.0, 180.0))
    points = []
    for _ in range(max(samples, 1)):
        joint_pos = np.array([rng.uniform(low, high) for low, high in joint_ranges])
        ee_pose = kinematics.forward_kinematics(joint_pos)
        points.append(ee_pose[:3, 3].copy())
    return np.array(points)


def workspace_bbox_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return mins, maxs


def workspace_hull_edges(points: np.ndarray) -> list[tuple[int, int]] | None:
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        print("WARNING: SciPy not installed; convex hull disabled.", flush=True)
        return None
    if points.shape[0] < 4:
        return None
    hull = ConvexHull(points)
    edges: set[tuple[int, int]] = set()
    for simplex in hull.simplices:
        i0, i1, i2 = (int(s) for s in simplex)
        for a, b in ((i0, i1), (i1, i2), (i0, i2)):
            if a > b:
                a, b = b, a
            edges.add((a, b))
    return list(edges)


def draw_workspace(
    viewer: mujoco.viewer.Handle,
    points: np.ndarray | None,
    bbox: tuple[np.ndarray, np.ndarray] | None,
    hull_edges: list[tuple[int, int]] | None,
    mode: str,
    point_max: int = 1200,
) -> bool:
    use_markers = hasattr(viewer, "add_marker")
    use_user_scn = hasattr(viewer, "user_scn")
    if not use_markers and not use_user_scn:
        return False

    scene = viewer.user_scn if use_user_scn else None
    max_geoms = int(getattr(scene, "maxgeom", 0)) if scene is not None else 0
    if scene is not None:
        scene.ngeom = 0
        mat = np.eye(3).flatten()

    def add_point(p: np.ndarray, radius: float, rgba: tuple[float, float, float, float]) -> None:
        if use_markers:
            viewer.add_marker(
                pos=p,
                size=[radius, radius, radius],
                rgba=rgba,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
            )
            return
        if scene is None or scene.ngeom >= max_geoms:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius]),
            p,
            mat,
            np.array(rgba),
        )
        scene.ngeom += 1

    mode = mode.lower()
    if points is not None and mode in ("points", "both", "all"):
        if len(points) > point_max:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(points), size=point_max, replace=False)
            sample = points[idx]
        else:
            sample = points
        rgba = (0.1, 0.6, 0.9, 0.35)
        radius = 0.003
        for p in sample:
            add_point(p, radius, rgba)

    if bbox is not None and mode in ("bbox", "both", "all"):
        mins, maxs = bbox
        corners = np.array(
            [
                [mins[0], mins[1], mins[2]],
                [maxs[0], mins[1], mins[2]],
                [maxs[0], maxs[1], mins[2]],
                [mins[0], maxs[1], mins[2]],
                [mins[0], mins[1], maxs[2]],
                [maxs[0], mins[1], maxs[2]],
                [maxs[0], maxs[1], maxs[2]],
                [mins[0], maxs[1], maxs[2]],
            ]
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        rgba = (0.9, 0.2, 0.2, 0.8)
        radius = 0.004
        spacing = 0.02
        for a, b in edges:
            p0 = corners[a]
            p1 = corners[b]
            length = float(np.linalg.norm(p1 - p0))
            steps = max(int(length / spacing), 1)
            for i in range(steps + 1):
                t = i / steps
                add_point(p0 + t * (p1 - p0), radius, rgba)

    if points is not None and hull_edges is not None and mode in ("hull", "all"):
        edges = hull_edges
        if len(edges) > point_max:
            rng = np.random.default_rng(1)
            idx = rng.choice(len(edges), size=point_max, replace=False)
            edges = [edges[i] for i in idx]
        rgba = (0.2, 0.9, 0.2, 0.7)
        radius = 0.0035
        spacing = 0.03
        for a, b in edges:
            p0 = points[a]
            p1 = points[b]
            length = float(np.linalg.norm(p1 - p0))
            steps = max(int(length / spacing), 1)
            for i in range(steps + 1):
                t = i / steps
                add_point(p0 + t * (p1 - p0), radius, rgba)
    return True


class _HeadlessViewer:
    """Minimal stand-in for mujoco.viewer.Handle when running headless.

    Allows the same viewer-gated loop to run without an open window.
    Marker drawing (user_scn, add_marker) is silently skipped because
    the ``hasattr`` guards inside the loop return False.
    """

    def is_running(self) -> bool:
        """Always True — loop is terminated by the SIGINT ``running`` flag."""
        return True

    def sync(self) -> None:
        """No-op — no window to sync."""

    def __enter__(self) -> "_HeadlessViewer":
        return self

    def __exit__(self, *args) -> None:
        pass


def run_with_controller(
    sim: MuJoCoSimulator,
    deadzone: float = 0.15,
    linear_scale: float | None = None,
    mode: str = "crane",
    controller_type: str = "xbox",
    keyboard_grab: bool = False,
    keyboard_record: str | None = None,
    keyboard_playback: str | None = None,
    debug_ik: bool = False,
    debug_ik_every: int = 10,
    ik_log_path: str | None = None,
    ik_max_err_mm: float = 30.0,
    ik_mean_err_mm: float = 10.0,
    routine_trace: bool = False,
    routine_trace_max: int = 300,
    routine_trace_step_mm: float = 2.0,
    workspace_draw: bool = False,
    workspace_mode: str = "bbox",
    workspace_samples: int = 2000,
    workspace_point_max: int = 1200,
    workspace_seed: int | None = 0,
    challenge_mode: bool = False,
    challenge_collect_radius: float = 0.025,
    challenge_initial_targets: int = 1,
    challenge_targets_per_level: int = 5,
    challenge_max_targets: int = 3,
    challenge_seed: int | None = None,
    challenge_layout: str = "random",
    camera_view: str = "front_right",
    headless: bool = False,
    benchmark: bool = False,
    rerun_mode: str | None = None,
    rerun_addr: str = "0.0.0.0:9876",
    rerun_save: str = "session.rrd",
) -> int:
    """Run with Xbox or Joy-Con controller and MuJoCo viewer (or headless physics loop)."""
    from xbox_soarm_teleop.config.modes import ControlMode

    runtime = build_control_runtime(
        controller_type=controller_type,
        mode=mode,
        deadzone=deadzone,
        linear_scale=linear_scale,
        keyboard_grab=keyboard_grab,
        keyboard_record=keyboard_record,
        keyboard_playback=keyboard_playback,
        loop_dt=LOOP_PERIOD,
        urdf_path=str(URDF_PATH),
        keyboard_focus_target="this window",
    )
    control_mode = runtime.control_mode
    print(f"Control mode: {control_mode.value.upper()}", flush=True)
    kinematics = runtime.kinematics
    controller = runtime.controller
    processor = runtime.processor
    mapper = runtime.mapper
    _proc_cfg = runtime.processor_config
    controller_cfg = runtime.controller_config

    print(f"Controller deadzone: {getattr(controller_cfg, 'deadzone', 'n/a')}", flush=True)
    print(f"Linear scale: {_proc_cfg.linear_scale} m/s", flush=True)
    orientation_enabled = bool(getattr(mapper, "enable_pitch", False) or getattr(mapper, "enable_yaw", False))
    if orientation_enabled:
        print(f"Orientation scale: {_proc_cfg.orientation_scale} rad/s", flush=True)
    else:
        print("Orientation controls: OFF (touch mode)", flush=True)
    if not headless:
        print(
            f"Camera view: {camera_view} | cycle with [ and ] | ESC for free camera",
            flush=True,
        )

    # Joint velocity limits for IK joints (4 joints, no wrist_roll)
    ik_joint_vel_limits = IK_JOINT_VEL_LIMITS_ARRAY

    if not controller.connect():
        print(f"ERROR: Failed to connect to {runtime.controller_label}")
        if controller_type == "keyboard":
            try:
                import evdev

                names = [
                    f"  {p}: {evdev.InputDevice(p).name}"
                    for p in evdev.list_devices()
                    if evdev.ecodes.EV_KEY in evdev.InputDevice(p).capabilities()
                ]
                print("  Keyboard devices with EV_KEY:")
                for n in names or ["  (none found)"]:
                    print(n)
            except ImportError:
                print("  evdev not installed — run: uv pip install evdev")
            print("  Check 'input' group: groups $USER | grep input")
            print("  Add with: sudo usermod -aG input $USER  (then re-login)")
        elif controller_type in {"joycon", "dual_joycon"}:
            try:
                import evdev

                names = []
                for path in evdev.list_devices():
                    try:
                        d = evdev.InputDevice(path)
                        names.append(f"  {path}: {d.name}")
                        d.close()
                    except (OSError, PermissionError):
                        pass
                if names:
                    print("  Input devices found:")
                    for n in names:
                        print(n)
                else:
                    print("  No input devices found (check permissions: sudo usermod -aG input $USER)")
            except ImportError:
                print("  evdev not installed — run: uv pip install evdev")
            if controller_type == "joycon":
                print("  Joy-Con setup: bluetooth connect + press SL+SR for single-controller mode")
            else:
                print("  Connect both Joy-Cons and keep the right Joy-Con awake for IMU input")
            print("  joycond must be running: systemctl is-active joycond")
        else:
            print("  - Check that controller is connected")
        print("  - Or use --no-controller for demo mode")
        sys.exit(1)

    print(f"{runtime.controller_label} connected", flush=True)
    print_controls(controller_type, mode, exit_hint="Close window              exit")
    if sim.has_stack_scene() or sim.has_pick_place_scene():
        print("  Y button                  reset scene", flush=True)


    # Initialize challenge mode if enabled
    challenge: ChallengeManager | None = None
    stack_challenge: StackChallenge | None = None
    pick_place_task: PickPlaceTask | None = None
    if challenge_mode and challenge_layout == "stack":
        if not sim.has_stack_scene():
            print("ERROR: stack challenge requested without stack scene.", flush=True)
            return 2
        stack_challenge = StackChallenge(sim)
        stack_challenge.start()
    elif challenge_mode:
        limits = parse_joint_limits(URDF_PATH, IK_JOINT_NAMES)
        limits_deg = limits_rad_to_deg(limits)
        challenge = ChallengeManager(
            kinematics=kinematics,
            joint_limits_deg=limits_deg,
            collect_radius=challenge_collect_radius,
            initial_targets=challenge_initial_targets,
            targets_per_level=challenge_targets_per_level,
            max_targets=challenge_max_targets,
            seed=challenge_seed,
            layout=challenge_layout,
        )
        challenge.start()
    if sim.has_pick_place_scene():
        pick_place_task = PickPlaceTask(sim)
        pick_place_task.start()

    # IK joint positions (4 joints: base, shoulder_lift, elbow_flex, wrist_flex)
    ik_joint_pos_deg = np.zeros(4)
    cartesian_state = make_cartesian_state(kinematics, ik_joint_pos_deg) if kinematics is not None else None
    gripper_rate = runtime.gripper_rate  # Position change per second
    trace_points: list[np.ndarray] = []
    trace_min_step_m = max(routine_trace_step_mm / 1000.0, 0.0005)
    homing_active = False

    running = True
    loop_counter = 0
    error_count = 0
    error_sum = 0.0
    error_max = 0.0
    error_start = time.monotonic()
    last_scene_reset_time = 0.0

    csv_file = None
    csv_writer = None
    if ik_log_path:
        csv_file = open(ik_log_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "t_s",
                "target_x",
                "target_y",
                "target_z",
                "actual_x",
                "actual_y",
                "actual_z",
                "pos_err_mm",
            ]
        )
        print(f"IK error logging: {ik_log_path}", flush=True)

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    from xbox_soarm_teleop.diagnostics.benchmark import LoopTimer

    bm_timer = LoopTimer(mode=mode) if benchmark else None

    from xbox_soarm_teleop.diagnostics.rerun_logger import RerunLogger

    rerun_logger: RerunLogger | None = None
    if rerun_mode is not None:
        rerun_logger = RerunLogger(
            app_id="xbox_soarm_teleop_sim",
            mode=rerun_mode,
            addr=rerun_addr,
            rrd_path=rerun_save,
        )

    if headless:
        import os

        # Set MUJOCO_GL for any sub-processes or late renderer init.
        # The viewer is skipped entirely in headless mode, so the physics
        # loop runs without any display requirement.
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "osmesa"
        print("Headless mode: viewer disabled (physics-only loop).", flush=True)

    workspace_points = None
    workspace_bbox = None
    workspace_edges = None
    if workspace_draw:
        limits = parse_joint_limits(URDF_PATH, IK_JOINT_NAMES)
        limits_deg = limits_rad_to_deg(limits)
        workspace_points = sample_workspace_points(
            kinematics,
            limits_deg,
            samples=workspace_samples,
            seed=workspace_seed,
        )
        workspace_bbox = workspace_bbox_from_points(workspace_points)
        if workspace_mode in ("hull", "all"):
            workspace_edges = workspace_hull_edges(workspace_points)

    def draw_trace(
        viewer: mujoco.viewer.Handle, points: list[np.ndarray], reset_scene: bool = True
    ) -> None:
        if not routine_trace:
            return
        rgba = (0.1, 0.9, 0.1, 1.0)
        radius = 0.003
        if hasattr(viewer, "add_marker"):
            for p in points:
                viewer.add_marker(
                    pos=p,
                    size=[radius, radius, radius],
                    rgba=rgba,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                )
            return
        if not hasattr(viewer, "user_scn"):
            return
        scene = viewer.user_scn
        if reset_scene:
            scene.ngeom = 0
        max_geoms = int(getattr(scene, "maxgeom", 0))
        mat = np.eye(3).flatten()
        for p in points:
            if scene.ngeom >= max_geoms:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, radius, radius]),
                p,
                mat,
                np.array(rgba),
            )
            scene.ngeom += 1

    camera_presets = available_camera_presets()
    initial_camera_name = camera_view if camera_view in camera_presets else None
    active_camera_name = initial_camera_name
    if not headless and camera_view not in camera_presets:
        print(
            f"WARNING: Camera preset '{camera_view}' not found. Available: "
            f"{', '.join(camera_presets) if camera_presets else '(none)'}",
            flush=True,
        )

    def key_callback(keycode: int) -> None:
        nonlocal active_camera_name
        if not camera_presets:
            return
        if keycode == 256:  # ESC
            active_camera_name = None
            return
        if keycode not in (ord("["), ord("]")):
            return
        if active_camera_name is None:
            next_index = 0 if keycode == ord("]") else len(camera_presets) - 1
        else:
            current_index = camera_presets.index(active_camera_name)
            step = 1 if keycode == ord("]") else -1
            next_index = (current_index + step) % len(camera_presets)
        active_camera_name = camera_presets[next_index]
        print(f"\nCamera: {active_camera_name}", flush=True)

    # Launch viewer (or use headless stub)
    _viewer_ctx = (
        _HeadlessViewer()
        if headless
        else mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key_callback)
    )
    with _viewer_ctx as viewer:
        if not headless:
            if active_camera_name is None:
                set_viewer_free_camera(viewer)
            else:
                set_viewer_camera(viewer, active_camera_name)
        workspace_drawn = False
        while viewer.is_running() and running:
            if not headless:
                if active_camera_name is None:
                    set_viewer_free_camera(viewer)
                else:
                    set_viewer_camera(viewer, active_camera_name)
            redraw_workspace = workspace_draw and (
                not workspace_drawn
                or (routine_trace and not hasattr(viewer, "add_marker") and hasattr(viewer, "user_scn"))
            )
            if redraw_workspace:
                if not draw_workspace(
                    viewer,
                    workspace_points,
                    workspace_bbox,
                    workspace_edges,
                    workspace_mode,
                    workspace_point_max,
                ):
                    print("WARNING: Workspace drawing not supported in this viewer.", flush=True)
                workspace_drawn = True
            loop_start = time.monotonic()
            if bm_timer is not None:
                bm_timer.start_frame()
            controller_ms = ik_ms = servo_ms = 0.0

            _t0 = time.perf_counter()
            state = controller.read()
            controller_ms = (time.perf_counter() - _t0) * 1000.0

            reset_requested = state.y_button or state.y_button_pressed
            now = time.monotonic()
            if reset_requested and now - last_scene_reset_time > 0.5:
                if reset_active_scene(
                    sim,
                    control_mode=control_mode,
                    processor=processor,
                    mapper=mapper,
                    cartesian_state=cartesian_state,
                    kinematics=kinematics,
                    stack_challenge=stack_challenge,
                    pick_place_task=pick_place_task,
                    trace_points=trace_points,
                ):
                    homing_active = False
                    last_scene_reset_time = now

            if control_mode in (ControlMode.JOINT, ControlMode.CRANE, ControlMode.PUPPET):
                joint_cmd = processor(state)
                positions_deg = np.array(
                    [joint_cmd.goals_deg[name] for name in JOINT_NAMES[:-1]], dtype=float
                )
                sim.set_joint_targets(positions_deg)
                g_lower, g_upper = JOINT_LIMITS_DEG["gripper"]
                g_deg = joint_cmd.goals_deg["gripper"]
                gripper_norm = float(
                    np.clip((g_deg - g_lower) / max(g_upper - g_lower, 1e-6), 0.0, 1.0)
                )
                sim.set_gripper(gripper_norm)
                _t0 = time.perf_counter()
                sim.step()
                pos = sim.get_ee_position()
                if stack_challenge is not None:
                    moved_now = stack_challenge.update()
                    for cube_name in moved_now:
                        print(f"\n  {cube_name} moved!", flush=True)
                if pick_place_task is not None:
                    for event in pick_place_task.update(gripper_norm):
                        print(f"\n  {event}", flush=True)
                viewer.sync()
                servo_ms = (time.perf_counter() - _t0) * 1000.0
                if bm_timer is not None:
                    bm_timer.record(loop_counter, controller_ms, 0.0, servo_ms)
                if rerun_logger is not None:
                    rerun_logger.log_frame(
                        loop_counter,
                        time.monotonic() - error_start,
                        joint_cmd.goals_deg,
                        ee_pos=pos,
                        mode=mode,
                    )
                if stack_challenge is not None:
                    print(
                        f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                        f"{stack_challenge.status_text()} | Grip: {gripper_norm:.2f}   ",
                        end="\r",
                        flush=True,
                    )
                elif pick_place_task is not None:
                    print(
                        f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                        f"{pick_place_task.status_text()} | Grip: {gripper_norm:.2f}   ",
                        end="\r",
                        flush=True,
                    )
                elapsed = time.monotonic() - loop_start
                if elapsed < LOOP_PERIOD:
                    time.sleep(LOOP_PERIOD - elapsed)
                loop_counter += 1
                continue

            if state.a_button_pressed:
                print("\nGoing home...", flush=True)
                homing_active = True

            ee_delta = mapper(state)
            ee_delta = apply_axis_mapping(ee_delta, swap_xy=False)
            assert cartesian_state is not None
            if homing_active:
                homing_active = not step_cartesian_home(
                    cartesian_state,
                    kinematics,
                    np.zeros(4, dtype=float),
                    home_wrist_roll_deg=0.0,
                    home_gripper_pos=0.0,
                    ik_joint_max_step_deg=ik_joint_vel_limits,
                    wrist_roll_vel_deg_s=90.0,
                    gripper_rate=gripper_rate,
                    dt=LOOP_PERIOD,
                )
                ee_delta = EEDelta(gripper=cartesian_state.gripper_pos)
            else:
                cartesian_state.gripper_pos = step_gripper_toward(
                    cartesian_state.gripper_pos,
                    ee_delta.gripper,
                    gripper_rate=gripper_rate,
                    dt=LOOP_PERIOD,
                )

            if not homing_active and not ee_delta.is_zero_motion():
                target_pose, target_pos, target_flags = advance_cartesian_target(
                    cartesian_state,
                    ee_delta,
                    dt=LOOP_PERIOD,
                    clip_position=lambda pos: (
                        np.array(
                            [
                                np.clip(pos[0], 0.05, 0.5),
                                pos[1],
                                np.clip(pos[2], 0.05, 0.45),
                            ],
                            dtype=float,
                        ),
                        {},
                    ),
                    pitch_limit_rad=np.pi / 2,
                    yaw_limit_rad=np.pi,
                )
                orientation_weight = float(target_flags["orientation_weight"])

                # Solve IK for 4 joints
                _t0 = time.perf_counter()
                new_joints = kinematics.inverse_kinematics(
                    cartesian_state.ik_joint_pos_deg,
                    target_pose,
                    position_weight=1.0,
                    orientation_weight=orientation_weight,
                )
                ik_ms = (time.perf_counter() - _t0) * 1000.0
                ik_result = new_joints[:4]

                # Apply joint velocity limiting to smooth IK output
                max_delta = ik_joint_vel_limits * LOOP_PERIOD
                joint_delta = ik_result - cartesian_state.ik_joint_pos_deg
                joint_delta = np.clip(joint_delta, -max_delta, max_delta)
                next_wrist_roll_deg = step_wrist_roll(
                    cartesian_state.wrist_roll_deg,
                    ee_delta.droll,
                    dt=LOOP_PERIOD,
                    roll_target=ee_delta.roll_target,
                )
                apply_ik_solution(
                    cartesian_state,
                    kinematics,
                    cartesian_state.ik_joint_pos_deg + joint_delta,
                    wrist_roll_deg=next_wrist_roll_deg,
                    target_pose=target_pose,
                )

                if debug_ik and (loop_counter % max(debug_ik_every, 1) == 0):
                    pos_error = np.linalg.norm(target_pose[:3, 3] - cartesian_state.ee_pose[:3, 3])
                    print(
                        f"\nIK: target=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, "
                        f"{target_pos[2]:.3f}] actual=[{cartesian_state.ee_pose[0, 3]:.3f}, "
                        f"{cartesian_state.ee_pose[1, 3]:.3f}, {cartesian_state.ee_pose[2, 3]:.3f}] "
                        f"err={pos_error * 1000.0:.1f}mm",
                        flush=True,
                    )

            # Combine base + IK joints for full 5-joint position
            full_joint_pos_deg = full_joint_positions(
                cartesian_state.ik_joint_pos_deg,
                cartesian_state.wrist_roll_deg,
            )

            # Update simulation
            sim.set_joint_targets(full_joint_pos_deg)
            sim.set_gripper(cartesian_state.gripper_pos)
            _t0 = time.perf_counter()
            sim.step()
            servo_ms = (time.perf_counter() - _t0) * 1000.0
            if bm_timer is not None:
                bm_timer.record(loop_counter, controller_ms, ik_ms, servo_ms)
            pos = sim.get_ee_position()
            if rerun_logger is not None:
                arm_joints = dict(zip(JOINT_NAMES, full_joint_pos_deg))
                rerun_logger.log_frame(
                    loop_counter,
                    time.monotonic() - error_start,
                    arm_joints,
                    ee_pos=pos,
                    mode=mode,
                )
            if routine_trace:
                if not trace_points:
                    trace_points.append(pos.copy())
                elif np.linalg.norm(pos - trace_points[-1]) >= trace_min_step_m:
                    trace_points.append(pos.copy())
                if len(trace_points) > routine_trace_max:
                    trace_points = trace_points[-routine_trace_max:]
                draw_trace(viewer, trace_points, reset_scene=not redraw_workspace)

            # Challenge mode update
            if stack_challenge is not None:
                moved_now = stack_challenge.update()
                for cube_name in moved_now:
                    print(f"\n  {cube_name} moved!", flush=True)
            if pick_place_task is not None:
                for event in pick_place_task.update(cartesian_state.gripper_pos):
                    print(f"\n  {event}", flush=True)
            if challenge is not None:
                challenge.update_with_touch_points(
                    pos,
                    LOOP_PERIOD,
                    touch_points=sim.get_gripper_touch_points(),
                )
                challenge.draw_targets(viewer)

            viewer.sync()

            # Status - show position and orientation
            pos_err = float(
                np.linalg.norm(
                    cartesian_state.last_ee_pose[:3, 3] - cartesian_state.last_target_pose[:3, 3]
                )
            )
            error_count += 1
            error_sum += pos_err
            error_max = max(error_max, pos_err)
            if csv_writer:
                t_s = time.monotonic() - error_start
                csv_writer.writerow(
                    [
                        f"{t_s:.3f}",
                        f"{cartesian_state.last_target_pose[0, 3]:.6f}",
                        f"{cartesian_state.last_target_pose[1, 3]:.6f}",
                        f"{cartesian_state.last_target_pose[2, 3]:.6f}",
                        f"{cartesian_state.last_ee_pose[0, 3]:.6f}",
                        f"{cartesian_state.last_ee_pose[1, 3]:.6f}",
                        f"{cartesian_state.last_ee_pose[2, 3]:.6f}",
                        f"{pos_err * 1000.0:.3f}",
                    ]
                )
            pitch_deg = np.rad2deg(cartesian_state.target_pitch)
            yaw_deg = np.rad2deg(cartesian_state.target_yaw)
            if stack_challenge is not None:
                print(
                    f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                    f"{stack_challenge.status_text()} | Grip: {cartesian_state.gripper_pos:.2f}   ",
                    end="\r",
                    flush=True,
                )
            elif pick_place_task is not None:
                print(
                    f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                    f"{pick_place_task.status_text()} | Grip: {cartesian_state.gripper_pos:.2f}   ",
                    end="\r",
                    flush=True,
                )
            elif challenge is not None:
                n_targets = len(challenge.active_targets)
                n_collected = len(challenge.collected_targets)
                print(
                    f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                    f"Targets: {n_targets} | Collected: {n_collected} | Grip: {cartesian_state.gripper_pos:.2f}   ",
                    end="\r",
                    flush=True,
                )
            else:
                print(
                    f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                    f"P:{pitch_deg:+5.1f}° Y:{yaw_deg:+5.1f}° | Grip: {cartesian_state.gripper_pos:.2f}   ",
                    end="\r",
                    flush=True,
                )

            loop_counter += 1
            elapsed = time.monotonic() - loop_start
            if elapsed < LOOP_PERIOD:
                time.sleep(LOOP_PERIOD - elapsed)

    controller.disconnect()
    if rerun_logger is not None:
        rerun_logger.close()
    if csv_file:
        csv_file.close()
    if bm_timer is not None and bm_timer.frame_count > 0:
        bm_path = LoopTimer.default_path("benchmark_sim")
        bm_timer.write_csv(bm_path)
        print(f"\nBenchmark data written to: {bm_path}", flush=True)
        print(bm_timer.summary(), flush=True)
    print("\nDisconnected.", flush=True)

    # Print challenge summary if in challenge mode
    if stack_challenge is not None:
        stack_challenge.print_summary()
    if pick_place_task is not None:
        pick_place_task.print_summary()
    if challenge is not None:
        challenge.print_summary()

    if error_count == 0:
        print("IK error summary: no samples collected.")
        return 0

    mean_err_mm = (error_sum / error_count) * 1000.0
    max_err_mm = error_max * 1000.0
    print("IK error summary (kinematics FK)")
    print(f"  samples: {error_count}")
    print(f"  max position error: {max_err_mm:.1f} mm")
    print(f"  mean position error: {mean_err_mm:.1f} mm")

    if max_err_mm > ik_max_err_mm or mean_err_mm > ik_mean_err_mm:
        print("FAIL: IK error exceeded thresholds.")
        return 1
    print("PASS")
    return 0


def run_demo_mode(
    sim: MuJoCoSimulator,
    routine_pattern: str = "lissajous",
    routine_plane: str = "xy",
    routine_square_size: float = 0.06,
    routine_square_speed: float = 0.03,
    routine_center_x: float = 0.0,
    routine_center_y: float = 0.0,
    routine_center_z: float = 0.0,
    routine_duration: float = 0.0,
    routine_scale: float = 1.0,
    routine_trace: bool = False,
    routine_trace_max: int = 300,
    routine_trace_step_mm: float = 2.0,
    workspace_draw: bool = False,
    workspace_mode: str = "bbox",
    workspace_samples: int = 2000,
    workspace_point_max: int = 1200,
    workspace_seed: int | None = 0,
    camera_view: str = "front_right",
    ik_log_path: str | None = None,
    ik_max_err_mm: float = 30.0,
    ik_mean_err_mm: float = 10.0,
) -> int:
    """Run demo mode with automatic movement."""
    from lerobot.model.kinematics import RobotKinematics

    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=IK_JOINT_NAMES,
    )

    print("\nDemo mode - automatic movement", flush=True)
    print("Close window to exit\n", flush=True)

    cartesian_state = make_cartesian_state(kinematics, np.zeros(4, dtype=float))
    ik_joint_vel_limits = IK_JOINT_VEL_LIMITS_ARRAY
    t = 0.0
    center_pos = cartesian_state.ee_pose[:3, 3].copy()
    center_pos += np.array([routine_center_x, routine_center_y, routine_center_z], dtype=float)
    trace_points: list[np.ndarray] = []
    trace_min_step_m = max(routine_trace_step_mm / 1000.0, 0.0005)

    def demo_target_offset(t_s: float) -> np.ndarray:
        if routine_pattern == "lissajous":
            return routine_scale * np.array(
                [
                    0.03 * np.sin(t_s * 0.5),
                    0.03 * np.cos(t_s * 0.5),
                    0.02 * np.sin(t_s * 0.3),
                ]
            )
        size = routine_square_size * routine_scale
        if routine_pattern == "square-xyz":
            period = max((4.0 * size) / max(routine_square_speed, 0.001), 0.1)
            phase = t_s / period
            plane_idx = int(phase) % 3
            u = phase - int(phase)
            plane = ["xy", "xz", "yz"][plane_idx]
            return plane_offset(plane, u, size)
        period = max((4.0 * size) / max(routine_square_speed, 0.001), 0.1)
        u = (t_s / period) % 1.0
        return plane_offset(routine_plane, u, size)

    def draw_trace(
        viewer: mujoco.viewer.Handle, points: list[np.ndarray], reset_scene: bool = True
    ) -> None:
        if not routine_trace:
            return
        rgba = (0.1, 0.9, 0.1, 1.0)
        radius = 0.003
        if hasattr(viewer, "add_marker"):
            for p in points:
                viewer.add_marker(
                    pos=p,
                    size=[radius, radius, radius],
                    rgba=rgba,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                )
            return
        if not hasattr(viewer, "user_scn"):
            return
        scene = viewer.user_scn
        if reset_scene:
            scene.ngeom = 0
        max_geoms = int(getattr(scene, "maxgeom", 0))
        mat = np.eye(3).flatten()
        for p in points:
            if scene.ngeom >= max_geoms:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, radius, radius]),
                p,
                mat,
                np.array(rgba),
            )
            scene.ngeom += 1

    running = True
    error_count = 0
    error_sum = 0.0
    error_max = 0.0
    error_start = time.monotonic()

    csv_file = None
    csv_writer = None
    if ik_log_path:
        csv_file = open(ik_log_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "t_s",
                "target_x",
                "target_y",
                "target_z",
                "actual_x",
                "actual_y",
                "actual_z",
                "pos_err_mm",
            ]
        )
        print(f"IK error logging: {ik_log_path}", flush=True)

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    workspace_points = None
    workspace_bbox = None
    workspace_edges = None
    if workspace_draw:
        limits = parse_joint_limits(URDF_PATH, IK_JOINT_NAMES)
        limits_deg = limits_rad_to_deg(limits)
        workspace_points = sample_workspace_points(
            kinematics,
            limits_deg,
            samples=workspace_samples,
            seed=workspace_seed,
        )
        workspace_bbox = workspace_bbox_from_points(workspace_points)
        if workspace_mode in ("hull", "all"):
            workspace_edges = workspace_hull_edges(workspace_points)

    camera_presets = available_camera_presets()
    active_camera_name = camera_view if camera_view in camera_presets else None
    if camera_view not in camera_presets:
        print(
            f"WARNING: Camera preset '{camera_view}' not found. Available: "
            f"{', '.join(camera_presets) if camera_presets else '(none)'}",
            flush=True,
        )

    def key_callback(keycode: int) -> None:
        nonlocal active_camera_name
        if not camera_presets:
            return
        if keycode == 256:  # ESC
            active_camera_name = None
            return
        if keycode not in (ord("["), ord("]")):
            return
        if active_camera_name is None:
            next_index = 0 if keycode == ord("]") else len(camera_presets) - 1
        else:
            current_index = camera_presets.index(active_camera_name)
            step = 1 if keycode == ord("]") else -1
            next_index = (current_index + step) % len(camera_presets)
        active_camera_name = camera_presets[next_index]
        print(f"\nCamera: {active_camera_name}", flush=True)

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key_callback) as viewer:
        if active_camera_name is None:
            set_viewer_free_camera(viewer)
        else:
            set_viewer_camera(viewer, active_camera_name)
        workspace_drawn = False
        while viewer.is_running() and running:
            if active_camera_name is None:
                set_viewer_free_camera(viewer)
            else:
                set_viewer_camera(viewer, active_camera_name)
            redraw_workspace = workspace_draw and (
                not workspace_drawn
                or (routine_trace and not hasattr(viewer, "add_marker") and hasattr(viewer, "user_scn"))
            )
            if redraw_workspace:
                if not draw_workspace(
                    viewer,
                    workspace_points,
                    workspace_bbox,
                    workspace_edges,
                    workspace_mode,
                    workspace_point_max,
                ):
                    print("WARNING: Workspace drawing not supported in this viewer.", flush=True)
                workspace_drawn = True
            loop_start = time.monotonic()

            if routine_duration > 0.0 and t >= routine_duration:
                break

            # Routine pattern
            offset = demo_target_offset(t)
            target_pos = center_pos + offset
            if routine_pattern == "lissajous":
                droll = 0.1 * np.sin(t * 0.4)
                gripper = 0.5 + 0.5 * np.sin(t * 0.2)
            else:
                droll = 0.0
                gripper = 0.0

            target_pos[0] = np.clip(target_pos[0], -0.1, 0.5)
            target_pos[1] = np.clip(target_pos[1], -0.3, 0.3)
            target_pos[2] = np.clip(target_pos[2], 0.05, 0.45)

            target_pose = cartesian_state.ee_pose.copy()
            target_pose[:3, 3] = target_pos

            # Solve IK for position only (4 joints)
            new_joints = kinematics.inverse_kinematics(
                cartesian_state.ik_joint_pos_deg,
                target_pose,
                position_weight=1.0,
                orientation_weight=0.0,
            )
            ik_result = new_joints[:4]

            # Apply joint velocity limiting to smooth IK output
            max_delta = ik_joint_vel_limits * LOOP_PERIOD
            joint_delta = ik_result - cartesian_state.ik_joint_pos_deg
            joint_delta = np.clip(joint_delta, -max_delta, max_delta)
            next_wrist_roll_deg = step_wrist_roll(
                cartesian_state.wrist_roll_deg,
                droll,
                dt=LOOP_PERIOD,
            )
            apply_ik_solution(
                cartesian_state,
                kinematics,
                cartesian_state.ik_joint_pos_deg + joint_delta,
                wrist_roll_deg=next_wrist_roll_deg,
                target_pose=target_pose,
            )
            full_joint_pos_deg = full_joint_positions(
                cartesian_state.ik_joint_pos_deg,
                cartesian_state.wrist_roll_deg,
            )

            sim.set_joint_targets(full_joint_pos_deg)
            sim.set_gripper(gripper)
            sim.step()
            pos = sim.get_ee_position()
            if routine_trace:
                if not trace_points:
                    trace_points.append(pos.copy())
                else:
                    if np.linalg.norm(pos - trace_points[-1]) >= trace_min_step_m:
                        trace_points.append(pos.copy())
                if len(trace_points) > routine_trace_max:
                    trace_points = trace_points[-routine_trace_max:]
                draw_trace(viewer, trace_points, reset_scene=not redraw_workspace)
            viewer.sync()

            pos_err = float(
                np.linalg.norm(
                    cartesian_state.last_ee_pose[:3, 3] - cartesian_state.last_target_pose[:3, 3]
                )
            )
            error_count += 1
            error_sum += pos_err
            error_max = max(error_max, pos_err)
            if csv_writer:
                t_s = time.monotonic() - error_start
                csv_writer.writerow(
                    [
                        f"{t_s:.3f}",
                        f"{cartesian_state.last_target_pose[0, 3]:.6f}",
                        f"{cartesian_state.last_target_pose[1, 3]:.6f}",
                        f"{cartesian_state.last_target_pose[2, 3]:.6f}",
                        f"{cartesian_state.last_ee_pose[0, 3]:.6f}",
                        f"{cartesian_state.last_ee_pose[1, 3]:.6f}",
                        f"{cartesian_state.last_ee_pose[2, 3]:.6f}",
                        f"{pos_err * 1000.0:.3f}",
                    ]
                )
            print(
                f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | Gripper: {gripper:.2f}   ",
                end="\r",
                flush=True,
            )

            t += LOOP_PERIOD

            elapsed = time.monotonic() - loop_start
            if elapsed < LOOP_PERIOD:
                time.sleep(LOOP_PERIOD - elapsed)

    if csv_file:
        csv_file.close()

    print("\nDone.", flush=True)
    if error_count == 0:
        print("IK error summary: no samples collected.")
        return 0

    mean_err_mm = (error_sum / error_count) * 1000.0
    max_err_mm = error_max * 1000.0
    print("IK error summary (kinematics FK)")
    print(f"  samples: {error_count}")
    print(f"  max position error: {max_err_mm:.1f} mm")
    print(f"  mean position error: {mean_err_mm:.1f} mm")

    if max_err_mm > ik_max_err_mm or mean_err_mm > ik_mean_err_mm:
        print("FAIL: IK error exceeded thresholds.")
        return 1
    print("PASS")
    return 0


def load_calibration_fractions(calibration_path: Path, joint_names: list[str]) -> dict[str, float]:
    if not calibration_path.exists():
        raise FileNotFoundError(calibration_path)
    data = json.loads(calibration_path.read_text())
    fractions: dict[str, float] = {}
    full = 4095.0
    for name in joint_names:
        if name not in data:
            continue
        entry = data[name]
        width = float(entry["range_max"]) - float(entry["range_min"])
        fractions[name] = max(min(width / full, 1.0), 0.05)
    return fractions


def find_default_calibration() -> Path | None:
    base = Path.home() / ".cache/huggingface/lerobot/calibration/robots"
    if not base.exists():
        return None
    candidates = list(base.rglob("*.json"))
    return candidates[0] if candidates else None


def apply_calibration_to_limits(
    limits: dict[str, tuple[float, float]],
    fractions: dict[str, float],
) -> dict[str, tuple[float, float]]:
    effective: dict[str, tuple[float, float]] = {}
    for name, (lower, upper) in limits.items():
        frac = fractions.get(name, 1.0)
        center = 0.5 * (lower + upper)
        span = (upper - lower) * frac
        effective[name] = (center - 0.5 * span, center + 0.5 * span)
    return effective


def square_points(plane: str, size: float, samples_per_side: int) -> list[np.ndarray]:
    half = size / 2.0
    samples = max(samples_per_side, 2)
    t = np.linspace(0.0, 1.0, samples, endpoint=False)
    points: list[np.ndarray] = []
    for u in t:
        s = u * 4.0
        seg = int(s)
        f = s - seg
        if seg == 0:
            a, b = -half + f * size, -half
        elif seg == 1:
            a, b = half, -half + f * size
        elif seg == 2:
            a, b = half - f * size, half
        else:
            a, b = -half, half - f * size
        if plane == "xy":
            points.append(np.array([a, b, 0.0]))
        elif plane == "xz":
            points.append(np.array([a, 0.0, b]))
        else:
            points.append(np.array([0.0, a, b]))
    return points


def estimate_max_square_size(
    kinematics,
    joint_limits: dict[str, tuple[float, float]],
    center_pos: np.ndarray,
    plane: str,
    max_size: float,
    samples_per_side: int,
    max_err_mm: float,
    iterations: int,
    verbose: bool,
) -> float:
    lower_bound = 0.0
    upper_bound = max_size
    max_err_m = max_err_mm / 1000.0
    joint_names = IK_JOINT_NAMES

    def square_ok(size: float) -> bool:
        points = square_points(plane, size, samples_per_side)
        guess = np.zeros(len(joint_names))
        for offset in points:
            target_pose = kinematics.forward_kinematics(guess)
            target_pose[:3, 3] = center_pos + offset
            try:
                solved = kinematics.inverse_kinematics(
                    guess, target_pose, position_weight=1.0, orientation_weight=0.0
                )
            except Exception:
                return False
            solved = np.array(solved[: len(joint_names)])
            if np.any(~np.isfinite(solved)):
                return False
            for idx, name in enumerate(joint_names):
                if name in joint_limits:
                    low, high = joint_limits[name]
                    if solved[idx] < low or solved[idx] > high:
                        return False
            fk_pose = kinematics.forward_kinematics(solved)
            err = np.linalg.norm(fk_pose[:3, 3] - (center_pos + offset))
            if err > max_err_m:
                return False
            guess = solved
        return True

    for i in range(iterations):
        mid = 0.5 * (lower_bound + upper_bound)
        ok = square_ok(mid)
        if verbose:
            status = "ok" if ok else "fail"
            print(f"  iter {i + 1:02d}: size={mid:.4f}m -> {status}", flush=True)
        if ok:
            lower_bound = mid
        else:
            upper_bound = mid
    return lower_bound


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo SO-ARM101 simulation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cartesian", "joint", "crane", "puppet"],
        default="crane",
        help="Control mode: crane (cylindrical, default), joint (direct per-joint), cartesian (full IK), puppet (IMU wrist).",
    )
    parser.add_argument(
        "--controller",
        choices=["xbox", "joycon", "dual_joycon", "keyboard"],
        default="xbox",
        help="Controller type: xbox, joycon, dual_joycon, or keyboard. Default: xbox.",
    )
    parser.add_argument(
        "--keyboard-grab",
        action="store_true",
        help="Grab keyboard device exclusively (keyboard mode only). "
        "Prevents keypresses from reaching other windows when focus changes.",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        default=None,
        help="Save keystroke recording to PATH when Tab is pressed (keyboard mode only). "
        "Defaults to recording_<timestamp>.json if Tab is pressed without this flag.",
    )
    parser.add_argument(
        "--playback",
        metavar="PATH",
        default=None,
        help="Replay a saved keystroke recording instead of live keyboard input.",
    )
    parser.add_argument("--no-controller", action="store_true", help="Run demo mode")
    parser.add_argument(
        "--motion-routine",
        action="store_true",
        help="Run automatic motion routine (alias for --no-controller).",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Run benchmark challenge mode with targets to collect.",
    )
    parser.add_argument(
        "--challenge-collect-radius",
        type=float,
        default=0.03,
        help="Distance to target for collection (m). Default: 0.03.",
    )
    parser.add_argument(
        "--challenge-initial-targets",
        type=int,
        default=1,
        help="Number of targets to start with. Default: 1.",
    )
    parser.add_argument(
        "--challenge-targets-per-level",
        type=int,
        default=5,
        help="Collections before difficulty increases. Default: 5.",
    )
    parser.add_argument(
        "--challenge-max-targets",
        type=int,
        default=3,
        help="Maximum simultaneous targets. Default: 3.",
    )
    parser.add_argument(
        "--challenge-seed",
        type=int,
        default=None,
        help="Random seed for target placement. Default: random.",
    )
    parser.add_argument(
        "--challenge-layout",
        choices=["random", "diagnostic", "stack"],
        default="random",
        help="Challenge target layout. 'diagnostic' spawns fixed targets in main test directions. 'stack' loads a physical cube stack.",
    )
    parser.add_argument(
        "--camera-view",
        choices=CAMERA_PRESET_NAMES,
        default="front_right",
        help="Initial viewer camera preset. Use [ and ] to cycle, ESC for free camera.",
    )
    parser.add_argument(
        "--scene",
        choices=["default", "pick_place_basic"],
        default="default",
        help="Physical scene to load. 'pick_place_basic' adds a table, cube, and goal box.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.15,
        help="Controller deadzone (0.0-1.0). Default: 0.15",
    )
    parser.add_argument(
        "--linear-scale",
        type=float,
        default=None,
        help="Linear velocity scale (m/s). Default from config.",
    )
    parser.add_argument(
        "--debug-ik",
        action="store_true",
        help="Print IK target/achieved error periodically.",
    )
    parser.add_argument(
        "--debug-ik-every",
        type=int,
        default=10,
        help="Print IK debug every N control loops. Default: 10.",
    )
    parser.add_argument(
        "--ik-log",
        type=str,
        default=None,
        help="Write IK position error CSV to this path.",
    )
    parser.add_argument(
        "--ik-max-err-mm",
        type=float,
        default=30.0,
        help="Fail if max position error exceeds this (mm). Default: 30.",
    )
    parser.add_argument(
        "--ik-mean-err-mm",
        type=float,
        default=10.0,
        help="Fail if mean position error exceeds this (mm). Default: 10.",
    )
    parser.add_argument(
        "--routine-pattern",
        "--demo-pattern",
        type=str,
        default="lissajous",
        choices=["lissajous", "square", "square-xyz"],
        help="Routine pattern when --no-controller. Default: lissajous.",
    )
    parser.add_argument(
        "--routine-plane",
        "--demo-square-plane",
        type=str,
        default="xy",
        choices=["xy", "xz", "yz"],
        help="Plane for square routine. Default: xy.",
    )
    parser.add_argument(
        "--routine-square-size",
        "--demo-square-size",
        type=float,
        default=0.06,
        help="Square side length in meters. Default: 0.06.",
    )
    parser.add_argument(
        "--routine-square-speed",
        type=float,
        default=0.03,
        help="Max square edge speed (m/s). Default: 0.03.",
    )
    parser.add_argument(
        "--routine-square-period",
        "--demo-square-period",
        type=float,
        default=None,
        help="Seconds per square trace (overrides square speed if set).",
    )
    parser.add_argument(
        "--routine-center-x",
        type=float,
        default=0.0,
        help="Additive offset to routine center X (meters). Default: 0.",
    )
    parser.add_argument(
        "--routine-center-y",
        type=float,
        default=0.0,
        help="Additive offset to routine center Y (meters). Default: 0.",
    )
    parser.add_argument(
        "--routine-center-z",
        type=float,
        default=0.0,
        help="Additive offset to routine center Z (meters). Default: 0.",
    )
    parser.add_argument(
        "--routine-duration",
        type=float,
        default=0.0,
        help="Auto-stop after this many seconds (0 = run until close). Default: 0.",
    )
    parser.add_argument(
        "--routine-scale",
        type=float,
        default=1.0,
        help="Scale factor for routine motion. Default: 1.0.",
    )
    parser.add_argument(
        "--routine-trace",
        "--demo-trace",
        action="store_true",
        help="Draw a virtual pen trace of the end effector.",
    )
    parser.add_argument(
        "--routine-trace-max",
        "--demo-trace-max",
        type=int,
        default=300,
        help="Max trace points to draw. Default: 300.",
    )
    parser.add_argument(
        "--routine-trace-step-mm",
        "--demo-trace-step-mm",
        type=float,
        default=2.0,
        help="Minimum distance between trace points (mm). Default: 2.0.",
    )
    parser.add_argument(
        "--workspace-draw",
        action="store_true",
        help="Draw an approximate reachable workspace in the viewer.",
    )
    parser.add_argument(
        "--workspace-mode",
        type=str,
        default="bbox",
        choices=["bbox", "points", "both", "hull", "all"],
        help="Workspace draw mode. Default: bbox.",
    )
    parser.add_argument(
        "--workspace-samples",
        type=int,
        default=2000,
        help="Number of random joint samples for workspace. Default: 2000.",
    )
    parser.add_argument(
        "--workspace-point-max",
        type=int,
        default=1200,
        help="Max workspace points to draw. Default: 1200.",
    )
    parser.add_argument(
        "--workspace-seed",
        type=int,
        default=0,
        help="Random seed for workspace sampling (-1 for random). Default: 0.",
    )
    parser.add_argument(
        "--estimate-max-square",
        action="store_true",
        help="Estimate max square size per plane (kinematics-only) and exit.",
    )
    parser.add_argument(
        "--estimate-max-square-plane",
        type=str,
        default="all",
        choices=["xy", "xz", "yz", "all"],
        help="Plane to estimate (xy/xz/yz/all). Default: all.",
    )
    parser.add_argument(
        "--estimate-max-square-max",
        type=float,
        default=0.18,
        help="Upper bound for square size search (m). Default: 0.18.",
    )
    parser.add_argument(
        "--estimate-max-square-samples",
        type=int,
        default=24,
        help="Samples per square side for feasibility check. Default: 24.",
    )
    parser.add_argument(
        "--estimate-max-square-iterations",
        type=int,
        default=10,
        help="Binary search iterations. Default: 10.",
    )
    parser.add_argument(
        "--estimate-max-square-err-mm",
        type=float,
        default=8.0,
        help="Max FK error allowed per point (mm). Default: 8.0.",
    )
    parser.add_argument(
        "--estimate-use-calibration",
        action="store_true",
        help="Shrink URDF joint limits based on calibration range (conservative).",
    )
    parser.add_argument(
        "--estimate-calibration-path",
        type=str,
        default=None,
        help="Path to calibration JSON. Default: auto-detect.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run without a display window (headless physics loop). "
            "Sets MUJOCO_GL=osmesa unless already set. "
            "Requires libosmesa6 on Linux: apt install libosmesa6-dev"
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Log per-frame timing data (controller read, IK solve, sim step). "
            "Writes benchmark_sim_<timestamp>.csv on exit."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Stream joint angles and EE position to a local Rerun viewer (requires rerun-sdk).",
    )
    parser.add_argument(
        "--rerun-mode",
        choices=["spawn", "serve", "connect", "save"],
        default=None,
        help="Rerun connection mode. --rerun is shorthand for --rerun-mode spawn.",
    )
    parser.add_argument(
        "--rerun-addr",
        default="0.0.0.0:9876",
        help="gRPC address for --rerun-mode serve/connect. Default: 0.0.0.0:9876.",
    )
    parser.add_argument(
        "--rerun-save",
        default="session.rrd",
        help="Output .rrd file path for --rerun-mode save. Default: session.rrd.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found at {URDF_PATH}")
        sys.exit(1)

    if args.estimate_max_square:
        from lerobot.model.kinematics import RobotKinematics

        kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name="gripper_frame_link",
            joint_names=IK_JOINT_NAMES,
        )
        limits = limits_rad_to_deg(parse_joint_limits(URDF_PATH, IK_JOINT_NAMES))
        if args.estimate_use_calibration:
            cal_path = (
                Path(args.estimate_calibration_path)
                if args.estimate_calibration_path
                else find_default_calibration()
            )
            if cal_path is None:
                print("ERROR: No calibration JSON found.")
                sys.exit(2)
            fractions = load_calibration_fractions(cal_path, IK_JOINT_NAMES)
            limits = apply_calibration_to_limits(limits, fractions)
            print(f"Using calibration limits: {cal_path}", flush=True)

        center = kinematics.forward_kinematics(np.zeros(len(IK_JOINT_NAMES)))[:3, 3].copy()
        planes = (
            ["xy", "xz", "yz"]
            if args.estimate_max_square_plane == "all"
            else [args.estimate_max_square_plane]
        )
        print("Max square estimate (kinematics-only)")
        print(f"  max_size search: {args.estimate_max_square_max:.3f} m")
        print(f"  samples/side: {args.estimate_max_square_samples}")
        print(f"  max_err: {args.estimate_max_square_err_mm:.1f} mm")
        for plane in planes:
            size = estimate_max_square_size(
                kinematics=kinematics,
                joint_limits=limits,
                center_pos=center,
                plane=plane,
                max_size=args.estimate_max_square_max,
                samples_per_side=args.estimate_max_square_samples,
                max_err_mm=args.estimate_max_square_err_mm,
                iterations=args.estimate_max_square_iterations,
                verbose=False,
            )
            print(f"  {plane}: {size:.3f} m")
        return

    if args.scene != "default" and args.challenge and args.challenge_layout == "stack":
        print("ERROR: --scene pick_place_basic cannot be combined with --challenge-layout stack.")
        sys.exit(1)

    scene = "stack" if args.challenge and args.challenge_layout == "stack" else args.scene
    print("Loading MuJoCo model...", flush=True)
    sim = MuJoCoSimulator(str(URDF_PATH), scene=scene)
    print("Model loaded!", flush=True)

    if args.routine_square_period is not None:
        if args.routine_square_period <= 0:
            print("ERROR: --routine-square-period must be > 0.")
            sys.exit(1)
        perimeter = 4.0 * max(args.routine_square_size, 0.001)
        args.routine_square_speed = perimeter / args.routine_square_period

    workspace_seed = None if args.workspace_seed < 0 else args.workspace_seed

    no_controller = args.no_controller or args.motion_routine

    if no_controller:
        exit_code = run_demo_mode(
            sim,
            routine_pattern=args.routine_pattern,
            routine_plane=args.routine_plane,
            routine_square_size=args.routine_square_size,
            routine_square_speed=args.routine_square_speed,
            routine_center_x=args.routine_center_x,
            routine_center_y=args.routine_center_y,
            routine_center_z=args.routine_center_z,
            routine_duration=args.routine_duration,
            routine_scale=args.routine_scale,
            routine_trace=args.routine_trace,
            routine_trace_max=args.routine_trace_max,
            routine_trace_step_mm=args.routine_trace_step_mm,
            workspace_draw=args.workspace_draw,
            workspace_mode=args.workspace_mode,
            workspace_samples=args.workspace_samples,
            workspace_point_max=args.workspace_point_max,
            workspace_seed=workspace_seed,
            camera_view=args.camera_view,
            ik_log_path=args.ik_log,
            ik_max_err_mm=args.ik_max_err_mm,
            ik_mean_err_mm=args.ik_mean_err_mm,
        )
    else:
        exit_code = run_with_controller(
            sim,
            deadzone=args.deadzone,
            linear_scale=args.linear_scale,
            mode=args.mode,
            controller_type=args.controller,
            keyboard_grab=args.keyboard_grab,
            keyboard_record=args.record,
            keyboard_playback=args.playback,
            debug_ik=args.debug_ik,
            debug_ik_every=args.debug_ik_every,
            ik_log_path=args.ik_log,
            ik_max_err_mm=args.ik_max_err_mm,
            ik_mean_err_mm=args.ik_mean_err_mm,
            routine_trace=args.routine_trace,
            routine_trace_max=args.routine_trace_max,
            routine_trace_step_mm=args.routine_trace_step_mm,
            workspace_draw=args.workspace_draw,
            workspace_mode=args.workspace_mode,
            workspace_samples=args.workspace_samples,
            workspace_point_max=args.workspace_point_max,
            workspace_seed=workspace_seed,
            challenge_mode=args.challenge,
            challenge_collect_radius=args.challenge_collect_radius,
            challenge_initial_targets=args.challenge_initial_targets,
            challenge_targets_per_level=args.challenge_targets_per_level,
            challenge_max_targets=args.challenge_max_targets,
            challenge_seed=args.challenge_seed,
            challenge_layout=args.challenge_layout,
            camera_view=args.camera_view,
            headless=args.headless,
            benchmark=args.benchmark,
            rerun_mode=args.rerun_mode or ("spawn" if args.rerun else None),
            rerun_addr=args.rerun_addr,
            rerun_save=args.rerun_save,
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
