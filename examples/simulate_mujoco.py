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

# Path to URDF
URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"

# Joint names (order matters - matches URDF joint order)
JOINT_NAMES = JOINT_NAMES_WITH_GRIPPER

# Control loop rate
CONTROL_RATE = 50  # Hz
LOOP_PERIOD = 1.0 / CONTROL_RATE


class MuJoCoSimulator:
    """MuJoCo-based SO-ARM101 simulator."""

    def __init__(self, urdf_path: str):
        """Initialize MuJoCo simulator.

        Args:
            urdf_path: Path to robot URDF file.
        """
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)

        # Get joint indices
        self.joint_ids = {}
        for name in JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                self.joint_ids[name] = jnt_id

        # Target joint positions (radians)
        self.target_pos = np.zeros(len(JOINT_NAMES))

        # Initialize to home position
        self.go_home()

    def go_home(self) -> None:
        """Reset to home position."""
        self.target_pos = np.zeros(len(JOINT_NAMES))
        # Set qpos directly
        for i, name in enumerate(JOINT_NAMES):
            if name in self.joint_ids:
                jnt_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] = 0.0
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
        # Find gripper body/site
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_frame_link")
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        # Fallback to last body
        return self.data.xpos[-1].copy()

    def step(self) -> None:
        """Update kinematics (no physics simulation)."""
        mujoco.mj_forward(self.model, self.data)


class ChallengeTarget:
    """A single target in the challenge mode."""

    def __init__(self, position: np.ndarray, target_id: int):
        self.position = position.copy()
        self.target_id = target_id
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
    ):
        self.kinematics = kinematics
        self.joint_limits_deg = joint_limits_deg
        self.collect_radius = collect_radius
        self.target_size = target_size
        self.initial_targets = initial_targets
        self.targets_per_level = targets_per_level
        self.max_targets = max_targets
        self.workspace_margin = workspace_margin
        self.initial_ee_position = initial_ee_position

        self.rng = np.random.default_rng(seed)
        self.active_targets: list[ChallengeTarget] = []
        self.collected_targets: list[ChallengeTarget] = []
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

    def start(self) -> None:
        """Start the challenge with initial targets."""
        self.spawn_targets(self.initial_targets)
        print("\n=== CHALLENGE MODE ===", flush=True)
        print(f"Collect targets by moving gripper within {self.collect_radius * 100:.1f}cm", flush=True)
        print(f"Starting with {self.initial_targets} target(s)", flush=True)
        print(f"Difficulty increases every {self.targets_per_level} collections\n", flush=True)

    def update(self, ee_position: np.ndarray, dt: float) -> list[ChallengeTarget]:
        """Update challenge state, return newly collected targets."""
        now = time.monotonic()
        collected_this_frame: list[ChallengeTarget] = []

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
            dist = np.linalg.norm(ee_position - target.position)
            if dist <= self.collect_radius:
                target.collected = True
                target.collect_time = now
                target.final_error = dist
                self.active_targets.remove(target)
                self.collected_targets.append(target)
                collected_this_frame.append(target)
                self.level_collected += 1

                # Print collection info
                ttc = target.time_to_collect()
                eff = target.path_efficiency()
                jerk = target.mean_jerk()
                print(f"\n  Target {target.target_id} collected!", flush=True)
                eff_str = f"{eff:.0%}" if eff is not None else "N/A"
                err_str = f"{target.final_error * 1000:.1f}mm" if target.final_error else "N/A"
                ttc_str = f"{ttc:.1f}s" if ttc is not None else "N/A"
                print(f"    Time: {ttc_str} | Efficiency: {eff_str} | Error: {err_str}", flush=True)
                if jerk is not None:
                    print(f"    Mean jerk: {jerk:.1f} m/s³", flush=True)

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


def _print_controls(controller_type: str, mode: str) -> None:
    """Print control instructions appropriate for the controller and mode."""
    print("\nControls:", flush=True)

    if controller_type == "keyboard":
        if mode == "joint":
            print("  A / D           shoulder_pan    (left / right)", flush=True)
            print("  W / S           shoulder_lift   (up / down)", flush=True)
            print("  R / F           elbow_flex      (flex / extend)", flush=True)
            print("  Q / E           wrist_flex      (up / down)", flush=True)
            print("  ↑ / ↓           wrist_roll      (+ / -)", flush=True)
            print("  Space (hold)    gripper close", flush=True)
            print("  H               home position", flush=True)
            print("  1–5             speed level  (default 3 = 75%)", flush=True)
            print("  Shift           2× speed multiplier", flush=True)
        else:
            print("  W / S           forward / back  (X)", flush=True)
            print("  A / D           left / right    (Y)", flush=True)
            print("  R / F           up / down       (Z)", flush=True)
            print("  Q / E           wrist roll", flush=True)
            print("  ↑ / ↓           pitch", flush=True)
            print("  ← / →           yaw", flush=True)
            print("  Space (hold)    gripper close", flush=True)
            print("  H               home position", flush=True)
            print("  Y               toggle coord frame (world/tool)", flush=True)
            print("  1–5             speed level  (default 3 = 75%)", flush=True)
            print("  Shift           2× speed multiplier", flush=True)
        print("  Ctrl+C / close window  exit\n", flush=True)

    elif controller_type == "joycon":
        if mode == "joint":
            print("  Stick left/right    drive selected joint", flush=True)
            print("  (no joint cycle on Joy-Con — use cartesian mode)", flush=True)
        else:
            print("  Stick              move arm (X/Y/Z/roll)", flush=True)
            print("  ZR                 gripper (hold=close)", flush=True)
        print("  SL (hold)          deadman switch", flush=True)
        print("  + button           home position", flush=True)
        print("  Close window       exit\n", flush=True)

    else:  # xbox
        if mode == "joint":
            print("  Hold LB + Left stick X    drive selected joint", flush=True)
            print("  D-pad left/right          cycle joint", flush=True)
            print("  Right trigger             gripper", flush=True)
            print("  A button                  home position", flush=True)
        else:
            print("  Hold LB + move sticks     control arm", flush=True)
            print("  Left stick X/Y            left-right / up-down", flush=True)
            print("  Right stick Y/X           forward-back / wrist roll", flush=True)
            print("  D-pad up/down             pitch", flush=True)
            print("  D-pad left/right          yaw", flush=True)
            print("  Right trigger             gripper", flush=True)
            print("  A button                  home position", flush=True)
            print("  Y button                  toggle coord frame", flush=True)
        print("  Close window              exit\n", flush=True)


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
    headless: bool = False,
    benchmark: bool = False,
    rerun_mode: str | None = None,
    rerun_addr: str = "0.0.0.0:9876",
    rerun_save: str = "session.rrd",
) -> int:
    """Run with Xbox or Joy-Con controller and MuJoCo viewer (or headless physics loop)."""
    from lerobot.model.kinematics import RobotKinematics

    from xbox_soarm_teleop.config.modes import ControlMode
    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.processors.factory import make_processor
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    control_mode = ControlMode(mode)
    print(f"Control mode: {control_mode.value.upper()}", flush=True)

    # IK joint names - include base, exclude wrist_roll (controlled directly)
    ik_joint_names = IK_JOINT_NAMES

    # Initialize kinematics only for modes that need IK
    kinematics = None
    if control_mode != ControlMode.JOINT:
        kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name="gripper_frame_link",
            joint_names=ik_joint_names,
        )

    if controller_type == "joycon":
        from xbox_soarm_teleop.config.joycon_config import JoyConConfig
        from xbox_soarm_teleop.teleoperators.joycon import JoyConController

        config = JoyConConfig(deadzone=deadzone)
        if linear_scale is not None:
            config.linear_scale = linear_scale
        controller = JoyConController(config)
        _proc_cfg = config
    elif controller_type == "keyboard":
        from xbox_soarm_teleop.config.keyboard_config import KeyboardConfig
        from xbox_soarm_teleop.teleoperators.keyboard import KeyboardController

        config = KeyboardConfig(
            grab=keyboard_grab,
            record_path=keyboard_record,
            playback_path=keyboard_playback,
        )
        if linear_scale is not None:
            config.speed_levels = tuple(s * linear_scale / 0.1 for s in config.speed_levels)
        if not keyboard_grab and not keyboard_playback:
            print(
                "WARNING: keyboard controller active without --keyboard-grab. "
                "Keypresses will be detected even when this window is not focused. "
                "Use --keyboard-grab for exclusive access.",
                flush=True,
            )
        if keyboard_playback:
            print(f"Keyboard playback mode: {keyboard_playback}", flush=True)
        controller = KeyboardController(config)
        # Processor scale values come from defaults; keyboard speed is internal to the controller
        _proc_cfg = XboxConfig()
        if linear_scale is not None:
            _proc_cfg.linear_scale = linear_scale
    else:
        config = XboxConfig(deadzone=deadzone)
        if linear_scale is not None:
            config.linear_scale = linear_scale
        controller = XboxController(config)
        _proc_cfg = config
    processor = make_processor(
        control_mode,
        linear_scale=_proc_cfg.linear_scale,
        angular_scale=_proc_cfg.angular_scale,
        orientation_scale=_proc_cfg.orientation_scale,
        invert_pitch=_proc_cfg.invert_pitch,
        invert_yaw=_proc_cfg.invert_yaw,
        loop_dt=LOOP_PERIOD,
        urdf_path=str(URDF_PATH),
        multi_joint=(controller_type == "keyboard" and control_mode.value == "joint"),
    )
    mapper = processor  # alias for cartesian/crane path compatibility

    print(f"Controller deadzone: {getattr(config, 'deadzone', 'n/a')}", flush=True)
    print(f"Linear scale: {_proc_cfg.linear_scale} m/s", flush=True)
    print(f"Orientation scale: {_proc_cfg.orientation_scale} rad/s", flush=True)

    # Joint velocity limits for IK joints (4 joints, no wrist_roll)
    ik_joint_vel_limits = IK_JOINT_VEL_LIMITS_ARRAY

    if not controller.connect():
        labels = {"joycon": "Joy-Con", "keyboard": "keyboard", "xbox": "Xbox controller"}
        label = labels.get(controller_type, controller_type)
        print(f"ERROR: Failed to connect to {label}")
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
        elif controller_type == "joycon":
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
            print("  Joy-Con setup: bluetooth connect + press SL+SR for single-controller mode")
            print("  joycond must be running: systemctl is-active joycond")
        else:
            print("  - Check that controller is connected")
        print("  - Or use --no-controller for demo mode")
        sys.exit(1)

    labels = {"joycon": "Joy-Con", "keyboard": "keyboard", "xbox": "Xbox controller"}
    label = labels.get(controller_type, controller_type)
    print(f"{label} connected", flush=True)
    _print_controls(controller_type, mode)


    # Initialize challenge mode if enabled
    challenge: ChallengeManager | None = None
    if challenge_mode:
        limits = parse_joint_limits(URDF_PATH, ik_joint_names)
        limits_deg = limits_rad_to_deg(limits)
        challenge = ChallengeManager(
            kinematics=kinematics,
            joint_limits_deg=limits_deg,
            collect_radius=challenge_collect_radius,
            initial_targets=challenge_initial_targets,
            targets_per_level=challenge_targets_per_level,
            max_targets=challenge_max_targets,
            seed=challenge_seed,
        )
        challenge.start()

    # IK joint positions (4 joints: base, shoulder_lift, elbow_flex, wrist_flex)
    ik_joint_pos_deg = np.zeros(4)
    wrist_roll_deg = 0.0

    # Target orientation (euler angles in radians)
    target_pitch = 0.0
    target_yaw = 0.0

    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert euler angles (ZYX convention) to rotation matrix."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )

    # Get initial EE pose (only needed for IK-based modes)
    if kinematics is not None:
        ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
        last_target_pose = ee_pose.copy()
        last_ee_pose = ee_pose.copy()
    else:
        ee_pose = None
        last_target_pose = None
        last_ee_pose = None
    gripper_pos = 0.0  # Current gripper position (smoothed)
    gripper_rate = getattr(_proc_cfg, "gripper_rate", 2.0)  # Position change per second
    trace_points: list[np.ndarray] = []
    trace_min_step_m = max(routine_trace_step_mm / 1000.0, 0.0005)

    running = True
    loop_counter = 0
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
        limits = parse_joint_limits(URDF_PATH, ik_joint_names)
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

    # Launch viewer (or use headless stub)
    _viewer_ctx = _HeadlessViewer() if headless else mujoco.viewer.launch_passive(sim.model, sim.data)
    with _viewer_ctx as viewer:
        workspace_drawn = False
        while viewer.is_running() and running:
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

            if control_mode in (ControlMode.JOINT, ControlMode.CRANE):
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
                viewer.sync()
                servo_ms = (time.perf_counter() - _t0) * 1000.0
                if bm_timer is not None:
                    bm_timer.record(loop_counter, controller_ms, 0.0, servo_ms)
                if rerun_logger is not None:
                    rerun_logger.log_frame(
                        loop_counter,
                        time.monotonic() - error_start,
                        joint_cmd.goals_deg,
                        ee_pos=sim.get_ee_position(),
                        mode=mode,
                    )
                elapsed = time.monotonic() - loop_start
                if elapsed < LOOP_PERIOD:
                    time.sleep(LOOP_PERIOD - elapsed)
                loop_counter += 1
                continue

            if state.a_button_pressed:
                print("\nGoing home...", flush=True)
                ik_joint_pos_deg = np.zeros(4)
                wrist_roll_deg = 0.0
                target_pitch = 0.0
                target_yaw = 0.0
                ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
                last_ee_pose = ee_pose.copy()
                last_target_pose = ee_pose.copy()
                sim.go_home()
                viewer.sync()
                continue

            ee_delta = mapper(state)
            # Rate-limit gripper movement toward target
            gripper_target = ee_delta.gripper
            gripper_diff = gripper_target - gripper_pos
            max_delta = gripper_rate * LOOP_PERIOD
            if abs(gripper_diff) > max_delta:
                gripper_pos += max_delta if gripper_diff > 0 else -max_delta
            else:
                gripper_pos = gripper_target

            if not ee_delta.is_zero_motion():
                # Update target EE pose (X/Y/Z)
                target_pos = ee_pose[:3, 3].copy()
                target_pos[0] += ee_delta.dx * LOOP_PERIOD  # Forward/back in arm plane
                target_pos[1] += ee_delta.dy * LOOP_PERIOD  # Left/right
                target_pos[2] += ee_delta.dz * LOOP_PERIOD  # Up/down

                # Workspace limits (X is reach, Z is height)
                target_pos[0] = np.clip(target_pos[0], 0.05, 0.5)  # Min reach to avoid singularity
                target_pos[2] = np.clip(target_pos[2], 0.05, 0.45)

                # Update target orientation
                if abs(ee_delta.dpitch) > 0.001:
                    target_pitch += ee_delta.dpitch * LOOP_PERIOD
                    target_pitch = np.clip(target_pitch, -np.pi / 2, np.pi / 2)

                if abs(ee_delta.dyaw) > 0.001:
                    target_yaw += ee_delta.dyaw * LOOP_PERIOD
                    target_yaw = np.clip(target_yaw, -np.pi, np.pi)

                target_pose = ee_pose.copy()
                target_pose[:3, 3] = target_pos

                # Build target orientation if pitch/yaw are set
                has_orientation_target = abs(target_pitch) > 0.01 or abs(target_yaw) > 0.01
                if has_orientation_target:
                    target_rotation = euler_to_rotation_matrix(0.0, target_pitch, target_yaw)
                    target_pose[:3, :3] = target_rotation
                    orientation_weight = 0.1  # Low weight - strongly prioritize position
                else:
                    orientation_weight = 0.0

                # Solve IK for 4 joints
                _t0 = time.perf_counter()
                new_joints = kinematics.inverse_kinematics(
                    ik_joint_pos_deg,
                    target_pose,
                    position_weight=1.0,
                    orientation_weight=orientation_weight,
                )
                ik_ms = (time.perf_counter() - _t0) * 1000.0
                ik_result = new_joints[:4]

                # Apply joint velocity limiting to smooth IK output
                max_delta = ik_joint_vel_limits * LOOP_PERIOD
                joint_delta = ik_result - ik_joint_pos_deg
                joint_delta = np.clip(joint_delta, -max_delta, max_delta)
                ik_joint_pos_deg = ik_joint_pos_deg + joint_delta

                # Apply wrist roll directly (not part of IK)
                if abs(ee_delta.droll) > 0.001:
                    roll_delta_deg = np.rad2deg(ee_delta.droll * LOOP_PERIOD)
                    wrist_roll_deg += roll_delta_deg
                    wrist_roll_deg = np.clip(wrist_roll_deg, -180.0, 180.0)

                ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
                last_ee_pose = ee_pose.copy()
                last_target_pose = target_pose.copy()

                if debug_ik and (loop_counter % max(debug_ik_every, 1) == 0):
                    pos_error = np.linalg.norm(target_pose[:3, 3] - ee_pose[:3, 3])
                    print(
                        f"\nIK: target=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, "
                        f"{target_pos[2]:.3f}] actual=[{ee_pose[0, 3]:.3f}, "
                        f"{ee_pose[1, 3]:.3f}, {ee_pose[2, 3]:.3f}] "
                        f"err={pos_error * 1000.0:.1f}mm",
                        flush=True,
                    )

            # Combine base + IK joints for full 5-joint position
            # Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
            full_joint_pos_deg = np.array(
                [
                    ik_joint_pos_deg[0],  # shoulder_pan
                    ik_joint_pos_deg[1],  # shoulder_lift
                    ik_joint_pos_deg[2],  # elbow_flex
                    ik_joint_pos_deg[3],  # wrist_flex
                    wrist_roll_deg,  # wrist_roll
                ]
            )

            # Update simulation
            sim.set_joint_targets(full_joint_pos_deg)
            sim.set_gripper(gripper_pos)
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
            if challenge is not None:
                challenge.update(pos, LOOP_PERIOD)
                challenge.draw_targets(viewer)

            viewer.sync()

            # Status - show position and orientation
            pos_err = float(np.linalg.norm(last_ee_pose[:3, 3] - last_target_pose[:3, 3]))
            error_count += 1
            error_sum += pos_err
            error_max = max(error_max, pos_err)
            if csv_writer:
                t_s = time.monotonic() - error_start
                csv_writer.writerow(
                    [
                        f"{t_s:.3f}",
                        f"{last_target_pose[0, 3]:.6f}",
                        f"{last_target_pose[1, 3]:.6f}",
                        f"{last_target_pose[2, 3]:.6f}",
                        f"{last_ee_pose[0, 3]:.6f}",
                        f"{last_ee_pose[1, 3]:.6f}",
                        f"{last_ee_pose[2, 3]:.6f}",
                        f"{pos_err * 1000.0:.3f}",
                    ]
                )
            pitch_deg = np.rad2deg(target_pitch)
            yaw_deg = np.rad2deg(target_yaw)
            if challenge is not None:
                n_targets = len(challenge.active_targets)
                n_collected = len(challenge.collected_targets)
                print(
                    f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                    f"Targets: {n_targets} | Collected: {n_collected} | Grip: {gripper_pos:.2f}   ",
                    end="\r",
                    flush=True,
                )
            else:
                print(
                    f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                    f"P:{pitch_deg:+5.1f}° Y:{yaw_deg:+5.1f}° | Grip: {gripper_pos:.2f}   ",
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

    wrist_roll_deg = 0.0
    ik_joint_pos_deg = np.zeros(4)
    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
    last_target_pose = ee_pose.copy()
    last_ee_pose = ee_pose.copy()
    ik_joint_vel_limits = IK_JOINT_VEL_LIMITS_ARRAY
    t = 0.0
    center_pos = ee_pose[:3, 3].copy()
    center_pos += np.array([routine_center_x, routine_center_y, routine_center_z], dtype=float)
    trace_points: list[np.ndarray] = []
    trace_min_step_m = max(routine_trace_step_mm / 1000.0, 0.0005)

    def square_offset(u: float, size: float) -> tuple[float, float]:
        half = size / 2.0
        s = (u % 1.0) * 4.0
        seg = int(s)
        f = s - seg
        if seg == 0:
            return (-half + f * size, -half)
        if seg == 1:
            return (half, -half + f * size)
        if seg == 2:
            return (half - f * size, half)
        return (-half, half - f * size)

    def plane_offset(plane: str, u: float, size: float) -> np.ndarray:
        a, b = square_offset(u, size)
        if plane == "xy":
            return np.array([a, b, 0.0])
        if plane == "xz":
            return np.array([a, 0.0, b])
        return np.array([0.0, a, b])

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

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        workspace_drawn = False
        while viewer.is_running() and running:
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

            target_pose = ee_pose.copy()
            target_pose[:3, 3] = target_pos

            # Solve IK for position only (4 joints)
            new_joints = kinematics.inverse_kinematics(
                ik_joint_pos_deg, target_pose, position_weight=1.0, orientation_weight=0.0
            )
            ik_result = new_joints[:4]

            # Apply joint velocity limiting to smooth IK output
            max_delta = ik_joint_vel_limits * LOOP_PERIOD
            joint_delta = ik_result - ik_joint_pos_deg
            joint_delta = np.clip(joint_delta, -max_delta, max_delta)
            ik_joint_pos_deg = ik_joint_pos_deg + joint_delta

            # Apply wrist roll directly
            if abs(droll) > 0.001:
                roll_delta_deg = np.rad2deg(droll * LOOP_PERIOD)
                wrist_roll_deg += roll_delta_deg
                wrist_roll_deg = np.clip(wrist_roll_deg, -180.0, 180.0)

            ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
            last_ee_pose = ee_pose.copy()
            last_target_pose = target_pose.copy()

            full_joint_pos_deg = np.array(
                [
                    ik_joint_pos_deg[0],
                    ik_joint_pos_deg[1],
                    ik_joint_pos_deg[2],
                    ik_joint_pos_deg[3],
                    wrist_roll_deg,
                ]
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

            pos_err = float(np.linalg.norm(last_ee_pose[:3, 3] - last_target_pose[:3, 3]))
            error_count += 1
            error_sum += pos_err
            error_max = max(error_max, pos_err)
            if csv_writer:
                t_s = time.monotonic() - error_start
                csv_writer.writerow(
                    [
                        f"{t_s:.3f}",
                        f"{last_target_pose[0, 3]:.6f}",
                        f"{last_target_pose[1, 3]:.6f}",
                        f"{last_target_pose[2, 3]:.6f}",
                        f"{last_ee_pose[0, 3]:.6f}",
                        f"{last_ee_pose[1, 3]:.6f}",
                        f"{last_ee_pose[2, 3]:.6f}",
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


def main():
    parser = argparse.ArgumentParser(description="MuJoCo SO-ARM101 simulation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cartesian", "joint", "crane"],
        default="crane",
        help="Control mode: crane (cylindrical, default), joint (direct per-joint), cartesian (full IK).",
    )
    parser.add_argument(
        "--controller",
        choices=["xbox", "joycon", "keyboard"],
        default="xbox",
        help="Controller type: xbox, joycon, or keyboard. Default: xbox.",
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
    args = parser.parse_args()

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

    print("Loading MuJoCo model...", flush=True)
    sim = MuJoCoSimulator(str(URDF_PATH))
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
            headless=args.headless,
            benchmark=args.benchmark,
            rerun_mode=args.rerun_mode or ("spawn" if args.rerun else None),
            rerun_addr=args.rerun_addr,
            rerun_save=args.rerun_save,
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
