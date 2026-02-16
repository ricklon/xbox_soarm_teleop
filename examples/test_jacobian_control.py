#!/usr/bin/env python3
"""Compare Jacobian-based vs IK-based control in MuJoCo simulation.

This script allows A/B testing between:
- IK mode: Integrate target pose, solve IK each cycle
- Jacobian mode: Compute joint velocities from EE velocity, integrate

Usage:
    uv run python examples/test_jacobian_control.py
    uv run python examples/test_jacobian_control.py --mode jacobian
    uv run python examples/test_jacobian_control.py --mode compare

Controls:
    - Standard Xbox controls (hold LB to move)
    - J key: Toggle between IK and Jacobian mode (in compare mode)
    - M key: Toggle metrics overlay
    - A button: Return to home position
    - Close window or Ctrl+C: Exit

Modes:
    --mode ik         Use IK-based control only
    --mode jacobian   Use Jacobian-based control only
    --mode compare    Start with IK, press J to toggle
"""

import argparse
import signal
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES, JOINT_NAMES_WITH_GRIPPER
from xbox_soarm_teleop.kinematics.jacobian import JacobianController

# Path to URDF
URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"

# Joint names
JOINT_NAMES = JOINT_NAMES_WITH_GRIPPER

# Control loop rate
CONTROL_RATE = 50  # Hz
LOOP_PERIOD = 1.0 / CONTROL_RATE


class ControlMetrics:
    """Track and report control loop metrics."""

    def __init__(self, history_size: int = 100):
        self.loop_times_ms = deque(maxlen=history_size)
        self.ik_times_ms = deque(maxlen=history_size)
        self.jacobian_times_ms = deque(maxlen=history_size)
        self.position_errors_mm = deque(maxlen=history_size)
        self.manipulability = deque(maxlen=history_size)
        self.near_singularity_count = 0
        self.total_cycles = 0

    def record_cycle(
        self,
        loop_time_ms: float,
        compute_time_ms: float,
        position_error_mm: float,
        manipulability: float,
        near_singularity: bool,
        mode: str,
    ):
        self.loop_times_ms.append(loop_time_ms)
        if mode == "jacobian":
            self.jacobian_times_ms.append(compute_time_ms)
        else:
            self.ik_times_ms.append(compute_time_ms)
        self.position_errors_mm.append(position_error_mm)
        self.manipulability.append(manipulability)
        if near_singularity:
            self.near_singularity_count += 1
        self.total_cycles += 1

    def get_summary(self, mode: str) -> dict:
        return {
            "mode": mode,
            "loop_time_mean_ms": np.mean(self.loop_times_ms) if self.loop_times_ms else 0,
            "loop_time_max_ms": np.max(self.loop_times_ms) if self.loop_times_ms else 0,
            "ik_time_mean_ms": np.mean(self.ik_times_ms) if self.ik_times_ms else 0,
            "jacobian_time_mean_ms": np.mean(self.jacobian_times_ms) if self.jacobian_times_ms else 0,
            "position_error_mean_mm": np.mean(self.position_errors_mm) if self.position_errors_mm else 0,
            "position_error_max_mm": np.max(self.position_errors_mm) if self.position_errors_mm else 0,
            "manipulability_mean": np.mean(self.manipulability) if self.manipulability else 0,
            "manipulability_min": np.min(self.manipulability) if self.manipulability else 0,
            "near_singularity_pct": 100 * self.near_singularity_count / max(self.total_cycles, 1),
        }

    def format_status(self, mode: str, current_manip: float, near_sing: bool) -> str:
        s = self.get_summary(mode)
        sing_str = "YES" if near_sing else "no"
        compute_time = s["jacobian_time_mean_ms"] if mode == "jacobian" else s["ik_time_mean_ms"]
        return (
            f"Mode: {mode.upper():8s} | "
            f"Compute: {compute_time:5.2f}ms | "
            f"Manip: {current_manip:.4f} | "
            f"Singularity: {sing_str}"
        )


class MuJoCoSimulator:
    """MuJoCo-based SO-ARM101 simulator."""

    def __init__(self, urdf_path: str):
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)

        self.joint_ids = {}
        for name in JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                self.joint_ids[name] = jnt_id

        self.go_home()

    def go_home(self) -> None:
        for name in JOINT_NAMES:
            if name in self.joint_ids:
                jnt_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_joint_targets(self, positions_deg: np.ndarray) -> None:
        positions_rad = np.deg2rad(positions_deg)
        for i, name in enumerate(JOINT_NAMES):
            if i < len(positions_rad) and name in self.joint_ids:
                jnt_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] = positions_rad[i]

    def set_gripper(self, position: float) -> None:
        if "gripper" in self.joint_ids:
            jnt_id = self.joint_ids["gripper"]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            gripper_open = 1.74533
            gripper_closed = -0.174533
            self.data.qpos[qpos_adr] = gripper_open - position * (gripper_open - gripper_closed)

    def get_ee_position(self) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_frame_link")
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return self.data.xpos[-1].copy()

    def step(self) -> None:
        mujoco.mj_forward(self.model, self.data)


def run_comparison(
    mode: str = "compare",
    deadzone: float = 0.15,
    linear_scale: float | None = None,
    damping: float = 0.05,
    show_metrics: bool = True,
):
    """Run the comparison test."""
    from lerobot.model.kinematics import RobotKinematics

    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    # Initialize kinematics
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=IK_JOINT_NAMES,
    )

    # Initialize Jacobian controller
    jacobian_ctrl = JacobianController(kinematics, damping=damping)

    # Initialize MuJoCo
    sim = MuJoCoSimulator(str(URDF_PATH))

    # Initialize Xbox controller
    config = XboxConfig(deadzone=deadzone)
    if linear_scale is not None:
        config.linear_scale = linear_scale
    controller = XboxController(config)
    mapper = MapXboxToEEDelta(
        linear_scale=config.linear_scale,
        angular_scale=config.angular_scale,
        orientation_scale=config.orientation_scale,
        invert_pitch=config.invert_pitch,
        invert_yaw=config.invert_yaw,
    )

    # Joint velocity limits for IK mode
    ik_joint_vel_limits = np.array([120.0, 90.0, 90.0, 90.0])  # deg/s

    # Joint limits for both modes
    joint_limits_deg = np.array([180.0, 90.0, 90.0, 90.0])  # symmetric limits

    if not controller.connect():
        print("ERROR: Failed to connect to Xbox controller")
        sys.exit(1)

    print("Xbox controller connected", flush=True)
    print(f"\nControl mode: {mode.upper()}", flush=True)
    print(f"Damping factor: {damping}", flush=True)
    print("\nControls:", flush=True)
    print("  Hold LB + move sticks to control arm", flush=True)
    if mode == "compare":
        print("  Press J to toggle between IK and Jacobian mode", flush=True)
    print("  Press M to toggle metrics display", flush=True)
    print("  A button to go home", flush=True)
    print("  Close window to exit\n", flush=True)

    # State
    current_mode = "ik" if mode in ("ik", "compare") else "jacobian"
    ik_joint_pos_deg = np.zeros(4)
    wrist_roll_deg = 0.0
    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
    gripper_pos = 0.0
    metrics = ControlMetrics()
    metrics_visible = show_metrics

    running = True
    mode_toggle_cooldown = 0.0

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Launch viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running() and running:
            loop_start = time.monotonic()

            state = controller.read()

            # Check for mode toggle (J key via Y button for now)
            if mode == "compare" and time.monotonic() > mode_toggle_cooldown:
                if state.y_button_pressed:
                    current_mode = "jacobian" if current_mode == "ik" else "ik"
                    print(f"\n  Switched to {current_mode.upper()} mode", flush=True)
                    mode_toggle_cooldown = time.monotonic() + 0.5

            # Home button
            if state.a_button_pressed:
                print("\nGoing home...", flush=True)
                ik_joint_pos_deg = np.zeros(4)
                wrist_roll_deg = 0.0
                ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
                sim.go_home()
                viewer.sync()
                continue

            ee_delta = mapper(state)

            # Rate-limit gripper
            gripper_target = ee_delta.gripper
            gripper_diff = gripper_target - gripper_pos
            max_gripper_delta = config.gripper_rate * LOOP_PERIOD
            if abs(gripper_diff) > max_gripper_delta:
                gripper_pos += max_gripper_delta if gripper_diff > 0 else -max_gripper_delta
            else:
                gripper_pos = gripper_target

            # Compute time tracking
            compute_start = time.monotonic()
            position_error_mm = 0.0

            if not ee_delta.is_zero_motion():
                if current_mode == "jacobian":
                    # Jacobian-based control: EE velocity -> joint velocity -> integrate
                    # Use position + pitch mode (4 DOF) for optimal control with 4 joints
                    ee_vel = np.array([ee_delta.dx, ee_delta.dy, ee_delta.dz, ee_delta.dpitch])

                    # Get joint velocities
                    joint_vel_deg = jacobian_ctrl.ee_vel_to_joint_vel(
                        ee_vel, ik_joint_pos_deg, mode="position_pitch"
                    )

                    # Apply velocity limits
                    joint_vel_deg = np.clip(joint_vel_deg, -ik_joint_vel_limits, ik_joint_vel_limits)

                    # Integrate to get new joint positions
                    ik_joint_pos_deg = ik_joint_pos_deg + joint_vel_deg * LOOP_PERIOD

                    # Apply joint limits
                    ik_joint_pos_deg = np.clip(ik_joint_pos_deg, -joint_limits_deg, joint_limits_deg)

                    # Update FK for display
                    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)

                else:
                    # IK-based control: integrate target pose, solve IK
                    target_pos = ee_pose[:3, 3].copy()
                    target_pos[0] += ee_delta.dx * LOOP_PERIOD
                    target_pos[1] += ee_delta.dy * LOOP_PERIOD
                    target_pos[2] += ee_delta.dz * LOOP_PERIOD

                    # Workspace limits
                    target_pos[0] = np.clip(target_pos[0], 0.05, 0.5)
                    target_pos[2] = np.clip(target_pos[2], 0.05, 0.45)

                    target_pose = ee_pose.copy()
                    target_pose[:3, 3] = target_pos

                    # Solve IK
                    new_joints = kinematics.inverse_kinematics(
                        ik_joint_pos_deg,
                        target_pose,
                        position_weight=1.0,
                        orientation_weight=0.0,
                    )
                    ik_result = new_joints[:4]

                    # Apply velocity limiting
                    max_delta = ik_joint_vel_limits * LOOP_PERIOD
                    joint_delta = ik_result - ik_joint_pos_deg
                    joint_delta = np.clip(joint_delta, -max_delta, max_delta)
                    ik_joint_pos_deg = ik_joint_pos_deg + joint_delta

                    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)

                    # Track position error (IK target vs achieved)
                    position_error_mm = np.linalg.norm(target_pose[:3, 3] - ee_pose[:3, 3]) * 1000

                # Apply wrist roll directly
                if abs(ee_delta.droll) > 0.001:
                    roll_delta_deg = np.rad2deg(ee_delta.droll * LOOP_PERIOD)
                    wrist_roll_deg += roll_delta_deg
                    wrist_roll_deg = np.clip(wrist_roll_deg, -180.0, 180.0)

            compute_time_ms = (time.monotonic() - compute_start) * 1000

            # Get manipulability
            manip = jacobian_ctrl.manipulability(ik_joint_pos_deg)
            near_sing = jacobian_ctrl.is_near_singularity(ik_joint_pos_deg)

            # Update simulation
            full_joint_pos_deg = np.array([
                ik_joint_pos_deg[0],
                ik_joint_pos_deg[1],
                ik_joint_pos_deg[2],
                ik_joint_pos_deg[3],
                wrist_roll_deg,
            ])
            sim.set_joint_targets(full_joint_pos_deg)
            sim.set_gripper(gripper_pos)
            sim.step()
            viewer.sync()

            # Record metrics
            loop_time_ms = (time.monotonic() - loop_start) * 1000
            metrics.record_cycle(
                loop_time_ms, compute_time_ms, position_error_mm, manip, near_sing, current_mode
            )

            # Status display
            pos = sim.get_ee_position()
            if metrics_visible:
                status = metrics.format_status(current_mode, manip, near_sing)
                print(f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | {status}   ", end="\r", flush=True)
            else:
                print(f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | Grip: {gripper_pos:.2f}   ", end="\r", flush=True)

            # Maintain loop rate
            elapsed = time.monotonic() - loop_start
            if elapsed < LOOP_PERIOD:
                time.sleep(LOOP_PERIOD - elapsed)

    controller.disconnect()
    print("\n\nDisconnected.", flush=True)

    # Print final summary
    summary = metrics.get_summary(current_mode)
    print("\n" + "=" * 60, flush=True)
    print("SESSION SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total cycles: {metrics.total_cycles}", flush=True)
    print(f"Loop time: {summary['loop_time_mean_ms']:.2f}ms mean, {summary['loop_time_max_ms']:.2f}ms max", flush=True)
    if summary["ik_time_mean_ms"] > 0:
        print(f"IK compute time: {summary['ik_time_mean_ms']:.3f}ms mean", flush=True)
    if summary["jacobian_time_mean_ms"] > 0:
        print(f"Jacobian compute time: {summary['jacobian_time_mean_ms']:.3f}ms mean", flush=True)
    print(f"Position error: {summary['position_error_mean_mm']:.2f}mm mean, {summary['position_error_max_mm']:.2f}mm max", flush=True)
    print(f"Manipulability: {summary['manipulability_mean']:.4f} mean, {summary['manipulability_min']:.4f} min", flush=True)
    print(f"Near singularity: {summary['near_singularity_pct']:.1f}% of cycles", flush=True)
    print("=" * 60, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Compare Jacobian vs IK control")
    parser.add_argument(
        "--mode",
        type=str,
        default="compare",
        choices=["ik", "jacobian", "compare"],
        help="Control mode: ik, jacobian, or compare (toggle with J). Default: compare.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.15,
        help="Controller deadzone (0.0-1.0). Default: 0.15.",
    )
    parser.add_argument(
        "--linear-scale",
        type=float,
        default=None,
        help="Linear velocity scale (m/s). Default from config.",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.05,
        help="Jacobian damping factor. Default: 0.05.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Hide metrics display.",
    )
    args = parser.parse_args()

    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found at {URDF_PATH}")
        sys.exit(1)

    run_comparison(
        mode=args.mode,
        deadzone=args.deadzone,
        linear_scale=args.linear_scale,
        damping=args.damping,
        show_metrics=not args.no_metrics,
    )


if __name__ == "__main__":
    main()
