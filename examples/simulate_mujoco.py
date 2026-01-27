#!/usr/bin/env python3
"""MuJoCo simulation with Xbox controller teleoperation.

This example runs the Xbox controller teleoperation with MuJoCo physics
simulation and real-time 3D visualization.

Usage:
    uv run python examples/simulate_mujoco.py

Options:
    --no-controller    Run without Xbox controller (demo mode)

Controls (Xbox):
    - Hold LB (left bumper) to enable arm movement
    - Left stick X: Move left/right (Y axis)
    - Left stick Y: Up/down (Z axis)
    - Right stick Y: Forward/back (X axis)
    - Right stick X: Wrist roll rotation (direct, not IK)
    - Right trigger: Gripper (released=open, pulled=closed)
    - A button: Return to home position
    - Ctrl+C or close window: Exit

Controls (Demo mode, --no-controller):
    - Automatic demo movement pattern
    - Ctrl+C or close window: Exit
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES, JOINT_NAMES_WITH_GRIPPER

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


def run_with_controller(
    sim: MuJoCoSimulator,
    deadzone: float = 0.15,
    linear_scale: float | None = None,
    debug_ik: bool = False,
    debug_ik_every: int = 10,
):
    """Run with Xbox controller and MuJoCo viewer."""
    from lerobot.model.kinematics import RobotKinematics

    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    # IK joint names - include base, exclude wrist_roll (controlled directly)
    ik_joint_names = IK_JOINT_NAMES

    # Initialize kinematics for IK (4 joints, not 5)
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=ik_joint_names,
    )

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

    print(f"Controller deadzone: {config.deadzone}", flush=True)
    print(f"Linear scale: {config.linear_scale} m/s", flush=True)
    print(f"Orientation scale: {config.orientation_scale} rad/s", flush=True)

    # Joint velocity limits for IK joints (4 joints, no wrist_roll)
    ik_joint_vel_limits = np.array([120.0, 90.0, 90.0, 90.0])  # deg/s

    if not controller.connect():
        print("ERROR: Failed to connect to Xbox controller")
        print("  - Check that controller is connected")
        print("  - Or use --no-controller for demo mode")
        sys.exit(1)

    print("Xbox controller connected", flush=True)
    print("\nControls:", flush=True)
    print("  Hold LB + move sticks to control arm", flush=True)
    print("  Left stick X: Move left/right (Y axis)", flush=True)
    print("  Left stick Y: Move up/down", flush=True)
    print("  Right stick Y: Move forward/back", flush=True)
    print("  Right stick X: Wrist roll (direct)", flush=True)
    print("  D-pad up/down: Pitch (tilt gripper)", flush=True)
    print("  D-pad left/right: Yaw (rotate heading)", flush=True)
    print("  Right trigger for gripper", flush=True)
    print("  A button to go home", flush=True)
    print("  Close window to exit\n", flush=True)

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
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])

    # Get initial EE pose (with base at 0)
    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
    gripper_pos = 0.0  # Current gripper position (smoothed)
    gripper_rate = config.gripper_rate  # Position change per second

    running = True
    loop_counter = 0

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Launch viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running() and running:
            loop_start = time.monotonic()

            state = controller.read()

            if state.a_button_pressed:
                print("\nGoing home...", flush=True)
                ik_joint_pos_deg = np.zeros(4)
                wrist_roll_deg = 0.0
                target_pitch = 0.0
                target_yaw = 0.0
                ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
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
                new_joints = kinematics.inverse_kinematics(
                    ik_joint_pos_deg, target_pose, position_weight=1.0, orientation_weight=orientation_weight
                )
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
            full_joint_pos_deg = np.array([
                ik_joint_pos_deg[0],  # shoulder_pan
                ik_joint_pos_deg[1],  # shoulder_lift
                ik_joint_pos_deg[2],  # elbow_flex
                ik_joint_pos_deg[3],  # wrist_flex
                wrist_roll_deg,  # wrist_roll
            ])

            # Update simulation
            sim.set_joint_targets(full_joint_pos_deg)
            sim.set_gripper(gripper_pos)
            sim.step()
            viewer.sync()

            # Status - show position and orientation
            pos = sim.get_ee_position()
            pitch_deg = np.rad2deg(target_pitch)
            yaw_deg = np.rad2deg(target_yaw)
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
    print("\nDisconnected.", flush=True)


def run_demo_mode(sim: MuJoCoSimulator):
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
    t = 0.0

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running() and running:
            loop_start = time.monotonic()

            # Demo pattern
            dx = 0.03 * np.sin(t * 0.5)
            dy = 0.03 * np.cos(t * 0.5)
            dz = 0.02 * np.sin(t * 0.3)
            droll = 0.1 * np.sin(t * 0.4)
            gripper = 0.5 + 0.5 * np.sin(t * 0.2)

            # Update EE
            target_pos = ee_pose[:3, 3].copy()
            target_pos[0] += dx * LOOP_PERIOD
            target_pos[1] += dy * LOOP_PERIOD
            target_pos[2] += dz * LOOP_PERIOD

            target_pos[0] = np.clip(target_pos[0], -0.1, 0.5)
            target_pos[1] = np.clip(target_pos[1], -0.3, 0.3)
            target_pos[2] = np.clip(target_pos[2], 0.05, 0.45)

            target_pose = ee_pose.copy()
            target_pose[:3, 3] = target_pos

            # Solve IK for position only (4 joints)
            new_joints = kinematics.inverse_kinematics(
                ik_joint_pos_deg, target_pose, position_weight=1.0, orientation_weight=0.0
            )
            ik_joint_pos_deg = new_joints[:4]

            # Apply wrist roll directly
            if abs(droll) > 0.001:
                roll_delta_deg = np.rad2deg(droll * LOOP_PERIOD)
                wrist_roll_deg += roll_delta_deg
                wrist_roll_deg = np.clip(wrist_roll_deg, -180.0, 180.0)

            ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)

            full_joint_pos_deg = np.array([
                ik_joint_pos_deg[0],
                ik_joint_pos_deg[1],
                ik_joint_pos_deg[2],
                ik_joint_pos_deg[3],
                wrist_roll_deg,
            ])

            sim.set_joint_targets(full_joint_pos_deg)
            sim.set_gripper(gripper)
            sim.step()
            viewer.sync()

            pos = sim.get_ee_position()
            print(
                f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | Gripper: {gripper:.2f}   ",
                end="\r",
                flush=True,
            )

            t += LOOP_PERIOD

            elapsed = time.monotonic() - loop_start
            if elapsed < LOOP_PERIOD:
                time.sleep(LOOP_PERIOD - elapsed)

    print("\nDone.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="MuJoCo SO-ARM101 simulation")
    parser.add_argument("--no-controller", action="store_true", help="Run demo mode")
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
    args = parser.parse_args()

    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found at {URDF_PATH}")
        sys.exit(1)

    print("Loading MuJoCo model...", flush=True)
    sim = MuJoCoSimulator(str(URDF_PATH))
    print("Model loaded!", flush=True)

    if args.no_controller:
        run_demo_mode(sim)
    else:
        run_with_controller(
            sim,
            deadzone=args.deadzone,
            linear_scale=args.linear_scale,
            debug_ik=args.debug_ik,
            debug_ik_every=args.debug_ik_every,
        )


if __name__ == "__main__":
    main()
