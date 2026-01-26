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
    - Left stick X: Left/right (Y axis)
    - Left stick Y: Up/down (Z axis)
    - Right stick Y: Forward/back (X axis)
    - Right stick X: Wrist roll rotation
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

# Path to URDF
URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"

# Joint names (order matters - matches URDF joint order)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

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


def run_with_controller(sim: MuJoCoSimulator):
    """Run with Xbox controller and MuJoCo viewer."""
    from lerobot.model.kinematics import RobotKinematics

    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    # Initialize kinematics for IK
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=JOINT_NAMES[:-1],  # Exclude gripper
    )

    config = XboxConfig()
    controller = XboxController(config)
    mapper = MapXboxToEEDelta(
        linear_scale=config.linear_scale,
        angular_scale=config.angular_scale,
    )

    if not controller.connect():
        print("ERROR: Failed to connect to Xbox controller")
        print("  - Check that controller is connected")
        print("  - Or use --no-controller for demo mode")
        sys.exit(1)

    print("Xbox controller connected", flush=True)
    print("\nControls:", flush=True)
    print("  Hold LB + move sticks to control arm", flush=True)
    print("  Right trigger for gripper", flush=True)
    print("  A button to go home", flush=True)
    print("  Close window to exit\n", flush=True)

    # Current EE pose and joint positions
    joint_pos_deg = np.zeros(5)
    ee_pose = kinematics.forward_kinematics(joint_pos_deg)
    gripper_pos = 0.0  # Current gripper position (smoothed)
    gripper_rate = config.gripper_rate  # Position change per second

    running = True

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
                joint_pos_deg = np.zeros(5)
                ee_pose = kinematics.forward_kinematics(joint_pos_deg)
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
                # Update target EE pose
                target_pos = ee_pose[:3, 3].copy()
                target_pos[0] += ee_delta.dx * LOOP_PERIOD
                target_pos[1] += ee_delta.dy * LOOP_PERIOD
                target_pos[2] += ee_delta.dz * LOOP_PERIOD

                # Workspace limits
                target_pos[0] = np.clip(target_pos[0], -0.1, 0.5)
                target_pos[1] = np.clip(target_pos[1], -0.3, 0.3)
                target_pos[2] = np.clip(target_pos[2], 0.05, 0.45)

                target_pose = ee_pose.copy()
                target_pose[:3, 3] = target_pos

                # Solve IK for position only
                new_joints = kinematics.inverse_kinematics(
                    joint_pos_deg, target_pose, position_weight=1.0, orientation_weight=0.0
                )
                joint_pos_deg = new_joints[:5]

                # Apply wrist roll directly (index 4 = wrist_roll)
                if abs(ee_delta.droll) > 0.001:
                    roll_delta_deg = np.rad2deg(ee_delta.droll * LOOP_PERIOD)
                    joint_pos_deg[4] += roll_delta_deg
                    joint_pos_deg[4] = np.clip(joint_pos_deg[4], -180.0, 180.0)

                ee_pose = kinematics.forward_kinematics(joint_pos_deg)

            # Update simulation
            sim.set_joint_targets(joint_pos_deg)
            sim.set_gripper(gripper_pos)
            sim.step()
            viewer.sync()

            # Status
            pos = sim.get_ee_position()
            print(
                f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | Gripper: {gripper_pos:.2f}   ",
                end="\r",
                flush=True,
            )

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
        joint_names=JOINT_NAMES[:-1],
    )

    print("\nDemo mode - automatic movement", flush=True)
    print("Close window to exit\n", flush=True)

    joint_pos_deg = np.zeros(5)
    ee_pose = kinematics.forward_kinematics(joint_pos_deg)
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

            # Solve IK for position only
            new_joints = kinematics.inverse_kinematics(
                joint_pos_deg, target_pose, position_weight=1.0, orientation_weight=0.0
            )
            joint_pos_deg = new_joints[:5]

            # Apply wrist roll directly
            if abs(droll) > 0.001:
                roll_delta_deg = np.rad2deg(droll * LOOP_PERIOD)
                joint_pos_deg[4] += roll_delta_deg
                joint_pos_deg[4] = np.clip(joint_pos_deg[4], -180.0, 180.0)

            ee_pose = kinematics.forward_kinematics(joint_pos_deg)

            sim.set_joint_targets(joint_pos_deg)
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
        run_with_controller(sim)


if __name__ == "__main__":
    main()
