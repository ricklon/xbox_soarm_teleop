#!/usr/bin/env python3
"""Digital twin mode: Real robot + MuJoCo simulation.

This example runs both the real SO-ARM101 robot and a MuJoCo simulation
simultaneously. The simulation serves as a real-time preview of the robot's
movements.

Usage:
    uv run python examples/teleoperate_dual.py --port /dev/ttyUSB0

Controls:
    - Hold LB (left bumper) to enable arm movement
    - Left stick X: Move left/right (Y axis)
    - Left stick Y: Up/down (Z axis)
    - Right stick Y: Forward/back (X axis)
    - Right stick X: Wrist roll rotation (direct, not IK)
    - Right trigger: Gripper (released=open, pulled=closed)
    - A button: Return to home position
    - Close window or Ctrl+C: Exit
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

# Joint names (order matters - matches URDF and robot)
JOINT_NAMES = JOINT_NAMES_WITH_GRIPPER

# Control loop rate
CONTROL_RATE = 30  # Hz
LOOP_PERIOD = 1.0 / CONTROL_RATE
GRIPPER_DEFAULT = 0.0  # 0=open, 1=closed


def gripper_to_robot(gripper_pos: float) -> float:
    """Map gripper position (0=open, 1=closed) to robot command."""
    return (1.0 - gripper_pos) * 100.0


def find_serial_port() -> str | None:
    """Find available serial port for the robot."""
    import glob

    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def deg_to_normalized(deg: float, joint_name: str) -> float:
    """Convert degrees to LeRobot normalized value [-100, 100]."""
    if joint_name == "gripper":
        return deg
    return deg * (100.0 / 180.0)


class MuJoCoSimulator:
    """MuJoCo-based SO-ARM101 simulator for visualization."""

    def __init__(self, urdf_path: str):
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)

        # Get joint indices
        self.joint_ids = {}
        for name in JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                self.joint_ids[name] = jnt_id

        self.go_home()

    def go_home(self) -> None:
        """Reset to home position."""
        for name in JOINT_NAMES:
            if name in self.joint_ids:
                jnt_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_joint_positions(self, positions_deg: np.ndarray, gripper_pos: float) -> None:
        """Set joint positions.

        Args:
            positions_deg: Joint positions in degrees (5 joints, excluding gripper).
            gripper_pos: Gripper position 0-1.
        """
        positions_rad = np.deg2rad(positions_deg)
        for i, name in enumerate(JOINT_NAMES[:-1]):
            if name in self.joint_ids:
                jnt_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] = positions_rad[i]

        # Set gripper
        if "gripper" in self.joint_ids:
            jnt_id = self.joint_ids["gripper"]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            gripper_open = 1.74533
            gripper_closed = -0.174533
            self.data.qpos[qpos_adr] = gripper_open - gripper_pos * (gripper_open - gripper_closed)

        mujoco.mj_forward(self.model, self.data)

    def get_ee_position(self) -> np.ndarray:
        """Get end effector position."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_frame_link")
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return self.data.xpos[-1].copy()


def run_dual_mode(port: str):
    """Run digital twin mode with real robot and simulation."""
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    # Initialize kinematics
    print("Loading kinematics model...", flush=True)
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=IK_JOINT_NAMES,
    )

    # Initialize MuJoCo simulation
    print("Loading MuJoCo model...", flush=True)
    sim = MuJoCoSimulator(str(URDF_PATH))

    # Initialize Xbox controller
    config = XboxConfig()
    controller = XboxController(config)
    mapper = MapXboxToEEDelta(
        linear_scale=config.linear_scale,
        angular_scale=config.angular_scale,
    )
    gripper_rate = config.gripper_rate

    if not controller.connect():
        print("ERROR: Failed to connect to Xbox controller")
        sys.exit(1)

    print("Xbox controller connected", flush=True)

    # Initialize real robot
    print(f"Connecting to real robot on {port}...", flush=True)
    robot_config = SOFollowerRobotConfig(port=port)
    robot = SOFollower(robot_config)

    try:
        robot.connect(calibrate=True)
    except Exception as e:
        print(f"ERROR: Failed to connect to robot: {e}")
        controller.disconnect()
        sys.exit(1)

    print("Robot connected!", flush=True)
    print("\n=== DIGITAL TWIN MODE ===", flush=True)
    print("Simulation window shows real-time preview of robot movements", flush=True)
    print("\nControls:", flush=True)
    print("  Hold LB + move sticks to control arm", flush=True)
    print("  Right trigger for gripper", flush=True)
    print("  A button to go home", flush=True)
    print("  Close window or Ctrl+C to exit\n", flush=True)

    # Current state
    wrist_roll_deg = 0.0
    ik_joint_pos_deg = np.zeros(4)
    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
    gripper_pos = GRIPPER_DEFAULT

    # Ensure sim + robot start with the same gripper state
    sim.set_joint_positions(np.zeros(5), gripper_pos)
    robot.send_action({"gripper.pos": gripper_to_robot(gripper_pos)})

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...", flush=True)
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Launch viewer
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running() and running:
                loop_start = time.monotonic()

                state = controller.read()

                if state.a_button_pressed:
                    print("\nGoing home...", flush=True)
                    wrist_roll_deg = 0.0
                    ik_joint_pos_deg = np.zeros(4)
                    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
                    gripper_pos = GRIPPER_DEFAULT

                    # Update simulation
                    sim.go_home()
                    sim.set_joint_positions(np.zeros(5), gripper_pos)
                    viewer.sync()

                    # Send to real robot
                    action = {f"{name}.pos": 0.0 for name in JOINT_NAMES[:-1]}
                    action["gripper.pos"] = gripper_to_robot(GRIPPER_DEFAULT)
                    robot.send_action(action)
                    continue

                ee_delta = mapper(state)

                # Rate-limit gripper
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
                    target_pos[0] += ee_delta.dx * LOOP_PERIOD
                    target_pos[1] += ee_delta.dy * LOOP_PERIOD
                    target_pos[2] += ee_delta.dz * LOOP_PERIOD

                    # Workspace limits
                    target_pos[0] = np.clip(target_pos[0], -0.1, 0.5)
                    target_pos[1] = np.clip(target_pos[1], -0.3, 0.3)
                    target_pos[2] = np.clip(target_pos[2], 0.05, 0.45)

                    target_pose = ee_pose.copy()
                    target_pose[:3, 3] = target_pos

                    # Solve IK
                    new_joints = kinematics.inverse_kinematics(
                        ik_joint_pos_deg, target_pose, position_weight=1.0, orientation_weight=0.0
                    )
                    ik_joint_pos_deg = new_joints[:4]

                    # Apply wrist roll directly
                    if abs(ee_delta.droll) > 0.001:
                        roll_delta_deg = np.rad2deg(ee_delta.droll * LOOP_PERIOD)
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

                # Update simulation (digital twin)
                sim.set_joint_positions(full_joint_pos_deg, gripper_pos)
                viewer.sync()

                # Send to real robot
                action = {}
                for i, name in enumerate(JOINT_NAMES[:-1]):
                    action[f"{name}.pos"] = deg_to_normalized(full_joint_pos_deg[i], name)
                action["gripper.pos"] = gripper_to_robot(gripper_pos)
                robot.send_action(action)

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

    finally:
        controller.disconnect()
        robot.disconnect()
        print("\nDisconnected.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Digital twin: real robot + MuJoCo simulation")
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port for robot (e.g., /dev/ttyUSB0). Auto-detected if not specified.",
    )
    args = parser.parse_args()

    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found at {URDF_PATH}")
        sys.exit(1)

    port = args.port
    if port is None:
        port = find_serial_port()
        if port is None:
            print("ERROR: No serial port found. Connect the robot and try again.")
            print("  Or specify port manually: --port /dev/ttyUSB0")
            sys.exit(1)
        print(f"Auto-detected port: {port}")

    run_dual_mode(port)


if __name__ == "__main__":
    main()
