#!/usr/bin/env python3
"""Xbox controller teleoperation for real SO-ARM101.

This example connects to a real SO-ARM101 robot and controls it
via Xbox controller using inverse kinematics.

Usage:
    uv run python examples/teleoperate_real.py --port /dev/ttyUSB0

Controls:
    - Hold LB (left bumper) to enable arm movement
    - Left stick X: Left/right (Y axis)
    - Left stick Y: Up/down (Z axis)
    - Right stick Y: Forward/back (X axis)
    - Right stick X: Wrist roll rotation
    - Right trigger: Gripper (released=open, pulled=closed)
    - A button: Return to home position
    - Ctrl+C: Exit
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Path to URDF
URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"

# Joint names (order matters - matches URDF and robot)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Control loop rate
CONTROL_RATE = 30  # Hz
LOOP_PERIOD = 1.0 / CONTROL_RATE


def find_serial_port() -> str | None:
    """Find available serial port for the robot."""
    import glob

    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def deg_to_normalized(deg: float, joint_name: str) -> float:
    """Convert degrees to LeRobot normalized value [-100, 100].

    The SO101 uses approximately +/-180 degrees range mapped to [-100, 100].
    """
    # Approximate mapping: 180 deg = 100 normalized
    if joint_name == "gripper":
        # Gripper uses [0, 100] range
        # Our gripper_pos is 0-1, map to 0-100
        return deg  # Actually we pass 0-100 directly for gripper
    return deg * (100.0 / 180.0)


def normalized_to_deg(normalized: float, joint_name: str) -> float:
    """Convert LeRobot normalized value [-100, 100] to degrees."""
    if joint_name == "gripper":
        return normalized
    return normalized * (180.0 / 100.0)


def run_teleoperation(port: str):
    """Run teleoperation with real robot."""
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    # Initialize kinematics for IK
    print("Loading kinematics model...", flush=True)
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=JOINT_NAMES[:-1],  # Exclude gripper
    )

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
        print("  - Check that controller is connected")
        sys.exit(1)

    print("Xbox controller connected", flush=True)

    # Initialize robot
    print(f"Connecting to robot on {port}...", flush=True)
    robot_config = SOFollowerRobotConfig(port=port)
    robot = SOFollower(robot_config)

    try:
        robot.connect(calibrate=True)
    except Exception as e:
        print(f"ERROR: Failed to connect to robot: {e}")
        controller.disconnect()
        sys.exit(1)

    print("Robot connected!", flush=True)
    print("\nControls:", flush=True)
    print("  Hold LB + move sticks to control arm", flush=True)
    print("  Right trigger for gripper", flush=True)
    print("  A button to go home", flush=True)
    print("  Ctrl+C to exit\n", flush=True)

    # Current EE pose and joint positions
    joint_pos_deg = np.zeros(5)
    ee_pose = kinematics.forward_kinematics(joint_pos_deg)
    gripper_pos = 0.0  # 0-1 range

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...", flush=True)
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running:
            loop_start = time.monotonic()

            state = controller.read()

            if state.a_button_pressed:
                print("\nGoing home...", flush=True)
                joint_pos_deg = np.zeros(5)
                ee_pose = kinematics.forward_kinematics(joint_pos_deg)
                gripper_pos = 0.0

                # Send home position to robot
                action = {}
                for i, name in enumerate(JOINT_NAMES[:-1]):
                    action[f"{name}.pos"] = 0.0
                action["gripper.pos"] = 0.0  # Open
                robot.send_action(action)
                continue

            ee_delta = mapper(state)

            # Rate-limit gripper movement
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

                # Solve IK
                new_joints = kinematics.inverse_kinematics(
                    joint_pos_deg, target_pose, position_weight=1.0, orientation_weight=0.0
                )
                joint_pos_deg = new_joints[:5]

                # Apply wrist roll directly
                if abs(ee_delta.droll) > 0.001:
                    roll_delta_deg = np.rad2deg(ee_delta.droll * LOOP_PERIOD)
                    joint_pos_deg[4] += roll_delta_deg
                    joint_pos_deg[4] = np.clip(joint_pos_deg[4], -180.0, 180.0)

                ee_pose = kinematics.forward_kinematics(joint_pos_deg)

            # Send to robot (convert to normalized values)
            action = {}
            for i, name in enumerate(JOINT_NAMES[:-1]):
                action[f"{name}.pos"] = deg_to_normalized(joint_pos_deg[i], name)
            # Gripper: 0-1 maps to 0-100
            action["gripper.pos"] = gripper_pos * 100.0

            robot.send_action(action)

            # Status
            pos = ee_pose[:3, 3]
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
    parser = argparse.ArgumentParser(description="Xbox teleoperation for real SO-ARM101")
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

    run_teleoperation(port)


if __name__ == "__main__":
    main()
