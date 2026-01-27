#!/usr/bin/env python3
"""Xbox controller teleoperation for real SO-ARM101.

This example connects to a real SO-ARM101 robot and controls it
via Xbox controller using inverse kinematics.

Usage:
    uv run python examples/teleoperate_real.py --port /dev/ttyUSB0
    uv run python examples/teleoperate_real.py --recalibrate  # Fresh calibration
    uv run python examples/teleoperate_real.py --no-calibrate # Skip calibration
    uv run python examples/teleoperate_real.py --motion-routine

Controls:
    - Hold LB (left bumper) to enable arm movement
    - Left stick X: Left/right (Y axis)
    - Left stick Y: Up/down (Z axis)
    - Right stick Y: Forward/back (X axis)
    - Right stick X: Wrist roll rotation (direct, not IK)
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

from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES, JOINT_NAMES_WITH_GRIPPER

# Path to URDF
URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"

# Joint names (order matters - matches URDF and robot)
JOINT_NAMES = JOINT_NAMES_WITH_GRIPPER

# Control loop rate
CONTROL_RATE = 30  # Hz
LOOP_PERIOD = 1.0 / CONTROL_RATE

# Workspace limits (meters)
WORKSPACE_LIMITS = {
    "x": (0.05, 0.5),
    "y": (-0.3, 0.3),
    "z": (0.05, 0.45),
}


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


def run_teleoperation(
    port: str,
    recalibrate: bool = False,
    no_calibrate: bool = False,
    deadzone: float = 0.15,
    linear_scale: float | None = None,
    debug_ik: bool = False,
    debug_ik_every: int = 10,
    motion_routine: bool = False,
    routine_duration: float = 15.0,
    routine_scale: float = 1.0,
):
    """Run teleoperation with real robot.

    Args:
        port: Serial port for robot.
        recalibrate: If True, delete existing calibration and run fresh.
        no_calibrate: If True, skip calibration (use existing only).
        deadzone: Controller deadzone (0.0-1.0).
        linear_scale: Linear velocity scale (m/s), or None for config default.
        debug_ik: If True, print IK debug output periodically.
        debug_ik_every: Print IK debug every N control loops.
        motion_routine: If True, run an automatic motion routine (no controller).
        routine_duration: Duration in seconds for motion routine.
        routine_scale: Scale for routine motion amplitudes.
    """
    import shutil

    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta, MapXboxToEEDelta
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    # Handle recalibration - delete existing calibration cache
    calibration_dir = Path.home() / ".cache/huggingface/lerobot/calibration/robots"
    if recalibrate and calibration_dir.exists():
        print("Deleting existing calibration for recalibration...", flush=True)
        shutil.rmtree(calibration_dir)
        print("Calibration cache cleared.", flush=True)

    # IK joint names - include base, exclude wrist_roll (controlled directly)
    ik_joint_names = IK_JOINT_NAMES

    # Initialize kinematics for IK (4 joints, not 5)
    print("Loading kinematics model...", flush=True)
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=ik_joint_names,
    )

    # Initialize Xbox controller with passed parameters
    config = XboxConfig(deadzone=deadzone)
    if linear_scale is not None:
        config.linear_scale = linear_scale
    controller = None
    mapper = None
    if not motion_routine:
        controller = XboxController(config)
        mapper = MapXboxToEEDelta(
            linear_scale=config.linear_scale,
            angular_scale=config.angular_scale,
            orientation_scale=config.orientation_scale,
            invert_pitch=config.invert_pitch,
            invert_yaw=config.invert_yaw,
        )
    gripper_rate = config.gripper_rate

    # Joint velocity limits for IK joints (4 joints, no wrist_roll)
    ik_joint_vel_limits = np.array([120.0, 90.0, 90.0, 90.0])

    print(f"Controller deadzone: {config.deadzone}", flush=True)
    print(f"Linear scale: {config.linear_scale} m/s", flush=True)
    print(f"Orientation scale: {config.orientation_scale} rad/s", flush=True)

    if not motion_routine:
        if not controller.connect():
            print("ERROR: Failed to connect to Xbox controller")
            print("  - Check that controller is connected")
            sys.exit(1)
        print("Xbox controller connected", flush=True)
    else:
        print("Motion routine enabled (no controller required).", flush=True)
        print("Ensure the arm is in home position and the area is clear.", flush=True)

    # Initialize robot
    print(f"Connecting to robot on {port}...", flush=True)
    robot_config = SOFollowerRobotConfig(port=port)
    robot = SOFollower(robot_config)

    # Determine calibration mode
    calibrate = not no_calibrate

    try:
        robot.connect(calibrate=calibrate)
    except Exception as e:
        print(f"ERROR: Failed to connect to robot: {e}")
        if no_calibrate and "calibration" in str(e).lower():
            print("  No existing calibration found. Run without --no-calibrate first.")
        if controller is not None:
            controller.disconnect()
        sys.exit(1)

    print("Robot connected!", flush=True)
    if motion_routine:
        print("\nMotion routine:", flush=True)
        print(f"  Duration: {routine_duration:.1f}s | Scale: {routine_scale:.2f}", flush=True)
        print("  Ctrl+C: Stop\n", flush=True)
    else:
        print("\nControls:", flush=True)
        print("  Hold LB + move sticks to control arm", flush=True)
        print("  Left stick X: Move left/right (Y axis)", flush=True)
        print("  Left stick Y: Move up/down", flush=True)
        print("  Right stick Y: Move forward/back", flush=True)
        print("  Right stick X: Wrist roll (direct)", flush=True)
        print("  D-pad up/down: Pitch (tilt gripper)", flush=True)
        print("  D-pad left/right: Yaw (rotate heading)", flush=True)
        print("  Right trigger: Gripper", flush=True)
        print("  A button: Go home", flush=True)
        print("  Ctrl+C: Exit\n", flush=True)

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

    # Get initial EE pose
    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
    gripper_pos = 0.0  # 0-1 range

    running = True
    loop_counter = 0
    routine_start = time.monotonic()

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...", flush=True)
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running:
            loop_start = time.monotonic()

            if motion_routine:
                t = time.monotonic() - routine_start
                if t >= routine_duration:
                    break
                dx = routine_scale * 0.03 * np.sin(t * 0.5)
                dy = routine_scale * 0.03 * np.cos(t * 0.5)
                dz = routine_scale * 0.02 * np.sin(t * 0.3)
                droll = routine_scale * 0.10 * np.sin(t * 0.4)
                gripper = 0.5 + 0.2 * np.sin(t * 0.2)
                gripper = float(np.clip(gripper, 0.0, 1.0))
                ee_delta = EEDelta(dx=dx, dy=dy, dz=dz, droll=droll, gripper=gripper)
            else:
                state = controller.read()

                if state.a_button_pressed:
                    print("\nGoing home...", flush=True)
                    ik_joint_pos_deg = np.zeros(4)
                    wrist_roll_deg = 0.0
                    target_pitch = 0.0
                    target_yaw = 0.0
                    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
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
                # Update target EE pose (X/Y/Z)
                target_pos = ee_pose[:3, 3].copy()
                target_pos[0] += ee_delta.dx * LOOP_PERIOD  # Forward/back
                target_pos[1] += ee_delta.dy * LOOP_PERIOD  # Left/right
                target_pos[2] += ee_delta.dz * LOOP_PERIOD  # Up/down

                # Workspace limits
                target_pos[0] = np.clip(target_pos[0], *WORKSPACE_LIMITS["x"])
                target_pos[1] = np.clip(target_pos[1], *WORKSPACE_LIMITS["y"])
                target_pos[2] = np.clip(target_pos[2], *WORKSPACE_LIMITS["z"])

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
                clipped_delta = np.clip(joint_delta, -max_delta, max_delta)
                ik_joint_pos_deg = ik_joint_pos_deg + clipped_delta

                # Apply wrist roll directly (not part of IK)
                if abs(ee_delta.droll) > 0.001:
                    roll_delta_deg = np.rad2deg(ee_delta.droll * LOOP_PERIOD)
                    wrist_roll_deg += roll_delta_deg
                    wrist_roll_deg = np.clip(wrist_roll_deg, -180.0, 180.0)

                ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)

                if debug_ik and (loop_counter % max(debug_ik_every, 1) == 0):
                    pos_error = np.linalg.norm(target_pose[:3, 3] - ee_pose[:3, 3])
                    raw = np.array2string(joint_delta, precision=2, separator=",")
                    clipped = np.array2string(clipped_delta, precision=2, separator=",")
                    print(
                        f"\nIK: target=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, "
                        f"{target_pos[2]:.3f}] actual=[{ee_pose[0, 3]:.3f}, "
                        f"{ee_pose[1, 3]:.3f}, {ee_pose[2, 3]:.3f}] "
                        f"err={pos_error * 1000.0:.1f}mm "
                        f"raw_delta={raw} clipped={clipped}",
                        flush=True,
                    )

            # Combine base + IK joints for full 5-joint position
            full_joint_pos_deg = np.array([
                ik_joint_pos_deg[0],  # shoulder_pan
                ik_joint_pos_deg[1],  # shoulder_lift
                ik_joint_pos_deg[2],  # elbow_flex
                ik_joint_pos_deg[3],  # wrist_flex
                wrist_roll_deg,  # wrist_roll
            ])

            # Send to robot (convert to normalized values)
            action = {}
            for i, name in enumerate(JOINT_NAMES[:-1]):
                action[f"{name}.pos"] = deg_to_normalized(full_joint_pos_deg[i], name)
            # Gripper: 0-1 maps to 0-100
            action["gripper.pos"] = gripper_pos * 100.0

            robot.send_action(action)

            # Status - show position and orientation
            pos = ee_pose[:3, 3]
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

    finally:
        if controller is not None:
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
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Delete existing calibration and run fresh calibration.",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip calibration (use existing). Fails if no calibration exists.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.15,
        help="Controller deadzone (0.0-1.0). Default: 0.15. Increase if sticks drift.",
    )
    parser.add_argument(
        "--linear-scale",
        type=float,
        default=None,
        help="Linear velocity scale (m/s at full stick). Default from config.",
    )
    parser.add_argument(
        "--debug-ik",
        action="store_true",
        help="Print IK target/achieved error and joint deltas periodically.",
    )
    parser.add_argument(
        "--debug-ik-every",
        type=int,
        default=10,
        help="Print IK debug every N control loops. Default: 10.",
    )
    parser.add_argument(
        "--motion-routine",
        action="store_true",
        help="Run automatic motion routine (no controller input).",
    )
    parser.add_argument(
        "--routine-duration",
        type=float,
        default=15.0,
        help="Motion routine duration in seconds. Default: 15.",
    )
    parser.add_argument(
        "--routine-scale",
        type=float,
        default=1.0,
        help="Scale factor for routine motion amplitudes. Default: 1.0.",
    )
    args = parser.parse_args()

    if args.recalibrate and args.no_calibrate:
        print("ERROR: Cannot use both --recalibrate and --no-calibrate")
        sys.exit(1)

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

    run_teleoperation(
        port,
        recalibrate=args.recalibrate,
        no_calibrate=args.no_calibrate,
        deadzone=args.deadzone,
        linear_scale=args.linear_scale,
        debug_ik=args.debug_ik,
        debug_ik_every=args.debug_ik_every,
        motion_routine=args.motion_routine,
        routine_duration=args.routine_duration,
        routine_scale=args.routine_scale,
    )


if __name__ == "__main__":
    main()
