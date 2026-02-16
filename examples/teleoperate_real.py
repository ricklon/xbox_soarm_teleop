#!/usr/bin/env python3
"""Xbox controller teleoperation for real SO-ARM101.

This example connects to a real SO-ARM101 robot and controls it
via Xbox controller using inverse kinematics or Jacobian-based control.

Usage:
    uv run python examples/teleoperate_real.py --port /dev/ttyUSB0
    uv run python examples/teleoperate_real.py --jacobian      # Use Jacobian control
    uv run python examples/teleoperate_real.py --recalibrate  # Fresh calibration
    uv run python examples/teleoperate_real.py --no-calibrate # Skip calibration
    uv run python examples/teleoperate_real.py --motion-routine
    uv run python examples/teleoperate_real.py --motion-routine --routine-pattern square-xyz

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
import csv
import signal
import sys
import time
from pathlib import Path

import numpy as np

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    IK_JOINT_NAMES,
    JOINT_NAMES_WITH_GRIPPER,
)

# Path to URDF
URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"

# Default calibration directory (project-local)
DEFAULT_CALIBRATION_DIR = Path(__file__).parent.parent / "calibration"

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
    calibration_dir: Path | None = None,
    recalibrate: bool = False,
    no_calibrate: bool = False,
    deadzone: float = 0.15,
    linear_scale: float | None = None,
    debug_ik: bool = False,
    debug_ik_every: int = 10,
    motion_routine: bool = False,
    routine_duration: float = 15.0,
    routine_scale: float = 1.0,
    routine_pattern: str = "lissajous",
    routine_plane: str = "xy",
    routine_square_size: float = 0.06,
    routine_square_speed: float = 0.03,
    routine_center_x: float = 0.0,
    routine_center_y: float = 0.0,
    routine_center_z: float = 0.0,
    ik_log_path: str | None = None,
    ik_max_err_mm: float = 30.0,
    ik_mean_err_mm: float = 10.0,
    ik_vel_scale: float = 1.0,
    use_jacobian: bool = False,
    jacobian_damping: float = 0.05,
):
    """Run teleoperation with real robot.

    Args:
        port: Serial port for robot.
        calibration_dir: Directory for calibration files. Default: ./calibration/
        recalibrate: If True, delete existing calibration and run fresh.
        no_calibrate: If True, skip calibration (use existing only).
        deadzone: Controller deadzone (0.0-1.0).
        linear_scale: Linear velocity scale (m/s), or None for config default.
        debug_ik: If True, print IK debug output periodically.
        debug_ik_every: Print IK debug every N control loops.
        motion_routine: If True, run an automatic motion routine (no controller).
        routine_duration: Duration in seconds for motion routine.
        routine_scale: Scale for routine motion amplitudes.
        routine_pattern: Motion routine pattern (lissajous, square, square-xyz).
        routine_plane: Plane for square routine (xy, xz, yz).
        routine_square_size: Square side length in meters.
        routine_square_speed: Max tracking speed for square (m/s).
        routine_center_x: Offset to add to routine center (meters).
        routine_center_y: Offset to add to routine center (meters).
        routine_center_z: Offset to add to routine center (meters).
        ik_log_path: Optional CSV path for IK error logging.
        ik_max_err_mm: Warn if max IK position error exceeds this (mm).
        ik_mean_err_mm: Warn if mean IK position error exceeds this (mm).
        ik_vel_scale: Scale factor for IK joint velocity limits.
        use_jacobian: If True, use Jacobian-based control instead of IK.
        jacobian_damping: Damping factor for Jacobian pseudo-inverse.
    """
    import shutil

    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.kinematics.jacobian import JacobianController
    from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta, MapXboxToEEDelta
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    # Use project-local calibration directory by default
    if calibration_dir is None:
        calibration_dir = DEFAULT_CALIBRATION_DIR
    calibration_dir = Path(calibration_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    print(f"Calibration directory: {calibration_dir}", flush=True)

    # Handle recalibration - delete existing calibration
    if recalibrate and calibration_dir.exists():
        print("Deleting existing calibration for recalibration...", flush=True)
        shutil.rmtree(calibration_dir)
        calibration_dir.mkdir(parents=True, exist_ok=True)
        print("Calibration cleared.", flush=True)

    # IK joint names - include base, exclude wrist_roll (controlled directly)
    ik_joint_names = IK_JOINT_NAMES

    # Initialize kinematics for IK (4 joints, not 5)
    print("Loading kinematics model...", flush=True)
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=ik_joint_names,
    )

    # Initialize Jacobian controller if needed
    jacobian_ctrl = None
    if use_jacobian:
        jacobian_ctrl = JacobianController(kinematics, damping=jacobian_damping)
        print(f"Control mode: JACOBIAN (damping={jacobian_damping})", flush=True)
    else:
        print("Control mode: IK", flush=True)

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
    ik_joint_vel_limits = np.array([120.0, 90.0, 90.0, 90.0]) * max(ik_vel_scale, 0.1)

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
    robot_config = SOFollowerRobotConfig(port=port, calibration_dir=calibration_dir)
    robot = SOFollower(robot_config)

    # Determine calibration mode
    calibrate = not no_calibrate

    try:
        robot.connect(calibrate=calibrate)
        # When --no-calibrate is used, we still need to write calibration to the motor bus
        # lerobot skips this when calibrate=False, so we do it manually
        if no_calibrate and robot.calibration and not robot.bus.is_calibrated:
            print("Loading existing calibration...", flush=True)
            robot.bus.write_calibration(robot.calibration)
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
        print(
            f"  Duration: {routine_duration:.1f}s | Scale: {routine_scale:.2f} | "
            f"Pattern: {routine_pattern}",
            flush=True,
        )
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
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )

    # Get initial EE pose
    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
    last_target_pose = ee_pose.copy()
    last_ee_pose = ee_pose.copy()
    gripper_pos = 0.0  # 0-1 range
    routine_center = ee_pose[:3, 3].copy()
    routine_center += np.array([routine_center_x, routine_center_y, routine_center_z], dtype=float)

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

    running = True
    loop_counter = 0
    routine_start = time.monotonic()
    error_count = 0
    error_sum = 0.0
    error_max = 0.0
    clip_count = 0
    clip_joints = 0

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
                "clipped",
            ]
        )

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...", flush=True)
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running:
            loop_start = time.monotonic()
            clipped_this = False

            if motion_routine:
                t = time.monotonic() - routine_start
                if t >= routine_duration:
                    break
                if routine_pattern == "lissajous":
                    dx = routine_scale * 0.03 * np.sin(t * 0.5)
                    dy = routine_scale * 0.03 * np.cos(t * 0.5)
                    dz = routine_scale * 0.02 * np.sin(t * 0.3)
                else:
                    size = routine_square_size * routine_scale
                    if routine_pattern == "square-xyz":
                        segment = max(routine_duration / 3.0, 0.1)
                        plane_idx = int(t / segment)
                        plane = ["xy", "xz", "yz"][min(plane_idx, 2)]
                        u = (t % segment) / segment
                    else:
                        plane = routine_plane
                        u = (t / max(routine_duration, 0.1)) % 1.0
                    desired = routine_center + plane_offset(plane, u, size)
                    delta = desired - ee_pose[:3, 3]
                    dist = float(np.linalg.norm(delta))
                    if dist < 1e-6:
                        dx = dy = dz = 0.0
                    else:
                        vel = delta / LOOP_PERIOD
                        speed = float(np.linalg.norm(vel))
                        max_speed = max(routine_square_speed, 0.001)
                        if speed > max_speed:
                            vel = vel * (max_speed / speed)
                        dx, dy, dz = vel
                if routine_pattern == "lissajous":
                    droll = routine_scale * 0.10 * np.sin(t * 0.4)
                    gripper = 0.5 + 0.2 * np.sin(t * 0.2)
                    gripper = float(np.clip(gripper, 0.0, 1.0))
                else:
                    droll = 0.0
                    gripper = gripper_pos
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

                if use_jacobian:
                    # Jacobian-based control: EE velocity -> joint velocity -> integrate
                    # Use position + pitch mode (4 DOF) for optimal control with 4 joints
                    ee_vel = np.array([ee_delta.dx, ee_delta.dy, ee_delta.dz, ee_delta.dpitch])
                    joint_vel_deg = jacobian_ctrl.ee_vel_to_joint_vel(
                        ee_vel, ik_joint_pos_deg, mode="position_pitch"
                    )

                    # Apply velocity limits
                    joint_vel_deg = np.clip(joint_vel_deg, -ik_joint_vel_limits, ik_joint_vel_limits)

                    # Integrate to get new joint positions
                    joint_delta = joint_vel_deg * LOOP_PERIOD
                    ik_joint_pos_deg = ik_joint_pos_deg + joint_delta

                    # Apply joint limits
                    joint_limits_deg = np.array([180.0, 90.0, 90.0, 90.0])
                    ik_joint_pos_deg = np.clip(ik_joint_pos_deg, -joint_limits_deg, joint_limits_deg)
                else:
                    # IK-based control: solve IK for target pose
                    new_joints = kinematics.inverse_kinematics(
                        ik_joint_pos_deg,
                        target_pose,
                        position_weight=1.0,
                        orientation_weight=orientation_weight,
                    )
                    ik_result = new_joints[:4]

                    # Apply joint velocity limiting to smooth IK output
                    max_delta = ik_joint_vel_limits * LOOP_PERIOD
                    joint_delta = ik_result - ik_joint_pos_deg
                    clipped_delta = np.clip(joint_delta, -max_delta, max_delta)
                    if np.any(np.abs(joint_delta) > max_delta):
                        clip_count += 1
                        clip_joints += int(np.sum(np.abs(joint_delta) > max_delta))
                        clipped_this = True
                    ik_joint_pos_deg = ik_joint_pos_deg + clipped_delta

                # Apply wrist roll directly (not part of IK)
                if abs(ee_delta.droll) > 0.001:
                    roll_delta_deg = np.rad2deg(ee_delta.droll * LOOP_PERIOD)
                    wrist_roll_deg += roll_delta_deg
                    wrist_roll_deg = np.clip(wrist_roll_deg, -180.0, 180.0)

                ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
                last_target_pose = target_pose.copy()
                last_ee_pose = ee_pose.copy()

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
            full_joint_pos_deg = np.array(
                [
                    ik_joint_pos_deg[0],  # shoulder_pan
                    ik_joint_pos_deg[1],  # shoulder_lift
                    ik_joint_pos_deg[2],  # elbow_flex
                    ik_joint_pos_deg[3],  # wrist_flex
                    wrist_roll_deg,  # wrist_roll
                ]
            )

            # Send to robot (convert to normalized values)
            action = {}
            for i, name in enumerate(JOINT_NAMES[:-1]):
                action[f"{name}.pos"] = deg_to_normalized(full_joint_pos_deg[i], name)
            # Gripper: 0-1 maps to 0-100
            action["gripper.pos"] = gripper_pos * 100.0

            robot.send_action(action)

            # Status - show position and orientation
            pos = ee_pose[:3, 3]
            pos_err = float(np.linalg.norm(last_ee_pose[:3, 3] - last_target_pose[:3, 3]))
            error_count += 1
            error_sum += pos_err
            error_max = max(error_max, pos_err)
            if csv_writer:
                t_s = time.monotonic() - routine_start
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
                        "1" if clipped_this else "0",
                    ]
                )
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
        print("\nReturning to home position...", flush=True)
        try:
            # Send home position command
            home_action = {}
            for name in JOINT_NAMES[:-1]:  # All except gripper
                home_action[f"{name}.pos"] = HOME_POSITION_DEG[name]
            home_action["gripper.pos"] = 0.0  # Gripper closed
            robot.send_action(home_action)
            time.sleep(2.0)  # Wait for movement
        except Exception as e:
            print(f"Warning: Could not return home: {e}")

        if controller is not None:
            controller.disconnect()
        robot.disconnect()
        if csv_file:
            csv_file.close()
        if error_count > 0:
            mean_err_mm = (error_sum / error_count) * 1000.0
            max_err_mm = error_max * 1000.0
            clip_rate = (clip_count / error_count) * 100.0
            print("\nIK routine summary")
            print(f"  samples: {error_count}")
            print(f"  max position error: {max_err_mm:.1f} mm")
            print(f"  mean position error: {mean_err_mm:.1f} mm")
            print(f"  velocity clipping: {clip_rate:.1f}% (joints clipped: {clip_joints})")
            if max_err_mm > ik_max_err_mm or mean_err_mm > ik_mean_err_mm:
                print("WARNING: IK error exceeded thresholds.")
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
        "--calibration-dir",
        type=str,
        default=None,
        help=f"Calibration directory. Default: {DEFAULT_CALIBRATION_DIR}",
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
    parser.add_argument(
        "--routine-pattern",
        type=str,
        default="lissajous",
        choices=["lissajous", "square", "square-xyz"],
        help="Motion routine pattern. Default: lissajous.",
    )
    parser.add_argument(
        "--routine-plane",
        type=str,
        default="xy",
        choices=["xy", "xz", "yz"],
        help="Plane for square routine. Default: xy.",
    )
    parser.add_argument(
        "--routine-square-size",
        type=float,
        default=0.06,
        help="Square side length in meters. Default: 0.06.",
    )
    parser.add_argument(
        "--routine-square-speed",
        type=float,
        default=0.03,
        help="Max tracking speed for square routine (m/s). Default: 0.03.",
    )
    parser.add_argument(
        "--routine-square-period",
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
        "--ik-log",
        type=str,
        default=None,
        help="Write IK position error CSV to this path.",
    )
    parser.add_argument(
        "--ik-max-err-mm",
        type=float,
        default=30.0,
        help="Warn if max position error exceeds this (mm). Default: 30.",
    )
    parser.add_argument(
        "--ik-mean-err-mm",
        type=float,
        default=10.0,
        help="Warn if mean position error exceeds this (mm). Default: 10.",
    )
    parser.add_argument(
        "--ik-vel-scale",
        type=float,
        default=1.0,
        help="Scale IK joint velocity limits. Default: 1.0.",
    )
    parser.add_argument(
        "--jacobian",
        action="store_true",
        help="Use Jacobian-based control instead of IK.",
    )
    parser.add_argument(
        "--jacobian-damping",
        type=float,
        default=0.05,
        help="Damping factor for Jacobian pseudo-inverse. Default: 0.05.",
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

    if args.routine_square_period is not None:
        if args.routine_square_period <= 0:
            print("ERROR: --routine-square-period must be > 0.")
            sys.exit(1)
        perimeter = 4.0 * max(args.routine_square_size, 0.001)
        args.routine_square_speed = perimeter / args.routine_square_period

    run_teleoperation(
        port,
        calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
        recalibrate=args.recalibrate,
        no_calibrate=args.no_calibrate,
        deadzone=args.deadzone,
        linear_scale=args.linear_scale,
        debug_ik=args.debug_ik,
        debug_ik_every=args.debug_ik_every,
        motion_routine=args.motion_routine,
        routine_duration=args.routine_duration,
        routine_scale=args.routine_scale,
        routine_pattern=args.routine_pattern,
        routine_plane=args.routine_plane,
        routine_square_size=args.routine_square_size,
        routine_square_speed=args.routine_square_speed,
        routine_center_x=args.routine_center_x,
        routine_center_y=args.routine_center_y,
        routine_center_z=args.routine_center_z,
        ik_log_path=args.ik_log,
        ik_max_err_mm=args.ik_max_err_mm,
        ik_mean_err_mm=args.ik_mean_err_mm,
        ik_vel_scale=args.ik_vel_scale,
        use_jacobian=args.jacobian,
        jacobian_damping=args.jacobian_damping,
    )


if __name__ == "__main__":
    main()
