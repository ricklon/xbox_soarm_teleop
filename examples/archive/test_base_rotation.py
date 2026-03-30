"""Simple base rotation test - oscillates shoulder_pan back and forth."""

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

try:
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    from xbox_soarm_teleop.config.joints import JOINT_NAMES
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure LeRobot is installed: uv pip install lerobot[kinematics]")
    sys.exit(1)


def find_port() -> str | None:
    """Find available serial port."""
    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def run_base_rotation_test(
    port: str,
    amplitude_deg: float = 15.0,
    period_sec: float = 2.0,
    duration_sec: float = 10.0,
    calibration_dir: Path | None = None,
):
    """Run base rotation test."""
    # Default calibration directory
    if calibration_dir is None:
        calibration_dir = Path(__file__).parent.parent / "calibration"
    calibration_dir = Path(calibration_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    # Connect to robot
    print(f"Connecting to robot on {port}...")
    robot_config = SOFollowerRobotConfig(port=port, calibration_dir=calibration_dir)
    robot = SOFollower(robot_config)

    # Try to connect, may need calibration
    try:
        robot.connect(calibrate=True)
    except Exception as e:
        print(f"Connection error: {e}")
        print("Trying with calibrate=False (use existing calibration)...")
        try:
            robot.connect(calibrate=False)
            # When calibrate=False is used, we need to write calibration manually
            if robot.calibration and not robot.bus.is_calibrated:
                print("Loading existing calibration...")
                robot.bus.write_calibration(robot.calibration)
        except Exception as e2:
            if "calibration" in str(e2).lower():
                print("ERROR: No existing calibration found. Run calibration first:")
                print(
                    f"  lerobot-calibrate --robot.type=so101_follower --robot.port={port} --robot.id=test_arm"
                )
                sys.exit(1)
            raise

    print(f"Connected! Calibration directory: {calibration_dir}")
    print()
    print("=" * 60)
    print("BASE ROTATION TEST")
    print("=" * 60)
    print(f"Amplitude: +/-{amplitude_deg}°")
    print(f"Period: {period_sec}s (one full back-and-forth)")
    print(f"Duration: {duration_sec}s")
    print()
    print("Press Ctrl+C to stop")
    print()

    # Get current joint positions
    current_obs = robot.get_observation()

    # Setup starting position (keep other joints at current, only move shoulder_pan)
    starting_joints = [current_obs.get(f"{name}.pos", 0.0) for name in JOINT_NAMES[:-1]]

    start_time = time.monotonic()
    last_update = start_time

    try:
        while True:
            now = time.monotonic()
            elapsed = now - start_time

            if elapsed >= duration_sec:
                break

            # Only update at control rate
            if now - last_update < 0.02:
                time.sleep(0.001)
                continue
            last_update = now

            # Calculate shoulder_pan angle (sinusoidal oscillation)
            shoulder_pan_deg = amplitude_deg * np.sin(2 * np.pi * elapsed / period_sec)

            # Build action dict
            action = {}
            for i, name in enumerate(JOINT_NAMES[:-1]):
                if name == "shoulder_pan":
                    # Send oscillating angle
                    action[f"{name}.pos"] = shoulder_pan_deg * (100.0 / 180.0)
                else:
                    # Keep other joints at starting position
                    action[f"{name}.pos"] = starting_joints[i]

            # Send to robot
            robot.send_action(action)

            # Print status
            print(
                f"\rT={elapsed:5.1f}s | shoulder_pan={shoulder_pan_deg:6.1f}°",
                end="",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        print("\n\nReturning to starting position...")
        action = {}
        for i, name in enumerate(JOINT_NAMES[:-1]):
            action[f"{name}.pos"] = starting_joints[i]
        robot.send_action(action)
        robot.disconnect()
        print("Disconnected.")


def main():
    parser = argparse.ArgumentParser(description="Test base rotation (shoulder_pan)")
    parser.add_argument("--port", help="Serial port (auto-detect if not specified)")
    parser.add_argument(
        "--amplitude",
        type=float,
        default=15.0,
        help="Rotation amplitude in degrees (default: 15)",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=2.0,
        help="Oscillation period in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Test duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="Calibration directory",
    )

    args = parser.parse_args()

    # Auto-detect port if not specified
    if args.port is None:
        args.port = find_port()
        if args.port is None:
            print("ERROR: No serial port found. Connect robot or specify --port")
            sys.exit(1)

    run_base_rotation_test(
        port=args.port,
        amplitude_deg=args.amplitude,
        period_sec=args.period,
        duration_sec=args.duration,
        calibration_dir=args.calibration_dir,
    )


if __name__ == "__main__":
    main()
