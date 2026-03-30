"""Read servo positions using LeRobot robot interface."""

import argparse
import glob
import sys
import time

try:
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure LeRobot is installed: uv pip install lerobot[kinematics]")
    sys.exit(1)


def find_port() -> str | None:
    """Find available serial port."""
    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def read_servo_positions(
    port: str,
    duration_sec: float = 10.0,
):
    """Read servo positions via LeRobot robot interface."""
    # Initialize robot
    print(f"Connecting to robot on {port}...")
    robot_config = SOFollowerRobotConfig(port=port)
    robot = SOFollower(robot_config)

    try:
        robot.connect(calibrate=False)
    except Exception as e:
        if "calibration" in str(e).lower():
            print("ERROR: No existing calibration found. Run calibration first:")
            print(
                f"  lerobot-calibrate --robot.type=so101_follower --robot.port={port} --robot.id=test_arm"
            )
            return
        raise

    print("Reading servo positions (press Ctrl+C to stop)...")
    print()
    print("=" * 110)
    print(
        f"{'ID':<3} {'Name':<15} {'Position (deg)':<20} {'Velocity (deg/s)':<20} {'Load (%)':<15} {'Temp (°C)':<10}"
    )
    print("=" * 110)

    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    start_time = time.monotonic()
    last_update = start_time

    try:
        while True:
            now = time.monotonic()
            elapsed = now - start_time

            # Limit update rate
            if now - last_update < 0.2:
                time.sleep(0.01)
                continue
            last_update = now

            print(f"\nT={elapsed:5.1f}s:", end="", flush=True)

            # Read observation
            obs = robot.get_observation()

            # Display joint positions
            for i, name in enumerate(joint_names):
                pos = obs.get(f"{name}.pos", 0.0)
                vel = obs.get(f"{name}.vel", 0.0)
                load = obs.get(f"{name}.current", 0.0)  # Current as proxy for load
                temp = obs.get(f"{name}.temperature", 0.0)

                # Convert normalized position to degrees
                pos_deg = pos * (180.0 / 100.0)

                print(
                    f"  [{i + 1}] {name:14s}: {pos_deg:7.1f}° vel={vel:5.1f} load={load:5.0f} temp={temp:5.0f}",
                    end="",
                    flush=True,
                )

            # Check for timeout
            if elapsed >= duration_sec:
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        robot.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Read servo positions via LeRobot")
    parser.add_argument("--port", help="Serial port (auto-detect if not specified)")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Read duration in seconds (default: 10.0)",
    )

    args = parser.parse_args()

    # Auto-detect port if not specified
    if args.port is None:
        args.port = find_port()
        if args.port is None:
            print("ERROR: No serial port found. Connect robot or specify --port")
            sys.exit(1)

    read_servo_positions(port=args.port, duration_sec=args.duration)


if __name__ == "__main__":
    main()
