"""Diagnose SO-ARM motors before teleoperation."""

from __future__ import annotations

import argparse


def find_serial_port() -> str | None:
    """Find available serial port for the robot."""
    import glob

    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def run_diagnostics(port: str, motors: list[str] | None = None) -> None:
    """Run full motor diagnostics."""
    try:
        from lerobot.motors import diagnose_motor_bus
    except ImportError:
        print("ERROR: Motor diagnostics not available.")
        print("Install from ricklon/lerobot fork:")
        print("  uv pip install git+https://github.com/ricklon/lerobot.git")
        raise SystemExit(1)

    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    print(f"Connecting to robot on {port}...", flush=True)
    config = SOFollowerRobotConfig(port=port)
    robot = SOFollower(config)

    try:
        robot.bus.connect()
        print("Connected!", flush=True)

        print("\n" + "=" * 60)
        print("MOTOR DIAGNOSTICS")
        print("=" * 60)
        print("\nThis will test each motor to ensure it's working correctly.")
        print("You'll be prompted to move each motor through its range.\n")

        results = diagnose_motor_bus(robot.bus, motors=motors, interactive=True)

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        all_healthy = True
        for motor_name, result in results.items():
            status = "PASS" if result.healthy else "FAIL"
            print(f"\n{motor_name}: {status}")

            print(f"  Position readable: {'Yes' if result.position_readable else 'No'}")
            if result.zero_position is not None:
                print(f"  Zero position: {result.zero_position}")
            if result.max_position is not None:
                print(f"  Max position: {result.max_position}")
            if result.position_range > 0:
                print(f"  Range: {result.position_range:.1f}")

            if result.errors:
                all_healthy = False
                for error in result.errors:
                    print(f"  ERROR: {error}")

            if result.warnings:
                for warning in result.warnings:
                    print(f"  WARNING: {warning}")

        print("\n" + "=" * 60)
        if all_healthy:
            print("All motors passed. Ready for teleoperation.")
            print("\nYou can now run:")
            print(f"  uv run python examples/teleoperate_real.py --port {port}")
            print(f"  uv run python examples/teleoperate_dual.py --port {port}")
        else:
            print("Some motors have issues. Fix them before teleoperation.")
            print("\nSee DIAGNOSTIC_GUIDE.md in ricklon/lerobot for troubleshooting.")

    except Exception as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        try:
            robot.bus.disconnect()
        except Exception:
            pass
        print("\nDisconnected.")


def run_simple_diagnostics(port: str) -> None:
    """Run simple diagnostics without the full diagnostic module."""
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    print(f"Connecting to robot on {port}...", flush=True)
    config = SOFollowerRobotConfig(port=port)
    robot = SOFollower(config)

    try:
        robot.bus.connect()
        print("Connected!", flush=True)

        print("\n" + "=" * 60)
        print("SIMPLE MOTOR TEST")
        print("=" * 60)

        motors = list(robot.bus.motors.keys())
        print(f"\nFound {len(motors)} motors: {', '.join(motors)}")

        print("\nReading current positions...")
        for motor_name in motors:
            try:
                pos = robot.bus.read("Present_Position", motor_name)
                print(f"  {motor_name}: {pos}")
            except Exception as exc:
                print(f"  {motor_name}: ERROR - {exc}")

        print("\n" + "=" * 60)
        print("INTERACTIVE TEST")
        print("=" * 60)

        for motor_name in motors:
            print(f"\nTesting {motor_name}...")

            try:
                pos1 = robot.bus.read("Present_Position", motor_name)
                print(f"  Current position: {pos1}")

                input(f"  Move {motor_name} and press ENTER...")

                pos2 = robot.bus.read("Present_Position", motor_name)
                print(f"  New position: {pos2}")

                diff = abs(pos2 - pos1)
                if diff > 10:
                    print(f"  Motor responding (moved {diff:.1f} units)")
                elif diff > 0:
                    print(f"  Small movement ({diff:.1f} units)")
                else:
                    print("  No movement detected")

            except Exception as exc:
                print(f"  Error: {exc}")

        print("\n" + "=" * 60)
        print("Done. If all motors responded, you're ready for teleoperation.")

    except Exception as exc:
        print(f"Error: {exc}")
    finally:
        try:
            robot.bus.disconnect()
        except Exception:
            pass
        print("\nDisconnected.")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Diagnose SO-ARM motors")
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port for robot (e.g., /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--motors",
        type=str,
        nargs="+",
        default=None,
        help="Specific motors to diagnose (e.g., gripper shoulder_pan)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Run simple diagnostics (no diagnostic module required)",
    )
    return parser


def main() -> int:
    """Run the robot-diagnostics CLI."""
    args = build_parser().parse_args()

    port = args.port
    if port is None:
        port = find_serial_port()
        if port is None:
            print("ERROR: No serial port found.")
            print("Connect the robot and try again, or specify: --port /dev/ttyUSB0")
            return 1
        print(f"Auto-detected port: {port}")

    if args.simple:
        run_simple_diagnostics(port)
        return 0

    try:
        import lerobot.motors  # noqa: F401

        if hasattr(lerobot.motors, "diagnose_motor_bus"):
            run_diagnostics(port, args.motors)
        else:
            print("Note: Full diagnostics not available, running simple test.\n")
            run_simple_diagnostics(port)
    except ImportError:
        print("Note: Full diagnostics not available, running simple test.\n")
        run_simple_diagnostics(port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
