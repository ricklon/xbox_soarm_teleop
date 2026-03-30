#!/usr/bin/env python3
"""Diagnostic test for all servos - checks each motor individually.

This script tests each motor ID (1-6) one by one to identify:
- Which motors respond correctly
- Which motors have communication issues
- Temperature and voltage status
"""

import argparse
import glob
import sys
import time


def find_port() -> str | None:
    """Find available serial port."""
    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def check_motor_status(bus, motor_id: int) -> dict:
    """Check status of a single motor."""
    motor_key = str(motor_id)
    try:
        # Try to read position
        pos_raw = bus.read("Present_Position", motor_key, normalize=False)
        # Use raw-only reads so this diagnostic works without bus calibration loaded.
        pos_normalized = None

        # Read additional diagnostics
        try:
            temp = bus.read("Temperature", motor_key, normalize=False)
        except Exception:
            temp = None

        try:
            voltage = bus.read("Input_Voltage", motor_key, normalize=False)
        except Exception:
            voltage = None

        try:
            load = bus.read("Present_Current", motor_key, normalize=False)
        except Exception:
            load = None

        return {
            "id": motor_id,
            "status": "OK",
            "position_raw": int(pos_raw),
            "position_norm": float(pos_normalized) if pos_normalized is not None else None,
            "temperature": int(temp) if temp is not None else None,
            "voltage": float(voltage) / 10 if voltage is not None else None,  # Usually in 0.1V
            "load": int(load) if load is not None else None,
            "error": None,
        }
    except Exception as e:
        return {
            "id": motor_id,
            "status": "ERROR",
            "position_raw": None,
            "position_norm": None,
            "temperature": None,
            "voltage": None,
            "load": None,
            "error": str(e),
        }


def run_motor_diagnostic(port: str, interactive: bool = False):
    """Run diagnostic test on all motors."""
    bus = None
    try:
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode

        print("=" * 80)
        print("SERVO MOTOR DIAGNOSTIC TEST")
        print("=" * 80)
        print(f"Port: {port}")
        print()

        # Initialize bus (without calibration check)
        motors = {
            str(i): Motor(id=i, model="sts3215", norm_mode=MotorNormMode.DEGREES) for i in range(1, 7)
        }
        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()

        print("Checking motor IDs 1-6...")
        print()

        results = []
        motor_names = {
            1: "shoulder_pan (base rotation)",
            2: "shoulder_lift",
            3: "elbow_flex",
            4: "wrist_flex",
            5: "wrist_roll",
            6: "gripper",
        }

        for motor_id in range(1, 7):
            name = motor_names[motor_id]
            print(f"  Testing ID {motor_id} ({name})... ", end="", flush=True)

            result = check_motor_status(bus, motor_id)
            results.append(result)

            if result["status"] == "OK":
                print(f"✓ OK - PosRaw: {result['position_raw']}", end="")
                if result["temperature"] is not None:
                    print(f" Temp: {result['temperature']}°C", end="")
                print()
            else:
                print(f"✗ FAILED - {result['error'][:50]}")

        # Summary
        print()
        print("=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)

        working = [r for r in results if r["status"] == "OK"]
        failed = [r for r in results if r["status"] == "ERROR"]

        print(f"Working motors: {len(working)}/6")
        for r in working:
            name = motor_names[r["id"]]
            print(f"  ✓ ID {r['id']}: {name}")

        if failed:
            print()
            print(f"Failed motors: {len(failed)}/6")
            for r in failed:
                name = motor_names[r["id"]]
                print(f"  ✗ ID {r['id']}: {name}")
                print(f"    Error: {r['error']}")

        # Interactive test for working motors
        if interactive and working:
            print()
            print("=" * 80)
            print("INTERACTIVE MOTOR TEST")
            print("=" * 80)
            print("Each working motor will move ±10° from current position")
            print("Press Ctrl+C to skip a motor, or wait for test to complete")
            print()

            for r in working:
                motor_id = r["id"]
                name = motor_names[motor_id]
                current_pos = r["position_norm"]

                print(f"\nTesting {name} (ID {motor_id}):")
                print(f"  Current position: {current_pos:.1f}°")

                # Move to +10°
                target_plus = current_pos + 10
                print(f"  Moving to +10° ({target_plus:.1f}°)...", end=" ", flush=True)
                try:
                    bus.write("Goal_Position", str(motor_id), target_plus, normalize=True)
                    time.sleep(1.0)
                    new_pos = bus.read("Present_Position", str(motor_id), normalize=True)
                    print(f"Actual: {new_pos:.1f}° ✓")
                except KeyboardInterrupt:
                    print("Skipped")
                    continue
                except Exception as e:
                    print(f"Error: {e}")
                    continue

                # Move to -10°
                target_minus = current_pos - 10
                print(f"  Moving to -10° ({target_minus:.1f}°)...", end=" ", flush=True)
                try:
                    bus.write("Goal_Position", str(motor_id), target_minus, normalize=True)
                    time.sleep(1.0)
                    new_pos = bus.read("Present_Position", str(motor_id), normalize=True)
                    print(f"Actual: {new_pos:.1f}° ✓")
                except KeyboardInterrupt:
                    print("Skipped")
                    continue
                except Exception as e:
                    print(f"Error: {e}")
                    continue

                # Return to original
                print(f"  Returning to original ({current_pos:.1f}°)...", end=" ", flush=True)
                try:
                    bus.write("Goal_Position", str(motor_id), current_pos, normalize=True)
                    time.sleep(1.0)
                    print("Done")
                except Exception:
                    print("Failed to return")

        print()
        print("=" * 80)
        if failed:
            print("RECOMMENDATIONS:")
            print("  - Check power supply (must be 12V)")
            print("  - Check USB cable connection")
            print("  - Verify servo IDs are configured correctly")
            print("  - Try different USB port")
            print("  - Check servo firmware versions are consistent")
            if any(r["id"] == 6 for r in failed):
                print()
                print("  NOTE: Motor 6 (gripper) failure is common if:")
                print("    - Gripper is not physically connected")
                print("    - Servo ID is misconfigured")
                print("    - Servo has different firmware version")
        else:
            print("All motors working correctly!")
        print("=" * 80)

    except ImportError as e:
        print(f"Import error: {e}")
        print("Ensure LeRobot is installed: uv pip install lerobot[kinematics]")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if bus is not None:
            try:
                bus.disconnect()
            except Exception:
                pass
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Diagnostic test for SO-ARM101 servos (IDs 1-6)")
    parser.add_argument(
        "--port",
        help="Serial port (auto-detect if not specified)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive test moving each working motor",
    )

    args = parser.parse_args()

    # Auto-detect port if not specified
    if args.port is None:
        args.port = find_port()
        if args.port is None:
            print("ERROR: No serial port found. Connect robot or specify --port")
            sys.exit(1)

    run_motor_diagnostic(port=args.port, interactive=args.interactive)


if __name__ == "__main__":
    main()
