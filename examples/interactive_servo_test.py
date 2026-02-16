#!/usr/bin/env python3
"""Interactive servo test - check and move individual motors.

Features:
1. Select motor (1-6) and view state (position, temp, velocity, voltage)
2. Jog mode - move incrementally with +/- keys
3. Move to specific position (raw 0-4095 or degrees)
4. Sweep test - move through full range
5. Save/load positions for calibration
6. Quick test all motors

Usage:
    uv run python examples/interactive_servo_test.py
    uv run python examples/interactive_servo_test.py --port /dev/ttyACM0
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

# Position save file
POSITIONS_FILE = Path.home() / ".cache" / "xbox_soarm_teleop" / "servo_positions.json"

# Raw to degrees conversion (STS3215: 4096 steps = 360 degrees)
RAW_PER_DEGREE = 4096 / 360  # ~11.38


def find_port() -> str | None:
    """Find available serial port."""
    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


MOTOR_NAMES = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}

# Default home position (raw values)
# This is the "parked" position for the arm
DEFAULT_HOME = {
    1: 2032,  # shoulder_pan (~0°)
    2: 958,  # shoulder_lift (-96°)
    3: 3139,  # elbow_flex (+96°)
    4: 2838,  # wrist_flex (+69°)
    5: 2078,  # wrist_roll (~0°)
    6: 2042,  # gripper (~0°)
}


def raw_to_deg(raw: int) -> float:
    """Convert raw position (0-4095) to degrees (-180 to +180)."""
    return (raw - 2048) / RAW_PER_DEGREE


def deg_to_raw(deg: float) -> int:
    """Convert degrees to raw position (0-4095)."""
    return int(2048 + deg * RAW_PER_DEGREE)


def read_servo_state(bus, motor_id: str) -> dict:
    """Read current state of a servo."""
    state = {}

    # Position
    try:
        state["position"] = bus.read("Present_Position", motor_id, normalize=False)
    except Exception as e:
        state["position"] = f"Error: {e}"

    # Temperature
    try:
        state["temperature"] = bus.read("Present_Temperature", motor_id, normalize=False)
    except Exception as e:
        state["temperature"] = f"Error: {e}"

    # Velocity
    try:
        state["velocity"] = bus.read("Present_Velocity", motor_id, normalize=False)
    except Exception as e:
        state["velocity"] = f"Error: {e}"

    # Voltage
    try:
        raw_voltage = bus.read("Present_Input_Voltage", motor_id, normalize=False)
        state["voltage"] = raw_voltage / 10.0  # Convert to volts
    except Exception:
        state["voltage"] = None

    # Load/current
    try:
        state["load"] = bus.read("Present_Load", motor_id, normalize=False)
    except Exception:
        state["load"] = None

    return state


def print_servo_state(motor_id: int, state: dict):
    """Print servo state in a formatted way."""
    name = MOTOR_NAMES[motor_id]
    print(f"\n{'=' * 60}")
    print(f"Motor ID {motor_id} - {name}")
    print(f"{'=' * 60}")

    pos = state.get("position", "N/A")
    temp = state.get("temperature", "N/A")
    vel = state.get("velocity", "N/A")
    voltage = state.get("voltage")
    load = state.get("load")

    if isinstance(pos, int):
        deg = raw_to_deg(pos)
        print(f"  Position:    {pos:5d} raw  ({deg:+7.1f} deg)")
    else:
        print(f"  Position:    {pos}")

    if isinstance(temp, int):
        print(f"  Temperature: {temp:5d} C")
    else:
        print(f"  Temperature: {temp}")

    if isinstance(vel, int):
        print(f"  Velocity:    {vel:5d}")
    else:
        print(f"  Velocity:    {vel}")

    if voltage is not None:
        print(f"  Voltage:     {voltage:5.1f} V")

    if load is not None:
        print(f"  Load:        {load:5d}")

    print(f"{'=' * 60}")


def move_servo(bus, motor_id: str, target_raw: int, wait: float = 0.5) -> tuple[bool, int | str]:
    """Move servo to target position and return (success, actual_position)."""
    target_raw = max(0, min(4095, target_raw))  # Clamp to valid range
    try:
        bus.write("Torque_Enable", motor_id, 1, normalize=False)
        bus.write("Goal_Position", motor_id, target_raw, normalize=False)
        time.sleep(wait)
        actual = bus.read("Present_Position", motor_id, normalize=False)
        return True, actual
    except Exception as e:
        return False, str(e)


def jog_mode(bus, motor_id: int):
    """Interactive jog mode for a single motor."""
    motor_key = str(motor_id)
    name = MOTOR_NAMES[motor_id]

    print(f"\n{'=' * 60}")
    print(f"JOG MODE - Motor {motor_id} ({name})")
    print(f"{'=' * 60}")
    print("Commands:")
    print("  +/=     : Move +50 raw (~4.4 deg)")
    print("  -       : Move -50 raw (~4.4 deg)")
    print("  ]/}     : Move +10 raw (~0.9 deg)")
    print("  [/{     : Move -10 raw (~0.9 deg)")
    print("  0       : Go to center (2048)")
    print("  r       : Read current position")
    print("  q       : Quit jog mode")
    print()

    # Enable torque
    try:
        bus.write("Torque_Enable", motor_key, 1, normalize=False)
    except Exception as e:
        print(f"Failed to enable torque: {e}")
        return

    while True:
        # Read current position
        try:
            pos = bus.read("Present_Position", motor_key, normalize=False)
            deg = raw_to_deg(pos)
            print(f"  Pos: {pos:4d} raw ({deg:+6.1f} deg) > ", end="", flush=True)
        except Exception as e:
            print(f"  Read error: {e} > ", end="", flush=True)
            pos = 2048

        cmd = input().strip().lower()

        if cmd == "q":
            break
        elif cmd in ["+", "="]:
            move_servo(bus, motor_key, pos + 50, wait=0.3)
        elif cmd == "-":
            move_servo(bus, motor_key, pos - 50, wait=0.3)
        elif cmd in ["]", "}"]:
            move_servo(bus, motor_key, pos + 10, wait=0.2)
        elif cmd in ["[", "{"]:
            move_servo(bus, motor_key, pos - 10, wait=0.2)
        elif cmd == "0":
            move_servo(bus, motor_key, 2048, wait=0.5)
        elif cmd == "r":
            pass  # Just re-read (happens at top of loop)
        elif cmd:
            # Try to parse as a number (raw position)
            try:
                target = int(cmd)
                if 0 <= target <= 4095:
                    move_servo(bus, motor_key, target, wait=0.5)
                else:
                    print("  (Enter 0-4095 for raw position)")
            except ValueError:
                print("  (Unknown command)")

    print("Exited jog mode")


def sweep_test(bus, motor_id: int, range_raw: int = 1000):
    """Sweep motor through a range to test movement."""
    motor_key = str(motor_id)
    name = MOTOR_NAMES[motor_id]

    print(f"\nSweep test for motor {motor_id} ({name})")

    # Get current position
    try:
        start_pos = bus.read("Present_Position", motor_key, normalize=False)
    except Exception as e:
        print(f"Failed to read position: {e}")
        return

    # Calculate sweep range (centered on current position, clamped to 0-4095)
    min_pos = max(0, start_pos - range_raw // 2)
    max_pos = min(4095, start_pos + range_raw // 2)

    print(f"  Start: {start_pos} ({raw_to_deg(start_pos):+.1f} deg)")
    print(
        f"  Range: {min_pos} to {max_pos} ({raw_to_deg(min_pos):+.1f} to {raw_to_deg(max_pos):+.1f} deg)"
    )

    confirm = input("  Proceed with sweep? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Cancelled")
        return

    # Enable torque
    bus.write("Torque_Enable", motor_key, 1, normalize=False)

    # Sweep: current -> min -> max -> current
    steps = [
        ("min", min_pos),
        ("max", max_pos),
        ("start", start_pos),
    ]

    for label, target in steps:
        print(f"  Moving to {label} ({target})...", end=" ", flush=True)
        success, actual = move_servo(bus, motor_key, target, wait=1.0)
        if success:
            error = abs(actual - target)
            print(f"reached {actual} (error: {error})")
        else:
            print(f"FAILED: {actual}")
            return

    print("  Sweep test PASSED")


def test_each_joint(bus):
    """Test each joint with movement (±200 steps)."""
    print("\n" + "=" * 60)
    print("JOINT MOVEMENT TEST")
    print("=" * 60)
    print("Testing each motor with ±200 step movement")
    print()

    results = []
    move_amount = 200

    for motor_id in range(1, 7):
        motor_key = str(motor_id)
        name = MOTOR_NAMES[motor_id]
        print(f"\nMotor {motor_id} ({name}):")

        # Get current position
        try:
            start_pos = bus.read("Present_Position", motor_key, normalize=False)
            print(f"  Start:  {start_pos:4d} ({raw_to_deg(start_pos):+6.1f} deg)")
        except Exception as e:
            print(f"  FAILED to read: {e}")
            results.append((motor_id, name, "READ_FAIL", None))
            continue

        # Test +200
        target_plus = min(4095, start_pos + move_amount)
        success, actual_plus = move_servo(bus, motor_key, target_plus, wait=0.8)
        if success:
            error_plus = abs(actual_plus - target_plus)
            print(f"  +{move_amount}:   {actual_plus:4d} (target {target_plus}, error {error_plus})")
        else:
            print(f"  +{move_amount}:   FAILED - {actual_plus}")
            results.append((motor_id, name, "MOVE_FAIL", None))
            continue

        # Test -200
        target_minus = max(0, start_pos - move_amount)
        success, actual_minus = move_servo(bus, motor_key, target_minus, wait=0.8)
        if success:
            error_minus = abs(actual_minus - target_minus)
            print(f"  -{move_amount}:   {actual_minus:4d} (target {target_minus}, error {error_minus})")
        else:
            print(f"  -{move_amount}:   FAILED - {actual_minus}")
            results.append((motor_id, name, "MOVE_FAIL", None))
            continue

        # Return to start
        success, actual_return = move_servo(bus, motor_key, start_pos, wait=0.8)
        if success:
            error_return = abs(actual_return - start_pos)
            print(f"  Return: {actual_return:4d} (target {start_pos}, error {error_return})")
        else:
            print(f"  Return: FAILED - {actual_return}")

        # Calculate max error
        max_error = max(error_plus, error_minus, error_return if success else 999)
        status = "PASS" if max_error <= 50 else "WARN" if max_error <= 200 else "FAIL"
        results.append((motor_id, name, status, max_error))

    # Summary
    print("\n" + "=" * 60)
    print("MOVEMENT TEST SUMMARY")
    print("=" * 60)
    print(f"{'Motor':<20} {'Status':<8} {'Max Error'}")
    print("-" * 40)
    for motor_id, name, status, error in results:
        error_str = str(error) if error is not None else "N/A"
        print(f"{motor_id}. {name:<15} {status:<8} {error_str}")

    passed = sum(1 for _, _, s, _ in results if s == "PASS")
    warned = sum(1 for _, _, s, _ in results if s == "WARN")
    failed = sum(1 for _, _, s, _ in results if s in ["FAIL", "READ_FAIL", "MOVE_FAIL"])
    print("-" * 40)
    print(f"PASS: {passed}  WARN: {warned}  FAIL: {failed}")
    print("=" * 60)


def test_all_motors(bus):
    """Quick test of all motors."""
    print("\n" + "=" * 60)
    print("QUICK TEST - ALL MOTORS")
    print("=" * 60)

    results = []
    for motor_id in range(1, 7):
        motor_key = str(motor_id)
        name = MOTOR_NAMES[motor_id]
        print(f"  {motor_id}. {name:15s} ", end="", flush=True)

        try:
            pos = bus.read("Present_Position", motor_key, normalize=False)
            temp = bus.read("Present_Temperature", motor_key, normalize=False)
            deg = raw_to_deg(pos)
            print(f"OK  pos={pos:4d} ({deg:+6.1f} deg)  temp={temp}C")
            results.append((motor_id, True, pos))
        except Exception as e:
            print(f"FAIL  {e}")
            results.append((motor_id, False, None))

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\nResult: {passed}/6 motors responding")
    return results


def save_positions(bus, name: str):
    """Save current positions of all motors."""
    positions = {}
    for motor_id in range(1, 7):
        try:
            pos = bus.read("Present_Position", str(motor_id), normalize=False)
            positions[str(motor_id)] = pos
        except Exception:
            positions[str(motor_id)] = None

    # Load existing file
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    if POSITIONS_FILE.exists():
        try:
            saved = json.loads(POSITIONS_FILE.read_text())
        except Exception:
            pass

    saved[name] = {
        "positions": positions,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    POSITIONS_FILE.write_text(json.dumps(saved, indent=2))
    print(f"Saved position '{name}': {positions}")


def load_positions(bus, name: str):
    """Load and move to saved positions."""
    if not POSITIONS_FILE.exists():
        print("No saved positions file")
        return

    try:
        saved = json.loads(POSITIONS_FILE.read_text())
    except Exception as e:
        print(f"Failed to read positions: {e}")
        return

    if name not in saved:
        print(f"Position '{name}' not found. Available: {list(saved.keys())}")
        return

    positions = saved[name]["positions"]
    print(f"Loading position '{name}'...")

    for motor_id in range(1, 7):
        motor_key = str(motor_id)
        target = positions.get(motor_key)
        if target is None:
            print(f"  Motor {motor_id}: skipped (no saved value)")
            continue

        print(f"  Motor {motor_id}: moving to {target}...", end=" ", flush=True)
        success, actual = move_servo(bus, motor_key, target, wait=0.8)
        if success:
            print(f"reached {actual}")
        else:
            print(f"FAILED: {actual}")


def list_saved_positions():
    """List all saved positions."""
    if not POSITIONS_FILE.exists():
        print("No saved positions")
        return

    try:
        saved = json.loads(POSITIONS_FILE.read_text())
    except Exception as e:
        print(f"Failed to read: {e}")
        return

    print("\nSaved positions:")
    for name, data in saved.items():
        ts = data.get("timestamp", "?")
        pos = data.get("positions", {})
        print(f"  {name}: {pos} ({ts})")


def get_home_position() -> dict[int, int]:
    """Get home position (from saved file or default)."""
    if POSITIONS_FILE.exists():
        try:
            saved = json.loads(POSITIONS_FILE.read_text())
            if "home" in saved:
                positions = saved["home"]["positions"]
                return {int(k): v for k, v in positions.items() if v is not None}
        except Exception:
            pass
    return DEFAULT_HOME.copy()


def go_home(bus, tolerance: int = 50):
    """Move all motors to home position.

    Args:
        bus: Motor bus connection
        tolerance: Skip motors already within this many steps of target
    """
    home = get_home_position()
    print("\nGoing to home position...")

    moved = 0
    for motor_id in range(1, 7):
        motor_key = str(motor_id)
        target = home.get(motor_id)
        if target is None:
            print(f"  Motor {motor_id}: skipped (no home value)")
            continue

        # Check current position - skip if already close
        try:
            current = bus.read("Present_Position", motor_key, normalize=False)
            diff = abs(current - target)
            if diff <= tolerance:
                print(
                    f"  Motor {motor_id} ({MOTOR_NAMES[motor_id]}): already at home ({current}, diff={diff})"
                )
                continue
        except Exception:
            pass  # If read fails, try to move anyway

        print(
            f"  Motor {motor_id} ({MOTOR_NAMES[motor_id]}): {current} -> {target}...",
            end=" ",
            flush=True,
        )
        success, actual = move_servo(bus, motor_key, target, wait=0.5)
        if success:
            error = abs(actual - target)
            print(f"done (error: {error})")
            moved += 1
        else:
            print(f"FAILED: {actual}")

    if moved == 0:
        print("Already at home - no movement needed.")
    else:
        print(f"Home position reached ({moved} motors moved).")


def set_home(bus):
    """Save current position as home."""
    print("\nSetting current position as HOME...")
    save_positions(bus, "home")
    print("Home position updated. Use 'h' to return here.")


def range_discovery(bus, motor_id: int):
    """Discover joint range by manually moving it with torque off.

    Disables torque so you can move the joint by hand.
    Continuously reads and displays position until you press Enter.
    Tracks min/max positions discovered.
    """
    motor_key = str(motor_id)
    name = MOTOR_NAMES[motor_id]

    print(f"\n{'=' * 60}")
    print(f"RANGE DISCOVERY - Motor {motor_id} ({name})")
    print(f"{'=' * 60}")
    print("Torque will be DISABLED so you can move the joint by hand.")
    print("Move the joint through its full range.")
    print("Press Enter when done to see min/max values.")
    print()

    # Disable torque
    try:
        bus.write("Torque_Enable", motor_key, 0, normalize=False)
        print("Torque DISABLED - you can now move the joint manually")
    except Exception as e:
        print(f"Failed to disable torque: {e}")
        return

    min_pos = 4095
    max_pos = 0
    readings = 0

    print("\nReading positions (press Enter to stop)...")
    print()

    import select
    import sys
    import termios
    import tty

    # Set terminal to raw mode for non-blocking input
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            # Check for Enter key (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch in ["\n", "\r"]:
                    break

            # Read position
            try:
                pos = bus.read("Present_Position", motor_key, normalize=False)
                deg = raw_to_deg(pos)
                min_pos = min(min_pos, pos)
                max_pos = max(max_pos, pos)
                readings += 1

                # Display with min/max
                print(
                    f"\r  Pos: {pos:4d} ({deg:+6.1f} deg)  |  "
                    f"Min: {min_pos:4d} ({raw_to_deg(min_pos):+6.1f} deg)  "
                    f"Max: {max_pos:4d} ({raw_to_deg(max_pos):+6.1f} deg)  ",
                    end="",
                    flush=True,
                )
            except Exception as e:
                print(f"\r  Read error: {e}  ", end="", flush=True)

            time.sleep(0.05)  # 20Hz update rate

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    # Summary
    print("\n")
    print(f"{'=' * 60}")
    print(f"RANGE DISCOVERY RESULTS - Motor {motor_id} ({name})")
    print(f"{'=' * 60}")
    if readings > 0:
        range_raw = max_pos - min_pos
        range_deg = raw_to_deg(max_pos) - raw_to_deg(min_pos)
        print(f"  Minimum: {min_pos:4d} raw ({raw_to_deg(min_pos):+6.1f} deg)")
        print(f"  Maximum: {max_pos:4d} raw ({raw_to_deg(max_pos):+6.1f} deg)")
        print(f"  Range:   {range_raw:4d} raw ({range_deg:6.1f} deg)")
        print(f"  Readings: {readings}")
    else:
        print("  No readings captured")
    print(f"{'=' * 60}")

    # Ask if user wants to re-enable torque
    enable = input("\nRe-enable torque? (y/n): ").strip().lower()
    if enable == "y":
        try:
            bus.write("Torque_Enable", motor_key, 1, normalize=False)
            print("Torque enabled")
        except Exception as e:
            print(f"Failed to enable torque: {e}")


def print_menu():
    """Print main menu."""
    print("\n" + "=" * 60)
    print("MAIN MENU")
    print("=" * 60)
    print("  1-6  : Select motor to inspect/control")
    print("  a    : Test ALL motors (quick check)")
    print("  t    : Test each joint with movement (±200 steps)")
    print("  r    : Range discovery (manually move joint)")
    print("  h    : Go HOME (move all to home position)")
    print("  H    : Set current position as HOME")
    print("  j    : Jog mode (select motor first)")
    print("  s    : Save current positions")
    print("  l    : Load saved positions")
    print("  p    : List saved positions")
    print("  q    : Quit")
    print()


def run_interactive_test(port: str):
    """Run interactive servo test."""
    try:
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode
    except ImportError:
        print("ERROR: LeRobot not installed")
        print("Install with: uv pip install lerobot")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("INTERACTIVE SERVO TEST")
    print("=" * 70)
    print(f"Port: {port}")
    print()

    # Initialize motors
    motors = {
        str(i): Motor(id=i, model="sts3215", norm_mode=MotorNormMode.DEGREES) for i in range(1, 7)
    }
    bus = FeetechMotorsBus(port=port, motors=motors)

    print("Connecting to robot...")
    try:
        bus.connect()
        print("Connected successfully!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    selected_motor = None

    try:
        while True:
            print_menu()
            if selected_motor:
                print(f"  (Selected: motor {selected_motor} - {MOTOR_NAMES[selected_motor]})")

            choice = input("Command: ").strip()

            if choice.lower() == "q":
                print("\nExiting...")
                break

            elif choice.lower() == "a":
                test_all_motors(bus)

            elif choice.lower() == "t":
                # Test each joint with movement
                test_each_joint(bus)

            elif choice.lower() == "r":
                # Range discovery mode
                motor_str = input("Which motor for range discovery? (1-6): ").strip()
                try:
                    motor_id = int(motor_str)
                    if motor_id in range(1, 7):
                        range_discovery(bus, motor_id)
                    else:
                        print("Invalid motor ID")
                except ValueError:
                    print("Invalid input")

            elif choice == "h":
                go_home(bus)

            elif choice == "H":
                set_home(bus)

            elif choice.lower() == "j":
                if selected_motor is None:
                    motor_str = input("Which motor to jog? (1-6): ").strip()
                    try:
                        selected_motor = int(motor_str)
                        if selected_motor not in range(1, 7):
                            print("Invalid motor ID")
                            selected_motor = None
                            continue
                    except ValueError:
                        print("Invalid input")
                        continue
                jog_mode(bus, selected_motor)

            elif choice.lower() == "s":
                name = input("Save name: ").strip()
                if name:
                    save_positions(bus, name)

            elif choice.lower() == "l":
                name = input("Load name: ").strip()
                if name:
                    load_positions(bus, name)

            elif choice.lower() == "p":
                list_saved_positions()

            elif choice in ["1", "2", "3", "4", "5", "6"]:
                motor_id = int(choice)
                selected_motor = motor_id
                motor_key = str(motor_id)

                # Read and display state
                print(f"\nReading motor {motor_id} ({MOTOR_NAMES[motor_id]})...")
                state = read_servo_state(bus, motor_key)
                print_servo_state(motor_id, state)

                # Sub-menu for this motor
                print("Actions:")
                print("  m : Move test (±200 steps)")
                print("  w : Sweep test (full range)")
                print("  j : Jog mode")
                print("  g : Go to position (enter raw 0-4095)")
                print("  Enter : Back to main menu")

                action = input("Action: ").strip().lower()

                if action == "m":
                    # Basic movement test
                    if isinstance(state.get("position"), int):
                        current_pos = state["position"]
                        move_amount = 200

                        print(f"\nMoving ±{move_amount} from {current_pos}...")
                        for target in [
                            current_pos + move_amount,
                            current_pos - move_amount,
                            current_pos,
                        ]:
                            target = max(0, min(4095, target))
                            success, actual = move_servo(bus, motor_key, target, wait=0.8)
                            if success:
                                print(f"  -> {target}: reached {actual}")
                            else:
                                print(f"  -> {target}: FAILED {actual}")
                                break

                elif action == "w":
                    sweep_test(bus, motor_id)

                elif action == "j":
                    jog_mode(bus, motor_id)

                elif action == "g":
                    pos_str = input("Target position (0-4095): ").strip()
                    try:
                        target = int(pos_str)
                        if 0 <= target <= 4095:
                            success, actual = move_servo(bus, motor_key, target)
                            if success:
                                print(f"Moved to {actual}")
                            else:
                                print(f"Failed: {actual}")
                        else:
                            print("Position must be 0-4095")
                    except ValueError:
                        print("Invalid number")

            else:
                print("Unknown command")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        bus.disconnect()
        print("Disconnected from robot")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive servo test - check and move individual motors"
    )
    parser.add_argument("--port", help="Serial port (auto-detect if not specified)")

    args = parser.parse_args()

    if args.port is None:
        args.port = find_port()
        if args.port is None:
            print("ERROR: No serial port found. Connect robot or specify --port")
            sys.exit(1)

    run_interactive_test(port=args.port)


if __name__ == "__main__":
    main()
