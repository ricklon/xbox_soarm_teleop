#!/usr/bin/env python3
"""Power cycle test for stuck elbow_flex servo (ID 3).

Instructions:
1. This script will power off the torque
2. You manually move the elbow joint
3. Power back on and test
"""

import glob
import sys
import time

from xbox_soarm_teleop.config.joints import MOTOR_IDS, deg_to_raw, raw_to_deg


def find_port():
    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def power_cycle_test():
    port = find_port()
    if not port:
        print("ERROR: No serial port found")
        sys.exit(1)

    print("=" * 60)
    print("ELBOW_FLEX (ID 3) POWER CYCLE TEST")
    print("=" * 60)
    print()

    import scservo_sdk as scs
    from scservo_sdk import PortHandler, sms_sts

    ph = PortHandler(port)
    handler = sms_sts(ph)

    if not ph.openPort():
        print("ERROR: Failed to open port")
        sys.exit(1)
    ph.setBaudRate(1000000)

    motor_id = MOTOR_IDS["elbow_flex"]

    # Step 1: Check current position
    pos_raw, _, _ = handler.read2ByteTxRx(motor_id, scs.SMS_STS_PRESENT_POSITION_L)
    print(f"Current position: {raw_to_deg(pos_raw):.1f}°")
    print()

    # Step 2: Disable torque (power off)
    print("Step 1: DISABLING TORQUE...")
    print("The servo is now free to move.")
    handler.write1ByteTxRx(motor_id, scs.SMS_STS_TORQUE_ENABLE, 0)
    print()

    print("=" * 60)
    print("MANUAL TEST REQUIRED:")
    print("=" * 60)
    print()
    print("1. Gently try to move the elbow joint (ID 3) by hand")
    print("   - It should move freely from ~70° to ~95°")
    print("   - If it feels stuck or gritty, there's mechanical damage")
    print()
    print("2. Check for cable snags around the elbow")
    print("   - Look for wires caught in the joint")
    print()
    print("3. Check if something is physically blocking the joint")
    print()

    input("Press ENTER when you've checked the joint manually...")
    print()

    # Step 3: Re-enable torque
    print("Step 2: RE-ENABLING TORQUE...")
    handler.write1ByteTxRx(motor_id, scs.SMS_STS_TORQUE_ENABLE, 1)
    time.sleep(0.5)

    # Step 4: Test movement to different positions
    print()
    print("Step 3: TESTING MOVEMENT")
    print("-" * 60)

    test_positions = [
        ("Current", raw_to_deg(pos_raw)),
        ("75°", 75.0),
        ("80°", 80.0),
        ("85°", 85.0),
        ("90°", 90.0),
        ("Home (96°)", 95.9),
    ]

    for name, target_deg in test_positions:
        target_raw = deg_to_raw(target_deg)
        print(f"\nMoving to {name} ({target_deg:.1f}°)...")

        handler.WritePosEx(motor_id, target_raw, 800, 100)
        time.sleep(2.0)

        pos_raw, _, _ = handler.read2ByteTxRx(motor_id, scs.SMS_STS_PRESENT_POSITION_L)
        actual = raw_to_deg(pos_raw)
        error = abs(actual - target_deg)

        status = "✓ PASS" if error < 3 else "✗ FAIL"
        print(f"  Target: {target_deg:.1f}° | Actual: {actual:.1f}° | Error: {error:.1f}° [{status}]")

        # If it's stuck at same position, stop testing
        if name != "Current" and abs(actual - 68.1) < 1:
            print("\n  ⚠️  SERVO IS STUCK - Cannot move from 68°")
            print("  Possible causes:")
            print("    - Internal gear damage")
            print("    - Encoder failure")
            print("    - Mechanical binding")
            print()
            break

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    # Return to home if possible
    print("\nReturning to home position...")
    home_raw = deg_to_raw(95.9)
    handler.WritePosEx(motor_id, home_raw, 800, 100)
    time.sleep(2.0)

    pos_raw, _, _ = handler.read2ByteTxRx(motor_id, scs.SMS_STS_PRESENT_POSITION_L)
    print(f"Final position: {raw_to_deg(pos_raw):.1f}°")

    ph.closePort()
    print("\nDone!")


if __name__ == "__main__":
    try:
        power_cycle_test()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
