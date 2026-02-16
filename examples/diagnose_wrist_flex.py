#!/usr/bin/env python3
"""Diagnostic script to test wrist_flex range and identify interference.

This script moves the arm to various positions and tests wrist_flex range
to identify mechanical interference.

Usage:
    uv run python examples/diagnose_wrist_flex.py --port /dev/ttyACM0
"""

import argparse
import glob
import sys
import time

import numpy as np

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    JOINT_LIMITS_DEG,
    JOINT_NAMES_WITH_GRIPPER,
    MOTOR_IDS,
    deg_to_raw,
    raw_to_deg,
)


def find_port() -> str | None:
    """Find available serial port."""
    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def test_wrist_flex_range(port: str):
    """Test wrist_flex range at different arm configurations."""
    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode

    print("=" * 70)
    print("WRIST_FLEX DIAGNOSTIC")
    print("=" * 70)
    print(f"Port: {port}")
    print()

    # Initialize motor bus
    motors = {
        str(mid): Motor(id=mid, model="sts3215", norm_mode=MotorNormMode.DEGREES)
        for mid in MOTOR_IDS.values()
    }
    bus = FeetechMotorsBus(port=port, motors=motors)

    try:
        bus.connect()
        print("Connected to robot")
        print()

        # Test configurations: (description, joint_positions_dict)
        test_configs = [
            ("Home position", HOME_POSITION_DEG),
            (
                "Extended forward",
                {
                    **HOME_POSITION_DEG,
                    "shoulder_lift": 90.0,
                    "elbow_flex": -90.0,
                    "wrist_flex": 0.0,
                },
            ),
            (
                "Elbow at 0° (straight)",
                {
                    **HOME_POSITION_DEG,
                    "elbow_flex": 0.0,
                    "wrist_flex": 0.0,
                },
            ),
            (
                "Elbow at 15° (ROM test pos)",
                {
                    **HOME_POSITION_DEG,
                    "elbow_flex": 15.0,
                    "wrist_flex": 0.0,
                },
            ),
            (
                "Elbow at 45°",
                {
                    **HOME_POSITION_DEG,
                    "elbow_flex": 45.0,
                    "wrist_flex": 0.0,
                },
            ),
        ]

        def read_wrist_flex():
            """Read current wrist_flex position."""
            raw = bus.read("Present_Position", str(MOTOR_IDS["wrist_flex"]), normalize=False)
            return raw_to_deg(raw)

        def set_joint(name: str, deg: float):
            """Set single joint position."""
            raw = deg_to_raw(deg)
            raw = max(0, min(4095, raw))
            motor_id = MOTOR_IDS[name]
            bus.write("Torque_Enable", str(motor_id), 1, normalize=False)
            bus.write("Goal_Position", str(motor_id), raw, normalize=False)

        def go_to_positions(positions: dict, duration: float = 2.0):
            """Move to position set."""
            for name, deg in positions.items():
                set_joint(name, deg)
            time.sleep(duration)

        results = []

        for desc, positions in test_configs:
            print(f"\nTest: {desc}")
            print("-" * 50)

            # Move to configuration
            go_to_positions(positions)

            # Test wrist_flex limits
            print("  Testing wrist_flex limits...")

            # Go to negative limit
            set_joint("wrist_flex", JOINT_LIMITS_DEG["wrist_flex"][0])
            time.sleep(1.0)
            neg_actual = read_wrist_flex()
            print(
                f"    Negative limit: commanded {JOINT_LIMITS_DEG['wrist_flex'][0]:.1f}°, actual {neg_actual:.1f}°"
            )

            # Go to positive limit
            set_joint("wrist_flex", 75.0)  # Test up to 75°
            time.sleep(1.0)
            pos_75 = read_wrist_flex()
            print(f"    Positive limit (75°): commanded 75.0°, actual {pos_75:.1f}°")

            # Try old 95° limit
            set_joint("wrist_flex", 95.0)
            time.sleep(1.0)
            pos_95 = read_wrist_flex()
            print(f"    Positive limit (95°): commanded 95.0°, actual {pos_95:.1f}°")

            results.append(
                {
                    "config": desc,
                    "negative": neg_actual,
                    "positive_75": pos_75,
                    "positive_95": pos_95,
                }
            )

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Configuration':<30} {'Neg (°)':<10} {'+75° (°)':<12} {'+95° (°)':<12}")
        print("-" * 70)
        for r in results:
            print(
                f"{r['config']:<30} {r['negative']:>8.1f}  {r['positive_75']:>10.1f}  {r['positive_95']:>10.1f}"
            )

        print("\nRecommendations:")
        max_pos = max(r["positive_75"] for r in results)
        if max_pos < 90.0:
            print(f"  • Wrist_flex upper limit should be ~{max_pos:.0f}° (not 95°)")
            print("  • This is due to mechanical interference between wrist and forearm")
        else:
            print("  • Wrist_flex can achieve 75°+ in some configurations")
            print("  • ROM test position may need adjustment")

        # Return to home
        print("\nReturning to home...")
        go_to_positions(HOME_POSITION_DEG, duration=3.0)

    finally:
        bus.disconnect()
        print("\nDisconnected")


def main():
    parser = argparse.ArgumentParser(description="Diagnose wrist_flex range issues")
    parser.add_argument("--port", help="Serial port (auto-detect if not specified)")
    args = parser.parse_args()

    port = args.port or find_port()
    if port is None:
        print("ERROR: No serial port found")
        sys.exit(1)

    test_wrist_flex_range(port)


if __name__ == "__main__":
    main()
