#!/usr/bin/env python3
"""Debug script to visualize Xbox controller input values.

Shows raw and processed values to help diagnose deadzone and scaling issues.

Usage:
    uv run python examples/debug_controller.py
    uv run python examples/debug_controller.py --deadzone 0.2
"""

import argparse
import sys
import time

from xbox_soarm_teleop.config.xbox_config import XboxConfig
from xbox_soarm_teleop.teleoperators.xbox import XboxController


def main():
    parser = argparse.ArgumentParser(description="Debug Xbox controller input")
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.15,
        help="Controller deadzone (0.0-1.0). Default: 0.15",
    )
    args = parser.parse_args()

    config = XboxConfig(deadzone=args.deadzone)
    controller = XboxController(config)

    print("Connecting to Xbox controller...")
    if not controller.connect():
        print("ERROR: Failed to connect to Xbox controller")
        sys.exit(1)

    print("Connected! Move sticks and triggers to see values.")
    print("Press Ctrl+C to exit.\n")
    print(f"Current deadzone: {config.deadzone}")
    print(f"Linear scale: {config.linear_scale} m/s")
    print(f"Angular scale: {config.angular_scale} rad/s")
    print("-" * 80)

    try:
        while True:
            state = controller.read()

            # Get raw values for comparison
            with controller._raw_state_lock:
                raw = controller._raw_state.copy()

            raw_lx = raw.get(config.left_stick_x_axis, 0)
            raw_ly = raw.get(config.left_stick_y_axis, 0)
            raw_rx = raw.get(config.right_stick_x_axis, 0)
            raw_ry = raw.get(config.right_stick_y_axis, 0)

            # Calculate stick magnitudes (for radial deadzone reference)
            import math

            left_mag = math.sqrt(state.left_stick_x**2 + state.left_stick_y**2)
            right_mag = math.sqrt(state.right_stick_x**2 + state.right_stick_y**2)

            line = (
                f"L({raw_lx:6d},{raw_ly:6d}) "
                f"n=({state.left_stick_x:+.3f},{state.left_stick_y:+.3f}) "
                f"m={left_mag:.3f} | "
                f"R({raw_rx:6d},{raw_ry:6d}) "
                f"n=({state.right_stick_x:+.3f},{state.right_stick_y:+.3f}) "
                f"m={right_mag:.3f} | "
                f"LB:{int(state.left_bumper)} RT:{state.right_trigger:.2f}"
            )
            print(f"\r\033[2K{line}", end="", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        try:
            controller.disconnect()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
