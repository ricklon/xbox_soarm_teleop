#!/usr/bin/env python3
"""Test Xbox controller using pygame (alternative to inputs library)."""

import sys
import time


def test_controller():
    try:
        import pygame
    except ImportError:
        print("ERROR: pygame not installed")
        print("Install with: uv add pygame")
        sys.exit(1)

    pygame.init()
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    print(f"Joysticks found: {joystick_count}")

    if joystick_count == 0:
        print("ERROR: No controllers found")
        pygame.quit()
        sys.exit(1)

    # Initialize first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"\nController: {joystick.get_name()}")
    print(f"Axes: {joystick.get_numaxes()}")
    print(f"Buttons: {joystick.get_numbuttons()}")
    print(f"Hats: {joystick.get_numhats()}")
    print()

    print("Testing for 5 seconds...")
    print("Move sticks, press LB, etc.")
    print()

    start = time.time()
    last_print = 0

    while time.time() - start < 5:
        pygame.event.pump()

        # Read axes
        left_x = joystick.get_axis(0)
        left_y = joystick.get_axis(1)
        right_x = joystick.get_axis(3)
        right_y = joystick.get_axis(4)
        left_trigger = joystick.get_axis(2)
        right_trigger = joystick.get_axis(5)

        # Read buttons (common Xbox mapping)
        a_btn = joystick.get_button(0)
        b_btn = joystick.get_button(1)
        x_btn = joystick.get_button(2)
        y_btn = joystick.get_button(3)
        lb = joystick.get_button(4)  # Left bumper
        rb = joystick.get_button(5)  # Right bumper
        back = joystick.get_button(6)
        start = joystick.get_button(7)
        l_stick = joystick.get_button(9)
        r_stick = joystick.get_button(10)

        # Read hat (D-pad)
        hat = joystick.get_hat(0) if joystick.get_numhats() > 0 else (0, 0)

        # Print only when values change significantly or every 1 second
        now = time.time()
        if now - last_print >= 1.0 or abs(left_x) > 0.5 or abs(left_y) > 0.5 or lb:
            print(f"LX={left_x:+.2f} LY={left_y:+.2f} RX={right_x:+.2f} RY={right_y:+.2f}")
            print(f"  LT={left_trigger:+.2f} RT={right_trigger:+.2f}")
            print(f"  A={a_btn} B={b_btn} X={x_btn} Y={y_btn}")
            print(f"  LB={lb} RB={rb} HAT={hat}")
            print()
            last_print = now

        time.sleep(0.05)

    pygame.quit()
    print("Test complete!")


if __name__ == "__main__":
    test_controller()
