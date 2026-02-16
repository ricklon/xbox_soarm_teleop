#!/usr/bin/env python3
"""Test Xbox controller using evdev (lower level than inputs/pygame)."""

import glob
import sys
import time


def find_controller():
    """Find Xbox controller event device."""
    try:
        import evdev
    except ImportError:
        print("ERROR: evdev not installed")
        print("Install with: uv add evdev")
        return None

    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    for device in devices:
        if "Microsoft" in device.name or "Xbox" in device.name:
            return device
    return None


def test_controller():
    device = find_controller()
    if not device:
        print("ERROR: No Xbox controller found")
        sys.exit(1)

    print(f"Controller: {device.name}")
    print(f"Path: {device.path}")
    print(f"Bus: {device.bus}\n")

    print("Capabilities:")
    for event_type, codes in device.capabilities().items():
        print(f"  {event_type}: {list(codes)[:5]}...")  # First 5 codes
    print()

    print("Reading events for 5 seconds...")
    print("Move sticks, press buttons, etc.")
    print()

    # Put device in non-blocking mode
    device.grab()

    try:
        start = time.time()
        last_print = 0
        count = 0

        while time.time() - start < 5:
            try:
                for event in device.read():
                    if event.type != evdev.ecodes.EV_SYN:  # Skip sync events
                        count += 1
                        if time.time() - last_print >= 0.5:  # Print every 0.5s
                            print(f"{event.code}: {event.value}")
                            last_print = time.time()
            except BlockingIOError:
                time.sleep(0.01)
                continue

        print(f"\nTotal events: {count}")
        if count == 0:
            print("\n⚠️  No events received!")
            print("Possible causes:")
            print("  - Controller needs to be reconnected")
            print("  - Different driver mode (try Xbox button + A)")
            print("  - Event device busy (another process using it)")
    finally:
        device.ungrab()


if __name__ == "__main__":
    import evdev
    from evdev import ecodes

    test_controller()
