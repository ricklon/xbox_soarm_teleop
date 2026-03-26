#!/usr/bin/env python3
"""Probe a connected gamepad and print all axis/button values as you move/press things.

Uses evdev directly — works with Joy-Cons via hid-nintendo + joycond.

Usage:
    uv run python examples/probe_controller.py
    uv run python examples/probe_controller.py --device /dev/input/event17
"""

import argparse
import sys
import time


def find_joycon_devices():
    """Return evdev devices that look like Joy-Cons or gamepads."""
    try:
        import evdev
    except ImportError:
        print("ERROR: evdev not installed. Run: uv pip install evdev")
        sys.exit(1)

    found = []
    for path in evdev.list_devices():
        try:
            d = evdev.InputDevice(path)
            caps = d.capabilities()
            # Must have absolute axes (sticks) or keys (buttons)
            has_abs = 3 in caps  # EV_ABS
            has_keys = 1 in caps  # EV_KEY
            is_imu = "IMU" in d.name
            if (has_abs or has_keys) and not is_imu:
                # Exclude pure keyboards/mice/touchscreens
                if any(k in d.name.lower() for k in ("joy", "nintendo", "switch", "gamepad", "controller")):
                    found.append(d)
        except PermissionError:
            pass
    return found


def main():
    parser = argparse.ArgumentParser(description="Probe gamepad event codes")
    parser.add_argument("--device", help="Explicit /dev/input/eventN path")
    args = parser.parse_args()

    try:
        import evdev
    except ImportError:
        print("ERROR: evdev not installed. Run: uv pip install evdev")
        sys.exit(1)

    if args.device:
        try:
            devices = [evdev.InputDevice(args.device)]
        except PermissionError:
            print(f"ERROR: Permission denied on {args.device}")
            print("Fix: sudo usermod -aG input $USER  (then log out/in)")
            sys.exit(1)
    else:
        devices = find_joycon_devices()
        if not devices:
            # Fall back: list all readable non-IMU abs/key devices
            all_devs = []
            for path in evdev.list_devices():
                try:
                    d = evdev.InputDevice(path)
                    caps = d.capabilities()
                    if (3 in caps or 1 in caps) and "IMU" not in d.name:
                        all_devs.append(d)
                except PermissionError:
                    pass
            if all_devs:
                print("No Joy-Con found by name. Available input devices:")
                for d in all_devs:
                    print(f"  {d.path}  {d.name}")
                print("\nRe-run with --device /dev/input/eventN")
            else:
                print("No readable input devices found.")
                print("Is the Joy-Con connected? Try: bluetoothctl connect 98:B6:AF:5E:5D:39")
            sys.exit(1)

    print(f"Found {len(devices)} device(s):")
    for d in devices:
        print(f"  {d.path}  {d.name}")
    print()

    dev = devices[0]
    print(f"Probing: {dev.name}  ({dev.path})")
    print("Move sticks, press buttons — Ctrl+C to quit\n")

    seen: dict = {}

    try:
        for event in dev.read_loop():
            if event.type not in (evdev.ecodes.EV_ABS, evdev.ecodes.EV_KEY):
                continue
            ts = time.monotonic()
            code_name = evdev.ecodes.bytype[event.type].get(event.code, str(event.code))
            key = (event.type, event.code)
            if seen.get(key) != event.value:
                seen[key] = event.value
                type_name = "ABS  " if event.type == evdev.ecodes.EV_ABS else "KEY  "
                print(f"  {ts:8.3f}  {type_name}  {code_name:20s}  = {event.value:7d}")
    except KeyboardInterrupt:
        pass
    except PermissionError:
        print(f"Permission denied on {dev.path}")
        print("Fix: sudo usermod -aG input $USER  (then log out/in)")

    print("\n--- All codes seen ---")
    for (etype, ecode), val in sorted(seen.items()):
        code_name = evdev.ecodes.bytype[etype].get(ecode, str(ecode))
        type_name = "ABS" if etype == evdev.ecodes.EV_ABS else "KEY"
        print(f"  {type_name}  {code_name:25s}  last={val}")


if __name__ == "__main__":
    main()
