#!/usr/bin/env python3
"""Print Joy-Con axis ranges and capabilities without needing to press buttons."""
import sys
import evdev

for path in evdev.list_devices():
    try:
        d = evdev.InputDevice(path)
        if "Joy-Con" in d.name and "IMU" not in d.name:
            print(f"Device: {d.name}  ({path})")
            caps = d.capabilities(verbose=True, absinfo=True)
            for etype, events in caps.items():
                if "ABS" in str(etype):
                    print(f"\nAxes:")
                    for name, absinfo in events:
                        print(f"  {name[0]:20s}  min={absinfo.min:7d}  max={absinfo.max:7d}  fuzz={absinfo.fuzz}  flat={absinfo.flat}")
                if "KEY" in str(etype):
                    print(f"\nButtons: {[n[0] if isinstance(n,tuple) else n for n in events]}")
            sys.exit(0)
    except PermissionError:
        pass

print("No Joy-Con found or permission denied.")
print("Run: sudo chmod 0660 /dev/input/event17")
