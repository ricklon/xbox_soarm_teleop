#!/usr/bin/env python3
"""Step-by-step Joy-Con connection test.

Run with: uv run python scripts/test_joycon.py
"""
import os
import sys
import time


def step(n, msg):
    print(f"\n[Step {n}] {msg}")


def ok(msg):
    print(f"  OK: {msg}")


def fail(msg):
    print(f"  FAIL: {msg}")


def info(msg):
    print(f"  {msg}")


# ── Step 1: evdev available ───────────────────────────────────────────────────
step(1, "Checking evdev library")
try:
    import evdev
    ok("evdev imported")
except ImportError:
    fail("evdev not installed — run: uv pip install evdev")
    sys.exit(1)

# ── Step 2: Find Joy-Con device ───────────────────────────────────────────────
step(2, "Looking for Joy-Con input device")
joycon_dev = None
for path in evdev.list_devices():
    try:
        d = evdev.InputDevice(path)
        if "Joy-Con" in d.name and "IMU" not in d.name:
            joycon_dev = d
            ok(f"Found: {d.name} at {path}")
            break
    except PermissionError:
        info(f"Permission denied: {path}")
    except Exception:
        pass

if joycon_dev is None:
    fail("No Joy-Con found")
    info("Make sure it's connected: sudo bluetoothctl connect 98:B6:AF:5E:5D:39")
    info("Then fix permissions: sudo chmod 0660 /dev/input/event17")
    sys.exit(1)

# ── Step 3: Print capabilities ────────────────────────────────────────────────
step(3, "Device capabilities")
caps = joycon_dev.capabilities(verbose=True)
for etype, events in caps.items():
    info(f"{etype[0]}: {[e[0] if isinstance(e, tuple) else e for e in events]}")

# ── Step 4: Check if grabbed by another process ───────────────────────────────
step(4, "Checking if device is grabbed (e.g. by joycond)")
try:
    joycon_dev.grab()
    joycon_dev.ungrab()
    ok("Device is not exclusively grabbed")
except OSError as e:
    fail(f"Device is grabbed by another process: {e}")
    info("joycond may be holding exclusive access")
    info("Try: sudo systemctl stop joycond  — then reconnect Joy-Con")

# ── Step 5: Read 5 seconds of raw events ─────────────────────────────────────
step(5, "Reading raw events for 5 seconds — press buttons/move stick now")
info("(if nothing appears, the Joy-Con is not sending events yet)")

deadline = time.monotonic() + 5.0
event_count = 0

joycon_dev.set_blocking(False)
while time.monotonic() < deadline:
    try:
        for event in joycon_dev.read():
            if event.type in (evdev.ecodes.EV_ABS, evdev.ecodes.EV_KEY):
                code_name = evdev.ecodes.bytype[event.type].get(event.code, str(event.code))
                type_name = "ABS" if event.type == evdev.ecodes.EV_ABS else "KEY"
                print(f"  EVENT: {type_name} {code_name} = {event.value}")
                event_count += 1
    except BlockingIOError:
        pass
    time.sleep(0.01)

if event_count == 0:
    fail("No events received")
    info("Possible causes:")
    info("  1. Press SL+SR on Joy-Con to activate single-controller mode")
    info("  2. joycond may need to initialize the controller first")
    info("  3. Try: sudo systemctl stop joycond, reconnect, run this again")
else:
    ok(f"Received {event_count} events — controller is working!")

# ── Step 6: Summary ───────────────────────────────────────────────────────────
step(6, "Summary")
info(f"Device: {joycon_dev.name}")
info(f"Path:   {joycon_dev.path}")
info(f"Phys:   {joycon_dev.phys}")
info(f"Events received: {event_count}")
