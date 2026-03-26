# SO-ARM101 Driving Guide

A practical reference for teleoperating the SO-ARM101 across all four control modes and three controller types.

---

## Quick-start

```bash
# Simulation (safe place to practise)
uv run python examples/simulate_mujoco.py --controller keyboard --mode crane

# Real robot
uv run python examples/teleoperate_real.py --port /dev/ttyUSB0 --mode crane
```

**Safety rule for all modes:** keep one hand free to Ctrl-C or hit the physical e-stop.
The deadman switch (LB on Xbox, SL on Joy-Con) must be held for any arm motion;
releasing it freezes the arm in place.  Keyboard mode has no deadman — motion starts
when movement keys are pressed.

---

## The arm at a glance

```
          wrist_flex  wrist_roll
               ↕           ↺
elbow_flex → [arm] ─── [gripper]
               |
shoulder_lift ↕
               |
shoulder_pan ← [base]
```

- **shoulder_pan** — rotates the whole arm left/right around the vertical axis
- **shoulder_lift** — raises/lowers the upper arm relative to the base
- **elbow_flex** — bends the forearm
- **wrist_flex** — tilts the gripper up/down (pitch)
- **wrist_roll** — rotates the gripper around the forearm axis (roll)
- **gripper** — open/close, position-controlled

---

## Mode 1 — Crane (default)

### Mental model

Think of the arm as a **crane on a turntable**.  You never think about individual
joint angles.  Instead you work in three intuitive quantities:

- **Pan** — which direction is the crane pointing? (left/right rotation)
- **Reach** — how far out is the load? (horizontal distance from base)
- **Height** — how high is the load?

Shoulder_lift and elbow_flex are solved automatically by a 2-DOF planar IK engine;
you never touch them.  The wrist is controlled separately as a final orientation
layer on top.

This separation means you can position the end-effector without worrying about
elbow collisions, and the arm naturally folds inward as you retract.

### Xbox controls

| Input | Action |
|-------|--------|
| Hold LB | Deadman — must be held for all motion |
| Left stick X | Pan left / right |
| Left stick Y | Height — push up to raise, down to lower |
| Right stick Y | Reach — push forward to extend, back to retract |
| Right stick X | Wrist roll |
| D-pad up / down | Wrist flex — tilt gripper up / down |
| Right trigger | Gripper — squeeze to close |
| A button | Return to home position |
| Y button | Toggle coordinate frame (world / tool) |

### Joy-Con (R) controls

| Input | Action |
|-------|--------|
| Hold SL | Deadman |
| Stick left / right | Pan |
| Stick up / down | Height |
| ZR (hold) | Gripper close |
| + button | Home position |

> Note: reach is not bound on a single Joy-Con.  Use Xbox or keyboard for full crane control.

### Keyboard controls

| Key | Action |
|-----|--------|
| A / D | Pan left / right |
| R / F | Height up / down |
| W / S | Reach extend / retract |
| Q / E | Wrist roll |
| ↑ / ↓ | Wrist flex |
| Space (hold) | Gripper close |
| H | Home position |
| 1–5 | Speed level (25 / 50 / 75 / 100 / 150%) |
| Shift | 2× speed while held |
| Tab | Toggle keystroke recording |

### Tips

- Start with small reach/height movements to get a feel for the IK.
- Reach and height are soft-clamped to the reachable workspace; the arm simply
  stops if you push into a limit.
- Wrist flex is independent of the rest — use it last, like adjusting your grip
  angle after positioning.

---

## Mode 2 — Cartesian

### Mental model

Think of your controller as a **6-DOF joystick attached to the gripper tip**.
Every stick motion moves the gripper tip through space in a straight line.
The IK engine computes which joints to move — you never see them.

There are two frames to choose from (toggle with Y):

- **World frame** — X is "into the scene", Y is left/right, Z is up.
  Good for precise positioning relative to a fixed reference.
- **Tool frame** — axes follow the gripper tip orientation.
  Good for tasks like "push straight into the object" regardless of wrist angle.

The trade-off: the arm has joint limits and singular configurations.  Near the
limits the IK may slow down or refuse to move further.  If you get stuck, press
A to go home and start over.

### Xbox controls

| Input | Action |
|-------|--------|
| Hold LB | Deadman |
| Left stick X | Move left / right (Y axis) |
| Left stick Y | Move up / down (Z axis) |
| Right stick Y | Move forward / back (X axis) |
| Right stick X | Wrist roll |
| D-pad up / down | Pitch — tilt gripper up / down |
| D-pad left / right | Yaw — rotate gripper heading |
| Right trigger | Gripper close |
| A button | Home |
| Y button | Toggle world / tool frame |

### Joy-Con (R) controls

The single Joy-Con stick maps to X/Y translation; Z and orientation axes are not
bound.  Cartesian mode is most useful with an Xbox controller.

### Keyboard controls

| Key | Action |
|-----|--------|
| W / S | Forward / back (X) |
| A / D | Left / right (Y) |
| R / F | Up / down (Z) |
| Q / E | Wrist roll |
| ↑ / ↓ | Pitch |
| ← / → | Yaw |
| Space (hold) | Gripper close |
| H | Home |
| Y | Toggle frame |
| 1–5 / Shift | Speed / boost |

### Tips

- Use world frame to move to a target position, then switch to tool frame to
  fine-tune approach angle.
- Keep stick deflections small — linear_scale is 0.1 m/s at full deflection,
  which is fast enough to cover the workspace in a couple of seconds.
- If the arm freezes mid-motion, the IK may have hit a joint limit.  Back off
  slightly in the direction you came from, or press A to home.

---

## Mode 3 — Joint

### Mental model

You are **directly rotating servos**.  There is no IK, no safety net, and
no coordinate transform.  The arm does exactly what the joints do.

This mode is most useful for:
- Calibration and range-of-motion verification
- Diagnosing a specific joint's behaviour
- Teaching precise joint-space poses

Because you control one joint at a time (Xbox/Joy-Con) or all joints on
dedicated keys (keyboard), you need to build a mental picture of the joint
chain: the base rotates, then the shoulder lifts the upper arm, then the elbow
bends the forearm, and so on.  Moving shoulder_pan with shoulder_lift extended
looks very different from moving it at home — the physical motion is always a
rotation around the joint's local axis.

### Xbox / Joy-Con single-joint controls

| Input | Action |
|-------|--------|
| Hold LB (SL) | Deadman |
| D-pad left / right | Cycle selected joint |
| Left stick X | Drive selected joint |
| Right trigger | Gripper |
| A button | Home all joints |

Joint cycle order: `shoulder_pan → shoulder_lift → elbow_flex → wrist_flex → wrist_roll → gripper → (wrap)`

The HUD display shows the currently selected joint and its angle.

### Keyboard multi-joint controls

All arm joints are driven simultaneously from dedicated key pairs:

| Keys | Joint |
|------|-------|
| A / D | shoulder_pan |
| W / S | shoulder_lift |
| R / F | elbow_flex |
| Q / E | wrist_flex |
| ↑ / ↓ | wrist_roll |
| Space | gripper close |
| H | Home all |
| 1–5 / Shift | Speed / boost |

### Tips

- Move slowly — there is no IK or workspace limit enforcement in this mode.
  The only guard is the per-joint degree limit.
- When cycling joints with D-pad, the HUD label updates immediately; confirm
  it shows the right joint before pushing the stick.
- For precise positioning, reduce speed to level 1 (25%) before making fine
  adjustments.

---

## Mode 4 — Puppet

### Mental model

**Your hand is the gripper.**  Hold the Joy-Con in your right hand as if you were
holding a TV remote, and the gripper mirrors your wrist orientation in real time.

The position of the gripper tip in space is still controlled with the stick
(crane-style pan, reach, and height), but as soon as you tilt or rotate your
wrist the gripper follows.  The link is made via the Joy-Con's built-in IMU
(accelerometer + gyroscope fused with a complementary filter):

- **Tilt hand forward / backward** → wrist_flex (gripper tips down / up)
- **Rotate hand left / right (like twisting a door handle)** → wrist_roll

The IMU reports orientation *relative to a neutral pose* captured when you press
the home button (or at startup).  Hold the Joy-Con in your natural resting
orientation before pressing A / + to calibrate — that becomes the zero reference.

### Joy-Con (R) controls

| Input | Action |
|-------|--------|
| Hold SL | Deadman |
| Stick left / right | Shoulder pan (base rotation) |
| Stick up / down | Reach (extend / retract) |
| SR (hold) | Height up |
| B face button | Height down |
| Tilt hand fwd / back | Wrist flex |
| Rotate hand | Wrist roll |
| ZR (hold) | Gripper close |
| + button | Home + recalibrate IMU neutral |

### Tips

- Calibrate with your arm relaxed and the Joy-Con in your natural grip.  An
  awkward calibration pose will make all subsequent orientation offsets feel wrong.
- Start with larger arm movements (pan/reach/height) before adding wrist
  orientation — it is easier to get the tip near the target first, then fine-tune
  the approach angle.
- If wrist motion feels sluggish, check that the hid-nintendo kernel module is
  loaded: `lsmod | grep hid_nintendo`.  The mode degrades gracefully if the IMU
  is not found — the wrist simply holds its last position.
- The complementary filter introduces a small lag on fast wrist movements
  (α = 0.95 favours gyro integration).  Slow, deliberate wrist movements track
  more accurately than flicks.

---

## Controller quick-reference

### Xbox controller

```
[LB]  [RB]
[LT]  [RT]  ← gripper

Left stick  ─────────  pan / height (crane) | Y / Z (cartesian) | selected joint (joint)
Right stick ─────────  reach / roll (crane) | X / roll (cartesian)

D-pad  ───────────────  wrist flex (crane) | pitch + yaw (cartesian) | cycle joint (joint)

[A]  home / reset IMU
[Y]  toggle frame (cartesian) / unused (other modes)
```

### Joy-Con (R) — horizontal hold, SL on left

```
[SL]  deadman (hold)     [SR]  height up (puppet)
[ZR]  gripper (hold)

Stick ────────────────── pan + reach/height (crane/puppet) | X+Y (cartesian)

[+]   home
[B]   height down (puppet)
IMU   wrist orientation (puppet)
```

### Keyboard

```
Movement keys always active — no deadman.

Speed:  1 / 2 / 3 / 4 / 5 keys   (25 / 50 / 75 / 100 / 150%)
Boost:  hold Shift                 (2× current speed, capped at 200%)

Tab          start / stop keystroke recording
Ctrl-C       exit
```

---

## Mode selection guide

| Situation | Recommended mode |
|-----------|-----------------|
| Learning the system, first session | **crane** |
| Picking up and placing objects | **crane** |
| Following a precise 3-D path | **cartesian** (world frame) |
| Approaching an object from a specific angle | **cartesian** (tool frame) |
| Debugging a single servo | **joint** |
| Calibration and ROM verification | **joint** |
| Keyboard-only operation | **joint** (multi-key) or **crane** |
| Demonstrating / recording manipulation | **puppet** |
| Tasks where wrist orientation tracks hand | **puppet** |

---

## Troubleshooting

**Arm freezes mid-movement**
- Cartesian/crane: the IK or workspace limit was hit.  Back off slightly or press A to home.
- Joint: the joint limit was reached.  Move in the opposite direction.

**D-pad does nothing (joint mode, Xbox)**
- D-pad cycles the selected joint.  Look at the HUD to confirm which joint is active.

**IMU wrist unresponsive (puppet mode)**
- Run `lsmod | grep hid_nintendo` — the kernel module must be loaded.
- Ensure the Joy-Con is connected over Bluetooth and joycond is running.
- The startup message will say "JoyConIMU: no IIO device found" if the IMU is unavailable.

**Joy-Con disconnects during use**
- The joycond daemon handles reconnection; run `journalctl -u joycond -f` to monitor.
- Increase Bluetooth TX power: `hciconfig hci0 sspmode 1`.

**Controller not detected**
```bash
# Xbox / Joy-Con
python -c "import evdev; print([d.name for d in map(evdev.InputDevice, evdev.list_devices())])"

# Keyboard (check group membership)
groups | grep input   # must include 'input'
sudo usermod -aG input $USER && newgrp input
```
