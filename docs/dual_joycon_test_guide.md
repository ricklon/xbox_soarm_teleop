# Dual Joy-Con Test Guide

This guide is the bring-up path for the split `dual_joycon` controller mode.
Use it in this order:

1. Pair both Joy-Cons
2. Verify Linux sees both controllers
3. Test in MuJoCo
4. Test on the real robot with small motions only

## Safety

- Test new mappings in simulation first.
- For live robot testing, keep the workspace clear and your hand near power-off or e-stop.
- Start with tiny motions and confirm deadman behavior before larger moves.

## Prerequisites

Install the Joy-Con dependencies and Linux setup:

```bash
uv pip install -e ".[joycon,dev]"
sudo bash scripts/setup_joycon.sh
```

The setup script installs the `hid-nintendo` path, udev rules, and `joycond`.

## 1. Pair Both Joy-Cons

Put each Joy-Con into pairing mode, then connect them with `bluetoothctl`.

List devices:

```bash
bluetoothctl devices | grep -i 'Joy-Con'
```

Check connection state for each controller:

```bash
bluetoothctl info <LEFT_MAC>
bluetoothctl info <RIGHT_MAC>
```

You want both controllers to show:

```text
Connected: yes
```

## 2. Confirm Linux Sees Both Joy-Cons

List evdev devices:

```bash
python -c "import evdev; [print(p, evdev.InputDevice(p).name) for p in evdev.list_devices()]"
```

Expected device names usually include:

- `Joy-Con (L)` or `Nintendo Switch Left Joy-Con`
- `Joy-Con (R)` or `Nintendo Switch Right Joy-Con`

You may also see a separate right-hand IMU device. That is expected.

For a quick axis and capability check:

```bash
uv run python scripts/joycon_axis_info.py
```

## 3. Verify Both Controllers Produce Events

Use this raw event probe and press buttons on both Joy-Cons:

```bash
python - <<'PY'
import evdev, select, time

devices = []
for path in evdev.list_devices():
    dev = evdev.InputDevice(path)
    if "Joy-Con" in dev.name:
        devices.append(dev)

print("Found devices:")
for dev in devices:
    print(dev.path, dev.name)

print("Press buttons on both Joy-Cons for 10 seconds...")
end = time.time() + 10
while time.time() < end:
    readable, _, _ = select.select([d.fd for d in devices], [], [], 0.2)
    for dev in devices:
        if dev.fd not in readable:
            continue
        for event in dev.read():
            if event.type in (evdev.ecodes.EV_KEY, evdev.ecodes.EV_ABS):
                print(dev.name, event.type, event.code, event.value)
PY
```

What you want:

- left Joy-Con buttons only trigger left-device events
- right Joy-Con buttons only trigger right-device events
- no missing controller

## 4. Simulation Test

Run MuJoCo first:

```bash
uv run simulate-mujoco --controller dual_joycon --mode cartesian
```

Current mapping:

- Left Joy-Con `ZL`: deadman/clutch
- Left stick: `X/Y` translation
- Left D-pad up/down: `Z` translation
- Right Joy-Con IMU: wrist orientation
- Right `ZR`: gripper
- Right `+`: home
- Release and re-hold `ZL`: reclutch IMU neutral

### Simulation Checklist

1. Hold `ZL` and move the left stick.
Expected: end effector translates in-plane.

2. Hold `ZL` and press D-pad up/down.
Expected: end effector moves up/down.

3. Hold `ZL`, put the right Joy-Con in a comfortable neutral pose, release `ZL`, then hold `ZL` again.
Expected: that pose becomes the new wrist neutral.

4. While holding `ZL`, rotate and tilt the right Joy-Con slowly.
Expected: wrist orientation changes without uncontrolled continuous spinning.

5. Press `ZR`.
Expected: gripper closes.

6. Press `+`.
Expected: home action runs.

## 5. What To Watch For

Good behavior:

- no motion when `ZL` is not held
- no large jump when pressing `ZL`
- wrist follows hand orientation
- reclutch feels natural

Problems to note:

- `X/Y` swapped
- vertical motion inverted
- IMU roll/pitch/yaw swapped
- one IMU axis inverted
- wrist jumps when clutching
- drift while holding still
- wrong button for gripper or home

## 6. Real Robot Test

Only do this after simulation looks acceptable.

Run:

```bash
uv run teleoperate-real --controller dual_joycon --mode cartesian --port /dev/ttyUSB0
```

Recommended live sequence:

1. Verify no motion without `ZL`
2. Verify tiny left-stick motions
3. Verify tiny up/down motions
4. Verify tiny wrist orientation motions
5. Verify gripper
6. Verify home

Keep the first live session focused on signs, swaps, and clutch feel, not task performance.

## 7. Report Format

After testing, record results like this:

- `left stick X`: correct / inverted / swapped
- `left stick Y`: correct / inverted / swapped
- `D-pad Z`: correct / inverted
- `IMU roll`: correct / inverted / mapped wrong
- `IMU pitch`: correct / inverted / mapped wrong
- `IMU yaw`: correct / inverted / mapped wrong
- `clutch`: good / jumps / drifts
- `gripper`: good / wrong button
- `home`: good / wrong button

That is enough to patch the mapping quickly.
