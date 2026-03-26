# Xbox SO-ARM Teleop

Xbox controller teleoperation for the SO-ARM100/101 robotic arm using inverse kinematics. Control the end effector in Cartesian space while IK computes joint angles in real-time.

Built as an extension to the [HuggingFace LeRobot](https://github.com/huggingface/lerobot) framework.

## Features

- Xbox controller input with deadzone filtering
- Cartesian end-effector control (X, Y, Z, roll)
- Position-controlled gripper via right trigger
- Deadman switch (LB) for safety
- Home position button (A)
- Workspace bounds enforcement
- **3D simulation** with meshcat visualization
- **MuJoCo simulation** with physics
- **Real robot control** via LeRobot
- **Digital twin mode** - simultaneous simulation and real robot

## Control Mapping

| Input | Function |
|-------|----------|
| Left Stick X | Left/right movement (Y axis) |
| Left Stick Y | Up/down movement (Z axis) |
| Right Stick Y | Forward/back movement (X axis) |
| Right Stick X | Wrist roll rotation |
| D-pad Up/Down | Pitch (tilt gripper up/down) |
| D-pad Left/Right | Yaw (rotate gripper heading) |
| Right Trigger | Gripper (released=open, pulled=closed) |
| Left Bumper | Deadman switch (hold to enable) |
| A Button | Return to home position |
| Y Button | Toggle coordinate frame (world/tool) |

## Installation

```bash
# Create virtual environment
uv venv --python 3.10
source .venv/bin/activate

# Install package with dev dependencies
uv pip install -e ".[dev]"
```

## Quick Start

### 1. Test Controller Input (no robot needed)

```bash
uv run python examples/teleoperate.py --sim
```

Move sticks while holding LB to see EE delta values printed.

### 2. Run 3D Simulation (meshcat)

```bash
uv run python examples/simulate.py
```

Then open http://127.0.0.1:7000/static/ in your browser to see the robot.

**Demo mode** (no controller):
```bash
uv run python examples/simulate.py --no-controller
```

### 3. Run MuJoCo Simulation (optional)

```bash
# Install MuJoCo first
uv pip install mujoco

# Run simulation
uv run python examples/simulate_mujoco.py
```

A window will open with the robot. Close window or Ctrl+C to exit.

**Routine mode (no controller)**:
```bash
uv run python examples/simulate_mujoco.py --motion-routine
```

**Routine square trace + pen trail (MuJoCo)**:
```bash
uv run python examples/simulate_mujoco.py --motion-routine \
  --routine-pattern square --routine-plane xy \
  --routine-square-size 0.06 --routine-trace
```

**IK error logging + thresholds (demo mode)**:
```bash
uv run mujoco-ik-check --no-controller --ik-log ik_error.csv --ik-max-err-mm 25 --ik-mean-err-mm 8
```

### 4. Run with Real Robot

Connect your SO-ARM101 via USB, then:

```bash
# Auto-detect port
uv run python examples/teleoperate_real.py

# Or specify port manually
uv run python examples/teleoperate_real.py --port /dev/ttyUSB0

# Routine mode (no controller)
uv run python examples/teleoperate_real.py --motion-routine --routine-pattern square --routine-plane xy
```

### 5. Digital Twin Mode (Real Robot + Simulation)

Run both the real robot and MuJoCo simulation simultaneously. The simulation shows a real-time preview of robot movements.

```bash
uv run python examples/teleoperate_dual.py --port /dev/ttyUSB0

# Routine mode (no controller)
uv run python examples/teleoperate_dual.py --port /dev/ttyUSB0 --motion-routine --routine-pattern square-xyz
```

## Examples

| Script | Description |
|--------|-------------|
| `examples/teleoperate.py --sim` | Test controller (terminal output only) |
| `examples/simulate.py` | 3D visualization with meshcat |
| `examples/simulate_mujoco.py` | MuJoCo simulation |
| `examples/xbox_joint_diagnostic.py` | Direct joint drive + telemetry logging (no IK) |
| `examples/diagnose_robot.py` | Pre-flight motor diagnostics |
| `examples/teleoperate_real.py` | Control real robot |
| `examples/teleoperate_dual.py` | Digital twin (real + simulation) |

Direct joint diagnostic (recommended when IK appears sluggish):

```bash
uv run python examples/xbox_joint_diagnostic.py --port /dev/ttyACM0
```

This bypasses IK and maps Xbox input directly to one selected servo at a time, with CSV logging of commanded velocity, goal, position, velocity, load, current, temperature, and voltage.
It automatically applies per-joint clearance poses (from `SWEEP_TEST_POSITIONS`) so each joint can reach fuller ROM without self-collision or gripper overload.

Control direction details:
- Hold `LB` to enable movement
- Left stick X:
  - right (`+X`) moves selected joint toward its configured upper limit
  - left (`-X`) moves selected joint toward its configured lower limit
- D-pad left/right:
  - left selects previous joint
  - right selects next joint
- `Y` toggles auto-sweep for selected joint
- `A` resets selected joint to its base test pose

If needed:
- Disable clearance poses: `--no-test-positions`
- Enable trigger-driven gripper during gripper test: `--gripper-trigger-control`

Analyze a captured diagnostic log:

```bash
uv run python examples/analyze_joint_diag.py --input joint_diag_YYYYMMDD_HHMMSS.csv
```

## Testing

```bash
# Run all tests
uv run pytest

# Controller input tests
uv run pytest tests/test_xbox_input.py -v

# IK roundtrip tests
uv run pytest tests/test_ik_roundtrip.py -v

# IK smoke test (opt-in)
IK_SMOKE=1 uv run pytest tests/test_ik_smoke.py -v
```

## IK Smoke Check (Routine)

Use this when you want a fast, repeatable IK sanity check without a robot:

```bash
uv run ik-smoke
```

Optional tuning:

```bash
uv run ik-smoke --duration 20 --hz 60 --max-err-mm 25 --mean-err-mm 8 --verbose
```

## MuJoCo IK Check (Routine)

Use this to log IK position error in MuJoCo and fail if thresholds are exceeded:

```bash
uv run mujoco-ik-check --no-controller --ik-log ik_error.csv
```

## Development

```bash
# Install pre-commit hooks
pre-commit install

# Run linting
uv run ruff check .
uv run ruff format .

# Run tests with coverage
uv run pytest --cov=xbox_soarm_teleop
```

## Project Structure

```
xbox_soarm_teleop/
├── src/xbox_soarm_teleop/
│   ├── teleoperators/xbox.py    # Xbox controller input
│   ├── processors/xbox_to_ee.py # EE delta mapping
│   └── config/                  # Configuration
├── examples/
│   ├── teleoperate.py           # Basic teleoperation (terminal)
│   ├── simulate.py              # Meshcat 3D visualization
│   ├── simulate_mujoco.py       # MuJoCo physics simulation
│   ├── teleoperate_real.py      # Real robot control
│   └── teleoperate_dual.py      # Digital twin mode
├── assets/                      # URDF and mesh files
└── tests/                       # Unit tests
```

## Requirements

- Python 3.10
- Xbox controller (tested with Xbox One/Series controllers)
- SO-ARM100 or SO-ARM101 robotic arm (for real robot control)
- LeRobot with kinematics extra

## Troubleshooting

### Controller not detected

```bash
# Check if controller is visible
uv run python -c "import inputs; print(inputs.devices.gamepads)"

# Linux: add user to input group
sudo usermod -aG input $USER
# Then logout and login
```

### Simulation window doesn't open (meshcat)

Open http://127.0.0.1:7000/static/ manually in your browser.

### MuJoCo viewer issues

```bash
# Install OpenGL dependencies (Linux)
sudo apt install libgl1-mesa-glx libosmesa6
```

### Real robot not detected

```bash
# List available serial ports
ls /dev/ttyUSB* /dev/ttyACM*

# Set permissions (temporary)
sudo chmod 666 /dev/ttyUSB0

# Or add user to dialout group (permanent, requires logout)
sudo usermod -aG dialout $USER
```

### Robot calibration

First time connecting, LeRobot will run calibration:
1. Move arm to middle of range when prompted
2. Move each joint through full range when prompted
3. Calibration is saved and reused on subsequent connections

### Pre-flight motor diagnostics

Before teleoperation, run diagnostics to check motor health:

```bash
# Full diagnostics (requires ricklon/lerobot fork)
uv run python examples/diagnose_robot.py --port /dev/ttyUSB0

# Simple motor test (works with standard lerobot)
uv run python examples/diagnose_robot.py --port /dev/ttyUSB0 --simple
```

Or use lerobot's built-in calibration with diagnostics:
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyUSB0 --diagnose
```

### Firmware version mismatch

If you see this error:
```
Error: Some Motors use different firmware versions:
{1: '3.9', 2: '3.9', 3: '3.10', 4: '3.9', 5: '3.9', 6: '3.9'}
```

All motors must have the same firmware. Update using Feetech's Windows software:

1. Download FD software from https://www.feetechrc.com/software
2. **Windows:** Run FD.exe directly
3. **Linux (Wine):**
   ```bash
   # Set up serial port for Wine
   mkdir -p ~/.wine/dosdevices
   ln -sf /dev/ttyACM0 ~/.wine/dosdevices/com3

   # Run Feetech software
   wine FD.exe
   ```
4. Connect one motor at a time, scan, and update firmware

**Warning:** Do not disconnect power/USB during firmware update - it can brick the motor.

### Common calibration issues

**Zero position difference:** Motor position not changing
- Check USB cable connection
- Verify motor ID matches configuration
- Ensure motor isn't mechanically stuck

**Small position range:** Motor moved but range is tiny
- Move motor through its COMPLETE range during calibration
- For gripper: fully open AND fully closed

## Joy-Con Support

A Nintendo Switch Right Joy-Con can be used as an alternative to an Xbox controller,
held sideways in single-controller mode.

### Control Mapping (Right Joy-Con, horizontal)

| Input | Function |
|-------|----------|
| Stick | Movement (same as Xbox left + right stick) |
| SL (left rail button) | Deadman switch — hold to enable motion |
| ZR (top shoulder) | Gripper (digital: open/closed) |
| A face button | Home position |
| Y face button | Toggle coordinate frame |
| SR, B, X, +, Home | Available for future mapping |

### One-time System Setup (Ubuntu 22.04+, kernel 6.8+)

```bash
sudo bash scripts/setup_joycon.sh
```

This script:
1. Loads the `hid-nintendo` kernel module (persists across reboots)
2. Configures BlueZ to accept non-bonded HID devices
3. Installs udev rules so the Joy-Con is accessible without sudo
4. Builds and installs `joycond` (keepalive daemon)

### Pairing

```bash
# Hold the sync button (flat rail side) for 3 seconds — LEDs will chase
bluetoothctl connect 98:B6:AF:5E:5D:39   # replace with your MAC

# When connected, press SL+SR simultaneously on the Joy-Con
# The second LED will become solid — single-controller mode is active
```

Find your MAC address with `bluetoothctl devices | grep Joy-Con`.

### Verify it works

```bash
uv pip install evdev            # one-time
uv run python scripts/joycon_axis_info.py   # shows axes and buttons
```

### Use in code

```python
from xbox_soarm_teleop.config.joycon_config import JoyConConfig
from xbox_soarm_teleop.teleoperators.joycon import JoyConController

ctrl = JoyConController(JoyConConfig())
if ctrl.connect():
    state = ctrl.read()   # returns XboxState — same interface as XboxController
    print(state.left_stick_x, state.right_trigger, state.left_bumper)
    ctrl.disconnect()
```

`JoyConController` is a drop-in replacement for `XboxController` — all processors
and control loops work unchanged.

### Troubleshooting

**`No Joy-Con found`** — Check `bluetoothctl info <MAC>` shows `Connected: yes`.
Press SL+SR to activate single-controller mode.

**`Permission denied on /dev/input/eventN`** — Re-run `setup_joycon.sh` or manually:
`sudo chmod 0660 /dev/input/event17`

**Joy-Con goes to sleep** — `joycond` must be running: `systemctl is-active joycond`.
If not, reconnect: `bluetoothctl connect <MAC>`.

**Connection fails immediately** — The Joy-Con is asleep. Hold the sync button
(3 sec, LEDs chase), then immediately run `bluetoothctl connect <MAC>`.

## License

Apache-2.0
