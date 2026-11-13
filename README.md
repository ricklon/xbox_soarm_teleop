# Xbox SO-ARM Teleop

Teleoperation for the SO-ARM100/101 robotic arm with Xbox, Joy-Con, dual Joy-Con, and keyboard input.
The project supports cartesian IK, crane, joint-direct, and puppet control modes for
simulation and real hardware.

Built as an extension to the [HuggingFace LeRobot](https://github.com/huggingface/lerobot) framework.

## Features

- Xbox, Joy-Con, dual Joy-Con, and keyboard teleoperation
- Multiple control modes: cartesian, crane, joint, and puppet
- Cartesian end-effector control (X, Y, Z, roll) with IK
- Gripper control across supported input paths
- Deadman-gated motion for safety
- Home position command
- Workspace bounds enforcement
- **3D simulation** with MuJoCo visualization
- **MuJoCo simulation** with physics
- **Real robot control** via LeRobot
- **Digital twin mode** - simultaneous simulation and real robot
- **Workshop twin** - a simulated arm shared by the Xbox controller and AI agents (MCP)

## Xbox Control Mapping

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

This table reflects the current cartesian-style Xbox mapping. For mode-specific
controls, see the [Driving Guide](docs/driving_guide.md).

## Installation

```bash
# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install package with dev dependencies
uv pip install -e ".[dev]"
```

## Guides

| Need | Use |
|------|-----|
| Run a workshop: drive the simulated arm, then let an AI agent drive it | [Workshop Guide](docs/workshop_guide.md) |
| Host a remote workshop: one simulated arm per student, in a browser | [Hosting a Remote Workshop](docs/workshop_hosting.md) |
| Fastest preflight and pass/fail check | [Operator Checklist](docs/operator_checklist.md) |
| Per-mode validation steps and failure patterns | [Mode Verification Guide](docs/mode_verification_guide.md) |
| Split Joy-Con pairing, bring-up, and natural-motion testing | [Dual Joy-Con Test Guide](docs/dual_joycon_test_guide.md) |
| Controller mappings and operating advice | [Driving Guide](docs/driving_guide.md) |
| Robot setup and recalibration | [Calibration Guide](docs/CALIBRATION.md) |
| Cartesian processor pipeline details | [LeRobot Cartesian Pipeline](docs/lerobot_pipeline.md) |
| Dataset action/observation layout | [Dataset Schema](docs/dataset_schema.md) |
| Project follow-up plan and audit notes | [Audit Response and Plan](docs/audit_plan.md) |
| Example script inventory | [Examples Inventory](examples/README.md) |

## Quick Start

### 1. Test Controller Input (no robot needed)

```bash
uv run python examples/debug_controller.py
```

Move sticks and triggers to inspect normalized controller values.

### 2. Run MuJoCo Simulation

```bash
# Install MuJoCo first
uv pip install mujoco

# Run simulation
uv run simulate-mujoco
```

A window will open with the robot. Close window or Ctrl+C to exit.

**Routine mode (no controller)**:
```bash
uv run simulate-mujoco --motion-routine
```

**Routine square trace + pen trail (MuJoCo)**:
```bash
uv run simulate-mujoco --motion-routine \
  --routine-pattern square --routine-plane xy \
  --routine-square-size 0.06 --routine-trace
```

**IK error logging + thresholds (demo mode)**:
```bash
uv run mujoco-ik-check --no-controller --ik-log ik_error.csv --ik-max-err-mm 25 --ik-mean-err-mm 8
```

### 3. Run with Real Robot

Connect your SO-ARM101 via USB, then:

```bash
# Auto-detect port
uv run teleoperate-real

# Or specify port manually
uv run teleoperate-real --port /dev/ttyUSB0

# Routine mode (no controller)
uv run teleoperate-real --motion-routine --routine-pattern square --routine-plane xy
```

### 4. Digital Twin Mode (Real Robot + Simulation)

Run both the real robot and MuJoCo simulation simultaneously. The simulation shows a real-time preview of robot movements.

```bash
uv run teleoperate-dual --port /dev/ttyUSB0

# Routine mode (no controller)
uv run teleoperate-dual --port /dev/ttyUSB0 --motion-routine --routine-pattern square-xyz
```

### 5. Workshop Twin (Simulation Shared With AI Agents)

No robot needed. One command runs the simulated arm, the Xbox controller and an
MCP server that AI agents connect to. The person holding the controller can take
over or stop the arm at any time, and every run is recorded to `twin_logs/`.

```bash
uv run soarm-twin

# Connect Claude Code (second terminal)
claude mcp add --transport http soarm-twin http://127.0.0.1:8765/mcp

# Check a laptop end to end: picks the cube into the goal box over MCP
uv run python scripts/twin_smoke.py
```

See the [Workshop Guide](docs/workshop_guide.md) for controls and exercises.

**Remote workshop.** `soarm-hub` hosts one simulated arm per student: each gets
a web page to drive it, an address and key for their AI agent, and a notebook.
Students install nothing.

```bash
uv run soarm-hub --seats 12
```

See [Hosting a Remote Workshop](docs/workshop_hosting.md) for the full guide.

### Try the remote workshop

Four stages, each proving more than the last. The first three need only this
computer.

**1. The hub and a student page (two minutes).**

```bash
uv run soarm-hub --seats 2
```

The instructor page opens in your browser. Click **Open** on Arm 1 and drive
the arm: W/A/S/D across the table, R/F up and down, O/C for the gripper, Space
to stop. This shows the hub, the student page and the camera pictures work.

Started like this, the arms can only be reached from this computer. The
notebook link on the student page opens Google Colab, which **cannot** connect
to an address starting with `http://127.0.0.1`; use stage 2 to try the
notebook now, or stage 4 for Colab.

No keys are printed in the terminal or left in the address bar, so the window
is safe to share. `--show-links` prints the instructor link if you need it.

**2. The notebook, on this computer.** Leave the hub running and, in a second
terminal:

```bash
uv run --with jupyterlab --with pip --with "anthropic[mcp]>=2,<3" jupyter lab notebooks/workshop_agent.ipynb
```

On the student page, click the **Address** to copy it and paste it into the
notebook's first cell; when the notebook asks for the key, click **Your key**
on the page and paste. Parts 0 to 4 need no model key. Part 5 needs one:
Option A for Claude, Option B for a model on OpenRouter or Hugging Face.

**3. See what your machine can carry.** Stop the hub first; this starts its own.

```bash
uv run python scripts/hub_load.py --seats 30
```

It plays 30 students' pages and prints the pictures per second and the upload
each one needs.

**4. From outside, as students will use it.** Start a tunnel first, because
the hub needs to know its address. This uses
[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/);
any HTTPS tunnel works.

```bash
cloudflared tunnel --url http://localhost:8800
```

Then start the hub in a second terminal. It finds the running tunnel and uses
its address for the students' links; the terminal says which address it found.

```bash
uv run soarm-hub --seats 2
```

With another kind of tunnel, or to be explicit, pass the address yourself with
`--public-url https://your-address`.

The instructor page opens on this computer (`127.0.0.1`). The links it copies
for students use the tunnel. To use the instructor page itself at the tunnel's
address or on another device, click **Copy instructor link** and paste it
there: opened without a key, the page shows a box for it that hides what you
paste.

Then, as a student would:

1. Click **Copy link** on an arm, open that link on a phone using mobile
   data, and drive the arm.
2. Follow the notebook link on that page into Google Colab and run it as in
   stage 2.

Stage 4 has not been run by the project yet: the hub through a tunnel, the
notebook in Colab and a real model driving the arm are all still to be proven.
Do it before you invite a class.

## Supported Commands

These are the primary commands to reach for first:

| Command | Purpose |
|--------|---------|
| `uv run python examples/debug_controller.py` | Inspect normalized Xbox input without a robot |
| `uv run simulate-mujoco` | MuJoCo simulation |
| `uv run teleoperate-real` | Main real-robot teleop entry point |
| `uv run teleoperate-dual --port /dev/ttyUSB0` | Digital twin mode |
| `uv run soarm-twin` | Workshop twin: simulated arm for the Xbox controller and AI agents |
| `uv run soarm-hub --seats 12` | Remote workshop: one simulated arm per student, in a browser |
| `uv run record-xbox --repo-id <user/dataset> --robot-port /dev/ttyUSB0 --task \"...\"` | Record project-native teleop demonstrations |
| `uv run python examples/lerobot_record_cartesian.py --repo-id <user/dataset> --task \"...\" --robot-port /dev/ttyUSB0` | Convenience wrapper for cartesian recording |

Legacy Meshcat path:
- `uv run python examples/simulate.py`
- Kept for lightweight demos, but not a maintained primary control path.

For the full script inventory, including diagnostics, compatibility shims, and
one-off investigation tools, see [examples/README.md](examples/README.md).

## Diagnostics And Utilities

Direct joint diagnostic is still the recommended first step when IK appears sluggish:

```bash
uv run xbox-joint-diagnostic --port /dev/ttyACM0
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
uv run analyze-joint-diag --input joint_diag_YYYYMMDD_HHMMSS.csv
```

Other maintenance and troubleshooting tools are cataloged in
[examples/README.md](examples/README.md).

Range-of-motion calibration helper:

```bash
uv run joint-rom-test --sim
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
│   ├── runtime/session.py       # Shared controller/runtime setup
│   └── config/                  # Configuration
├── examples/
│   ├── README.md                # Supported vs diagnostic script inventory
│   ├── simulate_mujoco.py       # MuJoCo simulation
│   ├── teleoperate_real.py      # Real robot control
│   ├── teleoperate_dual.py      # Digital twin mode
│   └── record_xbox.py           # Dataset recording
├── assets/                      # URDF and mesh files
└── tests/                       # Unit tests
```

## Requirements

- Python 3.12
- Supported input device:
  - Xbox controller
  - Nintendo Joy-Con (`joycon` or `dual_joycon` path)
  - Keyboard
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

### Legacy meshcat window doesn't open

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
uv run diagnose-robot --port /dev/ttyUSB0

# Simple motor test (works with standard lerobot)
uv run diagnose-robot --port /dev/ttyUSB0 --simple
```

Or use lerobot's built-in calibration with diagnostics:
see the [Calibration Guide](docs/CALIBRATION.md).

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

This project supports two Joy-Con workflows:

1. `joycon`
Right Joy-Con only, held sideways in single-controller mode. This is the older compatibility path.

2. `dual_joycon`
Split left + right Joy-Con control, with the right Joy-Con IMU used for more natural wrist orientation in cartesian mode.

Use these docs first:

- [Dual Joy-Con Test Guide](docs/dual_joycon_test_guide.md) for pairing, bring-up, and natural-motion cartesian testing
- [Mode Verification Guide](docs/mode_verification_guide.md) for per-mode pass/fail checks
- [Driving Guide](docs/driving_guide.md) for mode-specific mappings

### Dual Joy-Con cartesian mapping

| Input | Function |
|-------|----------|
| Left `ZL` | Deadman / clutch |
| Left stick | `X/Y` translation |
| Left D-pad up/down | `Z` translation |
| Right Joy-Con IMU | Wrist orientation |
| Right `ZR` | Gripper |
| Right `+` | Home |

Release and re-hold `ZL` to reclutch the IMU neutral pose.

### Single Joy-Con mapping

The single `joycon` mode still uses a right Joy-Con held sideways in single-controller mode.
It remains available for the existing single-controller crane, joint, and puppet workflows.

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
# Connect each Joy-Con with bluetoothctl
bluetoothctl connect <MAC>   # replace with your controller MAC

# For the single Joy-Con path only:
# press SL+SR simultaneously on the right Joy-Con to activate single-controller mode
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

Single Joy-Con compatibility:

- `JoyConController` is the single-controller compatibility path.

Dual Joy-Con path:

```python
from xbox_soarm_teleop.config.joycon_config import DualJoyConConfig
from xbox_soarm_teleop.teleoperators.joycon import DualJoyConController

ctrl = DualJoyConController(DualJoyConConfig())
if ctrl.connect():
    state = ctrl.read()
    print(state.left_stick_x, state.right_trigger, state.imu_orientation_valid)
    ctrl.disconnect()
```

### Troubleshooting

**`No Joy-Con found`** — Check `bluetoothctl info <MAC>` shows `Connected: yes`.
For the single Joy-Con path, press `SL+SR` to activate single-controller mode.
For the dual Joy-Con path, confirm that both left and right evdev devices appear.

**`Permission denied on /dev/input/eventN`** — Re-run `setup_joycon.sh` or manually:
`sudo chmod 0660 /dev/input/event17`

**Joy-Con goes to sleep** — `joycond` must be running: `systemctl is-active joycond`.
If not, reconnect: `bluetoothctl connect <MAC>`.

**Connection fails immediately** — The Joy-Con is asleep. Hold the sync button
(3 sec, LEDs chase), then immediately run `bluetoothctl connect <MAC>`.

## License

Apache-2.0
