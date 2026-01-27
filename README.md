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

**Demo mode**:
```bash
uv run python examples/simulate_mujoco.py --no-controller
```

### 4. Run with Real Robot

Connect your SO-ARM101 via USB, then:

```bash
# Auto-detect port
uv run python examples/teleoperate_real.py

# Or specify port manually
uv run python examples/teleoperate_real.py --port /dev/ttyUSB0
```

### 5. Digital Twin Mode (Real Robot + Simulation)

Run both the real robot and MuJoCo simulation simultaneously. The simulation shows a real-time preview of robot movements.

```bash
uv run python examples/teleoperate_dual.py --port /dev/ttyUSB0
```

## Examples

| Script | Description |
|--------|-------------|
| `examples/teleoperate.py --sim` | Test controller (terminal output only) |
| `examples/simulate.py` | 3D visualization with meshcat |
| `examples/simulate_mujoco.py` | MuJoCo simulation |
| `examples/diagnose_robot.py` | Pre-flight motor diagnostics |
| `examples/teleoperate_real.py` | Control real robot |
| `examples/teleoperate_dual.py` | Digital twin (real + simulation) |

## Testing

```bash
# Run all tests
uv run pytest

# Controller input tests
uv run pytest tests/test_xbox_input.py -v

# IK roundtrip tests
uv run pytest tests/test_ik_roundtrip.py -v
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

## License

MIT
