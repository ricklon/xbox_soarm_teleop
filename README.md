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
- **MuJoCo simulation** with physics (optional)

## Control Mapping

| Input | Function |
|-------|----------|
| Left Stick X | Left/right movement (Y axis) |
| Left Stick Y | Up/down movement (Z axis) |
| Right Stick Y | Forward/back movement (X axis) |
| Right Stick X | Wrist roll rotation |
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

```bash
uv run python examples/teleoperate.py
```

(Robot integration pending - currently runs in simulation mode)

## Examples

| Script | Description |
|--------|-------------|
| `examples/teleoperate.py` | Basic teleoperation (terminal output) |
| `examples/teleoperate.py --sim` | Simulation mode (no robot) |
| `examples/simulate.py` | 3D visualization with meshcat |
| `examples/simulate.py --no-controller` | Demo mode (automatic movement) |
| `examples/simulate_mujoco.py` | MuJoCo physics simulation |

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
│   ├── teleoperate.py           # Basic teleoperation
│   ├── simulate.py              # Meshcat 3D visualization
│   └── simulate_mujoco.py       # MuJoCo physics simulation
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

## License

MIT
