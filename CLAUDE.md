# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Xbox controller teleoperation for the SO-ARM100/101 robotic arm using inverse kinematics. The operator controls the end effector in Cartesian space while IK computes joint angles in real-time. Built as an extension to the HuggingFace LeRobot framework.

## Tech Stack

- **Framework**: LeRobot (HuggingFace) with `kinematics` extra
- **IK Solver**: Placo (wrapper around Pinocchio), integrated via LeRobot
- **Robot**: SO-ARM101, 6-DOF, Feetech STS3215 servos (30 kg·cm @ 12V)
- **Controller Input**: `inputs` library or `pygame` for Xbox gamepad
- **Package Manager**: uv
- **Build Backend**: hatchling
- **Linting/Formatting**: ruff
- **Testing**: pytest
- **Python Version**: 3.10 (LeRobot requirement)

## Project Structure

```
xbox_soarm_teleop/
├── pyproject.toml
├── CLAUDE.md
├── README.md
├── .gitignore
├── .pre-commit-config.yaml
├── src/
│   └── xbox_soarm_teleop/
│       ├── __init__.py
│       ├── teleoperators/
│       │   ├── __init__.py
│       │   └── xbox.py              # XboxController class, handles input reading
│       ├── processors/
│       │   ├── __init__.py
│       │   └── xbox_to_ee.py        # MapXboxToEEDelta processor
│       └── config/
│           ├── __init__.py
│           ├── xbox_config.py       # Controller configuration dataclass
│           └── workspace_limits.yaml # Safety bounds
├── examples/
│   └── teleoperate.py               # Main control loop example
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_xbox_input.py           # Standalone controller test
    └── test_ik_roundtrip.py         # FK/IK verification
```

## Architecture

```
XboxController (Teleoperator)
    │
    ▼
MapXboxToEEDelta (Processor)
    │
    ▼
EEReferenceAndDelta (LeRobot processor - latches reference, accumulates deltas)
    │
    ▼
EEBoundsAndSafety (LeRobot processor - workspace limits, rate limiting)
    │
    ▼
InverseKinematicsEEToJoints (LeRobot processor - Placo IK)
    │
    ▼
SO101Follower (Robot)
```

## Control Mapping

| Input | Function |
|-------|----------|
| Left Stick | Horizontal plane movement (X/Y) |
| Right Stick Y | Vertical movement (Z) |
| Right Stick X | Wrist roll rotation |
| Right Trigger | Gripper position (0=open, 1=closed) |
| Left Bumper | Deadman switch (hold to enable) |
| A Button | Return to home position |
| Y Button | Toggle coordinate frame (world/tool) |

## Development Commands

```bash
# Environment setup
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install LeRobot with kinematics
uv pip install "lerobot[kinematics]"

# Run the application
uv run python examples/teleoperate.py

# Code quality
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .

# Testing
uv run pytest                              # Run all tests
uv run pytest tests/test_xbox_input.py -v  # Test controller input
uv run pytest tests/test_ik_roundtrip.py -v # Test FK/IK verification

# Pre-commit
pre-commit install
pre-commit run --all-files
```

## Key LeRobot Integration Points

```python
# Kinematics solver from URDF
from lerobot.common.kinematics import RobotKinematics

# Processor pipeline steps
from lerobot.common.robot_devices.processors import (
    EEReferenceAndDelta,
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)

# Robot interface
from lerobot.robots.so101 import SO101Follower
```

### XboxController Pattern (Teleoperator)

```python
# src/xbox_soarm_teleop/teleoperators/xbox.py
from dataclasses import dataclass
import inputs  # or pygame

@dataclass
class XboxState:
    left_stick_x: float = 0.0
    left_stick_y: float = 0.0
    right_stick_x: float = 0.0
    right_stick_y: float = 0.0
    right_trigger: float = 0.0
    left_bumper: bool = False
    a_button: bool = False
    y_button: bool = False

class XboxController:
    """Xbox controller teleoperator for SO-ARM teleoperation."""
    
    DEADZONE = 0.1
    
    def __init__(self, config: "XboxConfig"):
        self.config = config
        self._state = XboxState()
    
    def read(self) -> XboxState:
        """Read and normalize controller state."""
        # Apply deadzone to analog inputs
        # Return normalized state
        pass
    
    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.DEADZONE:
            return 0.0
        return value
```

### MapXboxToEEDelta Pattern (Processor)

```python
# src/xbox_soarm_teleop/processors/xbox_to_ee.py
import numpy as np
from dataclasses import dataclass

@dataclass
class EEDelta:
    """End effector delta in Cartesian space."""
    dx: float = 0.0  # m/s
    dy: float = 0.0  # m/s
    dz: float = 0.0  # m/s
    droll: float = 0.0  # rad/s
    gripper: float = 0.0  # 0-1 position

class MapXboxToEEDelta:
    """Maps Xbox controller state to end effector velocity commands."""
    
    def __init__(
        self,
        linear_scale: float = 0.1,  # m/s at full stick
        angular_scale: float = 0.5,  # rad/s at full stick
    ):
        self.linear_scale = linear_scale
        self.angular_scale = angular_scale
    
    def __call__(self, state: "XboxState") -> EEDelta:
        # Deadman switch check
        if not state.left_bumper:
            return EEDelta(gripper=state.right_trigger)
        
        return EEDelta(
            dx=state.left_stick_y * self.linear_scale,
            dy=state.left_stick_x * self.linear_scale,
            dz=state.right_stick_y * self.linear_scale,
            droll=state.right_stick_x * self.angular_scale,
            gripper=state.right_trigger,
        )
```

### Control Loop Pattern

```python
# examples/teleoperate.py
from xbox_soarm_teleop.teleoperators.xbox import XboxController
from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta
from xbox_soarm_teleop.config.xbox_config import XboxConfig
from lerobot.robots.so101 import SO101Follower

def main():
    # Initialize components
    controller = XboxController(XboxConfig())
    mapper = MapXboxToEEDelta()
    robot = SO101Follower()
    
    # Control loop
    try:
        while True:
            state = controller.read()
            ee_delta = mapper(state)
            
            # Handle special buttons
            if state.a_button:
                robot.go_home()
                continue
            
            # Send to LeRobot processor pipeline
            # ...
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
```

## Constraints and Conventions

### Safety Requirements
- **Deadman switch (LB)**: All movement gated — if not held, arm does not move
- **Workspace bounds**: Enforced before IK to prevent unreachable targets
- **Controller deadzone**: 0.1 on all analog inputs

### Control Conventions
- Gripper is **position-controlled** (trigger maps directly to 0-100%), not velocity
- IK uses `initial_guess_current_joints=True` for closed-loop control
- Velocity units: **m/s** for linear, **rad/s** for angular
- SO-ARM100 and SO-ARM101 share identical kinematics (same URDF), differ only in servo torque/calibration

### Code Conventions
- Follow LeRobot processor pattern for pipeline compatibility
- Use dataclasses for state objects
- Type hints on all public methods
- Docstrings following Google style

## Development Phases

1. **Xbox input layer** — read and normalize controller state
2. **Standalone IK test** — verify FK/IK roundtrip with LeRobot's kinematics
3. **Processor integration** — MapXboxToEEDelta wired into LeRobot pipeline
4. **Control loop** — full teleoperation with safety limits
5. **Extras** — home button, frame toggle, calibration routine

## pyproject.toml

```toml
[project]
name = "xbox-soarm-teleop"
version = "0.1.0"
description = "Xbox controller teleoperation for SO-ARM100/101 using LeRobot IK"
readme = "README.md"
requires-python = ">=3.10,<3.11"
license = {text = "MIT"}
authors = [
    {name = "Rick Anderson", email = "rick.rickanderson@gmail.com"},
]
dependencies = [
    "lerobot[kinematics]",
    "inputs>=0.5",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff>=0.4.0",
    "pre-commit",
]
pygame = [
    "pygame>=2.5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/xbox_soarm_teleop"]

[tool.ruff]
line-length = 104
target-version = "py310"
exclude = [".git", ".venv", "__pycache__", "dist", "build"]

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]
fixable = ["ALL"]

[tool.ruff.lint.isort]
known-first-party = ["xbox_soarm_teleop"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

## Reference Resources

- **LeRobot phone teleoperation**: `examples/phone_so100_*.py` in LeRobot repo
- **LeRobot kinematics docs**: https://huggingface.co/docs/lerobot/en/phone_teleop
- **SO-ARM100 URDF/STEP files**: https://github.com/TheRobotStudio/SO-ARM100
- **Community kinematics package**: https://github.com/box2ai-robotics/lerobot-kinematics
- **Placo IK documentation**: https://github.com/Rhoban/placo

## Testing Patterns

### Controller Input Test

```python
# tests/test_xbox_input.py
import pytest
from xbox_soarm_teleop.teleoperators.xbox import XboxController, XboxState
from xbox_soarm_teleop.config.xbox_config import XboxConfig

def test_deadzone_filters_small_values():
    controller = XboxController(XboxConfig())
    assert controller._apply_deadzone(0.05) == 0.0
    assert controller._apply_deadzone(0.15) == 0.15

def test_state_initialization():
    state = XboxState()
    assert state.left_bumper == False
    assert state.right_trigger == 0.0
```

### IK Roundtrip Test

```python
# tests/test_ik_roundtrip.py
import pytest
import numpy as np
from lerobot.common.kinematics import RobotKinematics

@pytest.fixture
def kinematics():
    # Load SO-ARM101 URDF
    return RobotKinematics.from_urdf("path/to/so_arm101.urdf")

def test_fk_ik_roundtrip(kinematics):
    """FK then IK should return to original joint positions."""
    original_joints = np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0])
    
    # Forward kinematics
    ee_pose = kinematics.forward(original_joints)
    
    # Inverse kinematics
    recovered_joints = kinematics.inverse(
        ee_pose,
        initial_guess=original_joints
    )
    
    np.testing.assert_allclose(original_joints, recovered_joints, atol=1e-3)
```

## Troubleshooting

### Controller not detected

```bash
# List input devices
python -c "import inputs; print(inputs.devices.gamepads)"

# Check permissions (Linux)
sudo usermod -aG input $USER
# Logout and login again
```

### LeRobot kinematics import error

```bash
# Ensure kinematics extra is installed
uv pip install "lerobot[kinematics]"

# Verify Placo installation
python -c "from lerobot.common.kinematics import RobotKinematics; print('OK')"
```

### IK fails to converge

- Check that target pose is within workspace bounds
- Verify `initial_guess_current_joints=True` is set
- Reduce step size (linear_scale, angular_scale)
- Check URDF joint limits match physical robot

### Robot connection issues

```bash
# Find USB port
ls /dev/ttyUSB* /dev/ttyACM*

# Check permissions
sudo chmod 666 /dev/ttyUSB0
```

## Important Notes

### Development Workflow
- **Always write tests** before implementing new features (TDD)
- **Run pre-commit** before pushing: `pre-commit run --all-files`
- **Use conventional commits** for clear history
- **Test with robot disconnected first** using mock/simulation

### Safety First
- Always test new code with robot powered off or in simulation
- Deadman switch must be working before any live testing
- Keep workspace bounds conservative initially
- Have physical e-stop accessible during testing
