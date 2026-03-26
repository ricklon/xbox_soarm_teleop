"""Pytest fixtures for xbox_soarm_teleop tests."""

import pytest

from xbox_soarm_teleop.config.joycon_config import JoyConConfig
from xbox_soarm_teleop.config.xbox_config import XboxConfig
from xbox_soarm_teleop.teleoperators.joycon import JoyConController
from xbox_soarm_teleop.teleoperators.xbox import XboxController, XboxState


@pytest.fixture
def joycon_config() -> JoyConConfig:
    """Default Joy-Con configuration for testing."""
    return JoyConConfig()


@pytest.fixture
def joycon_controller(joycon_config: JoyConConfig) -> JoyConController:
    """JoyConController instance for testing (not connected)."""
    return JoyConController(joycon_config)


@pytest.fixture
def xbox_config() -> XboxConfig:
    """Default Xbox configuration for testing."""
    return XboxConfig()


@pytest.fixture
def xbox_controller(xbox_config: XboxConfig) -> XboxController:
    """Xbox controller instance for testing (not connected)."""
    return XboxController(xbox_config)


@pytest.fixture
def default_state() -> XboxState:
    """Default (zero) Xbox state."""
    return XboxState()


@pytest.fixture
def full_forward_state() -> XboxState:
    """Xbox state with full forward on right stick Y."""
    return XboxState(
        left_stick_x=0.0,
        left_stick_y=0.0,
        right_stick_x=0.0,
        right_stick_y=1.0,  # Forward/back is now right stick Y
        right_trigger=0.0,
        left_bumper=True,  # Deadman held
    )


@pytest.fixture
def deadman_released_state() -> XboxState:
    """Xbox state with deadman switch released."""
    return XboxState(
        left_stick_x=0.5,
        left_stick_y=0.5,
        right_stick_x=0.5,
        right_stick_y=0.5,
        right_trigger=0.5,
        left_bumper=False,  # Deadman NOT held
    )
