"""Tests for Xbox controller input handling."""

import pytest

from xbox_soarm_teleop.config.xbox_config import XboxConfig
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta, MapXboxToEEDelta
from xbox_soarm_teleop.teleoperators.xbox import XboxController, XboxState


class TestXboxState:
    """Tests for XboxState dataclass."""

    def test_default_initialization(self):
        """Default state should have all zeros and False buttons."""
        state = XboxState()
        assert state.left_stick_x == 0.0
        assert state.left_stick_y == 0.0
        assert state.right_stick_x == 0.0
        assert state.right_stick_y == 0.0
        assert state.right_trigger == 0.0
        assert state.left_bumper is False
        assert state.a_button is False
        assert state.y_button is False

    def test_custom_initialization(self):
        """State should accept custom values."""
        state = XboxState(
            left_stick_x=0.5,
            left_stick_y=-0.3,
            right_trigger=1.0,
            left_bumper=True,
        )
        assert state.left_stick_x == 0.5
        assert state.left_stick_y == -0.3
        assert state.right_trigger == 1.0
        assert state.left_bumper is True


class TestXboxController:
    """Tests for XboxController class."""

    def test_initialization_with_default_config(self):
        """Controller should initialize with default config."""
        controller = XboxController()
        assert controller.config.deadzone == 0.1
        assert controller.is_connected is False

    def test_initialization_with_custom_config(self, xbox_config: XboxConfig):
        """Controller should accept custom config."""
        controller = XboxController(xbox_config)
        assert controller.config is xbox_config

    def test_deadzone_filters_small_values(self, xbox_controller: XboxController):
        """Deadzone should filter values below threshold."""
        assert xbox_controller._apply_deadzone(0.05) == 0.0
        assert xbox_controller._apply_deadzone(-0.05) == 0.0
        assert xbox_controller._apply_deadzone(0.09) == 0.0

    def test_deadzone_passes_large_values(self, xbox_controller: XboxController):
        """Deadzone should pass and rescale values above threshold."""
        # Value just above deadzone (0.1)
        result = xbox_controller._apply_deadzone(0.15)
        assert result > 0.0
        # Should be rescaled: (0.15 - 0.1) / (1.0 - 0.1) = 0.0555...
        assert abs(result - 0.0555) < 0.01

    def test_deadzone_preserves_sign(self, xbox_controller: XboxController):
        """Deadzone should preserve sign of input."""
        positive = xbox_controller._apply_deadzone(0.5)
        negative = xbox_controller._apply_deadzone(-0.5)
        assert positive > 0
        assert negative < 0
        assert abs(positive) == abs(negative)

    def test_deadzone_full_range(self, xbox_controller: XboxController):
        """Full stick deflection should return +/- 1.0."""
        assert xbox_controller._apply_deadzone(1.0) == 1.0
        assert xbox_controller._apply_deadzone(-1.0) == -1.0

    def test_normalize_stick_center(self, xbox_controller: XboxController):
        """Center position should normalize to 0."""
        # Center of default range (-32768, 32767) is approximately 0
        result = xbox_controller._normalize_stick(0)
        assert abs(result) < 0.01  # Should be very close to 0

    def test_normalize_stick_extremes(self, xbox_controller: XboxController):
        """Extreme positions should normalize to +/- 1."""
        min_val, max_val = xbox_controller.config.stick_range
        result_max = xbox_controller._normalize_stick(max_val)
        result_min = xbox_controller._normalize_stick(min_val)
        # After deadzone, full deflection should still be ~1.0
        assert result_max > 0.9
        assert result_min < -0.9

    def test_normalize_stick_invert(self, xbox_controller: XboxController):
        """Inversion should flip the sign."""
        value = 16000  # Positive deflection
        normal = xbox_controller._normalize_stick(value, invert=False)
        inverted = xbox_controller._normalize_stick(value, invert=True)
        assert normal > 0
        assert inverted < 0

    def test_normalize_trigger_range(self, xbox_controller: XboxController):
        """Trigger should normalize to [0, 1]."""
        min_val, max_val = xbox_controller.config.trigger_range
        assert xbox_controller._normalize_trigger(min_val) == 0.0
        assert xbox_controller._normalize_trigger(max_val) == 1.0
        assert xbox_controller._normalize_trigger((min_val + max_val) // 2) == pytest.approx(
            0.5, abs=0.01
        )

    def test_read_returns_state_when_disconnected(self, xbox_controller: XboxController):
        """Read should return default state when not connected."""
        state = xbox_controller.read()
        assert isinstance(state, XboxState)
        assert state.left_stick_x == 0.0
        assert state.left_bumper is False


class TestEEDelta:
    """Tests for EEDelta dataclass."""

    def test_default_initialization(self):
        """Default delta should be all zeros."""
        delta = EEDelta()
        assert delta.dx == 0.0
        assert delta.dy == 0.0
        assert delta.dz == 0.0
        assert delta.droll == 0.0
        assert delta.gripper == 0.0

    def test_as_array(self):
        """as_array should return correct list."""
        delta = EEDelta(dx=0.1, dy=0.2, dz=0.3, droll=0.4, gripper=0.5)
        arr = delta.as_array()
        assert arr == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_is_zero_motion(self):
        """is_zero_motion should detect zero velocity."""
        zero_delta = EEDelta()
        assert zero_delta.is_zero_motion() is True

        moving_delta = EEDelta(dx=0.1)
        assert moving_delta.is_zero_motion() is False

        gripper_only = EEDelta(gripper=0.5)
        assert gripper_only.is_zero_motion() is True  # Gripper doesn't count as motion


class TestMapXboxToEEDelta:
    """Tests for MapXboxToEEDelta processor."""

    def test_initialization(self):
        """Processor should initialize with default scales."""
        mapper = MapXboxToEEDelta()
        assert mapper.linear_scale == 0.1
        assert mapper.angular_scale == 0.5

    def test_custom_scales(self):
        """Processor should accept custom scales."""
        mapper = MapXboxToEEDelta(linear_scale=0.2, angular_scale=1.0)
        assert mapper.linear_scale == 0.2
        assert mapper.angular_scale == 1.0

    def test_deadman_blocks_motion(self, deadman_released_state: XboxState):
        """Motion should be blocked when deadman is released."""
        mapper = MapXboxToEEDelta()
        delta = mapper(deadman_released_state)

        # All motion should be zero
        assert delta.dx == 0.0
        assert delta.dy == 0.0
        assert delta.dz == 0.0
        assert delta.droll == 0.0

        # Gripper should still work
        assert delta.gripper == 0.5

    def test_deadman_allows_motion(self, full_forward_state: XboxState):
        """Motion should be allowed when deadman is held."""
        mapper = MapXboxToEEDelta(linear_scale=0.1)
        delta = mapper(full_forward_state)

        # Forward motion from right_stick_y = 1.0 (negated)
        assert delta.dx == -0.1
        assert delta.dy == 0.0
        assert delta.dz == 0.0
        assert delta.droll == 0.0

    def test_gripper_always_active(self):
        """Gripper should respond regardless of deadman switch."""
        mapper = MapXboxToEEDelta()

        # Deadman released
        state_released = XboxState(right_trigger=0.7, left_bumper=False)
        delta_released = mapper(state_released)
        assert delta_released.gripper == 0.7

        # Deadman held
        state_held = XboxState(right_trigger=0.3, left_bumper=True)
        delta_held = mapper(state_held)
        assert delta_held.gripper == 0.3

    def test_full_motion_mapping(self):
        """Full stick deflection should map to full scale velocities."""
        mapper = MapXboxToEEDelta(linear_scale=0.1, angular_scale=0.5)
        state = XboxState(
            left_stick_x=1.0,
            left_stick_y=1.0,
            right_stick_x=1.0,
            right_stick_y=1.0,
            right_trigger=1.0,
            left_bumper=True,
        )
        delta = mapper(state)

        assert delta.dx == -0.1  # -right_stick_y * linear_scale (forward/back, negated)
        assert delta.dy == -0.1  # -left_stick_x * linear_scale (left/right, negated)
        assert delta.dz == -0.1  # -left_stick_y * linear_scale (up/down, negated)
        assert delta.droll == 0.5  # right_stick_x * angular_scale (roll)
        assert delta.gripper == 1.0
