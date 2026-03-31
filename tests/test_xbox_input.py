"""Tests for Xbox controller input handling."""

import pytest

from xbox_soarm_teleop.config.xbox_config import XboxConfig
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta, MapDualJoyConToEEDelta, MapXboxToEEDelta
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
        assert state.imu_roll == 0.0
        assert state.imu_pitch == 0.0
        assert state.imu_yaw == 0.0
        assert state.imu_orientation_valid is False

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

    def test_radial_deadzone_filters_small_values(self, xbox_controller: XboxController):
        """Radial deadzone should filter values below threshold."""
        # Small deflection in any direction
        x, y = xbox_controller._apply_radial_deadzone(0.05, 0.05)
        assert x == 0.0
        assert y == 0.0

        # Even diagonal small values should be filtered
        x, y = xbox_controller._apply_radial_deadzone(0.07, 0.07)
        assert x == 0.0
        assert y == 0.0

    def test_radial_deadzone_passes_large_values(self, xbox_controller: XboxController):
        """Radial deadzone should pass and rescale values above threshold."""
        # Value clearly above deadzone (0.1)
        x, y = xbox_controller._apply_radial_deadzone(0.5, 0.0)
        assert x > 0.0
        # Rescaled: (0.5 - 0.1) / (1.0 - 0.1) = 0.444...
        assert abs(x - 0.444) < 0.01
        assert y == 0.0

    def test_radial_deadzone_preserves_direction(self, xbox_controller: XboxController):
        """Radial deadzone should preserve direction of input."""
        x_pos, y_pos = xbox_controller._apply_radial_deadzone(0.5, 0.5)
        x_neg, y_neg = xbox_controller._apply_radial_deadzone(-0.5, -0.5)
        assert x_pos > 0 and y_pos > 0
        assert x_neg < 0 and y_neg < 0

    def test_radial_deadzone_full_range(self, xbox_controller: XboxController):
        """Full stick deflection should return magnitude ~1.0."""
        x, y = xbox_controller._apply_radial_deadzone(1.0, 0.0)
        assert abs(x - 1.0) < 0.01
        assert y == 0.0

    def test_minor_axis_attenuated_for_dominant_cardinal_motion(
        self, xbox_controller: XboxController
    ):
        """Small orthogonal bleed should be suppressed for strong cardinal input."""
        x, y = xbox_controller._apply_radial_deadzone(0.08, 0.9)
        assert y > 0.8
        assert abs(x) < 0.03

    def test_intentional_diagonal_motion_is_preserved(self, xbox_controller: XboxController):
        """Real diagonal motion should not be collapsed into a cardinal direction."""
        x, y = xbox_controller._apply_radial_deadzone(0.7, 0.6)
        assert x > 0.4
        assert y > 0.4

    def test_normalize_stick_raw_center(self, xbox_controller: XboxController):
        """Center position should normalize to ~0."""
        # Center of default range (-32768, 32767) is approximately 0
        result = xbox_controller._normalize_stick_raw(0)
        assert abs(result) < 0.01  # Should be very close to 0

    def test_normalize_stick_raw_extremes(self, xbox_controller: XboxController):
        """Extreme positions should normalize to +/- 1."""
        min_val, max_val = xbox_controller.config.stick_range
        result_max = xbox_controller._normalize_stick_raw(max_val)
        result_min = xbox_controller._normalize_stick_raw(min_val)
        assert result_max > 0.99
        assert result_min < -0.99

    def test_normalize_stick_raw_invert(self, xbox_controller: XboxController):
        """Inversion should flip the sign."""
        value = 16000  # Positive deflection
        normal = xbox_controller._normalize_stick_raw(value, invert=False)
        inverted = xbox_controller._normalize_stick_raw(value, invert=True)
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
        assert delta.dpitch == 0.0
        assert delta.dyaw == 0.0
        assert delta.gripper == 0.0

    def test_as_array(self):
        """as_array should return correct list."""
        delta = EEDelta(dx=0.1, dy=0.2, dz=0.3, droll=0.4, dpitch=0.5, dyaw=0.6, gripper=0.7)
        arr = delta.as_array()
        assert arr == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    def test_is_zero_motion(self):
        """is_zero_motion should detect zero velocity."""
        zero_delta = EEDelta()
        assert zero_delta.is_zero_motion() is True

        moving_delta = EEDelta(dx=0.1)
        assert moving_delta.is_zero_motion() is False

        gripper_only = EEDelta(gripper=0.5)
        assert gripper_only.is_zero_motion() is True  # Gripper doesn't count as motion

        target_only = EEDelta(roll_target=0.0)
        assert target_only.is_zero_motion() is False


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

        # Forward motion from left_stick_y = 1.0 (negated)
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

        assert delta.dx == -0.1  # -left_stick_y * linear_scale (forward/back, negated)
        assert delta.dy == -0.1  # -left_stick_x * linear_scale (left/right, negated)
        assert delta.dz == -0.1  # -right_stick_y * linear_scale (up/down, negated)
        assert delta.droll == 0.5  # right_stick_x * angular_scale (roll)
        assert delta.gripper == 1.0

    def test_dpad_up_maps_to_positive_pitch(self):
        """Cartesian/recording path should interpret D-pad up as pitch up."""
        mapper = MapXboxToEEDelta(orientation_scale=1.0, enable_pitch=True)
        delta = mapper(XboxState(left_bumper=True, dpad_y=-1.0))
        assert delta.dpitch == 1.0

    def test_dpad_orientation_disabled_by_default(self):
        mapper = MapXboxToEEDelta(orientation_scale=1.0)
        delta = mapper(XboxState(left_bumper=True, dpad_y=-1.0, dpad_x=1.0))
        assert delta.dpitch == 0.0
        assert delta.dyaw == 0.0


class TestMapDualJoyConToEEDelta:
    def test_deadman_blocks_motion_and_resets_clutch(self):
        mapper = MapDualJoyConToEEDelta()
        delta = mapper(
            XboxState(
                left_bumper=False,
                left_stick_x=1.0,
                left_stick_y=1.0,
                dpad_y=-1.0,
                right_trigger=0.5,
                imu_orientation_valid=True,
                imu_roll=0.1,
                imu_pitch=0.2,
                imu_yaw=0.3,
            )
        )
        assert delta.dx == 0.0
        assert delta.dy == 0.0
        assert delta.dz == 0.0
        assert delta.gripper == 0.5
        assert delta.roll_target is None

    def test_first_deadman_press_captures_imu_neutral(self):
        mapper = MapDualJoyConToEEDelta()
        state = XboxState(
            left_bumper=True,
            imu_orientation_valid=True,
            imu_roll=0.2,
            imu_pitch=-0.1,
            imu_yaw=0.3,
        )
        delta = mapper(state)
        assert delta.roll_target == pytest.approx(0.0)
        assert delta.pitch_target == pytest.approx(0.0)
        assert delta.yaw_target == pytest.approx(0.0)

    def test_translation_and_orientation_follow_dual_mapping(self):
        mapper = MapDualJoyConToEEDelta(linear_scale=0.1, vertical_scale=0.08)
        mapper(
            XboxState(
                left_bumper=True,
                imu_orientation_valid=True,
                imu_roll=0.2,
                imu_pitch=-0.1,
                imu_yaw=0.3,
            )
        )
        delta = mapper(
            XboxState(
                left_bumper=True,
                left_stick_x=0.5,
                left_stick_y=-1.0,
                dpad_y=-1.0,
                right_trigger=1.0,
                imu_orientation_valid=True,
                imu_roll=0.3,
                imu_pitch=0.2,
                imu_yaw=0.7,
            )
        )
        assert delta.dx == pytest.approx(0.1)
        assert delta.dy == pytest.approx(-0.05)
        assert delta.dz == pytest.approx(0.08)
        assert delta.gripper == 1.0
        assert delta.roll_target == pytest.approx(0.4)
        assert delta.pitch_target == pytest.approx(0.3)
        assert delta.yaw_target == pytest.approx(-0.075)

    def test_deadman_repress_reclutches_orientation(self):
        mapper = MapDualJoyConToEEDelta()
        mapper(
            XboxState(
                left_bumper=True,
                imu_orientation_valid=True,
                imu_roll=0.0,
                imu_pitch=0.0,
                imu_yaw=0.0,
            )
        )
        mapper(XboxState(left_bumper=False))
        delta = mapper(
            XboxState(
                left_bumper=True,
                imu_orientation_valid=True,
                imu_roll=0.5,
                imu_pitch=0.4,
                imu_yaw=0.3,
            )
        )
        assert delta.roll_target == pytest.approx(0.0)
        assert delta.pitch_target == pytest.approx(0.0)
        assert delta.yaw_target == pytest.approx(0.0)
