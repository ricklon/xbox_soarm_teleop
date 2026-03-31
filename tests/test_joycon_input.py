"""Tests for Joy-Con input handling."""


from xbox_soarm_teleop.config.joycon_config import DualJoyConConfig, JoyConConfig
from xbox_soarm_teleop.teleoperators.joycon import (
    DualJoyConController,
    JoyConController,
    _is_matching_joycon_device_name,
)
from xbox_soarm_teleop.teleoperators.xbox import XboxState


class TestJoyConConfig:
    def test_defaults(self):
        cfg = JoyConConfig()
        assert cfg.deadzone == 0.1
        assert cfg.deadman_button == "BTN_TL"
        assert cfg.zr_button == "BTN_TR2"
        assert cfg.left_stick_x_axis == "ABS_RX"
        assert cfg.left_stick_y_axis == "ABS_RY"
        assert cfg.stick_range == (-32767, 32767)
        assert "Joy-Con (R)" in cfg.device_name_patterns

    def test_single_stick_maps_to_both_sides(self):
        cfg = JoyConConfig()
        assert cfg.left_stick_x_axis == cfg.right_stick_x_axis
        assert cfg.left_stick_y_axis == cfg.right_stick_y_axis

    def test_no_dpad_axes(self):
        cfg = JoyConConfig()
        assert cfg.dpad_x_axis == ""
        assert cfg.dpad_y_axis == ""

    def test_device_name_matching_excludes_imu(self):
        cfg = JoyConConfig()
        assert _is_matching_joycon_device_name("Joy-Con (R)", cfg.device_name_patterns) is True
        assert (
            _is_matching_joycon_device_name("Joy-Con (R) (IMU)", cfg.device_name_patterns) is False
        )


class TestDualJoyConConfig:
    def test_defaults(self):
        cfg = DualJoyConConfig()
        assert cfg.deadzone == 0.1
        assert cfg.deadman_button == "BTN_TL2"
        assert cfg.gripper_button == "BTN_TR2"
        assert cfg.left_stick_x_axis == "ABS_X"
        assert cfg.right_stick_x_axis == "ABS_RX"
        assert "Joy-Con (L)" in cfg.left_device_name_patterns
        assert "Joy-Con (R)" in cfg.right_device_name_patterns


class TestJoyConController:
    def test_init_defaults(self):
        ctrl = JoyConController()
        assert ctrl.config.deadzone == 0.1
        assert ctrl.is_connected is False

    def test_init_custom_config(self):
        cfg = JoyConConfig(deadzone=0.2)
        ctrl = JoyConController(cfg)
        assert ctrl.config.deadzone == 0.2

    def test_read_when_disconnected_returns_zero_state(self):
        ctrl = JoyConController()
        state = ctrl.read()
        assert isinstance(state, XboxState)
        assert state.left_stick_x == 0.0
        assert state.left_bumper is False
        assert state.right_trigger == 0.0

    def test_deadzone_filters_small_values(self):
        ctrl = JoyConController()
        x, y = ctrl._apply_radial_deadzone(0.05, 0.05)
        assert x == 0.0
        assert y == 0.0

    def test_deadzone_passes_large_values(self):
        ctrl = JoyConController()
        x, y = ctrl._apply_radial_deadzone(0.5, 0.0)
        assert x > 0.0
        assert y == 0.0

    def test_deadzone_full_deflection(self):
        ctrl = JoyConController()
        x, y = ctrl._apply_radial_deadzone(1.0, 0.0)
        assert abs(x - 1.0) < 0.01

    def test_normalize_stick_center(self):
        ctrl = JoyConController()
        assert abs(ctrl._normalize_stick_raw(0)) < 0.01

    def test_normalize_stick_extremes(self):
        ctrl = JoyConController()
        assert ctrl._normalize_stick_raw(32767) > 0.99
        assert ctrl._normalize_stick_raw(-32767) < -0.99

    def test_normalize_stick_invert(self):
        ctrl = JoyConController()
        pos = ctrl._normalize_stick_raw(16000, invert=False)
        inv = ctrl._normalize_stick_raw(16000, invert=True)
        assert pos > 0
        assert inv < 0

    def test_zr_button_maps_to_trigger(self):
        ctrl = JoyConController()
        ctrl._connected = True
        ctrl._raw_state["BTN_TR2"] = 1
        state = ctrl.read()
        assert state.right_trigger == 1.0

    def test_zr_released_maps_to_zero(self):
        ctrl = JoyConController()
        ctrl._connected = True
        ctrl._raw_state["BTN_TR2"] = 0
        state = ctrl.read()
        assert state.right_trigger == 0.0

    def test_deadman_sl_button(self):
        ctrl = JoyConController()
        ctrl._connected = True
        ctrl._raw_state["BTN_TL"] = 1
        state = ctrl.read()
        assert state.left_bumper is True

    def test_stick_maps_to_both_left_and_right(self):
        ctrl = JoyConController()
        ctrl._connected = True
        ctrl._raw_state["ABS_RX"] = 32767
        ctrl._raw_state["ABS_RY"] = 0
        state = ctrl.read()
        assert abs(state.left_stick_x - state.right_stick_x) < 0.001
        assert abs(state.left_stick_y - state.right_stick_y) < 0.001

    def test_no_dpad_output(self):
        ctrl = JoyConController()
        ctrl._connected = True
        state = ctrl.read()
        assert state.dpad_x == 0.0
        assert state.dpad_y == 0.0

    def test_a_button_pressed_edge_detection(self):
        ctrl = JoyConController()
        ctrl._connected = True
        # First read: button up
        ctrl._raw_state["BTN_START"] = 0
        ctrl.read()
        # Second read: button pressed
        ctrl._raw_state["BTN_START"] = 1
        state = ctrl.read()
        assert state.a_button_pressed is True
        # Third read: held — no longer a fresh press
        state = ctrl.read()
        assert state.a_button_pressed is False

    def test_connect_fails_without_device(self, monkeypatch):
        """connect() returns False when no matching device is found."""
        import xbox_soarm_teleop.teleoperators.joycon as joycon_mod

        monkeypatch.setattr(joycon_mod, "__builtins__", __builtins__)
        ctrl = JoyConController()

        # Patch _find_device to return None
        ctrl._find_device = lambda *args, **kwargs: None
        result = ctrl.connect()
        assert result is False
        assert ctrl.is_connected is False


class TestDualJoyConController:
    def test_init_defaults(self):
        ctrl = DualJoyConController()
        assert ctrl.config.deadzone == 0.1
        assert ctrl.is_connected is False

    def test_read_when_disconnected_returns_zero_state(self):
        ctrl = DualJoyConController()
        state = ctrl.read()
        assert isinstance(state, XboxState)
        assert state.left_stick_x == 0.0
        assert state.imu_orientation_valid is False

    def test_read_merges_left_right_and_imu(self):
        ctrl = DualJoyConController()
        ctrl._connected = True
        ctrl._left_raw_state["ABS_X"] = 32767
        ctrl._left_raw_state["ABS_Y"] = 0
        ctrl._left_raw_state["BTN_TL2"] = 1
        ctrl._left_raw_state["BTN_DPAD_UP"] = 1
        ctrl._right_raw_state["ABS_RX"] = -32767
        ctrl._right_raw_state["ABS_RY"] = 0
        ctrl._right_raw_state["BTN_TR2"] = 1
        ctrl._right_raw_state["BTN_START"] = 1

        class _FakeIMU:
            available = True

            def read_orientation(self):
                return (0.1, -0.2, 0.3)

        ctrl._imu = _FakeIMU()
        state = ctrl.read()

        assert state.left_stick_x > 0.9
        assert abs(state.left_stick_y) < 0.01
        assert state.right_stick_x < -0.9
        assert state.left_bumper is True
        assert state.right_trigger == 1.0
        assert state.a_button is True
        assert state.dpad_y == -1.0
        assert state.imu_pitch == 0.1
        assert state.imu_roll == -0.2
        assert state.imu_yaw == 0.3
        assert state.imu_orientation_valid is True
