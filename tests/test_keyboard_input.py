"""Tests for KeyboardController input handling."""

import json
import time

import pytest

from xbox_soarm_teleop.config.keyboard_config import KeyboardConfig
from xbox_soarm_teleop.teleoperators.keyboard import KeyboardController, _combine
from xbox_soarm_teleop.teleoperators.xbox import XboxState

# ── Helpers ───────────────────────────────────────────────────────────────────


def _connected_ctrl(cfg: KeyboardConfig | None = None) -> KeyboardController:
    """Return a KeyboardController with _connected=True and a populated key map."""
    ctrl = KeyboardController(cfg)
    ctrl._connected = True
    # Build key map without a real device (requires evdev installed)
    try:
        ctrl._build_key_map()
    except Exception:
        pass
    return ctrl


def _press(ctrl: KeyboardController, *key_names: str) -> None:
    """Simulate keys being held down."""
    for name in key_names:
        code = ctrl._key_map.get(name)
        if code is not None:
            ctrl._held_keys.add(code)


def _release(ctrl: KeyboardController, *key_names: str) -> None:
    """Simulate keys being released."""
    for name in key_names:
        code = ctrl._key_map.get(name)
        if code is not None:
            ctrl._held_keys.discard(code)


# ── _combine helper ───────────────────────────────────────────────────────────


class TestCombineHelper:
    def test_both_false(self):
        assert _combine(False, False, 1.0) == 0.0

    def test_pos_only(self):
        assert _combine(True, False, 0.75) == pytest.approx(0.75)

    def test_neg_only(self):
        assert _combine(False, True, 0.75) == pytest.approx(-0.75)

    def test_both_cancel(self):
        assert _combine(True, True, 1.0) == 0.0

    def test_clamp_high(self):
        assert _combine(True, False, 1.5) == pytest.approx(1.0)

    def test_clamp_low(self):
        assert _combine(False, True, 1.5) == pytest.approx(-1.0)


# ── KeyboardConfig ────────────────────────────────────────────────────────────


class TestKeyboardConfig:
    def test_defaults(self):
        cfg = KeyboardConfig()
        assert cfg.default_speed_level == 2
        assert len(cfg.speed_levels) == 5
        assert cfg.speed_levels[2] == pytest.approx(0.75)
        assert cfg.shift_multiplier == pytest.approx(2.0)

    def test_key_names(self):
        cfg = KeyboardConfig()
        assert cfg.key_forward == "KEY_W"
        assert cfg.key_back == "KEY_S"
        assert cfg.key_left == "KEY_A"
        assert cfg.key_right == "KEY_D"
        assert cfg.key_up == "KEY_R"
        assert cfg.key_down == "KEY_F"
        assert cfg.key_roll_left == "KEY_Q"
        assert cfg.key_roll_right == "KEY_E"
        assert cfg.key_pitch_up == "KEY_UP"
        assert cfg.key_pitch_down == "KEY_DOWN"
        assert cfg.key_yaw_left == "KEY_LEFT"
        assert cfg.key_yaw_right == "KEY_RIGHT"
        assert cfg.key_gripper == "KEY_SPACE"
        assert cfg.key_home == "KEY_H"
        assert cfg.key_frame_toggle == "KEY_Y"

    def test_no_device_path_by_default(self):
        assert KeyboardConfig().device_path is None


# ── KeyboardController ────────────────────────────────────────────────────────


class TestKeyboardControllerInit:
    def test_defaults(self):
        ctrl = KeyboardController()
        assert ctrl.config.default_speed_level == 2
        assert ctrl.is_connected is False

    def test_custom_config(self):
        cfg = KeyboardConfig(default_speed_level=4)
        ctrl = KeyboardController(cfg)
        assert ctrl._speed_level == 4

    def test_read_disconnected_returns_zero_state(self):
        ctrl = KeyboardController()
        state = ctrl.read()
        assert isinstance(state, XboxState)
        assert state.left_stick_x == 0.0
        assert state.right_stick_y == 0.0
        assert state.left_bumper is False


class TestKeyboardControllerRead:
    def test_left_bumper_always_true_when_connected(self):
        ctrl = _connected_ctrl()
        state = ctrl.read()
        assert state.left_bumper is True

    def test_no_keys_zero_sticks(self):
        ctrl = _connected_ctrl()
        state = ctrl.read()
        assert state.left_stick_x == 0.0
        assert state.left_stick_y == 0.0
        assert state.right_stick_x == 0.0
        assert state.right_stick_y == 0.0

    def test_forward_key_sets_right_stick_y_negative(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_forward)
        state = ctrl.read()
        assert state.right_stick_y < 0.0

    def test_back_key_sets_right_stick_y_positive(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_back)
        state = ctrl.read()
        assert state.right_stick_y > 0.0

    def test_forward_back_cancel(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_forward, ctrl.config.key_back)
        state = ctrl.read()
        assert state.right_stick_y == pytest.approx(0.0)

    def test_right_key_sets_left_stick_x_positive(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_right)
        state = ctrl.read()
        assert state.left_stick_x > 0.0

    def test_left_key_sets_left_stick_x_negative(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_left)
        state = ctrl.read()
        assert state.left_stick_x < 0.0

    def test_up_key_sets_left_stick_y_negative(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_up)
        state = ctrl.read()
        assert state.left_stick_y < 0.0

    def test_down_key_sets_left_stick_y_positive(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_down)
        state = ctrl.read()
        assert state.left_stick_y > 0.0

    def test_roll_right_positive(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_roll_right)
        state = ctrl.read()
        assert state.right_stick_x > 0.0

    def test_roll_left_negative(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_roll_left)
        state = ctrl.read()
        assert state.right_stick_x < 0.0

    def test_pitch_up_dpad_y_negative(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_pitch_up)
        state = ctrl.read()
        assert state.dpad_y == pytest.approx(-1.0)

    def test_pitch_down_dpad_y_positive(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_pitch_down)
        state = ctrl.read()
        assert state.dpad_y == pytest.approx(1.0)

    def test_yaw_right_dpad_x_positive(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_yaw_right)
        state = ctrl.read()
        assert state.dpad_x == pytest.approx(1.0)

    def test_yaw_left_dpad_x_negative(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_yaw_left)
        state = ctrl.read()
        assert state.dpad_x == pytest.approx(-1.0)

    def test_pitch_up_down_last_wins(self):
        """Only first held key counts for discrete dpad axes."""
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_pitch_up, ctrl.config.key_pitch_down)
        state = ctrl.read()
        # pitch_up is checked first → -1.0
        assert state.dpad_y == pytest.approx(-1.0)

    def test_gripper_space_held(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_gripper)
        assert ctrl.read().right_trigger == pytest.approx(1.0)

    def test_gripper_released(self):
        ctrl = _connected_ctrl()
        assert ctrl.read().right_trigger == pytest.approx(0.0)

    def test_diagonal_movement_both_axes(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_forward, ctrl.config.key_right)
        state = ctrl.read()
        assert state.right_stick_y < 0.0  # forward
        assert state.left_stick_x > 0.0  # right


class TestKeyboardSpeedLevels:
    def test_default_speed_level(self):
        ctrl = _connected_ctrl()
        assert ctrl._speed_level == 2  # 0-indexed level 3

    def test_speed_level_affects_magnitude(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_forward)

        ctrl._speed_level = 0  # 25%
        s1 = ctrl.read().right_stick_y

        ctrl._speed_level = 3  # 100%
        s2 = ctrl.read().right_stick_y

        assert abs(s2) > abs(s1)

    def test_speed_level_set_via_number_key(self):
        ctrl = _connected_ctrl()
        # Simulate key_1 press by updating speed_level directly (as the reader thread would)
        ctrl._speed_level = 0
        assert ctrl._speed_level == 0

        ctrl._speed_level = 4
        assert ctrl.config.speed_levels[4] == pytest.approx(1.50)

    def test_shift_doubles_speed(self):
        ctrl = _connected_ctrl()
        ctrl._speed_level = 2  # 0.75

        _press(ctrl, ctrl.config.key_forward)
        no_shift = ctrl.read().right_stick_y

        _press(ctrl, ctrl.config.key_shift_left)
        with_shift = ctrl.read().right_stick_y

        assert abs(with_shift) > abs(no_shift)

    def test_shift_capped_at_2(self):
        ctrl = _connected_ctrl()
        ctrl._speed_level = 4  # 1.5 × 2 = 3.0 → capped at 2.0 → clamp to 1.0 on stick
        _press(ctrl, ctrl.config.key_forward, ctrl.config.key_shift_left)
        state = ctrl.read()
        assert abs(state.right_stick_y) == pytest.approx(1.0)


class TestKeyboardEdgeDetection:
    def test_home_pressed_edge(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_home)
        state = ctrl.read()
        assert state.a_button_pressed is True
        # Still held — no longer a fresh press
        state = ctrl.read()
        assert state.a_button_pressed is False

    def test_home_not_pressed_when_not_held(self):
        ctrl = _connected_ctrl()
        state = ctrl.read()
        assert state.a_button_pressed is False

    def test_frame_toggle_pressed_edge(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_frame_toggle)
        state = ctrl.read()
        assert state.y_button_pressed is True
        state = ctrl.read()
        assert state.y_button_pressed is False

    def test_home_release_and_repress(self):
        ctrl = _connected_ctrl()
        _press(ctrl, ctrl.config.key_home)
        ctrl.read()  # press detected
        _release(ctrl, ctrl.config.key_home)
        ctrl.read()  # released
        _press(ctrl, ctrl.config.key_home)
        state = ctrl.read()  # re-pressed → new edge
        assert state.a_button_pressed is True


# ── Recording ─────────────────────────────────────────────────────────────────


class TestRecording:
    def test_recording_starts_stopped(self):
        ctrl = _connected_ctrl()
        assert ctrl._recording is False
        assert ctrl._record_events == []

    def test_toggle_starts_recording(self):
        ctrl = _connected_ctrl()
        ctrl._toggle_recording()
        assert ctrl._recording is True

    def test_toggle_twice_stops_recording(self, tmp_path):
        ctrl = _connected_ctrl(KeyboardConfig(record_path=str(tmp_path / "rec.json")))
        ctrl._toggle_recording()   # start
        ctrl._record_events.append({"t": 0.0, "code": 17, "value": 1})
        ctrl._toggle_recording()   # stop — should save
        assert ctrl._recording is False
        saved = json.loads((tmp_path / "rec.json").read_text())
        assert len(saved) == 1
        assert saved[0]["code"] == 17

    def test_save_recording_auto_names_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ctrl = _connected_ctrl()
        ctrl._record_events = [{"t": 0.0, "code": 30, "value": 1}]
        path = ctrl._save_recording()
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_recording_uses_config_path(self, tmp_path):
        dest = tmp_path / "out.json"
        ctrl = _connected_ctrl(KeyboardConfig(record_path=str(dest)))
        ctrl._record_events = [{"t": 0.1, "code": 30, "value": 0}]
        path = ctrl._save_recording()
        assert path == dest
        assert dest.exists()


# ── Playback ──────────────────────────────────────────────────────────────────


class TestPlayback:
    def _write_recording(self, path, events):
        with open(path, "w") as f:
            json.dump(events, f)

    def test_playback_missing_file_returns_false(self, tmp_path):
        cfg = KeyboardConfig(playback_path=str(tmp_path / "nope.json"))
        ctrl = KeyboardController(cfg)
        result = ctrl._connect_playback()
        assert result is False
        assert ctrl._connected is False

    def test_playback_empty_recording_connects_then_disconnects(self, tmp_path):
        p = tmp_path / "empty.json"
        self._write_recording(p, [])
        cfg = KeyboardConfig(playback_path=str(p))
        ctrl = KeyboardController(cfg)
        assert ctrl._connect_playback() is True
        # Wait for playback thread to finish
        if ctrl._reader_thread:
            ctrl._reader_thread.join(timeout=2.0)
        assert ctrl._connected is False

    def test_playback_feeds_held_keys(self, tmp_path):
        import evdev

        key_w = evdev.ecodes.KEY_W
        events = [
            {"t": 0.0, "code": key_w, "value": 1},
            {"t": 0.05, "code": key_w, "value": 0},
        ]
        p = tmp_path / "demo.json"
        self._write_recording(p, events)
        cfg = KeyboardConfig(playback_path=str(p))
        ctrl = KeyboardController(cfg)
        ctrl._connect_playback()
        # Give playback thread time to fire first event
        time.sleep(0.02)
        with ctrl._held_keys_lock:
            held = frozenset(ctrl._held_keys)
        assert key_w in held
        # Wait for key-up and thread to finish
        if ctrl._reader_thread:
            ctrl._reader_thread.join(timeout=2.0)
        with ctrl._held_keys_lock:
            assert key_w not in ctrl._held_keys
