"""Tests for XboxTeleopConfig and XboxTeleoperator."""

from __future__ import annotations

import pytest

from xbox_soarm_teleop.config.joints import JOINT_NAMES_WITH_GRIPPER

# ---------------------------------------------------------------------------
# Helpers — avoid importing lerobot at collection time if not needed
# ---------------------------------------------------------------------------


def _make_config(**kwargs):
    from xbox_soarm_teleop.teleoperators.config_xbox_teleop import XboxTeleopConfig

    return XboxTeleopConfig(id="test_xbox", **kwargs)


def _make_teleop(**kwargs):
    from xbox_soarm_teleop.teleoperators.xbox_teleop import XboxTeleoperator

    cfg = _make_config(**kwargs)
    return XboxTeleoperator(cfg)


# ---------------------------------------------------------------------------
# XboxTeleopConfig
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = _make_config()
    assert cfg.mode == "crane"
    assert cfg.deadzone == 0.15
    assert cfg.loop_dt == pytest.approx(1.0 / 30.0)
    assert cfg.device_index == 0
    assert cfg.urdf_path is None


def test_config_joint_mode():
    cfg = _make_config(mode="joint")
    assert cfg.mode == "joint"


def test_config_cartesian_raises_on_teleop_init():
    """CARTESIAN mode should be rejected at __init__ time."""
    with pytest.raises(ValueError, match="cartesian"):
        _make_teleop(mode="cartesian")


# ---------------------------------------------------------------------------
# XboxTeleoperator — static properties (no hardware required)
# ---------------------------------------------------------------------------


def test_action_features_keys():
    teleop = _make_teleop(mode="joint")
    feats = teleop.action_features
    expected = {f"{m}.pos" for m in JOINT_NAMES_WITH_GRIPPER}
    assert set(feats.keys()) == expected


def test_action_features_types_are_float():
    teleop = _make_teleop(mode="joint")
    for v in teleop.action_features.values():
        assert v is float


def test_feedback_features_empty():
    teleop = _make_teleop(mode="crane")
    assert teleop.feedback_features == {}


def test_is_calibrated_always_true():
    teleop = _make_teleop(mode="joint")
    assert teleop.is_calibrated is True


def test_is_connected_false_before_connect():
    teleop = _make_teleop(mode="joint")
    assert not teleop.is_connected


# ---------------------------------------------------------------------------
# XboxTeleoperator — connect / get_action with mocked controller
# ---------------------------------------------------------------------------


class _FakeController:
    """Minimal stand-in for XboxController."""

    def __init__(self):
        self._connected = False
        self._state = None

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read(self):
        from xbox_soarm_teleop.teleoperators.xbox import XboxState

        return self._state or XboxState()


def _teleop_with_fake_controller(mode: str = "joint"):
    teleop = _make_teleop(mode=mode)
    teleop._controller = _FakeController()
    return teleop


def test_connect_sets_is_connected():
    teleop = _teleop_with_fake_controller()
    assert not teleop.is_connected
    teleop.connect()
    assert teleop.is_connected


def test_disconnect_clears_is_connected():
    teleop = _teleop_with_fake_controller()
    teleop.connect()
    teleop.disconnect()
    assert not teleop.is_connected


def test_get_action_not_connected_raises():
    teleop = _teleop_with_fake_controller()
    with pytest.raises(RuntimeError, match="not connected"):
        teleop.get_action()


def test_get_action_joint_mode_returns_correct_keys():
    teleop = _teleop_with_fake_controller(mode="joint")
    teleop.connect()
    action = teleop.get_action()
    expected = {f"{m}.pos" for m in JOINT_NAMES_WITH_GRIPPER}
    assert set(action.keys()) == expected


def test_get_action_crane_mode_returns_correct_keys():
    teleop = _teleop_with_fake_controller(mode="crane")
    teleop.connect()
    action = teleop.get_action()
    expected = {f"{m}.pos" for m in JOINT_NAMES_WITH_GRIPPER}
    assert set(action.keys()) == expected


def test_get_action_values_are_floats():
    teleop = _teleop_with_fake_controller(mode="joint")
    teleop.connect()
    action = teleop.get_action()
    for v in action.values():
        assert isinstance(v, float)


def test_send_feedback_is_noop():
    teleop = _teleop_with_fake_controller()
    teleop.connect()
    # Should not raise
    teleop.send_feedback({"shoulder_pan.pos": 10.0})


def test_calibrate_is_noop():
    teleop = _teleop_with_fake_controller()
    teleop.calibrate()  # must not raise


def test_configure_resets_processor():
    """configure() should reset the processor to home without error."""
    teleop = _teleop_with_fake_controller(mode="joint")
    teleop._processor._selected_idx = 3
    teleop.configure()
    assert teleop._processor._selected_idx == 0


def test_context_manager():
    teleop = _teleop_with_fake_controller()
    with teleop:
        assert teleop.is_connected
    assert not teleop.is_connected
