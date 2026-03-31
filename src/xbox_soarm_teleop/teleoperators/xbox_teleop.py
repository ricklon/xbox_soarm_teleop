"""LeRobot-compatible teleoperator backed by an Xbox controller.

Implements the LeRobot Teleoperator duck-type interface so that
XboxTeleoperator can be used with LeRobot's recording and evaluation
pipelines.  It does **not** inherit from ``lerobot.teleoperators.Teleoperator``
directly because lerobot has a circular import that prevents that base class
from being loaded cleanly in test / standalone contexts.  All required methods
and properties are implemented, so the object is fully compatible.

Supported control modes for recording:
- ``"joint"``  — direct per-joint velocity from left stick; produces absolute
  goal-position actions that can be replayed on the robot.
- ``"crane"``  — cylindrical decoupled control (pan / reach / height / wrist);
  also produces absolute goal-position actions.

Cartesian mode (``"cartesian"``) is **not** supported in :pymeth:`get_action`
because it produces EE-delta commands that require downstream IK integration;
use the standalone ``teleoperate_real.py`` example for that workflow instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xbox_soarm_teleop.config.joints import JOINT_NAMES_WITH_GRIPPER
from xbox_soarm_teleop.config.modes import ControlMode
from xbox_soarm_teleop.config.xbox_config import XboxConfig
from xbox_soarm_teleop.processors.factory import make_processor
from xbox_soarm_teleop.teleoperators.config_xbox_teleop import XboxTeleopConfig
from xbox_soarm_teleop.teleoperators.xbox import XboxController

# LeRobot calibration constants (used only when lerobot is available)
_HF_LEROBOT_CALIBRATION: Path = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
_TELEOPERATORS_DIR = "teleoperators"


class XboxTeleoperator:
    """LeRobot-compatible teleoperator driven by an Xbox controller.

    Wraps :class:`XboxController` and one of the existing processor classes
    (:class:`JointDirectProcessor` or :class:`CraneProcessor`) to implement
    the full Teleoperator interface (duck-type compatible with
    ``lerobot.teleoperators.Teleoperator``).

    Actions are returned as absolute joint-position goals in **degrees**, keyed
    as ``"<motor>.pos"`` to match the convention used by SOLeader.

    Args:
        config: Teleoperator configuration.
    """

    name = "xbox"

    def __init__(self, config: XboxTeleopConfig) -> None:
        self.config: XboxTeleopConfig = config
        self.id = config.id

        # Calibration bookkeeping (no-op for Xbox, kept for interface parity)
        _cal_root = (
            config.calibration_dir
            if config.calibration_dir
            else _HF_LEROBOT_CALIBRATION / _TELEOPERATORS_DIR / self.name
        )
        self.calibration_dir = Path(_cal_root)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict = {}

        self._mode = ControlMode(config.mode)
        if self._mode == ControlMode.CARTESIAN:
            raise ValueError(
                "XboxTeleoperator does not support 'cartesian' mode for recording. "
                "Use 'joint' or 'crane' mode instead, or use teleoperate_real.py for "
                "cartesian control without recording."
            )

        if config.controller_type == "joycon":
            from xbox_soarm_teleop.config.joycon_config import JoyConConfig
            from xbox_soarm_teleop.teleoperators.joycon import JoyConController

            ctrl_cfg = JoyConConfig(deadzone=config.deadzone)
            self._controller = JoyConController(ctrl_cfg)
        elif config.controller_type == "dual_joycon":
            from xbox_soarm_teleop.config.joycon_config import DualJoyConConfig
            from xbox_soarm_teleop.teleoperators.joycon import DualJoyConController

            ctrl_cfg = DualJoyConConfig(deadzone=config.deadzone)
            self._controller = DualJoyConController(ctrl_cfg)
        elif config.controller_type == "keyboard":
            from xbox_soarm_teleop.config.keyboard_config import KeyboardConfig
            from xbox_soarm_teleop.teleoperators.keyboard import KeyboardController

            ctrl_cfg = KeyboardConfig()
            self._controller = KeyboardController(ctrl_cfg)
        else:
            xbox_cfg = XboxConfig(device_index=config.device_index, deadzone=config.deadzone)
            self._controller = XboxController(xbox_cfg)
        self._processor = make_processor(
            self._mode,
            controller_type=config.controller_type,
            loop_dt=config.loop_dt,
            urdf_path=config.urdf_path,
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "XboxTeleoperator":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()

    def __del__(self) -> None:
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Interface properties
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        """Absolute joint-position goals in degrees, one per motor."""
        return {f"{motor}.pos": float for motor in JOINT_NAMES_WITH_GRIPPER}

    @property
    def feedback_features(self) -> dict:
        """No feedback channel for an Xbox controller."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._controller.is_connected

    @property
    def is_calibrated(self) -> bool:
        """Always True — Xbox controllers need no calibration."""
        return True

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the Xbox controller and reset the processor.

        Args:
            calibrate: Ignored (no calibration required).

        Raises:
            RuntimeError: If no Xbox controller is found.
        """
        if not self._controller.connect():
            raise RuntimeError(
                "Xbox controller not found. "
                "Ensure it is plugged in and you have input device permissions."
            )
        self.configure()

    def calibrate(self) -> None:
        """No-op — Xbox controllers need no calibration."""

    def configure(self) -> None:
        """Reset the processor to the home position."""
        self._processor.reset()

    def get_action(self) -> dict[str, float]:
        """Read the controller and return current joint-position goals.

        Returns:
            Flat dict ``{"<motor>.pos": degrees, ...}`` for all six joints.

        Raises:
            RuntimeError: If the controller is not connected.
        """
        if not self._controller.is_connected:
            raise RuntimeError("XboxTeleoperator is not connected. Call connect() first.")

        state = self._controller.read()
        cmd = self._processor(state)
        return {f"{motor}.pos": cmd.goals_deg[motor] for motor in JOINT_NAMES_WITH_GRIPPER}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """No-op — Xbox controllers have no force-feedback API exposed here."""

    def disconnect(self) -> None:
        """Disconnect from the controller."""
        self._controller.disconnect()
