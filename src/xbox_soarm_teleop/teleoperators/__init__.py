"""Teleoperator devices for SO-ARM teleoperation."""

from xbox_soarm_teleop.teleoperators.xbox import XboxController, XboxState

__all__ = [
    "JoyConController",
    "KeyboardController",
    "XboxController",
    "XboxState",
    "XboxTeleopConfig",
    "XboxTeleoperator",
]


def __getattr__(name: str):
    if name == "JoyConController":
        from xbox_soarm_teleop.teleoperators.joycon import JoyConController

        return JoyConController
    if name == "KeyboardController":
        from xbox_soarm_teleop.teleoperators.keyboard import KeyboardController

        return KeyboardController
    if name == "XboxTeleopConfig":
        from xbox_soarm_teleop.teleoperators.config_xbox_teleop import XboxTeleopConfig

        return XboxTeleopConfig
    if name == "XboxTeleoperator":
        from xbox_soarm_teleop.teleoperators.xbox_teleop import XboxTeleoperator

        return XboxTeleoperator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
