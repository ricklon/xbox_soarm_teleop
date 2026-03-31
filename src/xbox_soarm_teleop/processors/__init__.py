"""Processors for transforming controller input to robot commands."""

from xbox_soarm_teleop.processors.crane import CraneProcessor
from xbox_soarm_teleop.processors.factory import make_processor
from xbox_soarm_teleop.processors.joint_direct import JointCommand, JointDirectProcessor
from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta, MapDualJoyConToEEDelta, MapXboxToEEDelta

__all__ = [
    "CraneProcessor",
    "EEDelta",
    "JointCommand",
    "JointDirectProcessor",
    "MapDualJoyConToEEDelta",
    "MapXboxToEEDelta",
    "make_processor",
]
