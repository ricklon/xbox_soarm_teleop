"""Control mode enumeration for Xbox SO-ARM teleoperation."""

from __future__ import annotations

from enum import Enum


class ControlMode(Enum):
    """Teleoperation control mode.

    Attributes:
        CARTESIAN: End-effector Cartesian space control via IK (default).
        JOINT: Direct per-joint velocity control, bypasses IK.
        CRANE: Crane-style decoupled cylindrical control (Phase 2).
        PUPPET: Crane geometry with Joy-Con IMU wrist orientation (Phase 3).
    """

    CARTESIAN = "cartesian"
    JOINT = "joint"
    CRANE = "crane"
    PUPPET = "puppet"
