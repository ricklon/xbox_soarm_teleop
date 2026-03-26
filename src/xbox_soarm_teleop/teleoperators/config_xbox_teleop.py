"""Configuration for XboxTeleoperator (LeRobot-compatible interface)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class XboxTeleopConfig:
    """Configuration for XboxTeleoperator.

    Implements the same fields as LeRobot's TeleoperatorConfig so it can be
    used as a drop-in when the lerobot circular-import issue is resolved.

    Attributes:
        id: Optional identifier when multiple Xbox controllers are used.
        calibration_dir: Directory for calibration files (unused — Xbox controllers
            need no calibration, but kept for interface parity).
        mode: Control mode — "joint", "crane", or "cartesian".
            For dataset recording, prefer "joint" or "crane" as they produce
            absolute joint-position actions suitable for robot replay.
        urdf_path: Path to robot URDF (required for crane IK; optional for joint).
        deadzone: Radial deadzone applied to both sticks (0–1).
        loop_dt: Control-loop period in seconds (default 1/30 s).
        device_index: Which gamepad device to use when multiple are connected.
    """

    id: str | None = None
    calibration_dir: Path | None = None
    mode: str = "crane"
    urdf_path: str | None = None
    deadzone: float = 0.15
    loop_dt: float = field(default_factory=lambda: 1.0 / 30.0)
    device_index: int = 0

    @property
    def type(self) -> str:
        return "xbox"
