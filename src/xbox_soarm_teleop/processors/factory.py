"""Factory for creating the appropriate processor given a ControlMode."""

from __future__ import annotations

from xbox_soarm_teleop.config.modes import ControlMode


def make_processor(
    mode: ControlMode,
    linear_scale: float = 0.1,
    angular_scale: float = 0.5,
    orientation_scale: float = 1.0,
    invert_pitch: bool = False,
    invert_yaw: bool = False,
    joint_max_vel_deg_s: float = 70.0,
    loop_dt: float = 1.0 / 30.0,
    urdf_path: str | None = None,
    multi_joint: bool = False,
):
    """Create the processor for the given control mode.

    Args:
        mode: The desired ControlMode.
        linear_scale: m/s at full stick deflection (CARTESIAN/CRANE modes).
        angular_scale: rad/s at full stick for wrist roll (CARTESIAN/CRANE modes).
        orientation_scale: rad/s for pitch/yaw D-pad (CARTESIAN/CRANE modes).
        invert_pitch: Invert pitch axis (CARTESIAN/CRANE modes).
        invert_yaw: Invert yaw axis (CARTESIAN/CRANE modes).
        joint_max_vel_deg_s: Max joint velocity in deg/s (JOINT mode).
        loop_dt: Control loop period in seconds (JOINT/CRANE modes).
        urdf_path: Path to robot URDF (CRANE mode — enables the 2-DOF planar IK).
        multi_joint: When True, drive all joints simultaneously from separate axes
            (JOINT mode only — intended for keyboard control).

    Returns:
        A processor callable for the requested mode.

    Raises:
        ValueError: If mode is not a recognised ControlMode.
    """
    if mode == ControlMode.CARTESIAN:
        from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta

        return MapXboxToEEDelta(
            linear_scale=linear_scale,
            angular_scale=angular_scale,
            orientation_scale=orientation_scale,
            invert_pitch=invert_pitch,
            invert_yaw=invert_yaw,
        )

    if mode == ControlMode.JOINT:
        from xbox_soarm_teleop.processors.joint_direct import JointDirectProcessor

        return JointDirectProcessor(
            max_vel_deg_s=joint_max_vel_deg_s,
            dt=loop_dt,
            multi_joint=multi_joint,
        )

    if mode == ControlMode.CRANE:
        from xbox_soarm_teleop.processors.crane import CraneProcessor

        return CraneProcessor(urdf_path=urdf_path, loop_dt=loop_dt)

    raise ValueError(f"Unknown ControlMode: {mode!r}")
