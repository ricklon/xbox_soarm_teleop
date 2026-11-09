"""Shared control-session setup used by example entry points."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xbox_soarm_teleop.config.modes import ControlMode

_CONTROLLER_LABELS = {
    "dual_joycon": "Dual Joy-Con",
    "joycon": "Joy-Con",
    "keyboard": "keyboard",
    "xbox": "Xbox controller",
}


def controller_label(controller_type: str) -> str:
    """Return a human-readable controller label."""
    return _CONTROLLER_LABELS.get(controller_type, controller_type)


@dataclass
class ControlRuntime:
    """Shared control runtime components for an entry point."""

    control_mode: ControlMode
    controller_type: str
    controller_label: str
    controller_config: Any | None
    controller: Any | None
    processor: Any | None
    mapper: Any | None
    processor_config: Any
    kinematics: Any | None
    jacobian_controller: Any | None
    gripper_rate: float


def _build_controller_configs(
    *,
    controller_type: str,
    controller_backend: str,
    deadzone: float,
    linear_scale: float | None,
    keyboard_grab: bool,
    keyboard_record: str | None,
    keyboard_playback: str | None,
) -> tuple[Any, Any, Any]:
    if controller_type == "joycon":
        from xbox_soarm_teleop.config.joycon_config import JoyConConfig

        config = JoyConConfig(deadzone=deadzone)
        if linear_scale is not None:
            config.linear_scale = linear_scale
        return config, config, "joycon"

    if controller_type == "dual_joycon":
        from xbox_soarm_teleop.config.joycon_config import DualJoyConConfig

        config = DualJoyConConfig(deadzone=deadzone)
        if linear_scale is not None:
            config.linear_scale = linear_scale
            config.vertical_scale = linear_scale
        return config, config, "dual_joycon"

    if controller_type == "keyboard":
        from xbox_soarm_teleop.config.keyboard_config import KeyboardConfig
        from xbox_soarm_teleop.config.xbox_config import XboxConfig

        config = KeyboardConfig(
            backend=controller_backend,
            grab=keyboard_grab,
            record_path=keyboard_record,
            playback_path=keyboard_playback,
        )
        if linear_scale is not None:
            config.speed_levels = tuple(s * linear_scale / 0.1 for s in config.speed_levels)

        processor_config = XboxConfig()
        if linear_scale is not None:
            processor_config.linear_scale = linear_scale
        return config, processor_config, "keyboard"

    from xbox_soarm_teleop.config.xbox_config import XboxConfig

    config = XboxConfig(deadzone=deadzone, backend=controller_backend)
    if linear_scale is not None:
        config.linear_scale = linear_scale
    return config, config, "xbox"


def _resolve_keyboard_backend(controller_backend: str, keyboard_playback: str | None) -> str:
    """Resolve the effective keyboard backend for messaging and config."""
    if keyboard_playback:
        return "playback"
    if controller_backend != "auto":
        return controller_backend
    if sys.platform == "linux":
        return "evdev"
    return "pynput"


def _build_kinematics_stack(
    *,
    control_mode: ControlMode,
    urdf_path: str,
    use_jacobian: bool,
    jacobian_damping: float,
    announce: bool,
) -> tuple[Any | None, Any | None]:
    if control_mode == ControlMode.JOINT:
        return None, None

    from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES
    from xbox_soarm_teleop.kinematics.constrained import RobotKinematics

    urdf_path = _resolve_kinematics_urdf_path(urdf_path)

    if announce:
        print("Loading kinematics model...", flush=True)
    kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="gripper_frame_link",
        joint_names=IK_JOINT_NAMES,
    )

    jacobian_controller = None
    if use_jacobian:
        from xbox_soarm_teleop.kinematics.jacobian import JacobianController

        jacobian_controller = JacobianController(kinematics, damping=jacobian_damping)

    return kinematics, jacobian_controller


def _resolve_kinematics_urdf_path(urdf_path: str) -> str:
    """Prefer the portable sibling URDF when an absolute-mesh variant is passed."""
    path = Path(urdf_path)
    if path.name != "so101_abs.urdf":
        return urdf_path
    portable = path.with_name("so101.urdf")
    if portable.exists():
        return str(portable)
    return urdf_path


def build_control_runtime(
    *,
    controller_type: str,
    controller_backend: str = "auto",
    mode: str,
    deadzone: float,
    linear_scale: float | None,
    keyboard_grab: bool,
    keyboard_record: str | None,
    keyboard_playback: str | None,
    loop_dt: float,
    urdf_path: str,
    use_jacobian: bool = False,
    jacobian_damping: float = 0.05,
    announce_kinematics: bool = False,
    keyboard_focus_target: str = "this window",
    enable_controller: bool = True,
) -> ControlRuntime:
    """Build the shared controller/processor/kinematics stack for an entry point."""
    from xbox_soarm_teleop.processors.factory import make_processor

    control_mode = ControlMode(mode)
    urdf_path = _resolve_kinematics_urdf_path(urdf_path)
    kinematics, jacobian_controller = _build_kinematics_stack(
        control_mode=control_mode,
        urdf_path=urdf_path,
        use_jacobian=use_jacobian,
        jacobian_damping=jacobian_damping,
        announce=announce_kinematics,
    )

    controller_config, processor_config, controller_kind = _build_controller_configs(
        controller_type=controller_type,
        controller_backend=controller_backend,
        deadzone=deadzone,
        linear_scale=linear_scale,
        keyboard_grab=keyboard_grab,
        keyboard_record=keyboard_record,
        keyboard_playback=keyboard_playback,
    )

    controller = None
    processor = None
    if enable_controller:
        if controller_kind == "joycon":
            from xbox_soarm_teleop.teleoperators.joycon import JoyConController

            controller = JoyConController(controller_config)
        elif controller_kind == "dual_joycon":
            from xbox_soarm_teleop.teleoperators.joycon import DualJoyConController

            controller = DualJoyConController(controller_config)
        elif controller_kind == "keyboard":
            from xbox_soarm_teleop.teleoperators.keyboard import KeyboardController

            effective_keyboard_backend = _resolve_keyboard_backend(
                controller_backend, keyboard_playback
            )
            if effective_keyboard_backend == "evdev" and not keyboard_grab and not keyboard_playback:
                print(
                    "WARNING: keyboard controller active without --keyboard-grab. "
                    f"Keypresses will be detected even when {keyboard_focus_target} is not focused. "
                    "Use --keyboard-grab for exclusive access.",
                    flush=True,
                )
            if keyboard_playback:
                print(f"Keyboard playback mode: {keyboard_playback}", flush=True)
            controller = KeyboardController(controller_config)
        else:
            from xbox_soarm_teleop.teleoperators.xbox import XboxController

            controller = XboxController(controller_config)

        processor = make_processor(
            control_mode,
            controller_type=controller_type,
            linear_scale=processor_config.linear_scale,
            vertical_scale=getattr(processor_config, "vertical_scale", None),
            angular_scale=processor_config.angular_scale,
            orientation_scale=processor_config.orientation_scale,
            roll_scale=getattr(processor_config, "roll_scale", 1.0),
            pitch_scale=getattr(processor_config, "pitch_scale", 1.0),
            yaw_scale=getattr(processor_config, "yaw_scale", 0.75),
            invert_z=getattr(processor_config, "invert_z", False),
            invert_roll=getattr(processor_config, "invert_roll", False),
            invert_pitch=processor_config.invert_pitch,
            invert_yaw=processor_config.invert_yaw,
            loop_dt=loop_dt,
            urdf_path=urdf_path,
            multi_joint=(controller_type == "keyboard" and control_mode.value == "joint"),
        )
    return ControlRuntime(
        control_mode=control_mode,
        controller_type=controller_type,
        controller_label=controller_label(controller_type),
        controller_config=controller_config,
        controller=controller,
        processor=processor,
        mapper=processor,
        processor_config=processor_config,
        kinematics=kinematics,
        jacobian_controller=jacobian_controller,
        gripper_rate=float(getattr(processor_config, "gripper_rate", 2.0)),
    )
