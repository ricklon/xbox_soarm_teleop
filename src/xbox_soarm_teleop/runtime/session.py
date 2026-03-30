"""Shared control-session setup used by example entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xbox_soarm_teleop.config.modes import ControlMode

_CONTROLLER_LABELS = {
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

    if controller_type == "keyboard":
        from xbox_soarm_teleop.config.keyboard_config import KeyboardConfig
        from xbox_soarm_teleop.config.xbox_config import XboxConfig

        config = KeyboardConfig(
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

    config = XboxConfig(deadzone=deadzone)
    if linear_scale is not None:
        config.linear_scale = linear_scale
    return config, config, "xbox"


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

    from lerobot.model.kinematics import RobotKinematics

    from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES

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


def build_control_runtime(
    *,
    controller_type: str,
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
    kinematics, jacobian_controller = _build_kinematics_stack(
        control_mode=control_mode,
        urdf_path=urdf_path,
        use_jacobian=use_jacobian,
        jacobian_damping=jacobian_damping,
        announce=announce_kinematics,
    )

    controller_config, processor_config, controller_kind = _build_controller_configs(
        controller_type=controller_type,
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
        elif controller_kind == "keyboard":
            from xbox_soarm_teleop.teleoperators.keyboard import KeyboardController

            if not keyboard_grab and not keyboard_playback:
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
            linear_scale=processor_config.linear_scale,
            angular_scale=processor_config.angular_scale,
            orientation_scale=processor_config.orientation_scale,
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
