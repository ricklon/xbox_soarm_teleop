"""Digital twin service shared by the Xbox controller and AI agents."""

from __future__ import annotations

from xbox_soarm_teleop.twin.actions import ActionRejected
from xbox_soarm_teleop.twin.service import TwinClosed, TwinService

__all__ = ["ActionRejected", "TwinClosed", "TwinService", "build_twin"]


def build_twin(
    *,
    scene: str = "pick_place_basic",
    world: str | None = None,
    controller_backend: str = "auto",
    deadzone: float = 0.15,
    with_controller: bool = True,
) -> TwinService:
    """Create a twin with the same kinematics, mapping and bounds as teleop.

    The controller (if any) is created but not connected; call
    ``service.controller.connect()`` from the loop's process.
    """
    from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
    from xbox_soarm_teleop.runtime import build_control_runtime

    runtime = build_control_runtime(
        controller_type="xbox",
        controller_backend=controller_backend,
        mode="cartesian",
        deadzone=deadzone,
        linear_scale=None,
        keyboard_grab=False,
        keyboard_record=None,
        keyboard_playback=None,
        loop_dt=sim_cli.LOOP_PERIOD,
        urdf_path=str(sim_cli.URDF_PATH),
    )
    kwargs = {"world": world} if world is not None else {}
    sim = sim_cli.MuJoCoSimulator(str(sim_cli.URDF_PATH), scene=scene, **kwargs)
    return TwinService(
        sim,
        runtime.kinematics,
        mapper=runtime.mapper,
        workspace_bounds=sim_cli.SIM_WORKSPACE_BOUNDS,
        clip_position=sim_cli._clip_to_sim_workspace,
        gripper_rate=runtime.gripper_rate,
        dt=sim_cli.LOOP_PERIOD,
        controller=runtime.controller if with_controller else None,
    )
