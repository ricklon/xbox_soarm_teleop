"""Digital twin service shared by the Xbox controller and AI agents."""

from __future__ import annotations

from xbox_soarm_teleop.twin.actions import ActionRejected
from xbox_soarm_teleop.twin.service import TwinClosed, TwinService

__all__ = ["ActionRejected", "TwinClosed", "TwinService", "build_twin", "scene_workspace"]

# The tool may come no closer than this to a table top (m).
TABLE_CLEARANCE_M = 0.005


def table_top_z(scene: str) -> float | None:
    """Height of the scene's table top, or None when objects rest on the ground."""
    from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli

    return sim_cli.PICK_PLACE_TABLE_HEIGHT if scene == "pick_place_basic" else None


def scene_workspace(scene: str) -> dict:
    """Configured workspace bounds, with the floor raised above the scene's table.

    The arm is posed kinematically, so nothing physically stops it passing
    through the table; the bound does, for agents and the controller alike.
    """
    from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli

    bounds = dict(sim_cli.SIM_WORKSPACE_BOUNDS)
    top = table_top_z(scene)
    if top is not None:
        low, high = bounds["z"]
        bounds["z"] = (max(low, top + TABLE_CLEARANCE_M), high)
    return bounds


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
    from xbox_soarm_teleop.control.safety import make_workspace_clipper
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
    bounds = scene_workspace(scene)
    return TwinService(
        sim,
        runtime.kinematics,
        mapper=runtime.mapper,
        workspace_bounds=bounds,
        clip_position=make_workspace_clipper(bounds),
        table_top_z=table_top_z(scene),
        gripper_rate=runtime.gripper_rate,
        dt=sim_cli.LOOP_PERIOD,
        controller=runtime.controller if with_controller else None,
    )
