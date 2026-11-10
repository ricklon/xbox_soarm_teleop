"""Digital twin service shared by the Xbox controller and AI agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xbox_soarm_teleop.twin.actions import ActionRejected
from xbox_soarm_teleop.twin.service import TwinClosed, TwinService

__all__ = [
    "ActionRejected",
    "TwinClosed",
    "TwinParts",
    "TwinService",
    "build_parts",
    "build_twin",
    "scene_workspace",
]

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


@dataclass
class TwinParts:
    """The heavy, read-only pieces of a twin, which twins of one scene can share.

    Compiling the robot model and loading the IK solver take about 2 s and
    500 MB. A process that hosts many twins (``twin.hub``) builds these once
    and gives every twin the same ones: each twin then adds only its own
    simulator state. Sharing is safe when all the twins are ticked from one
    thread, because the model is never modified and the solver takes its
    whole state as arguments on every call.
    """

    scene: str
    world: str
    runtime: Any
    model: Any
    workspace_bounds: dict


def build_parts(
    *,
    scene: str = "pick_place_basic",
    world: str | None = None,
    controller_backend: str = "auto",
    deadzone: float = 0.15,
) -> TwinParts:
    """Build the shareable pieces of a twin (see :class:`TwinParts`)."""
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
    world = world or sim_cli.DEFAULT_WORLD
    model = sim_cli.load_model_with_cameras(str(sim_cli.URDF_PATH), scene=scene, world=world)
    return TwinParts(
        scene=scene,
        world=world,
        runtime=runtime,
        model=model,
        workspace_bounds=scene_workspace(scene),
    )


_NO_CONTROLLER = object()


def build_twin(
    *,
    scene: str = "pick_place_basic",
    world: str | None = None,
    controller_backend: str = "auto",
    deadzone: float = 0.15,
    with_controller: bool = True,
    log_path: str | Path | None = None,
    parts: TwinParts | None = None,
    controller: Any = _NO_CONTROLLER,
    cameras: Any = None,
) -> TwinService:
    """Create a twin with the same kinematics, mapping and bounds as teleop.

    The controller (if any) is created but not connected; call
    ``service.controller.connect()`` from the loop's process.

    Args:
        scene: Scene name.
        world: Lighting/floor preset.
        controller_backend: Xbox backend (``auto``, ``sdl``, ``inputs``, ``subprocess``).
        deadzone: Controller deadzone.
        with_controller: Read the Xbox controller. Ignored when ``controller`` is given.
        log_path: Record the session as JSONL here (``twin.session_log``).
        parts: Shared pieces from :func:`build_parts`; built here when None
            (then ``scene``, ``world``, ``controller_backend`` and ``deadzone`` apply).
        controller: Use this controller object (anything with ``read()`` and
            ``is_connected``) in place of the Xbox pad; None for no controller.
        cameras: A camera rig shared between twins of the same model.
    """
    from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
    from xbox_soarm_teleop.control.safety import make_workspace_clipper

    if parts is None:
        parts = build_parts(
            scene=scene, world=world, controller_backend=controller_backend, deadzone=deadzone
        )
    runtime = parts.runtime
    sim = sim_cli.MuJoCoSimulator(
        str(sim_cli.URDF_PATH), scene=parts.scene, world=parts.world, model=parts.model
    )
    bounds = parts.workspace_bounds
    if controller is _NO_CONTROLLER:
        controller = runtime.controller if with_controller else None
    log = None
    if log_path is not None:
        from xbox_soarm_teleop.twin import service as twin_service
        from xbox_soarm_teleop.twin.session_log import SessionLog, session_header

        header = session_header(
            scene=parts.scene,
            world=sim.world,
            urdf_path=sim_cli.URDF_PATH,
            limits={
                "workspace": bounds,
                "dt_s": sim_cli.LOOP_PERIOD,
                "gripper_rate_per_s": runtime.gripper_rate,
                "lease_timeout_s": twin_service.LEASE_TIMEOUT_S,
                "human_idle_s": twin_service.HUMAN_IDLE_S,
            },
        )
        log = SessionLog(log_path, header, dt=sim_cli.LOOP_PERIOD)
    return TwinService(
        sim,
        runtime.kinematics,
        mapper=runtime.mapper,
        workspace_bounds=bounds,
        clip_position=make_workspace_clipper(bounds),
        table_top_z=table_top_z(parts.scene),
        gripper_rate=runtime.gripper_rate,
        dt=sim_cli.LOOP_PERIOD,
        controller=controller,
        log=log,
        cameras=cameras,
    )
