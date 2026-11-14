"""World presets: lighting, sky and floor around the robot and task scenes."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from xbox_soarm_teleop.cli import simulate_mujoco as sim_cli
from xbox_soarm_teleop.sim.world import DEFAULT_WORLD, WORLD_NAMES

URDF = str(Path(__file__).resolve().parents[1] / "assets/so101.urdf")
SCENES = ["default", "stack", "pick_place_basic"]


def _planes(model: mujoco.MjModel) -> int:
    return int(np.sum(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))


def test_world_names_and_default():
    assert set(WORLD_NAMES) == {"studio", "bright", "dark", "plain"}
    assert DEFAULT_WORLD == "studio"


@pytest.mark.parametrize("world", ["studio", "bright", "dark", "plain"])
@pytest.mark.parametrize("scene", SCENES)
def test_every_world_loads_and_steps_with_every_scene(world, scene):
    sim = sim_cli.MuJoCoSimulator(URDF, scene=scene, world=world)
    joints = np.array([10.0, 30.0, -45.0, 20.0, 35.0, 0.0])
    for _ in range(20):
        sim.set_joint_targets(joints)
        sim.step()
    assert np.all(np.isfinite(sim.data.qpos))
    # Exactly one ground plane: the world's or the scene's, never both (z-fighting).
    expected = 0 if (world == "plain" and scene == "default") else 1
    assert _planes(sim.model) == expected


def test_plain_default_matches_bare_robot_model():
    bare = mujoco.MjModel.from_xml_path(URDF)
    plain = sim_cli.load_model_with_cameras(URDF, scene="default", world="plain")
    # The robot itself, plus the invisible boxes the jaws collide as (not their hulls).
    added = sum(
        1
        for g in range(plain.ngeom)
        if (mujoco.mj_id2name(plain, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_collision")
    )
    assert added > 0
    assert plain.ngeom - added == bare.ngeom
    assert plain.nlight == bare.nlight
    assert plain.nbody == bare.nbody


def test_studio_has_shadowed_key_light_floor_and_sky():
    model = sim_cli.load_model_with_cameras(URDF, world="studio")
    assert model.nlight >= 2
    assert np.any(model.light_castshadow)
    assert _planes(model) == 1
    assert np.any(model.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX)


def test_bright_has_no_shadows():
    model = sim_cli.load_model_with_cameras(URDF, world="bright")
    assert model.nlight >= 2
    assert not np.any(model.light_castshadow)


def test_dark_is_a_single_shadowed_spot_with_dim_headlight():
    model = sim_cli.load_model_with_cameras(URDF, world="dark")
    assert model.nlight == 1
    assert model.light_type[0] == mujoco.mjtLightType.mjLIGHT_SPOT
    assert model.light_castshadow[0]
    assert np.max(model.vis.headlight.diffuse) <= 0.1


def test_unknown_world_is_rejected():
    with pytest.raises(ValueError, match="Unknown world"):
        sim_cli.load_model_with_cameras(URDF, world="disco")


def test_cli_world_flag_defaults_to_studio_and_is_forwarded(monkeypatch):
    assert sim_cli.build_parser().parse_args([]).world == "studio"
    created = []

    class FakeSim:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(sys, "argv", ["simulate-mujoco", "--headless", "--world", "dark"])
    monkeypatch.setattr(sim_cli, "MuJoCoSimulator", FakeSim)
    monkeypatch.setattr(sim_cli, "run_with_controller", lambda *a, **kw: 0)
    with pytest.raises(SystemExit):
        sim_cli.main()
    assert created[0]["world"] == "dark"
