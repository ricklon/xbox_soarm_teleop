"""Small real-engine compatibility checks, runnable without LeRobot imports."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from xbox_soarm_teleop.cli.simulate_mujoco import PICK_PLACE_CUBE_START, MuJoCoSimulator

URDF = str(Path(__file__).resolve().parents[1] / "assets/so101.urdf")


@pytest.mark.parametrize("scene", ["default", "stack", "pick_place_basic"])
def test_scene_load_step_reset(scene):
    sim = MuJoCoSimulator(URDF, scene=scene)
    joints = np.array([10.0, 30.0, -45.0, 20.0, 35.0, 0.0])
    sim.set_joint_targets(joints)
    mujoco.mj_forward(sim.model, sim.data)
    np.testing.assert_allclose(sim.get_joint_positions_deg(), joints, atol=1e-10)
    assert np.all(np.isfinite(sim.get_ee_position()))
    start_time = sim.data.time
    for _ in range(100):
        sim.set_joint_targets(joints)
        sim.step()
    assert np.all(np.isfinite(sim.data.qpos))
    assert np.all(np.isfinite(sim.data.qvel))
    if scene == "default":
        # This path is kinematic FK, not a dynamic actuator simulation.
        assert sim.data.time == start_time
        sim.go_home()
        np.testing.assert_allclose(sim.get_joint_positions_deg(), 0.0, atol=1e-10)
    else:
        assert sim.data.time > start_time
        if scene == "stack":
            sim.reset_stack_scene()
            assert max(sim.get_stack_cube_displacements().values()) < 1e-6
        else:
            sim.reset_pick_place_scene()
            np.testing.assert_allclose(sim.get_pick_cube_position(), PICK_PLACE_CUBE_START, atol=1e-6)
        assert np.all(np.isfinite(sim.data.qpos))


def _robot_body_ids(model):
    robot = {"shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link"}
    robot |= {"moving_jaw_so101_v1_link"}
    return {
        b for b in range(model.nbody) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) in robot
    }


@pytest.mark.parametrize("scene", ["stack", "pick_place_basic"])
def test_dynamic_scene_arm_tracks_targets_without_base_collision(scene):
    """The base meshes must not wedge the shoulder in physics scenes.

    The URDF base is fused into the world body, which MuJoCo does not filter
    against its child, so the base hulls used to interpenetrate the shoulder
    and hold every joint off target.
    """
    sim = MuJoCoSimulator(URDF, scene=scene)
    joints = np.array([10.0, 30.0, -45.0, 20.0, 35.0, 0.0])
    for _ in range(200):
        sim.set_joint_targets(joints)
        sim.step()
    np.testing.assert_allclose(sim.get_joint_positions_deg(), joints, atol=0.25)
    robot = _robot_body_ids(sim.model)
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        bodies = {sim.model.geom_bodyid[c.geom1], sim.model.geom_bodyid[c.geom2]}
        assert not (bodies & robot and c.dist < -1e-3), (scene, bodies, c.dist)


def test_stack_reset_returns_arm_home():
    sim = MuJoCoSimulator(URDF, scene="stack")
    joints = np.array([10.0, 30.0, -45.0, 20.0, 35.0, 0.0])
    for _ in range(100):
        sim.set_joint_targets(joints)
        sim.step()
    sim.reset_stack_scene()
    np.testing.assert_allclose(sim.get_joint_positions_deg(), 0.0, atol=0.25)
