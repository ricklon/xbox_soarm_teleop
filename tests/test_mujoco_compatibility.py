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
