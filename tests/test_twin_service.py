"""Twin service: agent actions, virtual stop, human override and ownership.

Covers the design's milestone-1 exercise (docs/mcp_control_interface.md §14)
headlessly, driving ``tick()`` by hand: no threads, deterministic time.
"""

import logging
import warnings

import numpy as np
import pytest

from xbox_soarm_teleop.schemas.agent import ActionState, ReasonCode
from xbox_soarm_teleop.teleoperators.xbox import XboxState
from xbox_soarm_teleop.twin import ActionRejected, TwinClosed, build_twin
from xbox_soarm_teleop.twin.grasp import GraspSettings
from xbox_soarm_teleop.twin.service import HUMAN_IDLE_S, TwinService

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

TABLE_TOP = 0.12
CUBE_REST_Z = TABLE_TOP + 0.02
CUBE_FIT = GraspSettings().gripper_for_opening(0.04)  # jaws closed to the cube's width


@pytest.fixture
def twin() -> TwinService:
    service = build_twin(scene="pick_place_basic", with_controller=False)
    service.tick()
    return service


def resolve(twin: TwinService, future, human=None):
    twin.tick(human)
    return future.result(timeout=0)


def rejected(twin: TwinService, future, human=None) -> ActionRejected:
    twin.tick(human)
    with pytest.raises(ActionRejected) as info:
        future.result(timeout=0)
    return info.value


def run_until_done(twin: TwinService, action_id: str, max_ticks: int = 1500, human=None):
    for _ in range(max_ticks):
        status = twin.wait(action_id, 0)
        if status.state.terminal:
            return status
        twin.tick(human)
    return twin.wait(action_id, 0)


def move(twin, xyz, *, client_id="agent", preempt=False, **payload):
    request = {"type": "move_ee_to", "params": {"xyz_m": [float(v) for v in xyz]}, **payload}
    return twin.submit(request, client_id=client_id, preempt=preempt)


def lb(**fields) -> XboxState:
    state = XboxState(left_bumper=True)
    for key, value in fields.items():
        setattr(state, key, value)
    return state


def test_observation_reports_enforced_workspace(twin):
    obs = twin.observe()
    assert obs["schema"] == "soarm.observation/v1"
    assert obs["workspace"] == {k: list(v) for k, v in twin.workspace_bounds.items()}
    assert obs["scene"]["objects"]["pick_cube"]["attached"] is False
    assert obs["control"]["stop_latched_by"] is None


def test_milestone_move_stop_reset_resume(twin):
    # 2. reachable move succeeds within tolerance
    target = np.array([0.30, 0.05, 0.22])
    status = resolve(twin, move(twin, target))
    assert status.state is ActionState.EXECUTING
    final = run_until_done(twin, status.id)
    assert final.state is ActionState.SUCCEEDED, final
    assert np.linalg.norm(twin.tool_xyz() - target) < 0.003

    # 3. below the floor is rejected at submission
    error = rejected(twin, move(twin, [0.30, 0.0, 0.0]))
    assert error.reason is ReasonCode.WORKSPACE_BOUND and error.detail.startswith("z_min")

    # 4. stop mid-motion -> STOPPED, and the tool holds
    status = resolve(twin, move(twin, [0.40, -0.10, 0.15]))
    for _ in range(10):
        twin.tick()
    twin.stop(by="agent")
    twin.tick()
    assert twin.wait(status.id, 0).state is ActionState.STOPPED
    held = twin.tool_xyz()
    for _ in range(50):  # 1 s
        twin.tick()
    assert np.linalg.norm(twin.tool_xyz() - held) < 0.001

    # 5. motion is rejected while latched; 6. observation shows it
    assert rejected(twin, move(twin, [0.3, 0.0, 0.2])).reason is ReasonCode.STOPPED
    assert twin.observe()["stop_latched"] is True
    assert rejected(twin, twin.reset_stop(confirm=False)).reason is ReasonCode.INVALID_REQUEST
    assert resolve(twin, twin.reset_stop(confirm=True)) is True

    # 7. resume
    status = resolve(twin, move(twin, [0.30, 0.0, 0.20]))
    assert run_until_done(twin, status.id).state is ActionState.SUCCEEDED


def test_stop_beats_a_queued_move(twin):
    future = move(twin, [0.3, 0.0, 0.2])
    twin.stop()
    assert rejected(twin, future).reason is ReasonCode.STOPPED


def test_controller_stop_only_released_by_controller(twin):
    twin.tick(XboxState(b_button=True, b_button_pressed=True))
    assert twin.observe()["control"]["stop_latched_by"] == "human"
    error = rejected(twin, twin.reset_stop(confirm=True))
    assert error.reason is ReasonCode.NOT_OWNER and "Start" in error.detail
    twin.tick(XboxState(start_button=True, start_button_pressed=True))
    assert twin.observe()["stop_latched"] is False


def test_human_preempts_agent_and_blocks_until_idle(twin):
    status = resolve(twin, move(twin, [0.25, 0.10, 0.25]))
    twin.tick(lb())
    final = twin.wait(status.id, 0)
    assert final.state is ActionState.CANCELLED
    assert (final.reason, final.detail) == (ReasonCode.PREEMPTED, "human")
    assert twin.observe()["control"]["human_in_control"] is True

    error = rejected(twin, move(twin, [0.3, 0.0, 0.2]), human=XboxState())
    assert (error.reason, error.detail) == (ReasonCode.NOT_OWNER, "human")
    for _ in range(int(HUMAN_IDLE_S / twin.dt) + 1):
        twin.tick(XboxState())
    status = resolve(twin, move(twin, [0.3, 0.0, 0.2]), human=XboxState())
    assert status.state is ActionState.EXECUTING


def test_busy_and_preempt(twin):
    first = resolve(twin, move(twin, [0.25, 0.10, 0.25]))
    assert rejected(twin, move(twin, [0.3, 0.0, 0.2])).reason is ReasonCode.BUSY
    second = resolve(twin, move(twin, [0.3, 0.0, 0.2], preempt=True))
    old = twin.wait(first.id, 0)
    assert (old.state, old.reason) == (ActionState.CANCELLED, ReasonCode.PREEMPTED)
    assert run_until_done(twin, second.id).state is ActionState.SUCCEEDED


def test_ownership_handover(twin):
    a = resolve(twin, move(twin, [0.25, 0.10, 0.25], client_id="claude-a"))
    error = rejected(twin, move(twin, [0.3, 0.0, 0.2], client_id="claude-b"))
    assert (error.reason, error.detail) == (ReasonCode.NOT_OWNER, "claude-a")
    run_until_done(twin, a.id)
    # Owner idle (no running action): a restarted agent can take over at once.
    b = resolve(twin, move(twin, [0.3, 0.0, 0.2], client_id="claude-b"))
    assert b.state is ActionState.EXECUTING and twin.observe()["owner"] == "claude-b"


def test_timeout_and_unreachable_fail(twin):
    status = resolve(twin, move(twin, [0.45, -0.25, 0.15], timeout_s=0.2))
    final = run_until_done(twin, status.id)
    assert (final.state, final.reason) == (ActionState.FAILED, ReasonCode.TIMEOUT)

    status = resolve(twin, move(twin, [0.5, 0.3, 0.45]))
    final = run_until_done(twin, status.id)
    assert final.state is ActionState.FAILED
    assert final.reason in {ReasonCode.OUT_OF_REACH, ReasonCode.JOINT_LIMIT, ReasonCode.WORKSPACE_BOUND}


def test_wrist_roll_gripper_home_hold(twin):
    for request in (
        {"type": "set_wrist_roll", "params": {"angle_rad": 0.8}},
        {"type": "set_gripper", "params": {"position": 0.6}},
        {"type": "hold", "params": {"duration_s": 0.3}},
        {"type": "home"},
    ):
        status = resolve(twin, twin.submit(request))
        assert run_until_done(twin, status.id).state is ActionState.SUCCEEDED, request
    assert twin.observe()["joints_deg"]["wrist_roll"] == pytest.approx(0.0, abs=0.5)
    assert (
        rejected(twin, twin.submit({"type": "set_wrist_roll", "params": {"angle_rad": 9.0}})).reason
        is ReasonCode.JOINT_LIMIT
    )


def _pick_cube(twin):
    cube = twin.grasp.position("pick_cube")
    above = cube + [0.0, 0.0, 0.06]
    for request in (
        {"type": "move_ee_to", "params": {"xyz_m": list(above)}},
        {"type": "move_ee_to", "params": {"xyz_m": list(cube + [0.0, 0.0, 0.025])}},
        {"type": "set_gripper", "params": {"position": CUBE_FIT}},
    ):
        status = resolve(twin, twin.submit(request))
        assert run_until_done(twin, status.id).state is ActionState.SUCCEEDED, request
    assert twin.grasp.attached == "pick_cube"


def test_agent_assisted_pick_carry_release(twin):
    _pick_cube(twin)
    status = resolve(twin, move(twin, [0.28, 0.12, 0.22]))
    assert run_until_done(twin, status.id).state is ActionState.SUCCEEDED
    carried = twin.grasp.position("pick_cube")
    assert np.linalg.norm(carried[:2] - [0.28, 0.12]) < 0.02
    status = resolve(twin, twin.submit({"type": "set_gripper", "params": {"position": 0.0}}))
    run_until_done(twin, status.id)
    assert twin.grasp.attached is None
    for _ in range(100):
        twin.tick()
    assert twin.grasp.position("pick_cube")[2] < carried[2]  # it fell
    assert "pick_cube released" in twin.observe()["control"]["events"]


def test_human_takeover_does_not_drop_carried_cube(twin):
    _pick_cube(twin)
    for _ in range(5):
        twin.tick(lb(right_trigger=0.0))  # trigger at rest: keep holding
    assert twin.grasp.attached == "pick_cube"
    assert twin.state.gripper_pos == pytest.approx(CUBE_FIT, abs=0.01)
    for _ in range(5):
        twin.tick(lb(right_trigger=CUBE_FIT))  # picked up the trigger
    eased = CUBE_FIT + 0.02  # a touch tighter: still within what holds the cube
    for _ in range(40):
        twin.tick(lb(right_trigger=eased))  # now follows it
    assert twin.state.gripper_pos == pytest.approx(eased, abs=0.01)
    assert twin.grasp.attached == "pick_cube"
    for _ in range(int(HUMAN_IDLE_S / twin.dt) + 5):
        twin.tick(XboxState(right_trigger=eased))
    assert twin.agent_gripper == pytest.approx(eased, abs=0.01)  # agent inherits it


@pytest.mark.parametrize("speed", [0.05, 0.1])
def test_push_through_service_keeps_cube_on_table(twin, speed):
    cube = twin.grasp.position("pick_cube")
    for xyz in ([0.25, cube[1], 0.22], [0.25, cube[1], CUBE_REST_Z]):
        status = resolve(twin, move(twin, xyz))
        assert run_until_done(twin, status.id).state is ActionState.SUCCEEDED
    start = twin.grasp.position("pick_cube")
    status = resolve(twin, move(twin, [0.40, cube[1], CUBE_REST_Z], max_speed_m_s=speed))
    peak = 0.0
    obj = twin.grasp.objects["pick_cube"]
    for _ in range(600):
        twin.tick()
        peak = max(peak, float(np.linalg.norm(twin.sim.data.qvel[obj.qvel_adr : obj.qvel_adr + 3])))
        if twin.wait(status.id, 0).state.terminal:
            break
    for _ in range(50):
        twin.tick()
    end = twin.grasp.position("pick_cube")
    assert end[0] - start[0] > 0.02  # pushed forward
    assert abs(end[2] - start[2]) < 0.003  # still on the table, upright height
    assert peak < 3 * speed


def test_close_fails_pending_and_wakes_waiters(twin):
    status = resolve(twin, move(twin, [0.25, 0.1, 0.25]))
    pending = move(twin, [0.3, 0.0, 0.2])
    twin.close()
    assert twin.wait(status.id, 1.0).state is ActionState.CANCELLED
    with pytest.raises(TwinClosed):
        pending.result(timeout=0)
    with pytest.raises(TwinClosed):
        move(twin, [0.3, 0.0, 0.2]).result(timeout=0)


def test_closed_gripper_pushes_without_grabbing(twin):
    """Grasp attaches only when the gripper closes near an object, not when a
    closed gripper sweeps past one (the push task keeps it closed)."""
    status = resolve(twin, twin.submit({"type": "set_gripper", "params": {"position": 1.0}}))
    assert run_until_done(twin, status.id).state is ActionState.SUCCEEDED
    cube = twin.grasp.position("pick_cube")
    for xyz in ([0.25, cube[1], 0.22], [0.25, cube[1], CUBE_REST_Z]):
        status = resolve(twin, move(twin, xyz))
        assert run_until_done(twin, status.id).state is ActionState.SUCCEEDED
    start = twin.grasp.position("pick_cube")
    status = resolve(twin, move(twin, [0.40, cube[1], CUBE_REST_Z], max_speed_m_s=0.05))
    run_until_done(twin, status.id)
    assert twin.grasp.attached is None
    assert twin.grasp.position("pick_cube")[0] - start[0] > 0.02


def test_grasp_rearms_after_opening(twin):
    _pick_cube(twin)
    for position in (0.0, CUBE_FIT):  # release, then close again on the same cube
        status = resolve(twin, twin.submit({"type": "set_gripper", "params": {"position": position}}))
        run_until_done(twin, status.id)
    assert twin.grasp.attached == "pick_cube"


def test_table_is_the_floor_in_the_pick_place_scene(twin):
    """The tool cannot be driven through the table, by agent or controller."""
    z_min = twin.workspace_bounds["z"][0]
    assert z_min == pytest.approx(TABLE_TOP + 0.005)
    assert twin.observe()["workspace"]["z"][0] == pytest.approx(z_min)
    error = rejected(twin, move(twin, [0.30, -0.10, 0.10]))
    assert error.reason is ReasonCode.WORKSPACE_BOUND and error.detail.startswith("z_min")
    for _ in range(150):  # hold LB, right stick down for 3 s
        twin.tick(lb(right_stick_y=1.0))
    assert z_min - 1e-3 <= twin.tool_xyz()[2] <= z_min + 0.01


def test_object_knocked_off_the_table_is_flagged(twin):
    obj = twin.grasp.objects["pick_cube"]
    twin.sim.data.qpos[obj.qpos_adr : obj.qpos_adr + 3] = [0.48, -0.14, 0.02]
    twin.tick()
    assert twin.observe()["scene"]["objects"]["pick_cube"]["off_table"] is True
    assert resolve(twin, twin.reset_scene()) is True
    assert twin.observe()["scene"]["objects"]["pick_cube"]["off_table"] is False
