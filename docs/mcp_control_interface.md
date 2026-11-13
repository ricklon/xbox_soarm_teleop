# Agent Control Interface and MCP Server — Design

Status: **Draft, partly implemented** (2026-11-09) · Scope: **v1 = simulation, cartesian mode**

> **Implementation notes (workshop build).** `twin/service.py` implements §3–§9
> with the amendments marked *Amended* below; `control/step.py` is the shared
> step (D2). Workshop students use Apple Silicon Macs or Linux.
Parent plan: [`audit_plan.md` → "MCP and simulation-first milestones"](audit_plan.md)

## 1. Goal

Let software agents (Claude, Codex, other MCP clients) observe and move the
SO-ARM101 through a **bounded, stoppable, recorded** interface, validated first
in MuJoCo and later reused unchanged for hardware.

The first milestone (from the audit plan) is a recorded, fully simulated
exercise: **request → move → observe → stop → inspect → reset → resume.**

### Non-goals for v1
- Hardware control (see §11 — blocked on physical validation).
- Joint, crane, and puppet modes (cartesian only; others reuse the same
  interface later).
- Trajectory planning, collision-free paths, or grasp planning. Actions are
  bounded straight-line goals executed by the existing IK/Jacobian step.
- Concurrent multi-agent control. One owner at a time (§7).

## 2. Layering

```
 MCP client (Claude / Codex / ...)
        │  MCP tools/resources
 ┌──────▼───────────────────────┐
 │ MCP adapter  (optional extra) │  thin: schema in/out, no control logic
 └──────┬───────────────────────┘
        │  Python API
 ┌──────▼───────────────────────┐
 │ ControlInterface              │  actions, status, stop latch, ownership,
 │  + ActionExecutor             │  recording — no MCP, no MuJoCo imports
 └──────┬───────────────────────┘
        │  Backend protocol
 ┌──────▼───────────────────────┐   ┌───────────────────────┐
 │ MuJoCoBackend (v1)            │   │ HardwareBackend (later)│
 └──────┬───────────────────────┘   └───────────────────────┘
        │
 shared cartesian step  (same code teleop uses)
```

Decisions:
- **D1. The interface knows nothing about MCP.** Everything below the adapter is
  plain Python, testable with pytest and without the `mcp` package. MCP becomes
  an optional dependency (`uv sync --extra mcp`).
- **D2. One motion path.** Agent actions and human teleop drive the *same*
  cartesian step. Today that step is embedded in the 684-line
  `run_with_controller` in `cli/simulate_mujoco.py`
  (advance target → `solve_cartesian_motion` or `jacobian_joint_step` →
  `apply_ik_solution` → `describe_motion_limit`). It must be extracted into a
  reusable function first, so the agent path inherits the joint-limit,
  workspace-bound and IK-acceptance fixes instead of re-implementing them.

## 3. Threading and loop ownership

- **D3. One control thread owns the backend** and ticks at the existing 50 Hz
  (`LOOP_PERIOD`). It is the only code that steps the simulator or mutates
  `CartesianControlState`.
- MCP handlers (and any other client) **only enqueue commands and read
  snapshots**. Commands go through a thread-safe queue; observations are
  published as immutable snapshots after each tick.
- A handler that waits for completion polls status or blocks on a per-action
  event with a timeout; it never steps the simulation itself.
- Headless by default. A viewer may be attached for humans, but the control
  thread must not depend on it (reuse `_HeadlessViewer`).
- *Amended:* the control thread is the process's **main Python thread**, which
  also syncs the viewer and owns offscreen rendering (camera requests are
  queued like commands, at most one render per tick). Under `mjpython` on macOS
  the viewer owns the **OS** main thread and Python runs on a secondary one, so
  nothing in the twin process may call SDL/Cocoa event APIs: the Xbox pad is
  read by a helper process (`teleoperators/xbox_helper.py`, the `subprocess`
  backend), whose stale or missing reports release the deadman.
- *Amended:* `stop()` sets a flag checked before the command queue is drained,
  so a move queued just before a stop is rejected, not started. Commands
  resolve `concurrent.futures.Future`s; shutdown fails pending ones and wakes
  every waiter.
- *Amended:* time is `frame × dt` (control time). Each tick runs
  `dt / timestep` physics substeps with the kinematically posed arm
  interpolated across them, so sim time equals control time and contacts see
  small steps.

## 4. Actions

An action is a **goal executed over many ticks**, not a single setpoint.

| Action | Parameters | Completes when |
|---|---|---|
| `move_ee_to` | target `xyz` (m, base frame), optional `max_speed` (m/s) | tool within tolerance of target |
| `move_ee_by` | delta `xyz` (m) | as above, relative to pose at start |
| `set_wrist_roll` | angle (rad) | roll within tolerance |
| `set_gripper` | position 0–1 | gripper within tolerance |
| `home` | — | existing `step_cartesian_home` reports done |
| `hold` | duration (s) | duration elapsed |

Every action has: a server-assigned **`action_id`**, a **timeout** (default
derived from distance / speed, capped), and a **speed cap** that can lower but
never raise the configured limits (`velocity.linear_max`, joint velocity
limits).

- **D4. Out-of-bounds targets are rejected, not silently clipped.** A target
  outside the workspace is refused at submission with the offending axis and
  bound. (Teleop clips, because a human is steering continuously; an agent
  should learn the target was invalid.) An optional `clip=true` flag may be
  added later; if used, the clipped target is returned explicitly.
- Execution converts the goal into a per-tick EE velocity toward the target
  (speed-capped, slowing near the goal) and feeds it to the shared step — the
  same input shape a joystick produces.
- Only one action executes at a time. Submitting a new action while one is
  running is rejected unless `preempt=true`, which cancels the current one.

## 5. Status state machine

```
           submit            tick ok            reached
  IDLE ───────────► EXECUTING ───────► EXECUTING ───────► SUCCEEDED
                       │  │  │
          no progress  │  │  └── cancel() ─────────────► CANCELLED
          + limit      │  └──── timeout ───────────────► FAILED(timeout)
                       ▼
                  HELD(reason) ── persists > hold_timeout ─► FAILED(reason)
  any state ── stop() ─────────────────────────────────► STOPPED
```

- **D5. Terminal states carry a machine-readable reason**, mapped from what the
  control step already reports:

| Source | Reason code | Meaning |
|---|---|---|
| `describe_motion_limit` "at floor/ceiling/front/back/side limit" | `workspace_bound` (+ which) | target reachable only through a bound |
| `describe_motion_limit` "joint limit: \<joint\>" | `joint_limit` (+ joint) | a joint at its limit is blocking progress |
| `IKResult.reason == "not_converged"` | `out_of_reach` | IK held: target not reachable from here |
| `IKResult.reason in {"invalid_output","solver_error"}` | `ik_error` | solver failure |
| executor | `timeout`, `cancelled`, `stopped`, `preempted` | |

- `HELD` is not immediately terminal: a brief hold (e.g. sliding along a bound)
  can recover. If no progress is made for `hold_timeout` (default 1 s) the
  action ends `FAILED(<reason>)`.

## 6. Virtual stop

- **D6. Latching.** `stop()` sets a latch that:
  1. cancels the executing action (`STOPPED`),
  2. holds the current joint targets (in sim: hold position; physics keeps
     running so stopping behaviour can be observed — see §9),
  3. rejects every new motion command with `stopped` until reset.
- Anyone may call `stop()` (any client, the human teleop, a watchdog).
- **`reset_stop()` must be deliberate**: it requires the current owner (§7) and
  an explicit confirmation argument; it never clears implicitly (no
  auto-reset on reconnect, new action, or timeout).
- *Amended:* the Xbox **B** button latches the stop and **Start** releases it.
  A stop latched on the controller can only be released on the controller;
  agents get `not_owner` ("press Start"). Observations report `stop_latched_by`:
  `agent`, `human`, or `instructor`. The hub's instructor latches the third
  kind with **STOP ALL**; only the instructor releases it, and it outranks the
  other two (a student's or agent's stop cannot replace it).
- Read-only calls (`observe`, `status`, `get_log`) keep working while latched.
- Hardware later adds an **independent physical stop**; the virtual stop is not
  a substitute (§11).

## 7. Command ownership

- **D7. Single owner via lease.** Motion commands require holding the control
  lease (`acquire_control(client_id)` → `lease_id`, renewed by heartbeat or by
  any command). Observation needs no lease.
- *Amended for LLM agents:* agents think for tens of seconds between calls, so
  the lease is **60 s**, motion commands **acquire it automatically** when it
  is free, and a different client may take over whenever the owner has no
  running action (a restarted agent is not locked out). Expiry only frees
  ownership; it does not cancel (actions have their own capped timeouts). The
  MCP adapter keys clients by the `mcp-session-id` header.
- *Amended (human override):* holding **LB**, or pressing **A** (home) / **Y**
  (reset scene), cancels the agent's action (`preempted`, detail `human`) and
  blocks agent motion (`not_owner`, detail `human`) until the controller has
  been idle for 2 s. The trigger is absolute, so on takeover the gripper holds
  until the trigger reaches it (no dropped cube); on handback the agent keeps
  the human's gripper position.
- Lease expiry (heartbeat missed for `lease_timeout`, default 2 s) or client
  disconnect **cancels the running action and holds** (does not latch stop, but
  does not resume anything either).
- Human teleop and agent control are mutually exclusive: a teleop session holds
  the lease; an agent cannot acquire it until released. (Initially the MCP
  server simply runs without a human controller.)

## 8. Observation schema (versioned)

`observe()` returns a snapshot, schema `soarm.observation/v1`:

```jsonc
{
  "schema": "soarm.observation/v1",
  "frame": 12345,              // control tick counter
  "sim_time_s": 246.9,
  "mode": "cartesian",
  "ee": {"xyz_m": [0.39, 0.0, 0.23], "rpy_rad": [0.0, 0.0, 0.0]},
  "joints_deg": {"shoulder_pan": 0.0, "...": 0.0},
  "gripper": 0.0,
  "stop_latched": false,
  "owner": "claude-session-1",
  "action": {"id": "a-17", "type": "move_ee_to", "state": "EXECUTING",
             "reason": null, "progress": 0.62},
  "last_ik": {"accepted": true, "reason": "accepted", "position_error_m": 0.0},
  "limit": "",                 // describe_motion_limit message
  "workspace": {"x": [0.05, 0.5], "y": [-0.3, 0.3], "z": [0.05, 0.45]},
  "scene": {"name": "pick_place_basic", "objects": {"pick_cube": [0.3, 0.1, 0.02]}}
}
```

Units are SI (m, rad) on the wire; joints additionally in degrees because the
whole codebase and calibration are degree-based. The schema is versioned and
should be defined together with the per-mode action/observation schemas in
**#20**.

## 9. Simulation backend specifics

- Built on `MuJoCoSimulator` (scenes and `--world` presets apply).
- **Physics caveat:** the `default` scene is *kinematic only* (`step()` calls
  `mj_forward`, and `test_mujoco_compatibility` asserts sim time does not
  advance). "Observe stopping behaviour" is only meaningful in physics scenes
  (`stack`, `pick_place_basic`). Decision **D8**: the milestone exercise runs in
  `pick_place_basic`; the default scene stays kinematic for speed.
- Scene control exposed to agents: `reset_scene()` (existing
  `reset_stack_scene` / `reset_pick_place_scene`), and scene object poses in
  observations.

## 10. Recording

- **D9. Every session writes a JSONL log**: one line per request, response,
  status transition, stop/reset, ownership change, and a downsampled
  observation stream (e.g. 10 Hz, full rate around transitions).
- Each line has `t`, `frame`, `kind`, `client_id`, `action_id`, payload. The
  file header records schema versions, git commit, URDF hash, world/scene,
  limits in force, and calibration identity (from `audit_calibration`).
- **Implemented** in `twin/session_log.py` as `soarm.session_log/v1`; the
  line format is specified in `docs/schemas.md`. `soarm-twin` writes
  `twin_logs/twin-<date>-<time>-<pid>.jsonl` unless `--no-log` is given.
- This log is the milestone evidence and the basis for later sim-vs-hardware
  comparison (audit plan gate 5).

## 11. Hardware (later — not v1)

The same `ControlInterface` gets a `HardwareBackend`. Blocked on, and **open
questions** to resolve during hardware validation rather than assume:

- Audit plan gate 5 (independently measured physical accuracy).
- **#15** — IK seeded from measured joint feedback.
- An **independent physical stop** that works without software.
- **Hold vs. torque-disable** on virtual stop: holding can resist contact;
  torque-off can drop the arm. Decide per measurement, not assumption.
- Real-time behaviour of serial I/O inside the 50 Hz control thread.

## 12. Prerequisites and constraints found in the code

- **Workspace bounds disagree in three places — #18 is a hard prerequisite.**
  An agent will be told the workspace in observations; it must be the bounds
  actually enforced.

  | Source | x (m) | y (m) | z (m) |
  |---|---|---|---|
  | `simulate_mujoco.SIM_WORKSPACE_BOUNDS` | 0.05–0.5 | **unbounded** | 0.05–0.45 |
  | `teleoperate_dual` (inline) | **−0.1**–0.5 | ±0.3 | 0.05–0.45 |
  | `config/workspace_limits.yaml` | 0.05–0.5 | ±0.3 | 0.05–0.45 |

  The YAML `home_position` is also stale (joint names `elbow`/`wrist_pitch`,
  gripper 0.5) and should be fixed or removed under #18.
- **#20** — action/observation schemas: define `soarm.observation/v1` and the
  action schema there.
- **PR #27** must merge first (constrained IK, `IKResult`, motion feedback,
  Jacobian bounds).
- **#25** — Jacobian drift: v1 agent actions use IK mode; `--jacobian` is not
  offered to agents until #25 is fixed.
- The MCP Python SDK is not currently a dependency. Its current API and
  transports (stdio vs. HTTP) must be checked against the SDK documentation when
  the adapter is implemented; this doc intentionally does not pin them.

## 13. MCP surface (sketch)

Thin mapping of the interface; names indicative only.

| MCP tool | Interface call | Needs lease |
|---|---|---|
| `observe` | `observe()` | no |
| `acquire_control` / `release_control` | ownership | — |
| `move_ee_to`, `move_ee_by`, `set_wrist_roll`, `set_gripper`, `home` | `submit(action)` | yes |
| `wait_for_action` | block until terminal or timeout | no |
| `cancel_action` | `cancel(action_id)` | yes |
| `stop` | `stop()` | **no** |
| `reset_stop` | `reset_stop(confirm=True)` | yes |
| `reset_scene` | `reset_scene()` | yes |
| `list_scenes` | `list_scenes()` | no |
| `load_scene` | `load_scene(name)` | yes |

Scenes of one compiled model are *layouts* (`twin/layouts.py`): the model holds
every object, and a layout says which are on the table. `pick_place_easy` has a
bean bag the arm passes through instead of pushing (a rigid box to the physics,
but it does not collide with the robot); `pick_place_basic` has the rigid cube,
which the gripper knocks away if it touches it before closing. Switching never
recompiles, so each hub room switches on its own.

Resources: the current observation, the workspace/limits in force, and the
session log path. Tool descriptions must state units, frames and bounds, so an
agent can form valid requests without trial and error.

## 14. Acceptance (milestone 1)

A **headless pytest** (no `mcp` package needed) and an equivalent MCP-client
script both perform, in `pick_place_basic`:

1. `acquire_control` → `observe`.
2. `move_ee_to` a reachable target → `SUCCEEDED`, tool within tolerance.
3. `move_ee_to` below the floor → rejected at submission (`workspace_bound z`).
4. `move_ee_to` far target; `stop()` mid-motion → `STOPPED`; tool holds
   (pose change < 1 mm over the next 1 s of physics).
5. New `move_ee_to` while latched → rejected `stopped`.
6. `observe` shows `stop_latched: true`; `reset_stop(confirm=True)`.
7. Resume: `move_ee_to` → `SUCCEEDED`.
8. Lease expiry mid-action → `CANCELLED`, arm holds.

The JSONL log of the run contains every step above and is attached to the PR
that completes the milestone.

## 15. Implementation breakdown

Tracking epic plus the issues below (dependency order). Existing issues
**#18**, **#20**, **#15**, **#25** are linked, not duplicated.

1. **Extract the shared cartesian control step** from `run_with_controller`
   (D2). Teleop behaviour unchanged; covered by existing tests + playback runs.
   *Blocked by #27.*
2. **ControlInterface + headless MuJoCo backend**: control thread, command
   queue, observation snapshots (D3, §8, §9). *Blocked by 1, #18, #20.*
3. **ActionExecutor**: action types, IDs, timeouts, speed caps, rejection of
   out-of-bounds targets, state machine + reason mapping (§4, §5). *Blocked by 2.*
4. **Virtual stop latch** with deliberate reset (§6). *Blocked by 3.*
5. **Ownership lease + heartbeat/disconnect handling** (§7). *Blocked by 3.*
6. **Session JSONL recording** (§10). *Blocked by 2.*
7. **MCP adapter** as optional `[mcp]` extra; verify current SDK API/transport
   first (§13). *Blocked by 3–6.*
8. **Milestone 1 scripted exercise** as pytest + MCP client script (§14).
   *Blocked by 7.*
9. **Hardware backend** — design + implementation. *Blocked by gate 5, #15,
   physical stop, hold-vs-torque decision (§11).*
