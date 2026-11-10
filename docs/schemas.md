# Data Schemas

Two families of versioned contracts:

| Schema | Version | Defined in | Used by |
|---|---|---|---|
| Recorded datasets (per mode) | `1.0.0` (`DATASET_SCHEMA_VERSION`) | `recording/features.py` | `record-xbox` → LeRobot dataset + `meta/schema.json` |
| Agent observation | `soarm.observation/v1` | `schemas/agent.py` | agent control interface / MCP server |
| Agent action request | `soarm.action/v1` | `schemas/agent.py` | agent control interface / MCP server |

The code is the source of truth; this page explains it. Tests in
`tests/test_dataset_features.py` and `tests/test_agent_schemas.py` pin the
contracts.

## Conventions

- **Frame:** robot base frame, +x forward, +y left, +z up, origin at the base.
- **Units:** SI (m, s, rad) unless stated. Joint angles are **degrees** in agent
  schemas (the codebase and calibration are degree-based) and **LeRobot
  normalized** values in recorded datasets (what the follower arm reports).
- **Gripper:** `0` = open, `1` = closed in agent schemas and `action.ee_delta`;
  `0..100` (open..closed) in recorded joint vectors.
- **Versioning:** bump `DATASET_SCHEMA_VERSION` (semver) whenever a recorded field
  is added, removed, reordered, or changes meaning or units. Agent schemas change
  their ID (`/v2`) on any incompatible change.

## Recorded datasets

`record-xbox` writes a LeRobot dataset plus `meta/schema.json` containing the
output of `build_schema_metadata(mode, joint_names)`: for every feature its
`dtype`, `shape`, `names`, `unit` and `description`. Recordable modes:
`joint`, `crane`, `cartesian`. (`puppet` is not recordable yet.)

### All modes

| Feature | Shape | Names | Unit |
|---|---|---|---|
| `observation.state` | (6,) | `shoulder_pan.pos` … `gripper.pos` | LeRobot normalized: arm −100..100, gripper 0..100 |
| `action` | (6,) | same | same |

In `joint` and `crane` mode `action` is the joint target each processor
produced; the operator's stick input is not recorded separately.

### Cartesian mode adds

| Feature | Shape | Names | Unit |
|---|---|---|---|
| `action.ee_delta` | (7,) | `dx dy dz droll dpitch dyaw gripper` | m/s, rad/s, gripper 0..1 |
| `action.ee_target` | (16,) | `m00`…`m33` | 4×4 transform, row-major, m |
| `observation.ee_pose` | (16,) | `m00`…`m33` | 4×4 transform, row-major, m |
| `safety.flags` | (7,) | `ws_clip_x ws_clip_y ws_clip_z speed_clip orient_clip joint_clip reject` | 0/1 |

`safety.flags` let datasets be filtered for clipped or rejected steps (see #22).

## Agent observation — `soarm.observation/v1`

Produced by `Observation.to_dict()`:

| Field | Type | Meaning |
|---|---|---|
| `schema` | str | `"soarm.observation/v1"` |
| `frame` | int | control tick counter |
| `sim_time_s` | float | simulator (or wall) time |
| `mode` | str | `cartesian`, `joint`, `crane`, `puppet` |
| `ee.xyz_m`, `ee.rpy_rad` | [3], [3] | tool position and roll/pitch/yaw |
| `joints_deg` | {name: deg} | arm joint angles |
| `gripper` | 0..1 | gripper position |
| `stop_latched` | bool | virtual stop latched |
| `owner` | str \| null | client holding the control lease |
| `action` | object \| null | `{id, type, state, reason, detail, progress}` |
| `last_ik` | object \| null | `{accepted, reason, position_error_m}` |
| `limit` | str | human-readable stop reason, e.g. `"at floor"` |
| `workspace` | {x,y,z: [min,max]} | bounds actually enforced (from `workspace_limits.yaml`) |
| `scene` | object | `{name, objects: {name: {xyz_m, attached}}}` |
| `control` | object | `{stop_latched_by, human_in_control, controller_connected, events}` (added in v1; additive) |

## Agent action request — `soarm.action/v1`

Parsed and validated by `ActionRequest.from_dict()`:

```json
{"type": "move_ee_to", "params": {"xyz_m": [0.3, 0.0, 0.2]}, "max_speed_m_s": 0.05, "timeout_s": 5}
```

| Type | Params | Meaning |
|---|---|---|
| `move_ee_to` | `xyz_m` [3] | absolute tool position (m) |
| `move_ee_by` | `dxyz_m` [3] | relative tool offset (m) |
| `set_wrist_roll` | `angle_rad` | absolute wrist roll |
| `set_gripper` | `position` 0..1 | gripper target |
| `home` | — | joint-space return to home |
| `hold` | `duration_s` > 0 | hold current pose |

Optional for every type: `max_speed_m_s` (> 0, can only lower configured limits)
and `timeout_s` (> 0). Validation rejects unknown types, missing or unexpected
params, non-finite numbers, wrong vector lengths, and out-of-range gripper
values. **Workspace bounds are checked by the executor**, not the schema.

### Action states

`IDLE → EXECUTING → (HELD) → SUCCEEDED | FAILED | CANCELLED | STOPPED`.
Terminal: `SUCCEEDED`, `FAILED`, `CANCELLED`, `STOPPED`.

### Reason codes

| Code | Source |
|---|---|
| `workspace_bound` (detail `x_min`, `x_max`, `y`, `z_min`, `z_max`) | stop messages "at floor/ceiling/front/back/side limit" |
| `joint_limit` (detail = joint) | "joint limit: \<joint\>", IK `joint_limit` |
| `out_of_reach` | "out of reach", IK `not_converged` |
| `ik_error` | IK `invalid_output` / `solver_error` |
| `timeout`, `cancelled`, `stopped`, `preempted` | executor |
| `invalid_request`, `not_owner`, `busy` | submission rejected |

`reason_from_feedback(limit, ik_result)` performs the mapping; a test fails if a
new stop message is added without one.
