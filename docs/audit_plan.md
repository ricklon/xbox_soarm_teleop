---
title: Audit Response and Plan
date: 2026-03-26
scope: xbox_soarm_teleop
---

# Audit Response and Plan

Related docs:
- [Driving Guide](driving_guide.md)
- [LeRobot Cartesian Pipeline](lerobot_pipeline.md)
- [Dataset Schema](dataset_schema.md)

## Scope
Workshop preparation now prioritizes the validation gates below. The original
findings and phased plan remain historical context, not a current defect list.

## Workshop critical path — validation first

Evidence review: 2026-11-09. No hardware was commanded during this review.

### Evidence available

- Local reference fork: `/Users/rianders/Documents/ricklon/personal/lerobot`,
  commit `6600e03143f6568abdf976804a001d6926a35d72` (2026-01-14), version
  0.4.3. Inspection found an untracked `gym_hil_env.json` and no tracked changes.
  This teleop environment also reports LeRobot 0.4.3, installed in its own
  site-packages with no direct-source metadata. Matching version numbers do not
  prove the installed package contains the fork's changes.
- The fork contains tracked `.cache/calibration/so100/main_follower.json` and
  `main_leader.json`. These use arrays of homing offsets, drive modes, start/end
  positions, calibration modes, and motor names. They are historical calibration
  candidates, not yet identified as calibration for the workshop SO-ARM101.
  Their tracked status means file presence alone does not establish local use.
- Commit `0c3ebfea06461247a11f17fc05884c37bbe4dae0` (2026-02-15)
  reports a verified +72-degree wrist-flex hardstop, calibration passing for
  all six joints, and nine passing wrist IK tests. These are historical reports,
  not independently reproducible measurements: the calibration JSON and raw
  telemetry mentioned in the message are absent from the local Git history.
- Searching the checkout, including ignored files outside the environment and
  caches, found no measurement CSVs or calibration JSONs. The standard local
  `~/.cache/huggingface/lerobot/calibration/robots` directory is absent. Data
  could still exist on the original robot-control computer or another branch
  unavailable locally.
- The preceding full test run produced 291 passes, 35 skips, and 30 setup errors.
  The errors stem from machine-specific mesh paths in `so101_abs.urdf` and do
  not establish whether the IK calculations are correct.
- A fresh FK probe using portable `assets/so101.urdf`, IK joints
  `[10, 30, -45, 20]` degrees, and wrist roll changed from 0 to 90 degrees
  produced 11.1765 mm displacement of `gripper_frame_link`. Wrist roll must be
  represented in the full pose model even when controlled separately.
- The installed LeRobot normalization uses calibrated motor range endpoints;
  `control/units.py` instead scales by the largest absolute configured joint
  limit. Agreement requires verification of motor-specific scale, zero, and sign.
- Real-robot Cartesian feedback seeding already exists and defaults to enabled.
  The old finding below that it is missing is superseded.

### Ordered validation gates

1. **Recover and identify physical evidence.** Locate the original
   `so101_calibration.json`, LeRobot calibration, ROM telemetry, and joint
   diagnostic logs. Record robot identity, firmware, calibration date, joint
   zero convention, and source commit where known. Preserve original files.
   Synthetic test fixtures and commit claims must not be treated as measurements.
2. **Establish a reproducible model baseline.** Unify portable asset loading;
   reconcile URDF, configured, and historically measured limits; run the real
   FK/IK/Jacobian tests. Record dependency versions and residuals. A passing
   roundtrip proves model consistency only, not physical accuracy.
3. **Validate the command-to-model mapping.** For each arm joint, establish the
   relationship between encoder readings, calibration, normalized commands, and
   URDF angles. Test asymmetric ranges, direction, offsets, and roundtrips against
   recovered calibration. Do not infer an actual robot's zero from URDF limits.
4. **Validate solver degrees of freedom.** Explicitly test whether excluded
   joints move inside the solver, constrain uncommanded joints, and include
   measured wrist roll in FK. Check reachable XYZ/pitch targets, neutral
   orientation, infeasible targets, and limits. Apply the same checks to Crane's
   reduced-chain solver. Define which orientation controls the arm can satisfy.
5. **Validate physical pose and motion.** After the software gates, compare FK
   with independently measured gripper positions at several configurations,
   including different roll angles. Use a defined base frame and tool point.
   Log target, measured joints, predicted pose, independently measured pose,
   uncertainty, and calibration identity. Agree workshop tolerances before
   evaluating results; servo tracking alone is not tool-position ground truth.
6. **Freeze the workshop workflow.** Select a controller and default mode after
   validation, rehearse the same task in simulation and on hardware, and document
   installation and recovery. Additional modes remain advanced options.

Immediate software priority: gates 2–4 can proceed while gate 1's external data
is recovered. Physical accuracy and workshop readiness remain unverified until
gate 5. Controller expansion and cosmetic cleanup are secondary to these gates.

### MCP and simulation-first milestones

After establishing the kinematics baseline, build one shared control interface
for observation, bounded actions, execution status, cancellation, and recorded
results. First implement and validate its MuJoCo backend, then expose it through
MCP for Codex, Claude, and other clients.

The virtual stop must latch, cancel pending motion, reject new movement, and
require deliberate reset. Keep physics running to observe stopping behavior.
Test interruption, disconnects, conflicting command ownership, limits, and
recovery. Hardware subsequently uses the same interface with an independent
physical stop; determine hold-versus-torque-disable behavior during hardware
validation rather than assuming they are interchangeable.

First interaction milestone: record a complete simulated request, move, observe,
stop, inspect, reset, and resume exercise. Then compare equivalent sequences
against independently measured physical poses before freezing workshop setup.

### Baseline implementation progress

The three FK/IK, wrist-limit, and Jacobian test modules now load the portable
URDF. `test_kinematics_contract.py` independently checks the roll/tool-offset
geometry and includes a strict expected failure for excluded-wrist movement.
A 10 mm X target step from `[10, 30, -45, 20]` degrees changed the excluded wrist
roll from zero to approximately -0.000170592 radians in installed LeRobot 0.4.3.
This confirms that filtering returned joint names does not constrain that joint.
The expected failure documents the installed upstream implementation; production now uses
the constrained adapter described below. It is not workshop acceptance.

### Crane/Puppet radial geometry (2026-11-09)

- Replaced the world-X/Z target with Y fixed to zero by a two-variable solve for
  tool radius `hypot(world_x, world_y)` and world height. Pan, wrist flex, and
  wrist roll are held at their proposed commanded values in the full URDF.
  Tool azimuth and orientation are resulting geometry, not extra constraints.
- The bounded damped least-squares solver uses NumPy and Placo FK, caps work at
  40 iterations with up to eight backtracking trials per iteration, and accepts
  only radius/height residuals <= 10 micrometers. It restores model state after
  planning and returns rejection for nonconvergence. No global reachability or
  collision-free-path guarantee is implied.
- The installed SciPy optimizer could not load a native sparse-library module
  on this machine. The implementation does not require that import; the existing
  dependency environment was preserved.
- Crane's old `(0.16 m, 0.12 m)` home target did not solve with its commanded
  wrist geometry. Home radius/height now come from FK of the existing parked
  shoulder/elbow settings, existing home pan/roll, and Crane wrist flex of 13
  degrees. This changes the Cartesian home target, not those joint settings.
- Puppet samples IMU wrist targets before solving and initializes/resets
  reach/height from full-model FK, including roll. Rejected coupled targets hold
  all five arm joints and restore the prior reach/height target. Joint-step
  scaling coordinates direct joints with shoulder/elbow movement.
- Both processors now gate homing and gripper commands with LB, closing a
  pre-existing deadman bypass. Gripper remains independent of IK acceptance
  while enabled. Puppet's enabled home command still uses its existing direct
  reset behavior; trajectory/recovery validation remains outstanding.

Regression coverage checks independent full-arm FK at several pan/wrist angles,
Crane home and a combined reach/roll command, unreachable-target rejection,
model restoration, same-sample IMU geometry, coupled rejection, and deadman
behavior. Intermediate rate-limited joint trajectories and physical poses still
need simulation and hardware validation; these tests establish model consistency.

Validation after radial geometry changes: **389 passed, 35 skipped, 1 expected
upstream failure**. Changed-file lint and whitespace checks pass. No hardware
was commanded.

### Neutral orientation and coupled roll (2026-11-09)

- Cartesian state now latches an explicit orientation request independently of
  its numeric angle. Returning pitch/yaw to zero, including by velocity input,
  keeps orientation enabled on subsequent translation. Fresh translation-only
  state leaves orientation disabled; completing home clears the latch.
- Added a shared coupled motion planner. It evaluates IK with the proposed
  wrist roll already in the model, compensating the tool offset in the same
  solve. Rejection holds both the selected chain and roll. Planning restores the
  current model state before returning and does not commit controller state.
- Uniform joint-step scaling applies to both arm joints and roll. The accepted
  endpoint satisfies the numerical pose gate; intermediate joint-space steps
  need not trace an exact Cartesian line, and later safety clipping can further
  alter motion. Roll endpoints are clipped to configured wrist-roll limits.
- Real, dual, MuJoCo teleop/routine, and recording paths use the shared planner.
  Requests involving orientation or roll use constrained IK even when Jacobian
  mode is selected; translation-only requests retain the resolved-rate path.
- Regression coverage includes absolute and velocity-driven neutral return,
  orientation persistence/reset, simultaneous roll/translation, rejection of a
  coupled move, coordinated step limits, and the recording processor.

Orientation targets still use the existing world Euler convention with roll
zero; zero pitch/yaw is therefore a world identity orientation, not a newly
calibrated physical neutral. This change fixes request persistence, not the
arm's underactuation: arbitrary XYZ + orientation targets may be rejected.
Defining a reachable position-plus-tool-pitch control manifold, and correcting
Crane/Puppet geometry, remain open. No physical neutral or hardware accuracy
claim follows from these tests.

Validation after neutral/roll changes: **379 passed, 35 skipped, 1 expected
upstream failure**. Changed-file lint and whitespace checks pass. No hardware
was commanded.

### IK acceptance gate (2026-11-09)

Production IK now distinguishes an accepted solution from a held command:

- Validate seed angles, rigid target transforms, and task weights before solving.
- Enable URDF joint limits inside Placo; reject outputs outside the intersection
  of URDF and configured limits. The configured wrist-flex upper bound is
  1.256 radians (about 71.963 degrees), not exactly 72 degrees.
- Allow at most 30 iterations. Enabled tasks must reach 10 micrometers of
  position error and 0.03 radians of orientation error in the model. These are
  software acceptance thresholds, not demonstrated physical tolerances.
- Reject nonfinite/wrong-shaped outputs, missing results, solver runtime errors,
  joint-limit violations, and failure to converge. Restore model joint state and
  return the supplied seed. `last_ik_result` records acceptance, reason, solve
  count, and residuals; warning logs report transitions into rejection states.
- Crane homing must not report completion when the planar solve was rejected.

The hold applies to selected IK joints, not an emergency stop of every channel:
independently controlled wrist/gripper channels retain their existing behavior.
A rejected solve does not prove global unreachability. Caller velocity limiting
still applies after an accepted solve. Invalid inputs raise `ValueError` before
solver mutation; they are not treated as accepted hold commands.

This replaces permissive one-step best-effort output with explicit acceptance.
Infeasible full-orientation requests and inconsistent Crane/Puppet planar tool
geometry can therefore hold motion where they previously produced approximate
commands. Defining achievable orientation controls and reconciling planar tool
geometry remain prerequisites for workshop readiness. Neutral-orientation input
semantics are not changed by this gate.

A local 10-call timing probe for each of a 1 mm X step and a distant target from
`[10, 30, -45, 20]` degrees measured maxima of 0.324 ms and 0.206 ms respectively.
This is a limited software measurement, not a control-loop deadline guarantee.

Validation after the acceptance gate: **371 passed, 35 skipped, 1 expected
upstream failure**. Changed-file lint and whitespace checks pass. All work was
performed offline; no hardware was commanded.

### Calibration mapping investigation (2026-11-09)

The operator confirms the same physical arm has been used for all development.
This makes the historical follower calibration a candidate for that arm, while
its active calibration revision and relation to the URDF remain unverified.

- The historical `main_follower.json` is a legacy array-format calibration,
  last changed by `d93386ec7` in the reference fork. Its fingerprint and values
  are preserved in [the offline report](evidence/legacy_follower_calibration_audit.json).
  The original file was not modified.
- Historical Feetech code at reference-fork commit
  `b536f47e3ff8c3b340fc5efa52f0ece0a7212a57`, in
  `lerobot/common/robot_devices/motors/feetech.py`, computes DEGREE values by
  applying direction, adding a software homing offset, then scaling by encoder
  resolution. The report identifies those nominal zeros without claiming they
  equal URDF zeros. Legacy `start_pos`/`end_pos` must not be copied into modern
  range endpoints without establishing their coordinate meaning.
- Installed LeRobot 0.4.3 normalizes arm values using calibrated range endpoints
  and drive mode. Its DEGREES option uses the range midpoint too, so switching
  modes alone would not establish the model's zero. Encoder readings already
  reflect the active hardware homing offset; applying it again would be wrong.
- Added `CalibratedJointMapping` with explicit encoder zero, model direction,
  scale, and calibrated range. It rejects nonfinite values and out-of-range
  commands. Synthetic tests compare both drive directions against the installed
  LeRobot bus conversion methods without opening a serial connection.
- Added an offline calibration audit command supporting modern and legacy
  formats, fingerprints, and separate gripper semantics. No calibration is
  rewritten and no live-control conversion is switched automatically.

Validation: **75 calibration-related tests passed** (30 new tests plus 45 existing
ROM-calibration tests). These establish software conversion contracts, not the
physical reference values. Existing `control/units.py` remains in live paths and
is now explicitly documented as an unverified limit-scaled approximation.

Next evidence needed: identify the active calibration revision and hardware
homing-offset state, then relate known encoder positions to URDF angles for each
joint. The ROM exporter also derives a homing offset from parked-position
constants while retaining sweep encoder extrema; its coordinate consistency
needs verification before treating its output as a migration path.

### Constrained solver implementation (2026-11-09)

- Added a project-owned LeRobot adapter that masks every joint outside the
  selected chain. Runtime, simulation, recording, Crane, Puppet, and smoke-test
  model construction now use it; the installed dependency is unchanged.
- Cartesian initialization, feedback synchronization, accepted solutions, and
  home targets explicitly include wrist roll when computing tool poses.
- Regression tests first reproduced excluded-joint movement in both four-joint
  and two-joint chains, plus omitted roll in Cartesian state. They now verify
  fixed excluded joints and agreement with a separate full-arm FK model.
- Reachable position targets are checked at -90, 0, and +90 degrees of roll,
  with a model-space position residual threshold of 10 micrometers. This is a
  numerical test tolerance, not a claim of physical accuracy.
- Existing FK/IK, wrist-limit, and Jacobian tests now exercise the production
  adapter. The strict expected failure in `test_kinematics_contract.py` remains
  an explicit probe of unmodified upstream LeRobot.

Validation: full suite **328 passed, 35 skipped, 1 expected upstream failure**.
Changed-file lint and whitespace checks pass. No hardware was commanded.

Remaining gates: calibration/angle mapping; achievable orientation semantics
(including neutral orientation); rehearsal of the new IK acceptance and limit
policy; Crane/Puppet tool geometry with directly commanded wrist joints; and
independent physical measurements. The subsequent coupled-roll change above now accounts for roll-induced tool
displacement within the Cartesian solve; direct Crane/Puppet wrist geometry
still requires separate validation.

Audit focus is the SO-ARM101 movement models and code organization in this repository, with specific attention to DOF assumptions and safety behavior. The SO-ARM101 URDF defines five revolute arm joints plus a gripper. The current cartesian control path uses a 4-joint IK chain (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex) and drives wrist_roll directly.

## Movement Model Summary
- **Cartesian (IK/Jacobian)**: EE delta commands integrated into a target pose; IK solves for 4 joints; wrist_roll is direct. Jacobian mode uses a 4-DOF position+pitch mapping.
- **Crane**: Cylindrical reach/height with a 2-DOF planar IK; pan and wrist are direct joints.
- **Puppet**: Crane + Joy-Con IMU for wrist orientation.
- **Joint-direct**: Per-joint velocity integration for diagnostics and keyboard multi-joint control.

## DOF Decision Notes
- The SO-ARM101 arm provides 5 DOF for pose (excluding gripper), so full 6-DOF pose control is underactuated.
- Current implementation effectively supports 4-DOF IK + 1-DOF wrist roll. Pitch/yaw are optional and low-weight in IK; Jacobian mode ignores yaw.
- Any move toward 6-DOF targets must be treated as best-effort and explicitly documented in UI and safety constraints.

## Findings (High Signal)
1. Axis conventions and XY swap are inconsistent between real and sim entry points.
2. Workspace limits are duplicated across modules; YAML config is unused.
3. Jacobian mode accepts yaw input but does not use it, while the UI implies it does.
4. Crane IK path does not guard against a None return from inverse_kinematics.
5. IK is seeded from internal state only; closed-loop seeding from feedback is missing.

## Plan (Phased)
**Phase 1 — Safety and Clarity**
- Centralize axis mapping and swap_xy behavior.
- Unify workspace limits configuration and usage.
- Fix Jacobian yaw handling and UI messaging.
- Guard Crane IK failure path.

**Phase 2 — Robustness**
- Seed IK from measured joint feedback where available.

**Phase 3 — Data/LeRobot Alignment**
- Add a LeRobot-compatible cartesian recording path (EEReferenceAndDelta → EEBoundsAndSafety → IK) so cartesian control can generate consistent training data.
- Define and document action/observation schemas per control mode (joint/crane/puppet/cartesian).
- Track and log safety clipping events so datasets can be filtered or annotated.

**Phase 4 — Maintainability**
- Refactor teleoperate_real.py into smaller, reusable control modules.

## Issue Tracker
- Axis mapping unification: https://github.com/ricklon/xbox_soarm_teleop/issues/17
- Workspace limits config: https://github.com/ricklon/xbox_soarm_teleop/issues/18
- Jacobian yaw clarity: https://github.com/ricklon/xbox_soarm_teleop/issues/16
- Crane IK robustness: https://github.com/ricklon/xbox_soarm_teleop/issues/19
- Closed-loop IK seeding: https://github.com/ricklon/xbox_soarm_teleop/issues/15
