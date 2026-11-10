# MuJoCo compatibility and control-service handoff

Give this document to the next AI working on `xbox_soarm_teleop`.

## Goal

Continue the simulator-first control work. Establish a reproducible MuJoCo 4.0
compatibility check while preserving the project's Python 3.10 / MuJoCo 3.4
environment, then improve the shared simulation interface so both a CLI and an
MCP server can observe state, execute bounded moves, stop, reset, and report
results. Simulation remains the first backend; do not connect to or command the
physical robot as part of these tasks.

## Repository status at handoff

- Branch: `feat/versioned-schemas`.
- The work discussed as "not yet committed" was in fact already present in
  commit `10ef505` (`feat: constrained IK, coupled wrist roll, and macOS input
  backends`) and pushed to `origin/feat/versioned-schemas`.
- The checkout was clean at the last inspection. Verify `git status`, branch
  tracking, and ancestry before making changes; don't recreate or recommit
  existing work.
- MuJoCo 3.4.0 is the project's installed version and remains the declared
  dependency. The project requires Python 3.10. MuJoCo 4.0.0 requires Python
  3.11 or newer, so it was tested in an isolated temporary Python 3.11
  environment, not installed into the project environment.
- Compatibility snapshots are checked in at
  `docs/evidence/mujoco_3_4_compatibility.json` and
  `docs/evidence/mujoco_4_0_compatibility.json`. They cover the same portable
  SO-ARM101 URDF and default, stack, and pick/place scenes.
- `tests/test_mujoco_compatibility.py` exercises scene loading, state updates,
  stepping, finite state, and reset in the installed MuJoCo version.
- `tests/test_mujoco_headless.py` checks headless routing without launching a
  viewer. The CLI now forwards `--headless` into automatic-routine mode.
- The bounded automatic routine completed under both MuJoCo 3.4.0 and 4.0.0
  with 11 samples and zero reported FK position error. This is a short smoke
  check, not a broad physics validation.
- Opening the MuJoCo 4.0 GUI from the restricted environment exited with code
  134. An escalated launch did run the short routine and exited successfully,
  but macOS printed a sandbox-extension diagnostic for the temporary
  `mjpython` launcher. Treat GUI validation as limited; headless validation is
  the repeatable result.
- The temporary 4.0 environment lives at `/tmp/xbox-mujoco4-validation` and
  its uv cache at `/tmp/xbox-mujoco-uv-cache`. These are machine-local and are
  not part of the project. The checked-in snapshots and tests are the portable
  handoff; recreate an isolated environment when needed.

## Instructions for the next AI

1. Inspect repository status and read `docs/audit_plan.md`, this handoff, and
   the two compatibility snapshots. Confirm which commits are on the remote.
2. Run the targeted headless and scene-compatibility tests under the project's
   Python 3.10 / MuJoCo 3.4 setup. Recreate an isolated Python 3.11 / MuJoCo
   4.0 environment and run those same tests there. Keep LeRobot and Placo
   versions aligned where possible so the MuJoCo version is the main changed
   variable. Do not change `requires-python`, `pyproject.toml`, or `uv.lock`
   merely to make MuJoCo 4.0 the project default.
3. If compatibility differs, report the smallest failing case and fix it with
   a regression test. Regenerate the JSON snapshots only from a documented,
   repeatable probe, and explain any numeric differences rather than hiding
   them behind loose tolerances.
4. Move toward one persistent simulation service that owns the MuJoCo model
   and exposes typed observation, bounded action, execution status,
   cancellation, and recorded results. Keep one command owner at a time. A
   virtual emergency stop must latch, stop pending motion, reject new motion,
   and require a deliberate reset while the physics engine remains observable.
5. Expose that same service through a useful CLI first, then MCP. MCP should
   call the service; it must not own physics or implement a second motion path.
   Include tests for interruption, client disconnect, command conflicts,
   limits, stop/reset, and recovery in simulation.
6. Preserve existing user changes. Commit only task-related changes, and push
   only after checking the active branch and remote.

The physical arm, calibration-to-model angle mapping, collision clearance, and
hardware stop behavior remain unverified. Do not present simulation agreement
as proof of physical accuracy or safety.
