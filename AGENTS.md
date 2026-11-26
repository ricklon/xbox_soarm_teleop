# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, and others) working in this repository.
`CLAUDE.md` imports this file; edit this one.

## What this project is

Teleoperation for the SO-ARM100/101 6-DOF arm (Feetech STS3215 servos), built on HuggingFace
LeRobot. A person drives the arm with an Xbox controller, Joy-Con(s) or keyboard, in simulation
(MuJoCo), on the real arm, or both at once (digital twin). It is also a workshop platform:
a simulated arm that students and AI agents (over MCP) share, and a hub that hosts one arm per
student in the browser, with pick-and-place and sorting challenges.

Start with `README.md` for features and quick start, and `docs/` for the detailed guides.

## Tech stack

- Python 3.12 only (`>=3.12,<3.13`), managed with **uv**, built with hatchling
- LeRobot `>=0.7,<0.8` with `kinematics,feetech,dataset` extras; IK via Placo
- MuJoCo `>=4,<5` for simulation and camera rendering
- `mcp` for the agent server; `pygame`/`inputs` for gamepads
- ruff (lint + format, line length 104), pytest, pre-commit
- Robot model: `assets/so101.urdf` (SO-ARM100 and 101 share kinematics)

## Layout

```
src/xbox_soarm_teleop/
  cli/            Entry points for the console scripts in pyproject.toml
  teleoperators/  Input devices: xbox.py, joycon.py, joycon_imu.py, keyboard.py
  processors/     Input state -> commands, one per mode: xbox_to_ee (cartesian), crane,
                  joint_direct, puppet; factory.py picks one
  control/        Cartesian stepping, safety clipping, home, routines, unit conversion
  kinematics/     Constrained / cylindrical IK helpers, Jacobian
  lerobot_steps/  `soarm_cartesian_ik` LeRobot processor step (see docs/lerobot_pipeline.md)
  config/         Controller configs, ControlMode enum, joints, workspace_limits.yaml
  sim/            MuJoCo world, table scenes, printable props
  twin/           Workshop twin: MCP server, hub, cameras, challenges, judge, web UI
  runtime/        Session plumbing and on-screen control help
  recording/      LeRobot dataset features
  diagnostics/    Benchmarks, joint diagnostics, rerun logging
  schemas/        Agent-facing schemas
examples/         Runnable scripts; diagnostics/ has hardware probes (see examples/README.md)
scripts/          Benchmarks, agent trials, notebook builders, home-rig tools
tests/            pytest suite (runs headless; see CI below)
notebooks/        Workshop notebooks
spikes/           Experiments, not part of the package
twin_logs/, captures/  Run output; don't edit or commit casually
```

## Control modes

`ControlMode` in `config/modes.py`: `cartesian` (default; end-effector deltas through IK),
`joint` (per-joint velocity, no IK), `crane` (cylindrical, decoupled), `puppet` (crane plus
Joy-Con IMU wrist). Select with `--mode` and the device with `--controller`.

Xbox mapping in cartesian mode: left stick X = Y axis, left stick Y = Z axis, right stick Y = X
axis, right stick X = wrist roll, D-pad = pitch/yaw, right trigger = gripper, LB = deadman,
A = home. `docs/driving_guide.md` is the source of truth for every mode and device; update it
when you change a mapping.

## Commands

```bash
uv sync --extra dev                 # install (CI uses: uv sync --locked --extra dev)
uv run pytest                       # full suite
uv run pytest tests/test_x.py -v    # one file
uv run ruff check . --fix
uv run ruff format <changed files>  # not the whole tree; see below
pre-commit run --files <changed files>

uv run simulate-mujoco [--mode crane --controller keyboard]   # sim, no robot
uv run teleoperate-real --port /dev/ttyUSB0                    # real arm
uv run teleoperate-dual --port /dev/ttyUSB0                    # real + sim
uv run soarm-twin                                              # workshop twin + MCP on :8765
uv run soarm-hub --seats 2                                     # one browser arm per student
uv run python scripts/twin_smoke.py                            # end-to-end twin check over MCP
```

Other entry points (`ik-smoke`, `mujoco-ik-check`, `soarm-challenge`, `record-xbox`,
`diagnose-robot`, ...) are listed under `[project.scripts]` in `pyproject.toml`.

## Safety (non-negotiable)

- Every motion path is gated by a deadman (Xbox LB, single Joy-Con SL, dual Joy-Con ZL).
  Releasing it stops the arm. Don't add a motion path that bypasses it.
- Workspace bounds and safety clipping apply **before** IK; velocity caps apply after.
- Analog deadzone is 0.1.
- Gripper is position-controlled (0 = open, 1 = closed), not velocity.
- Units: m/s linear, rad/s angular; IK seeds from the current joints.
- Test in simulation first. Never run code against the real arm without the user's go-ahead,
  and assume a person must be at the e-stop.
- In the twin, the person holding the controller can always take over or stop an agent.

## Conventions

- Write or update tests with every change (TDD preferred). Tests must pass headless
  (CI sets `MUJOCO_GL=egl`, `SDL_VIDEODRIVER=dummy`) and must not need a controller or robot.
- Run `ruff format` only on files you changed; formatting the whole tree rewrites ~30
  unrelated files.
- Type hints on public functions, Google-style docstrings, dataclasses for state.
- Conventional commits (`feat:`, `fix:`, `docs:`, `ci:`, ...). Work on a branch; `main` is
  merged through PRs, and CI (`.github/workflows/tests.yml`) runs pytest on every PR.
- Never print, log, screenshot or commit student keys, API keys or tokens. Hub output is
  designed to be safe to share on screen; keep it that way.
- New fields in twin session logs need a test on a twin that keeps a session log (past
  crashes came from field names colliding with the logger's own).

## Working on agents and challenges

Read `docs/lessons_learned.md` before changing the twin, perception or challenges. Key habits:
measure in layers (perception, grasp, understanding) before end-to-end trials; prove a
challenge is solvable with agent-like aim error before giving it to an agent; look at the
camera images, not just the numbers; replay an agent's actions before theorizing; count runs
that died on quota or service errors as invalid, not as agent failures.

## Docs map

| Topic | File |
|-------|------|
| Controls per mode/device | `docs/driving_guide.md` |
| Per-mode verification | `docs/mode_verification_guide.md` |
| Calibration | `docs/CALIBRATION.md` |
| Cartesian IK processor step | `docs/lerobot_pipeline.md` |
| Dataset layout | `docs/dataset_schema.md`, `docs/schemas.md` |
| MCP interface for agents | `docs/mcp_control_interface.md` |
| Workshop / hosting / students | `docs/workshop_guide.md`, `docs/workshop_hosting.md`, `docs/students.md` |
| Sorting and challenges | `docs/try_sorting.md`, `docs/challenges.md`, `docs/building_scenes.md` |
| Preflight checklist | `docs/operator_checklist.md` |

## Troubleshooting

- Controller not found: `uv run python examples/debug_controller.py`; on Linux add the user to
  the `input` group. Joy-Con udev rules: `scripts/70-joycon.rules`, `scripts/setup_joycon.sh`.
- Robot port: `ls /dev/ttyUSB* /dev/ttyACM*` (Linux) or `/dev/tty.usbmodem*` (macOS).
- IK not converging: check the target is inside `config/workspace_limits.yaml`, reduce
  speed scales, and confirm URDF joint limits match the calibrated arm.
- Headless rendering errors: set `MUJOCO_GL=egl` (Linux) as CI does.
