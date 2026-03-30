# Examples Inventory

This directory contains a mix of primary entry points, maintenance tools, and
one-off investigation scripts. They are not all equally supported.

## Supported Entry Points

Use these first. They are the scripts the main docs and current cleanup work are
optimizing around.

| Script | Preferred Command | Purpose |
|--------|-------------------|---------|
| `debug_controller.py` | `uv run python examples/debug_controller.py` | Inspect normalized Xbox controller input without a robot |
| `simulate.py` | `uv run python examples/simulate.py` | Meshcat visualization for lightweight simulation and demos |
| `simulate_mujoco.py` | `uv run simulate-mujoco` | MuJoCo simulation for controller, mode, and routine testing |
| `teleoperate_real.py` | `uv run teleoperate-real` | Main real-robot teleoperation entry point |
| `teleoperate_dual.py` | `uv run teleoperate-dual` | Digital twin mode: real robot plus MuJoCo |
| `record_xbox.py` | `uv run record-xbox` | Project-native dataset recording for joint, crane, and cartesian modes |
| `lerobot_record_cartesian.py` | `uv run python examples/lerobot_record_cartesian.py` | Convenience wrapper for cartesian recording |

## Diagnostics And Maintenance

These are useful tools, but they are narrower in scope and often assume a
specific debugging or calibration workflow.

| Script | Purpose |
|--------|---------|
| `diagnostics/xbox_joint_diagnostic.py` | Bypass IK and drive one joint at a time with telemetry logging |
| `diagnostics/analyze_joint_diag.py` | Summarize a captured joint diagnostic CSV |
| `diagnostics/diagnose_robot.py` | Pre-flight motor diagnostics using the diagnostics-enabled LeRobot fork |
| `diagnostics/diagnose_motors.py` | Low-level per-servo diagnostic sweep |
| `diagnostics/probe_controller.py` | Print raw gamepad/evdev event codes for controller discovery |
| `diagnostics/joint_rom_test.py` | Range-of-motion sweep and calibration helper |
| `diagnostics/interactive_servo_test.py` | Manual motor inspection and jogging utility |
| `diagnostics/read_servo_positions.py` | Poll servo positions through the LeRobot interface |
| `diagnostics/go_home.py` | Send the arm to its home pose without entering teleop |

Preferred packaged commands for the documented diagnostics:
- `uv run xbox-joint-diagnostic`
- `uv run analyze-joint-diag`
- `uv run diagnose-robot`
- `uv run joint-rom-test`

## Compatibility Shims

These exist so older commands or bookmarks still work, but they are not the
preferred entry points.

| Script | Preferred Command |
|--------|-------------------|
| `teleoperate.py` | `uv run teleoperate-real` |
| `ik_smoke_test.py` | `uv run ik-smoke` |

## One-Off And Investigative Scripts

These are still useful, but they should be treated as ad hoc tools rather than
stable interfaces.

| Script | Purpose |
|--------|---------|
| `archive/compare_direct_vs_ik.py` | Compare joint-diagnostic logs against IK logs |
| `archive/diagnose_wrist_flex.py` | Investigate wrist-flex interference and travel limits |
| `archive/power_cycle_test.py` | Manual recovery flow for a stuck elbow joint |
| `archive/test_base_rotation.py` | Simple shoulder-pan oscillation script |
| `archive/test_controller_evdev.py` | Manual evdev controller probe |
| `archive/test_controller_pygame.py` | Manual pygame controller probe |
| `archive/test_jacobian_control.py` | A/B exploration of Jacobian and IK control in MuJoCo |

## Conventions

- Scripts under `examples/` are not pytest tests, even if their filenames start
  with `test_`.
- Prefer the packaged CLIs (`ik-smoke`, `mujoco-ik-check`, `simulate-mujoco`,
  `teleoperate-real`, `teleoperate-dual`, `record-xbox`, `xbox-joint-diagnostic`,
  `analyze-joint-diag`, `diagnose-robot`, `joint-rom-test`) over wrapper scripts
  when both exist.
- If a tool proves broadly useful, it should graduate into the supported entry
  point set or move into `src/` as a reusable CLI module.
