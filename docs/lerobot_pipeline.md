---
title: LeRobot Processor Pipeline (Cartesian IK)
date: 2026-03-26
---

# LeRobot Processor Pipeline (Cartesian IK)

This project exposes a LeRobot processor step `soarm_cartesian_ik` that converts
end-effector delta actions into joint-position actions using IK. It is registered
through `ProcessorStepRegistry` and can be inserted into a teleop action pipeline.

See also:
- [Dataset Schema](dataset_schema.md)
- [Audit Response and Plan](audit_plan.md)
- [`examples/record_xbox.py`](../examples/record_xbox.py)
- [`configs/soarm_cartesian_processor.json`](../configs/soarm_cartesian_processor.json)

## Step: `soarm_cartesian_ik`

Inputs (RobotAction):
- `dx`, `dy`, `dz` (m/s)
- `droll`, `dpitch`, `dyaw` (rad/s)
- `gripper` (0–1 position)

Output (RobotAction):
- `"<joint>.pos"` for all joints, in LeRobot normalized units.

The step reads `observation` from the transition to seed IK and applies
workspace limits, strict safety clipping, and velocity caps.

## Minimal Pipeline Config (JSON)

```json
{
  "name": "soarm_cartesian_teleop",
  "steps": [
    {
      "registry_name": "soarm_cartesian_ik",
      "config": {
        "urdf_path": "/abs/path/to/so101_abs.urdf",
        "dt": 0.0333333333,
        "swap_xy": true,
        "strict_safety": true,
        "ik_vel_scale": 1.0,
        "gripper_rate": 2.0
      }
    }
  ]
}
```

In [`examples/record_xbox.py`](../examples/record_xbox.py), cartesian recording
now uses this pipeline step directly for teleop actions.

An example config is provided in
[`configs/soarm_cartesian_processor.json`](../configs/soarm_cartesian_processor.json).
