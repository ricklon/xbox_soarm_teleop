---
title: Dataset Schemas by Control Mode
date: 2026-03-26
---

# Dataset Schemas by Control Mode

This project records datasets through the LeRobotDataset interface. Each control
mode produces different action semantics, but observation/state is always the
measured joint positions from the robot.

See also:
- [LeRobot Cartesian Pipeline](lerobot_pipeline.md)
- [`examples/record_xbox.py`](../examples/record_xbox.py)

## Shared Observation
- `observation.state` (float32, shape = 6)
  - Ordered joint positions: shoulder_pan, shoulder_lift, elbow_flex,
    wrist_flex, wrist_roll, gripper.
  - Values are the robot’s normalized units (LeRobot `*.pos`).

## Joint Mode
- `action` (float32, shape = 6)
  - Absolute joint position goals in normalized units.

## Crane Mode
- `action` (float32, shape = 6)
  - Absolute joint position goals in normalized units.
  - Generated via cylindrical control + 2-DOF planar IK.

## Puppet Mode
- `action` (float32, shape = 6)
  - Absolute joint position goals in normalized units.
  - Wrist joints driven by Joy-Con IMU orientation.

## Cartesian Mode (recording path)
- `action` (float32, shape = 6)
  - Absolute joint position goals in normalized units (IK/Jacobian output).
- `action.ee_delta` (float32, shape = 7)
  - [dx, dy, dz, droll, dpitch, dyaw, gripper] in SI units.
- `action.ee_target` (float32, shape = 16)
  - Flattened 4x4 target end-effector pose (row-major).
- `observation.ee_pose` (float32, shape = 16)
  - Flattened 4x4 end-effector pose computed from measured joints.
- `safety.flags` (float32, shape = 7)
  - [ws_clip_x, ws_clip_y, ws_clip_z, speed_clip, orient_clip, joint_clip, reject]

## Notes
- Safety clipping should be logged alongside frames when available to allow
  dataset filtering or annotation.
- If a dataset mixes control modes, treat it as a new schema version.
