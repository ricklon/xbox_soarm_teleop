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
