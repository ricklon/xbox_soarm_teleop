# Mode Verification Guide

This guide gives a repeatable smoke-test for each control mode so an operator can confirm that the mode is functioning before deeper use.

If you want the shortest field version, use the [Operator Checklist](operator_checklist.md).

Use the same progression for every mode:

1. Validate in MuJoCo first
2. Confirm the expected control mapping
3. Look for obvious sign, swap, and deadman errors
4. Only then try the real robot

## Safety

- Simulation first.
- On real hardware, start from home, clear the workspace, and keep one hand free for stop/power-off.
- For controller modes, verify the deadman behavior before any larger motion.

## Recommended Controller / Mode Matrix

| Mode | Recommended controller | Also usable | Notes |
|------|------------------------|-------------|-------|
| `crane` | `xbox` | `keyboard`, `joycon` | Single Joy-Con crane is intentionally limited. |
| `cartesian` | `xbox` | `keyboard`, `dual_joycon` | `dual_joycon` is the natural-motion path. |
| `joint` | `xbox` | `keyboard`, `joycon` | Best mode for direct servo confirmation. |
| `puppet` | `joycon` | none | Uses the Joy-Con IMU path. |

If you are specifically validating the new split Joy-Con path, also use the [Dual Joy-Con Test Guide](dual_joycon_test_guide.md).

## Mode 1: Crane

### What crane should feel like

Crane mode should feel like controlling:

- base pan
- reach
- height
- wrist flex
- wrist roll
- gripper

without directly thinking about shoulder and elbow joint angles.

### Recommended simulation command

```bash
uv run simulate-mujoco --controller xbox --mode crane
```

Keyboard alternative:

```bash
uv run simulate-mujoco --controller keyboard --mode crane
```

### Recommended real-robot command

```bash
uv run teleoperate-real --controller xbox --mode crane --port /dev/ttyUSB0
```

### Crane verification checklist

1. Verify no motion without the deadman.
2. Move pan left/right only.
Expected: base rotates without unexpected reach or height motion.

3. Move reach only.
Expected: end effector extends/retracts while the arm folds naturally.

4. Move height only.
Expected: end effector rises/lowers without large sideways drift.

5. Move wrist roll only.
Expected: wrist roll changes without the arm translating.

6. Move wrist flex only.
Expected: gripper tips up/down without unexpected pan.

7. Squeeze the gripper input.
Expected: gripper closes and reopens correctly.

8. Press home.
Expected: arm returns to the neutral crane pose smoothly.

### Crane pass criteria

- deadman works
- pan/reach/height are independent enough to be usable
- wrist controls do not move the arm body unexpectedly
- home works

### Common crane failures

- reach and height swapped
- one axis inverted
- wrist flex reversed
- gripper bound to wrong input
- deadman not gating motion

## Mode 2: Cartesian

### What cartesian should feel like

Cartesian mode should feel like commanding the gripper tip directly in space rather than commanding individual joints.

### Recommended simulation commands

Xbox:

```bash
uv run simulate-mujoco --controller xbox --mode cartesian
```

Dual Joy-Con:

```bash
uv run simulate-mujoco --controller dual_joycon --mode cartesian
```

Keyboard alternative:

```bash
uv run simulate-mujoco --controller keyboard --mode cartesian
```

### Recommended real-robot commands

Xbox:

```bash
uv run teleoperate-real --controller xbox --mode cartesian --port /dev/ttyUSB0
```

Dual Joy-Con:

```bash
uv run teleoperate-real --controller dual_joycon --mode cartesian --port /dev/ttyUSB0
```

### Cartesian verification checklist

1. Verify no motion without the deadman.
2. Command pure `X`, then pure `Y`, then pure `Z`.
Expected: each direction moves mostly along that axis with no obvious swap.

3. Command a small square path in the horizontal plane.
Expected: the end effector roughly traces the intended path.

4. Command wrist orientation only.
Expected: orientation changes without a large positional jump.

5. Command gripper only.
Expected: no arm motion, only gripper motion.

6. Press home.
Expected: arm returns to home without snapping.

### Additional dual Joy-Con cartesian checks

1. Hold `ZL`, choose a comfortable right-hand pose, release `ZL`, then hold it again.
Expected: reclutch captures a new neutral wrist orientation.

2. While holding `ZL`, tilt/rotate the right Joy-Con slowly.
Expected: wrist orientation follows hand orientation rather than continuously spinning.

### Cartesian pass criteria

- translation axes are not swapped
- no unexpected free-running drift
- orientation control is usable
- gripper and home work
- dual Joy-Con reclutch feels natural if used

### Common cartesian failures

- `X/Y/Z` swapped
- one axis inverted
- orientation mapped to the wrong physical hand motion
- wrist jumps when clutching
- IK freezes immediately because the command mapping is wrong

## Mode 3: Joint

### What joint mode should feel like

Joint mode should feel literal: the selected joint moves, and only that joint moves.

### Recommended simulation commands

Xbox:

```bash
uv run simulate-mujoco --controller xbox --mode joint
```

Keyboard:

```bash
uv run simulate-mujoco --controller keyboard --mode joint
```

### Recommended real-robot command

```bash
uv run teleoperate-real --controller xbox --mode joint --port /dev/ttyUSB0
```

For low-level confirmation of joint motion outside the main teleop path:

```bash
uv run xbox-joint-diagnostic --port /dev/ttyUSB0
```

### Joint verification checklist

1. Verify no motion without the deadman for controller-driven single-joint mode.
2. Cycle the selected joint.
Expected: HUD/selection changes immediately.

3. Drive the selected joint in the positive direction, then negative direction.
Expected: only that joint target changes.

4. Repeat for:
- shoulder pan
- shoulder lift
- elbow flex
- wrist flex
- wrist roll
- gripper

5. Press home.
Expected: all joints return toward home.

### Joint pass criteria

- selection logic works
- each joint responds on its own
- commanded direction matches expectation
- no hidden IK behavior

### Common joint failures

- D-pad cycle not working
- selected joint label does not match actual motion
- positive/negative sign reversed
- gripper command mapped incorrectly

## Mode 4: Puppet

### What puppet should feel like

Puppet mode should feel like:

- stick/buttons position the arm in a crane-like way
- Joy-Con IMU directly controls the wrist

This is the mode where “your hand is the gripper” should feel most obvious.

### Recommended simulation command

```bash
uv run simulate-mujoco --controller joycon --mode puppet
```

### Recommended real-robot command

```bash
uv run teleoperate-real --controller joycon --mode puppet --port /dev/ttyUSB0
```

### Puppet verification checklist

1. Verify no motion without the deadman.
2. Move pan only.
Expected: base rotates.

3. Move reach only.
Expected: arm extends/retracts.

4. Move height only.
Expected: arm raises/lowers.

5. Hold the Joy-Con in a neutral pose and recalibrate/home.
Expected: wrist neutral resets to the current hand pose.

6. Tilt the Joy-Con forward/back.
Expected: wrist flex follows that motion.

7. Rotate the Joy-Con like a door handle.
Expected: wrist roll follows that motion.

8. Command gripper.
Expected: gripper closes/opens correctly.

### Puppet pass criteria

- crane-style positioning still works
- IMU wrist tracks hand motion
- recalibration resets the neutral pose cleanly
- no severe drift while holding still

### Common puppet failures

- IMU unavailable
- wrist flex and roll swapped
- one IMU axis inverted
- neutral pose captured incorrectly
- Joy-Con sleeps or disconnects during use

## Test Result Template

Use this format after any mode test:

- `mode`: crane / cartesian / joint / puppet
- `controller`: xbox / keyboard / joycon / dual_joycon
- `deadman`: good / bad / not applicable
- `axis mapping`: correct / inverted / swapped
- `orientation`: correct / inverted / wrong axis / not tested
- `gripper`: good / bad
- `home`: good / bad
- `notes`: short free-form summary

That is enough to turn operator feedback into a concrete mapping or control fix.
