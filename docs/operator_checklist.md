# Operator Checklist

Use this as the fast preflight and smoke-test sheet.

Rule: `simulation first`, then `real robot`.

## Preflight

- [ ] Workspace clear
- [ ] Robot starts at home
- [ ] Power-off or e-stop is reachable
- [ ] Correct controller connected
- [ ] For Joy-Con paths: both pairing and input visibility confirmed

## Crane

### Simulation

```bash
uv run simulate-mujoco --controller xbox --mode crane
```

- [ ] No motion without deadman
- [ ] Pan works
- [ ] Reach works
- [ ] Height works
- [ ] Wrist flex works
- [ ] Wrist roll works
- [ ] Gripper works
- [ ] Home works

### Real robot

```bash
uv run teleoperate-real --controller xbox --mode crane --port /dev/ttyUSB0
```

- [ ] Tiny motions only first
- [ ] Same behavior as simulation

## Cartesian

### Simulation

Xbox:

```bash
uv run simulate-mujoco --controller xbox --mode cartesian
```

Dual Joy-Con:

```bash
uv run simulate-mujoco --controller dual_joycon --mode cartesian
```

- [ ] No motion without deadman
- [ ] `X` translation correct
- [ ] `Y` translation correct
- [ ] `Z` translation correct
- [ ] Orientation works
- [ ] Gripper works
- [ ] Home works

Dual Joy-Con only:

- [ ] Reclutch works
- [ ] Wrist follows hand orientation
- [ ] No wrist jump on clutch

### Real robot

Xbox:

```bash
uv run teleoperate-real --controller xbox --mode cartesian --port /dev/ttyUSB0
```

Dual Joy-Con:

```bash
uv run teleoperate-real --controller dual_joycon --mode cartesian --port /dev/ttyUSB0
```

- [ ] Tiny motions only first
- [ ] Same behavior as simulation

## Joint

### Simulation

```bash
uv run simulate-mujoco --controller xbox --mode joint
```

- [ ] No motion without deadman
- [ ] Joint selection works
- [ ] Selected joint moves alone
- [ ] Positive direction matches expectation
- [ ] Negative direction matches expectation
- [ ] Gripper works
- [ ] Home works

### Real robot

```bash
uv run teleoperate-real --controller xbox --mode joint --port /dev/ttyUSB0
```

Optional direct diagnostic:

```bash
uv run xbox-joint-diagnostic --port /dev/ttyUSB0
```

- [ ] Tiny motions only first
- [ ] Same behavior as simulation

## Puppet

### Simulation

```bash
uv run simulate-mujoco --controller joycon --mode puppet
```

- [ ] No motion without deadman
- [ ] Pan works
- [ ] Reach works
- [ ] Height works
- [ ] IMU wrist flex works
- [ ] IMU wrist roll works
- [ ] Recalibrate/home works
- [ ] Gripper works

### Real robot

```bash
uv run teleoperate-real --controller joycon --mode puppet --port /dev/ttyUSB0
```

- [ ] Tiny motions only first
- [ ] Same behavior as simulation

## Result Summary

- [ ] `crane` passed
- [ ] `cartesian` passed
- [ ] `joint` passed
- [ ] `puppet` passed

## Failure Notes

- `mode`:
- `controller`:
- `deadman`:
- `axis/sign issue`:
- `orientation issue`:
- `gripper`:
- `home`:
- `notes`:
