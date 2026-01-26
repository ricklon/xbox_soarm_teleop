# SO-ARM101 Calibration Guide

This guide explains how calibration works for the SO-ARM101 and how to perform it correctly.

## What Calibration Does

Calibration maps the raw encoder values from each servo to a normalized position range. It records:

1. **Midpoint position** - The encoder value when each joint is at the center of its range
2. **MIN/MAX values** - The encoder values at each joint's physical limits

This allows:
- Consistent position values across different robots
- Neural networks trained on one robot to work on another
- Proper joint limit enforcement

## Prerequisites

- Robot powered at **12V** (for Pro Edition follower arm)
- USB connected to your computer
- Know your serial port (run `lerobot-find-port` if unsure)

## Calibration Command

```bash
# For follower arm (the one you control via teleoperation)
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_follower_arm
```

Replace `/dev/ttyUSB0` with your actual port.

## Step-by-Step Process

### Step 1: Set Midpoint Position

When prompted, move the arm so **all joints are at the middle of their range**:

```
         Midpoint Position Reference
         ============================

         Side View:

              [2]
               |  \
               |   \ Upper Arm
               |    \
              [1]    [3]----[4]--[5]--[6]
               |            Forearm    Gripper
               |
             Base

         Joint positions at midpoint:
         - [1] Shoulder Pan: Pointing forward (0°)
         - [2] Shoulder Lift: ~45° up from horizontal
         - [3] Elbow Flex: ~90° bend
         - [4] Wrist Flex: Straight/neutral
         - [5] Wrist Roll: Neutral (0°)
         - [6] Gripper: Half open
```

**Visual check:** The arm should look like it's in a relaxed "ready" pose, not fully extended or fully retracted.

Press **Enter** when the arm is in position.

### Step 2: Move Through Full Range

After setting midpoint, you must move **each joint** through its complete range of motion:

| Joint | Movement |
|-------|----------|
| Shoulder Pan | Rotate base left and right (full sweep) |
| Shoulder Lift | Move upper arm up and down |
| Elbow Flex | Bend and straighten the elbow fully |
| Wrist Flex | Tilt wrist up and down |
| Wrist Roll | Rotate wrist clockwise and counter-clockwise |
| Gripper | Open fully, close fully |

**Important:** Move each joint to its **physical limits** (where it stops). Don't be gentle - the system needs to see the actual range.

Press **Enter** when done moving all joints.

## Verifying Calibration

After calibration completes, the calibration file is saved to:
```
~/.cache/huggingface/lerobot/calibration/robots/my_follower_arm/
```

You can verify by running your teleoperation script. The arm should:
- Respond smoothly to controller input
- Not have any joints "jump" on startup
- Respect joint limits properly

## Recalibration

If you need to recalibrate (e.g., after reassembly or if something seems off):

```bash
# Delete existing calibration
rm -rf ~/.cache/huggingface/lerobot/calibration/robots/my_follower_arm

# Run calibration again
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=my_follower_arm
```

**Note:** The `.cache` folder is hidden. Use `Ctrl+H` in file manager or `ls -la ~` to see it.

## Common Issues

### "Calibration file already exists"
Delete the existing calibration folder and try again (see Recalibration above).

### Arm moves wildly after calibration
- Midpoint was set incorrectly - recalibrate
- Didn't move joints through full range - recalibrate and ensure you hit physical limits

### Joint seems reversed or offset
- The midpoint position was wrong for that joint
- Recalibrate, paying attention to that specific joint's center position

### Communication errors during calibration
- Check USB connection
- Verify power supply is 12V and stable
- Try a different USB port
- Check servo firmware is up to date

## Safety Warning

**Secure the robot before calibrating.** At the end of calibration or on first command after calibration, the arm may move quickly to its initial position. Keep hands clear and ensure the workspace is free of obstacles.

## Quick Reference

```bash
# Find your port
lerobot-find-port

# Set permissions (Linux)
sudo chmod 666 /dev/ttyUSB0

# Calibrate follower
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyUSB0 --robot.id=my_arm

# Delete calibration to start fresh
rm -rf ~/.cache/huggingface/lerobot/calibration/robots/my_arm

# Check calibration exists
ls ~/.cache/huggingface/lerobot/calibration/robots/
```

## References

- [LeRobot SO-101 Docs](https://huggingface.co/docs/lerobot/so101)
- [Waveshare Calibration Wiki](https://www.waveshare.com/wiki/SO-ARM100/101_Robotic_Arm_Calibration_and_Remote_Control)
- [TheRobotStudio SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100)
