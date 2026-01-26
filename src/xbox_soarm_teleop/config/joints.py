"""Shared joint name definitions for SO-ARM teleoperation."""

# Joint names (order matters - matches URDF and robot)
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Joint names including gripper (order matters - matches URDF and robot)
JOINT_NAMES_WITH_GRIPPER = JOINT_NAMES + ["gripper"]

# IK joint names (include base, exclude wrist_roll/gripper)
IK_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
