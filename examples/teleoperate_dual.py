#!/usr/bin/env python3
"""Digital twin mode: Real robot + MuJoCo simulation.

This example runs both the real SO-ARM101 robot and a MuJoCo simulation
simultaneously. The simulation serves as a real-time preview of the robot's
movements.

Usage:
    uv run python examples/teleoperate_dual.py --port /dev/ttyUSB0
    uv run python examples/teleoperate_dual.py --port /dev/ttyUSB0 --motion-routine --routine-pattern square

Controls:
    - Hold LB (left bumper) to enable arm movement
    - Left stick X: Move left/right (Y axis)
    - Left stick Y: Up/down (Z axis)
    - Right stick Y: Forward/back (X axis)
    - Right stick X: Wrist roll rotation (direct, not IK)
    - Right trigger: Gripper (released=open, pulled=closed)
    - A button: Return to home position
    - Close window or Ctrl+C: Exit
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from xbox_soarm_teleop.config.joints import IK_JOINT_NAMES, JOINT_NAMES_WITH_GRIPPER

# Path to URDF
URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"

# Joint names (order matters - matches URDF and robot)
JOINT_NAMES = JOINT_NAMES_WITH_GRIPPER

# Control loop rate
CONTROL_RATE = 30  # Hz
LOOP_PERIOD = 1.0 / CONTROL_RATE
GRIPPER_DEFAULT = 0.0  # 0=open, 1=closed


def gripper_to_robot(gripper_pos: float) -> float:
    """Map gripper position (0=open, 1=closed) to robot command."""
    return (1.0 - gripper_pos) * 100.0


def find_serial_port() -> str | None:
    """Find available serial port for the robot."""
    import glob

    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def deg_to_normalized(deg: float, joint_name: str) -> float:
    """Convert degrees to LeRobot normalized value [-100, 100]."""
    if joint_name == "gripper":
        return deg
    return deg * (100.0 / 180.0)


class MuJoCoSimulator:
    """MuJoCo-based SO-ARM101 simulator for visualization."""

    def __init__(self, urdf_path: str):
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)

        # Get joint indices
        self.joint_ids = {}
        for name in JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                self.joint_ids[name] = jnt_id

        self.go_home()

    def go_home(self) -> None:
        """Reset to home position."""
        for name in JOINT_NAMES:
            if name in self.joint_ids:
                jnt_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_joint_positions(self, positions_deg: np.ndarray, gripper_pos: float) -> None:
        """Set joint positions.

        Args:
            positions_deg: Joint positions in degrees (5 joints, excluding gripper).
            gripper_pos: Gripper position 0-1.
        """
        positions_rad = np.deg2rad(positions_deg)
        for i, name in enumerate(JOINT_NAMES[:-1]):
            if name in self.joint_ids:
                jnt_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] = positions_rad[i]

        # Set gripper
        if "gripper" in self.joint_ids:
            jnt_id = self.joint_ids["gripper"]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            gripper_open = 1.74533
            gripper_closed = -0.174533
            self.data.qpos[qpos_adr] = gripper_open - gripper_pos * (gripper_open - gripper_closed)

        mujoco.mj_forward(self.model, self.data)

    def get_ee_position(self) -> np.ndarray:
        """Get end effector position."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_frame_link")
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return self.data.xpos[-1].copy()


def run_dual_mode(
    port: str,
    motion_routine: bool = False,
    routine_duration: float = 15.0,
    routine_scale: float = 1.0,
    routine_pattern: str = "lissajous",
    routine_plane: str = "xy",
    routine_square_size: float = 0.06,
    routine_square_speed: float = 0.03,
    routine_trace: bool = False,
    routine_trace_max: int = 300,
    routine_trace_step_mm: float = 2.0,
):
    """Run digital twin mode with real robot and simulation."""
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta, MapXboxToEEDelta
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    # Initialize kinematics
    print("Loading kinematics model...", flush=True)
    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=IK_JOINT_NAMES,
    )

    # Initialize MuJoCo simulation
    print("Loading MuJoCo model...", flush=True)
    sim = MuJoCoSimulator(str(URDF_PATH))

    # Initialize Xbox controller
    config = XboxConfig()
    controller = None
    mapper = None
    if not motion_routine:
        controller = XboxController(config)
        mapper = MapXboxToEEDelta(
            linear_scale=config.linear_scale,
            angular_scale=config.angular_scale,
            orientation_scale=config.orientation_scale,
            invert_pitch=config.invert_pitch,
            invert_yaw=config.invert_yaw,
        )
        if not controller.connect():
            print("ERROR: Failed to connect to Xbox controller")
            sys.exit(1)
        print("Xbox controller connected", flush=True)
    gripper_rate = config.gripper_rate

    # Initialize real robot
    print(f"Connecting to real robot on {port}...", flush=True)
    robot_config = SOFollowerRobotConfig(port=port)
    robot = SOFollower(robot_config)

    try:
        robot.connect(calibrate=True)
    except Exception as e:
        print(f"ERROR: Failed to connect to robot: {e}")
        controller.disconnect()
        sys.exit(1)

    print("Robot connected!", flush=True)
    print("\n=== DIGITAL TWIN MODE ===", flush=True)
    print("Simulation window shows real-time preview of robot movements", flush=True)
    if motion_routine:
        print("\nMotion routine:", flush=True)
        print(
            f"  Duration: {routine_duration:.1f}s | Scale: {routine_scale:.2f} | "
            f"Pattern: {routine_pattern}",
            flush=True,
        )
        print("  Close window or Ctrl+C to exit\n", flush=True)
    else:
        print("\nControls:", flush=True)
        print("  Hold LB + move sticks to control arm", flush=True)
        print("  D-pad up/down: Pitch (tilt gripper)", flush=True)
        print("  D-pad left/right: Yaw (rotate heading)", flush=True)
        print("  Right trigger for gripper", flush=True)
        print("  A button to go home", flush=True)
        print("  Close window or Ctrl+C to exit\n", flush=True)

    # Current state
    wrist_roll_deg = 0.0
    ik_joint_pos_deg = np.zeros(4)
    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
    gripper_pos = GRIPPER_DEFAULT
    routine_center = ee_pose[:3, 3].copy()

    # Target orientation (euler angles in radians)
    target_pitch = 0.0
    target_yaw = 0.0

    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert euler angles (ZYX convention) to rotation matrix."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])

    def square_offset(u: float, size: float) -> tuple[float, float]:
        half = size / 2.0
        s = (u % 1.0) * 4.0
        seg = int(s)
        f = s - seg
        if seg == 0:
            return (-half + f * size, -half)
        if seg == 1:
            return (half, -half + f * size)
        if seg == 2:
            return (half - f * size, half)
        return (-half, half - f * size)

    def plane_offset(plane: str, u: float, size: float) -> np.ndarray:
        a, b = square_offset(u, size)
        if plane == "xy":
            return np.array([a, b, 0.0])
        if plane == "xz":
            return np.array([a, 0.0, b])
        return np.array([0.0, a, b])

    def draw_trace(viewer: mujoco.viewer.Handle, points: list[np.ndarray]) -> None:
        if not routine_trace:
            return
        rgba = (0.1, 0.9, 0.1, 1.0)
        radius = 0.003
        if hasattr(viewer, "add_marker"):
            for p in points:
                viewer.add_marker(
                    pos=p,
                    size=[radius, radius, radius],
                    rgba=rgba,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                )
            return
        if not hasattr(viewer, "user_scn"):
            return
        scene = viewer.user_scn
        scene.ngeom = 0
        max_geoms = int(getattr(scene, "maxgeom", 0))
        mat = np.eye(3).flatten()
        for p in points:
            if scene.ngeom >= max_geoms:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, radius, radius]),
                p,
                mat,
                np.array(rgba),
            )
            scene.ngeom += 1

    # Ensure sim + robot start with the same gripper state
    sim.set_joint_positions(np.zeros(5), gripper_pos)
    robot.send_action({"gripper.pos": gripper_to_robot(gripper_pos)})

    running = True
    routine_start = time.monotonic()
    trace_points: list[np.ndarray] = []
    trace_min_step_m = max(routine_trace_step_mm / 1000.0, 0.0005)

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...", flush=True)
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Launch viewer
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running() and running:
                loop_start = time.monotonic()

                if motion_routine:
                    t = time.monotonic() - routine_start
                    if t >= routine_duration:
                        break
                    if routine_pattern == "lissajous":
                        dx = routine_scale * 0.03 * np.sin(t * 0.5)
                        dy = routine_scale * 0.03 * np.cos(t * 0.5)
                        dz = routine_scale * 0.02 * np.sin(t * 0.3)
                    else:
                        size = routine_square_size * routine_scale
                        if routine_pattern == "square-xyz":
                            segment = max(routine_duration / 3.0, 0.1)
                            plane_idx = int(t / segment)
                            plane = ["xy", "xz", "yz"][min(plane_idx, 2)]
                            u = (t % segment) / segment
                        else:
                            plane = routine_plane
                            u = (t / max(routine_duration, 0.1)) % 1.0
                        desired = routine_center + plane_offset(plane, u, size)
                        delta = desired - ee_pose[:3, 3]
                        dist = float(np.linalg.norm(delta))
                        if dist < 1e-6:
                            dx = dy = dz = 0.0
                        else:
                            vel = delta / LOOP_PERIOD
                            speed = float(np.linalg.norm(vel))
                            max_speed = max(routine_square_speed, 0.001)
                            if speed > max_speed:
                                vel = vel * (max_speed / speed)
                            dx, dy, dz = vel
                    droll = routine_scale * 0.10 * np.sin(t * 0.4)
                    gripper_target = 0.5 + 0.2 * np.sin(t * 0.2)
                    gripper_target = float(np.clip(gripper_target, 0.0, 1.0))
                else:
                    state = controller.read()

                if state.a_button_pressed:
                    print("\nGoing home...", flush=True)
                    wrist_roll_deg = 0.0
                    ik_joint_pos_deg = np.zeros(4)
                    target_pitch = 0.0
                    target_yaw = 0.0
                    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)
                    gripper_pos = GRIPPER_DEFAULT

                    # Update simulation
                    sim.go_home()
                    sim.set_joint_positions(np.zeros(5), gripper_pos)
                    viewer.sync()

                    # Send to real robot
                    action = {f"{name}.pos": 0.0 for name in JOINT_NAMES[:-1]}
                    action["gripper.pos"] = gripper_to_robot(GRIPPER_DEFAULT)
                    robot.send_action(action)
                    continue

                if motion_routine:
                    ee_delta = EEDelta(dx=dx, dy=dy, dz=dz, droll=droll, gripper=gripper_target)
                else:
                    ee_delta = mapper(state)

                # Rate-limit gripper
                gripper_target = ee_delta.gripper
                gripper_diff = gripper_target - gripper_pos
                max_delta = gripper_rate * LOOP_PERIOD
                if abs(gripper_diff) > max_delta:
                    gripper_pos += max_delta if gripper_diff > 0 else -max_delta
                else:
                    gripper_pos = gripper_target

                if not ee_delta.is_zero_motion():
                    # Update target EE pose (X/Y/Z)
                    target_pos = ee_pose[:3, 3].copy()
                    target_pos[0] += ee_delta.dx * LOOP_PERIOD
                    target_pos[1] += ee_delta.dy * LOOP_PERIOD
                    target_pos[2] += ee_delta.dz * LOOP_PERIOD

                    # Workspace limits
                    target_pos[0] = np.clip(target_pos[0], -0.1, 0.5)
                    target_pos[1] = np.clip(target_pos[1], -0.3, 0.3)
                    target_pos[2] = np.clip(target_pos[2], 0.05, 0.45)

                    # Update target orientation
                    if abs(ee_delta.dpitch) > 0.001:
                        target_pitch += ee_delta.dpitch * LOOP_PERIOD
                        target_pitch = np.clip(target_pitch, -np.pi / 2, np.pi / 2)

                    if abs(ee_delta.dyaw) > 0.001:
                        target_yaw += ee_delta.dyaw * LOOP_PERIOD
                        target_yaw = np.clip(target_yaw, -np.pi, np.pi)

                    target_pose = ee_pose.copy()
                    target_pose[:3, 3] = target_pos

                    # Build target orientation if pitch/yaw are set
                    has_orientation_target = abs(target_pitch) > 0.01 or abs(target_yaw) > 0.01
                    if has_orientation_target:
                        target_rotation = euler_to_rotation_matrix(0.0, target_pitch, target_yaw)
                        target_pose[:3, :3] = target_rotation
                        orientation_weight = 0.1  # Low weight - strongly prioritize position
                    else:
                        orientation_weight = 0.0

                    # Solve IK
                    new_joints = kinematics.inverse_kinematics(
                        ik_joint_pos_deg, target_pose, position_weight=1.0, orientation_weight=orientation_weight
                    )
                    ik_joint_pos_deg = new_joints[:4]

                    # Apply wrist roll directly
                    if abs(ee_delta.droll) > 0.001:
                        roll_delta_deg = np.rad2deg(ee_delta.droll * LOOP_PERIOD)
                        wrist_roll_deg += roll_delta_deg
                        wrist_roll_deg = np.clip(wrist_roll_deg, -180.0, 180.0)

                    ee_pose = kinematics.forward_kinematics(ik_joint_pos_deg)

                full_joint_pos_deg = np.array([
                    ik_joint_pos_deg[0],
                    ik_joint_pos_deg[1],
                    ik_joint_pos_deg[2],
                    ik_joint_pos_deg[3],
                    wrist_roll_deg,
                ])

                # Update simulation (digital twin)
                sim.set_joint_positions(full_joint_pos_deg, gripper_pos)
                if routine_trace:
                    pos = sim.get_ee_position()
                    if not trace_points:
                        trace_points.append(pos.copy())
                    else:
                        if np.linalg.norm(pos - trace_points[-1]) >= trace_min_step_m:
                            trace_points.append(pos.copy())
                    if len(trace_points) > routine_trace_max:
                        trace_points = trace_points[-routine_trace_max :]
                    draw_trace(viewer, trace_points)
                viewer.sync()

                # Send to real robot
                action = {}
                for i, name in enumerate(JOINT_NAMES[:-1]):
                    action[f"{name}.pos"] = deg_to_normalized(full_joint_pos_deg[i], name)
                action["gripper.pos"] = gripper_to_robot(gripper_pos)
                robot.send_action(action)

                # Status
                pos = sim.get_ee_position()
                pitch_deg = np.rad2deg(target_pitch)
                yaw_deg = np.rad2deg(target_yaw)
                print(
                    f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                    f"P:{pitch_deg:+5.1f}° Y:{yaw_deg:+5.1f}° | Grip: {gripper_pos:.2f}   ",
                    end="\r",
                    flush=True,
                )

                elapsed = time.monotonic() - loop_start
                if elapsed < LOOP_PERIOD:
                    time.sleep(LOOP_PERIOD - elapsed)

    finally:
        if controller is not None:
            controller.disconnect()
        robot.disconnect()
        print("\nDisconnected.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Digital twin: real robot + MuJoCo simulation")
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port for robot (e.g., /dev/ttyUSB0). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--motion-routine",
        action="store_true",
        help="Run automatic motion routine (no controller input).",
    )
    parser.add_argument(
        "--routine-duration",
        type=float,
        default=15.0,
        help="Motion routine duration in seconds. Default: 15.",
    )
    parser.add_argument(
        "--routine-scale",
        type=float,
        default=1.0,
        help="Scale factor for routine motion amplitudes. Default: 1.0.",
    )
    parser.add_argument(
        "--routine-pattern",
        type=str,
        default="lissajous",
        choices=["lissajous", "square", "square-xyz"],
        help="Motion routine pattern. Default: lissajous.",
    )
    parser.add_argument(
        "--routine-plane",
        type=str,
        default="xy",
        choices=["xy", "xz", "yz"],
        help="Plane for square routine. Default: xy.",
    )
    parser.add_argument(
        "--routine-square-size",
        type=float,
        default=0.06,
        help="Square side length in meters. Default: 0.06.",
    )
    parser.add_argument(
        "--routine-square-speed",
        type=float,
        default=0.03,
        help="Max tracking speed for square routine (m/s). Default: 0.03.",
    )
    parser.add_argument(
        "--routine-square-period",
        type=float,
        default=None,
        help="Seconds per square trace (overrides square speed if set).",
    )
    parser.add_argument(
        "--routine-trace",
        action="store_true",
        help="Draw a virtual pen trace of the end effector in the sim.",
    )
    parser.add_argument(
        "--routine-trace-max",
        type=int,
        default=300,
        help="Max trace points to draw. Default: 300.",
    )
    parser.add_argument(
        "--routine-trace-step-mm",
        type=float,
        default=2.0,
        help="Minimum distance between trace points (mm). Default: 2.0.",
    )
    args = parser.parse_args()

    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found at {URDF_PATH}")
        sys.exit(1)

    port = args.port
    if port is None:
        port = find_serial_port()
        if port is None:
            print("ERROR: No serial port found. Connect the robot and try again.")
            print("  Or specify port manually: --port /dev/ttyUSB0")
            sys.exit(1)
        print(f"Auto-detected port: {port}")

    if args.routine_square_period is not None:
        if args.routine_square_period <= 0:
            print("ERROR: --routine-square-period must be > 0.")
            sys.exit(1)
        perimeter = 4.0 * max(args.routine_square_size, 0.001)
        args.routine_square_speed = perimeter / args.routine_square_period

    run_dual_mode(
        port,
        motion_routine=args.motion_routine,
        routine_duration=args.routine_duration,
        routine_scale=args.routine_scale,
        routine_pattern=args.routine_pattern,
        routine_plane=args.routine_plane,
        routine_square_size=args.routine_square_size,
        routine_square_speed=args.routine_square_speed,
        routine_trace=args.routine_trace,
        routine_trace_max=args.routine_trace_max,
        routine_trace_step_mm=args.routine_trace_step_mm,
    )


if __name__ == "__main__":
    main()
