#!/usr/bin/env python3
"""Simulated teleoperation with 3D visualization.

This example runs the Xbox controller teleoperation in simulation,
visualizing the SO-ARM101 in a web browser using meshcat.

Usage:
    uv run python examples/simulate.py

    Then open http://127.0.0.1:7000/static/ in your browser.

Options:
    --no-controller    Run without Xbox controller (demo mode)

Controls (Xbox):
    - Hold LB (left bumper) to enable arm movement
    - Left stick X: Rotate base (turret)
    - Left stick Y: Up/down (Z axis)
    - Right stick Y: Forward/back (X axis)
    - Right stick X: Wrist roll rotation (direct, not IK)
    - Right trigger: Gripper position
    - A button: Return to home position
    - Ctrl+C: Exit

Controls (Keyboard demo, --no-controller):
    - Automatic demo movement pattern
    - Ctrl+C: Exit
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Path to URDF with absolute mesh paths
URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"

# Joint configuration
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
EE_FRAME = "gripper_frame_link"

# Control loop rate
CONTROL_RATE = 30  # Hz
LOOP_PERIOD = 1.0 / CONTROL_RATE

# Workspace limits (meters)
WORKSPACE_LIMITS = {
    "x": (-0.1, 0.5),
    "y": (-0.3, 0.3),
    "z": (0.05, 0.45),
}


class ArmSimulator:
    """Simulated SO-ARM101 with meshcat visualization.

    Uses direct base control (not IK) for smooth turret-style rotation.
    IK is used for the base and 3 arm joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex).
    """

    # IK joint names (including base, excluding wrist_roll)
    IK_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

    def __init__(self, urdf_path: str):
        """Initialize the simulator.

        Args:
            urdf_path: Path to the robot URDF file.
        """
        import pinocchio as pin
        from lerobot.model.kinematics import RobotKinematics
        from pinocchio.visualize import MeshcatVisualizer

        self.urdf_path = urdf_path

        # Load robot model for visualization
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, package_dirs=[str(Path(urdf_path).parent)]
        )
        self.data = self.model.createData()

        # Get joint indices for all joints (for visualization)
        self.joint_ids = [self.model.getJointId(name) for name in JOINT_NAMES]

        # Get end effector frame ID
        self.ee_frame_id = self.model.getFrameId(EE_FRAME)

        # IK joint positions (4 joints) in degrees
        self.ik_joint_pos_deg = np.zeros(len(self.IK_JOINT_NAMES))
        self.wrist_roll_deg = 0.0

        # Gripper position (0-1)
        self.gripper_pos = 0.0

        # Current EE pose (4x4 transform) - in arm plane
        self.ee_pose = None

        # Create IK solver for 4 joints (excluding wrist_roll)
        self.kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=EE_FRAME,
            joint_names=self.IK_JOINT_NAMES,
        )

        self._update_ee_pose()

        # Joint velocity limits for IK joints (4 joints)
        self.ik_joint_vel_limits = np.array([120.0, 90.0, 90.0, 90.0])

        # Create visualizer
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(open=False)
        self.viz.loadViewerModel()
        self.port = 7000  # Default meshcat port

    def _update_ee_pose(self) -> None:
        """Update end effector pose from IK joint positions."""
        self.ee_pose = self.kinematics.forward_kinematics(self.ik_joint_pos_deg)

    def _get_full_joint_pos_deg(self) -> np.ndarray:
        """Get full 5-joint position (base + IK joints) in degrees."""
        return np.array([
            self.ik_joint_pos_deg[0],  # shoulder_pan
            self.ik_joint_pos_deg[1],  # shoulder_lift
            self.ik_joint_pos_deg[2],  # elbow_flex
            self.ik_joint_pos_deg[3],  # wrist_flex
            self.wrist_roll_deg,  # wrist_roll
        ])

    def _joints_to_q(self) -> np.ndarray:
        """Convert joint positions to pinocchio q vector for visualization."""
        q = np.zeros(self.model.nq)
        full_joint_pos_deg = self._get_full_joint_pos_deg()
        joint_pos_rad = np.deg2rad(full_joint_pos_deg)
        for i, jid in enumerate(self.joint_ids):
            idx = self.model.joints[jid].idx_q
            q[idx] = joint_pos_rad[i]
        # Set gripper: map 0-1 to joint limits (open=1.74, closed=-0.17)
        gripper_jid = self.model.getJointId("gripper")
        gripper_idx = self.model.joints[gripper_jid].idx_q
        gripper_open = 1.74533  # Upper limit (open)
        gripper_closed = -0.174533  # Lower limit (closed)
        q[gripper_idx] = gripper_open - self.gripper_pos * (gripper_open - gripper_closed)
        return q

    def get_ee_position(self) -> np.ndarray:
        """Get current end effector position in arm plane."""
        return self.ee_pose[:3, 3].copy()

    def apply_delta(self, dx: float, dy: float, dz: float, droll: float, dt: float) -> bool:
        """Apply movement delta with IK for base and arm.

        Args:
            dx: X velocity (m/s) - forward/back
            dy: Y velocity (m/s) - left/right
            dz: Z velocity (m/s) - up/down
            droll: Wrist roll velocity (rad/s)
            dt: Time step (seconds)

        Returns:
            True if successful.
        """
        # Update target EE pose (X/Y/Z)
        target_pos = self.get_ee_position()
        target_pos[0] += dx * dt  # Forward/back
        target_pos[1] += dy * dt  # Left/right
        target_pos[2] += dz * dt  # Up/down

        # Workspace limits
        target_pos[0] = np.clip(target_pos[0], 0.05, 0.5)  # Min reach to avoid singularity
        target_pos[1] = np.clip(target_pos[1], *WORKSPACE_LIMITS["y"])
        target_pos[2] = np.clip(target_pos[2], *WORKSPACE_LIMITS["z"])

        target_pose = self.ee_pose.copy()
        target_pose[:3, 3] = target_pos

        # Solve IK for 4 joints (position only)
        new_joints = self.kinematics.inverse_kinematics(
            self.ik_joint_pos_deg,
            target_pose,
            position_weight=1.0,
            orientation_weight=0.0,
        )
        ik_result = new_joints[:4]

        # Apply joint velocity limiting to smooth IK output
        max_delta = self.ik_joint_vel_limits * dt
        joint_delta = ik_result - self.ik_joint_pos_deg
        joint_delta = np.clip(joint_delta, -max_delta, max_delta)
        self.ik_joint_pos_deg = self.ik_joint_pos_deg + joint_delta

        # Apply wrist roll directly (not part of IK)
        if abs(droll) > 0.001:
            roll_delta_deg = np.rad2deg(droll * dt)
            self.wrist_roll_deg += roll_delta_deg
            self.wrist_roll_deg = np.clip(self.wrist_roll_deg, -180.0, 180.0)

        self._update_ee_pose()
        return True

    def set_gripper(self, position: float) -> None:
        """Set gripper position (0=open, 1=closed)."""
        self.gripper_pos = np.clip(position, 0.0, 1.0)

    def go_home(self) -> None:
        """Move to home position."""
        self.ik_joint_pos_deg = np.zeros(len(self.IK_JOINT_NAMES))
        self.wrist_roll_deg = 0.0
        self.gripper_pos = 0.0
        self._update_ee_pose()

    def update_visualization(self) -> None:
        """Update meshcat visualization."""
        q = self._joints_to_q()
        self.viz.display(q)


def run_with_controller(
    sim: ArmSimulator, deadzone: float = 0.15, linear_scale: float | None = None
):
    """Run simulation with Xbox controller input."""
    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    config = XboxConfig(deadzone=deadzone)
    if linear_scale is not None:
        config.linear_scale = linear_scale
    controller = XboxController(config)
    mapper = MapXboxToEEDelta(
        linear_scale=config.linear_scale,
        angular_scale=config.angular_scale,
    )
    gripper_rate = config.gripper_rate  # Position change per second

    print(f"Controller deadzone: {config.deadzone}", flush=True)
    print(f"Linear scale: {config.linear_scale} m/s", flush=True)

    if not controller.connect():
        print("ERROR: Failed to connect to Xbox controller")
        print("  - Check that controller is connected and powered on")
        print('  - Run: uv run python -c "import inputs; print(inputs.devices.gamepads)"')
        print("  - Or use --no-controller for demo mode")
        sys.exit(1)

    print("Xbox controller connected", flush=True)
    print("\nSimulation started!", flush=True)
    print(f"  Open http://127.0.0.1:{sim.port}/static/ in your browser", flush=True)
    print("\nControls:", flush=True)
    print("  Hold LB + move sticks to control arm", flush=True)
    print("  Left stick X: Move left/right (Y axis)", flush=True)
    print("  Left stick Y: Move up/down", flush=True)
    print("  Right stick Y: Move forward/back", flush=True)
    print("  Right stick X: Wrist roll", flush=True)
    print("  Right trigger: Gripper", flush=True)
    print("  A button: Return to home", flush=True)
    print("  Ctrl+C: Exit\n", flush=True)

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...", flush=True)
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running:
            loop_start = time.monotonic()

            state = controller.read()

            if state.a_button_pressed:
                print("\nMoving to home position...", flush=True)
                sim.go_home()
                sim.update_visualization()
                continue

            ee_delta = mapper(state)
            # Rate-limit gripper movement toward target
            gripper_target = ee_delta.gripper
            gripper_diff = gripper_target - sim.gripper_pos
            max_delta = gripper_rate * LOOP_PERIOD
            if abs(gripper_diff) > max_delta:
                sim.set_gripper(sim.gripper_pos + (max_delta if gripper_diff > 0 else -max_delta))
            else:
                sim.set_gripper(gripper_target)

            if not ee_delta.is_zero_motion():
                sim.apply_delta(
                    ee_delta.dx,
                    ee_delta.dy,
                    ee_delta.dz,
                    ee_delta.droll,
                    LOOP_PERIOD,
                )

            sim.update_visualization()

            pos = sim.get_ee_position()
            print(
                f"EE: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f} | "
                f"Base: {sim.ik_joint_pos_deg[0]:+6.1f}° | "
                f"Gripper: {sim.gripper_pos:.2f}   ",
                end="\r",
                flush=True,
            )

            elapsed = time.monotonic() - loop_start
            sleep_time = LOOP_PERIOD - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        controller.disconnect()
        print("\nDisconnected.", flush=True)


def run_demo_mode(sim: ArmSimulator):
    """Run simulation in demo mode (no controller required)."""
    print("\nDemo mode - automatic movement pattern", flush=True)
    print(f"  Open http://127.0.0.1:{sim.port}/static/ in your browser", flush=True)
    print("  Press Ctrl+C to exit\n", flush=True)

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...", flush=True)
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    t = 0.0
    try:
        while running:
            loop_start = time.monotonic()

            # Demo pattern: XYZ motion + wrist roll
            dx = 0.03 * np.sin(t * 0.5)
            dy = 0.03 * np.cos(t * 0.5)
            dz = 0.02 * np.sin(t * 0.3)
            droll = 0.2 * np.sin(t * 0.4)

            sim.apply_delta(dx, dy, dz, droll, LOOP_PERIOD)
            sim.set_gripper(0.5 + 0.5 * np.sin(t * 0.2))
            sim.update_visualization()

            pos = sim.get_ee_position()
            print(
                f"EE: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f} | "
                f"Base: {sim.ik_joint_pos_deg[0]:+6.1f}° | "
                f"Gripper: {sim.gripper_pos:.2f}   ",
                end="\r",
                flush=True,
            )

            t += LOOP_PERIOD

            elapsed = time.monotonic() - loop_start
            sleep_time = LOOP_PERIOD - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nDone.", flush=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SO-ARM101 simulation with Xbox controller")
    parser.add_argument(
        "--no-controller",
        action="store_true",
        help="Run in demo mode without Xbox controller",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.15,
        help="Controller deadzone (0.0-1.0). Default: 0.15",
    )
    parser.add_argument(
        "--linear-scale",
        type=float,
        default=None,
        help="Linear velocity scale (m/s). Default from config.",
    )
    args = parser.parse_args()

    # Check URDF exists
    if not URDF_PATH.exists():
        print(f"ERROR: URDF not found at {URDF_PATH}")
        print("The URDF and mesh files should be in the assets/ directory.")
        print("If missing, they were downloaded during IK test setup.")
        sys.exit(1)

    # Initialize simulator
    print("Loading robot model...", flush=True)
    sim = ArmSimulator(str(URDF_PATH))
    sim.update_visualization()
    print("Robot model loaded!", flush=True)

    if args.no_controller:
        run_demo_mode(sim)
    else:
        run_with_controller(sim, deadzone=args.deadzone, linear_scale=args.linear_scale)


if __name__ == "__main__":
    main()
