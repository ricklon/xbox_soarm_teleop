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
    - Left stick: X/Y movement in horizontal plane
    - Right stick Y: Z movement (up/down)
    - Right stick X: Wrist roll rotation
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
    """Simulated SO-ARM101 with meshcat visualization."""

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

        # Get joint indices
        self.joint_ids = [self.model.getJointId(name) for name in JOINT_NAMES]

        # Get end effector frame ID
        self.ee_frame_id = self.model.getFrameId(EE_FRAME)

        # Initialize joint positions (degrees for consistency with LeRobot)
        self.joint_pos_deg = np.zeros(len(JOINT_NAMES))

        # Gripper position (0-1)
        self.gripper_pos = 0.0

        # Current EE pose (4x4 transform)
        self.ee_pose = None
        self._update_ee_pose()

        # Create IK solver (reused for all IK calls)
        self.kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=EE_FRAME,
            joint_names=JOINT_NAMES,
        )

        # Create visualizer
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        self.viz.initViewer(open=False)
        self.viz.loadViewerModel()
        self.port = 7000  # Default meshcat port

    def _update_ee_pose(self) -> None:
        """Update end effector pose from current joint positions."""
        import pinocchio as pin

        q = self._joints_to_q()
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        self.ee_pose = self.data.oMf[self.ee_frame_id].homogeneous

    def _joints_to_q(self) -> np.ndarray:
        """Convert joint positions to pinocchio q vector."""
        q = np.zeros(self.model.nq)
        joint_pos_rad = np.deg2rad(self.joint_pos_deg)
        for i, jid in enumerate(self.joint_ids):
            idx = self.model.joints[jid].idx_q
            q[idx] = joint_pos_rad[i]
        # Set gripper: map 0-1 to joint limits (open=1.74, closed=-0.17)
        # gripper_pos=0 (not pulled) -> open (1.74)
        # gripper_pos=1 (fully pulled) -> closed (-0.17)
        gripper_jid = self.model.getJointId("gripper")
        gripper_idx = self.model.joints[gripper_jid].idx_q
        gripper_open = 1.74533  # Upper limit (open)
        gripper_closed = -0.174533  # Lower limit (closed)
        q[gripper_idx] = gripper_open - self.gripper_pos * (gripper_open - gripper_closed)
        return q

    def get_ee_position(self) -> np.ndarray:
        """Get current end effector position."""
        return self.ee_pose[:3, 3].copy()

    def apply_ee_delta(self, dx: float, dy: float, dz: float, droll: float, dt: float) -> bool:
        """Apply end effector delta and solve IK.

        Args:
            dx: X velocity (m/s)
            dy: Y velocity (m/s)
            dz: Z velocity (m/s)
            droll: Roll velocity (rad/s)
            dt: Time step (seconds)

        Returns:
            True if IK succeeded, False otherwise.
        """
        # Compute target position
        target_pos = self.get_ee_position()
        target_pos[0] += dx * dt
        target_pos[1] += dy * dt
        target_pos[2] += dz * dt

        # Apply workspace limits
        target_pos[0] = np.clip(target_pos[0], *WORKSPACE_LIMITS["x"])
        target_pos[1] = np.clip(target_pos[1], *WORKSPACE_LIMITS["y"])
        target_pos[2] = np.clip(target_pos[2], *WORKSPACE_LIMITS["z"])

        # Create target pose (position only, IK handles arm joints)
        target_pose = self.ee_pose.copy()
        target_pose[:3, 3] = target_pos

        # Solve IK for position (arm joints only, excluding wrist_roll)
        new_joints = self.kinematics.inverse_kinematics(
            self.joint_pos_deg,
            target_pose,
            position_weight=1.0,
            orientation_weight=0.0,  # Ignore orientation, we control roll directly
        )

        self.joint_pos_deg = new_joints[: len(JOINT_NAMES)]

        # Apply wrist roll directly (index 4 = wrist_roll)
        if abs(droll) > 0.001:
            roll_delta_deg = np.rad2deg(droll * dt)
            self.joint_pos_deg[4] += roll_delta_deg
            # Clamp wrist roll to reasonable limits
            self.joint_pos_deg[4] = np.clip(self.joint_pos_deg[4], -180.0, 180.0)

        self._update_ee_pose()
        return True

    def set_gripper(self, position: float) -> None:
        """Set gripper position (0=open, 1=closed)."""
        self.gripper_pos = np.clip(position, 0.0, 1.0)

    def go_home(self) -> None:
        """Move to home position."""
        self.joint_pos_deg = np.zeros(len(JOINT_NAMES))
        self.gripper_pos = 0.0
        self._update_ee_pose()

    def update_visualization(self) -> None:
        """Update meshcat visualization."""
        q = self._joints_to_q()
        self.viz.display(q)


def run_with_controller(sim: ArmSimulator):
    """Run simulation with Xbox controller input."""
    from xbox_soarm_teleop.config.xbox_config import XboxConfig
    from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta
    from xbox_soarm_teleop.teleoperators.xbox import XboxController

    config = XboxConfig()
    controller = XboxController(config)
    mapper = MapXboxToEEDelta(
        linear_scale=config.linear_scale,
        angular_scale=config.angular_scale,
    )
    gripper_rate = config.gripper_rate  # Position change per second

    if not controller.connect():
        print("ERROR: Failed to connect to Xbox controller")
        print("  - Check that controller is connected and powered on")
        print('  - Run: uv run python -c "import inputs; print(inputs.devices.gamepads)"')
        print("  - Or use --no-controller for demo mode")
        sys.exit(1)

    print("Xbox controller connected", flush=True)
    print("\nSimulation started!", flush=True)
    print(f"  Open http://127.0.0.1:{sim.port}/static/ in your browser", flush=True)
    print("  Hold LB (left bumper) to enable arm movement", flush=True)
    print("  Press A to return to home position", flush=True)
    print("  Press Ctrl+C to exit\n", flush=True)

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
                sim.apply_ee_delta(
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
                f"Gripper: {sim.gripper_pos:.2f} | "
                f"Joints: {np.round(sim.joint_pos_deg, 1)}   ",
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

            # Demo pattern: circular motion
            dx = 0.05 * np.sin(t * 0.5)
            dy = 0.05 * np.cos(t * 0.5)
            dz = 0.02 * np.sin(t * 0.3)
            droll = 0.2 * np.sin(t * 0.4)

            sim.apply_ee_delta(dx, dy, dz, droll, LOOP_PERIOD)
            sim.set_gripper(0.5 + 0.5 * np.sin(t * 0.2))
            sim.update_visualization()

            pos = sim.get_ee_position()
            print(
                f"EE: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f} | "
                f"Gripper: {sim.gripper_pos:.2f} | "
                f"Joints: {np.round(sim.joint_pos_deg, 1)}   ",
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
        run_with_controller(sim)


if __name__ == "__main__":
    main()
