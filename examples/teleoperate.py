#!/usr/bin/env python3
"""Main teleoperation control loop for SO-ARM with Xbox controller.

This example demonstrates the full teleoperation pipeline:
1. Read Xbox controller input
2. Map to end effector delta commands
3. Apply safety limits and IK
4. Send commands to the robot

Usage:
    python examples/teleoperate.py

Requirements:
    - Xbox controller connected
    - SO-ARM101 connected via USB
    - LeRobot installed with kinematics extra
"""

import signal
import sys
import time
from typing import NoReturn

from xbox_soarm_teleop.config.xbox_config import XboxConfig
from xbox_soarm_teleop.processors.xbox_to_ee import MapXboxToEEDelta
from xbox_soarm_teleop.teleoperators.xbox import XboxController

# Control loop rate (Hz)
CONTROL_RATE = 50
LOOP_PERIOD = 1.0 / CONTROL_RATE


class TeleoperationController:
    """Main teleoperation controller integrating Xbox input with robot control."""

    def __init__(
        self,
        config: XboxConfig | None = None,
        linear_scale: float = 0.1,
        angular_scale: float = 0.5,
        use_robot: bool = True,
    ):
        """Initialize teleoperation controller.

        Args:
            config: Xbox controller configuration.
            linear_scale: Maximum linear velocity (m/s).
            angular_scale: Maximum angular velocity (rad/s).
            use_robot: If True, connect to real robot. If False, run in simulation mode.
        """
        self.config = config or XboxConfig()
        self.controller = XboxController(self.config)
        self.mapper = MapXboxToEEDelta(
            linear_scale=linear_scale,
            angular_scale=angular_scale,
        )
        self.use_robot = use_robot
        self.robot = None
        self.running = False

        # State tracking
        self.world_frame = True  # vs tool frame

    def connect(self) -> bool:
        """Connect to Xbox controller and optionally robot.

        Returns:
            True if all connections successful.
        """
        # Connect to Xbox controller
        if not self.controller.connect():
            print("ERROR: Failed to connect to Xbox controller")
            print("  - Check that controller is connected and powered on")
            print('  - Run: python -c "import inputs; print(inputs.devices.gamepads)"')
            return False
        print("Xbox controller connected", flush=True)

        # Connect to robot if enabled
        if self.use_robot:
            try:
                # TODO: Integrate with LeRobot robot interface
                # from lerobot.robots.so101 import SO101Follower
                # self.robot = SO101Follower()
                # self.robot.connect()
                print("WARNING: Robot connection not yet implemented")
                print("  Running in simulation mode")
                self.use_robot = False
            except ImportError:
                print("WARNING: LeRobot not installed, running in simulation mode")
                self.use_robot = False
            except Exception as e:
                print(f"WARNING: Failed to connect to robot: {e}")
                print("  Running in simulation mode")
                self.use_robot = False

        return True

    def disconnect(self) -> None:
        """Disconnect from controller and robot."""
        self.controller.disconnect()
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:
                pass
        print("Disconnected")

    def run(self) -> NoReturn:
        """Run the main control loop.

        This method runs until interrupted by Ctrl+C or a fatal error.
        """
        self.running = True
        print("\nTeleoperation started")
        print("  Hold LB (left bumper) to enable arm movement")
        print("  Press A to return to home position")
        print("  Press Y to toggle world/tool frame")
        print("  Press Ctrl+C to exit\n", flush=True)

        try:
            while self.running:
                loop_start = time.monotonic()

                # Read controller state
                state = self.controller.read()

                # Check for special buttons
                if state.a_button_pressed:
                    self._go_home()
                    continue

                if state.y_button_pressed:
                    self._toggle_frame()
                    continue

                # Map to end effector delta
                ee_delta = self.mapper(state)

                # Apply to robot (or print in simulation mode)
                self._apply_delta(ee_delta)

                # Maintain loop rate
                elapsed = time.monotonic() - loop_start
                sleep_time = LOOP_PERIOD - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nShutdown requested")
        finally:
            self.running = False

    def _go_home(self) -> None:
        """Move robot to home position."""
        print("Moving to home position...")
        if self.robot is not None:
            try:
                # TODO: Implement home position command
                # self.robot.go_home()
                pass
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("  (simulation mode - no robot connected)")

    def _toggle_frame(self) -> None:
        """Toggle between world and tool coordinate frames."""
        self.world_frame = not self.world_frame
        frame_name = "world" if self.world_frame else "tool"
        print(f"Switched to {frame_name} frame")

    def _apply_delta(self, ee_delta) -> None:
        """Apply end effector delta to robot.

        Args:
            ee_delta: End effector delta command.
        """
        if self.robot is not None:
            # TODO: Integrate with LeRobot processor pipeline
            # This would involve:
            # 1. EEReferenceAndDelta - latch reference, accumulate delta
            # 2. EEBoundsAndSafety - enforce workspace limits
            # 3. InverseKinematicsEEToJoints - compute joint angles
            # 4. Send joint commands to robot
            pass
        else:
            # Simulation mode - print non-zero commands
            if not ee_delta.is_zero_motion() or ee_delta.gripper > 0.01:
                print(
                    f"EE Delta: dx={ee_delta.dx:+.3f} dy={ee_delta.dy:+.3f} "
                    f"dz={ee_delta.dz:+.3f} droll={ee_delta.droll:+.3f} "
                    f"gripper={ee_delta.gripper:.2f}",
                    flush=True,
                )


def main():
    """Main entry point for teleoperation."""
    # Setup signal handler for graceful shutdown
    controller = None

    def signal_handler(sig, frame):
        print("\nInterrupt received, shutting down...")
        if controller is not None:
            controller.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse command line arguments (simple implementation)
    use_robot = "--sim" not in sys.argv

    # Create and run controller
    config = XboxConfig()
    controller = TeleoperationController(
        config=config,
        linear_scale=config.linear_scale,
        angular_scale=config.angular_scale,
        use_robot=use_robot,
    )

    if not controller.connect():
        sys.exit(1)

    try:
        controller.run()
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
