"""Direct joint diagnostic using an Xbox controller, bypassing IK."""

from __future__ import annotations

import argparse
import csv
import glob
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    JOINT_LIMITS_DEG,
    JOINT_NAMES_WITH_GRIPPER,
    MOTOR_IDS,
    SWEEP_TEST_POSITIONS,
    deg_to_raw,
    raw_to_deg,
)
from xbox_soarm_teleop.config.xbox_config import XboxConfig
from xbox_soarm_teleop.diagnostics.xbox_joint_drive import (
    advance_goal,
    dpad_edge,
    map_trigger_to_gripper_deg,
)
from xbox_soarm_teleop.teleoperators.xbox import XboxController

CONTROL_HZ = 30.0


@dataclass
class SweepState:
    active: bool = False
    direction: int = 1


def find_serial_port() -> str | None:
    """Find available serial port for the robot."""
    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def read_voltage_v(bus, motor_key: str) -> float | None:
    """Read servo voltage in volts from whichever register is available."""
    for register in ("Present_Voltage", "Present_Input_Voltage"):
        try:
            raw = bus.read(register, motor_key, normalize=False)
            return float(raw) / 10.0
        except Exception:
            continue
    return None


def read_position_deg(bus, joint_name: str) -> float:
    """Read a joint position in degrees."""
    motor_key = str(MOTOR_IDS[joint_name])
    pos_raw = int(bus.read("Present_Position", motor_key, normalize=False))
    return float(raw_to_deg(pos_raw))


def write_goal_deg(bus, joint_name: str, goal_deg: float) -> int:
    """Write a degree target to a joint and return raw goal."""
    lower, upper = JOINT_LIMITS_DEG[joint_name]
    clamped = max(lower, min(upper, goal_deg))
    goal_raw = max(0, min(4095, deg_to_raw(clamped)))
    motor_key = str(MOTOR_IDS[joint_name])
    bus.write("Goal_Position", motor_key, goal_raw, normalize=False)
    return goal_raw


def build_base_positions(joint_name: str, use_test_positions: bool = True) -> dict[str, float]:
    """Build per-joint base pose for safe full-range motion tests."""
    base = {name: HOME_POSITION_DEG.get(name, 0.0) for name in JOINT_NAMES_WITH_GRIPPER}
    if use_test_positions and joint_name in SWEEP_TEST_POSITIONS:
        base.update(SWEEP_TEST_POSITIONS[joint_name])
    return base


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Xbox direct-joint diagnostic for SO-ARM")
    parser.add_argument("--port", type=str, default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--deadzone", type=float, default=0.15, help="Controller deadzone")
    parser.add_argument(
        "--max-vel",
        type=float,
        default=70.0,
        help="Max selected-joint command speed in deg/s at full stick",
    )
    parser.add_argument(
        "--sweep-vel",
        type=float,
        default=35.0,
        help="Auto sweep speed in deg/s",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=CONTROL_HZ,
        help=f"Control loop frequency in Hz (default: {CONTROL_HZ})",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="CSV log path (default: ./joint_diag_YYYYMMDD_HHMMSS.csv)",
    )
    parser.add_argument(
        "--no-test-positions",
        action="store_true",
        help="Disable per-joint base poses and always test from home pose.",
    )
    parser.add_argument(
        "--gripper-trigger-control",
        action="store_true",
        help="When gripper is selected, map trigger directly to gripper position.",
    )
    return parser


def main() -> int:
    """Run the Xbox joint-diagnostic CLI."""
    args = build_parser().parse_args()

    if args.port is None:
        args.port = find_serial_port()
    if args.port is None:
        print("ERROR: No serial port found. Pass --port explicitly.")
        return 1

    loop_hz = max(5.0, float(args.hz))
    loop_dt = 1.0 / loop_hz
    max_vel = max(1.0, float(args.max_vel))
    sweep_vel = max(1.0, float(args.sweep_vel))

    if args.log is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log = f"joint_diag_{stamp}.csv"
    log_path = Path(args.log)

    try:
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode
    except ImportError as exc:
        print(f"Import error: {exc}")
        print("Install dependencies with: uv pip install -e '.[dev]'")
        return 1

    print(f"Connecting to robot on {args.port}...")
    motors = {
        str(mid): Motor(id=mid, model="sts3215", norm_mode=MotorNormMode.DEGREES)
        for mid in MOTOR_IDS.values()
    }
    bus = FeetechMotorsBus(port=args.port, motors=motors)

    try:
        bus.connect()
    except Exception as exc:
        print(f"ERROR: Failed to connect to bus: {exc}")
        return 1

    for joint_name in JOINT_NAMES_WITH_GRIPPER:
        bus.write("Torque_Enable", str(MOTOR_IDS[joint_name]), 1, normalize=False)

    config = XboxConfig(deadzone=args.deadzone)
    controller = XboxController(config)
    if not controller.connect():
        print("ERROR: Xbox controller not detected")
        bus.disconnect()
        return 1

    goals_deg: dict[str, float] = {}
    for joint_name in JOINT_NAMES_WITH_GRIPPER:
        try:
            goals_deg[joint_name] = read_position_deg(bus, joint_name)
        except Exception:
            goals_deg[joint_name] = HOME_POSITION_DEG[joint_name]

    selected_idx = 0
    selected_joint = JOINT_NAMES_WITH_GRIPPER[selected_idx]
    prev_dpad_x = 0.0
    sweep = SweepState()
    use_test_positions = not args.no_test_positions
    gripper_faulted = False

    def apply_base_pose_for_selected() -> None:
        nonlocal gripper_faulted
        base = build_base_positions(selected_joint, use_test_positions=use_test_positions)
        for joint_name, base_deg in base.items():
            goals_deg[joint_name] = base_deg
        for joint_name in JOINT_NAMES_WITH_GRIPPER:
            if gripper_faulted and joint_name == "gripper":
                continue
            try:
                write_goal_deg(bus, joint_name, goals_deg[joint_name])
            except RuntimeError as exc:
                if joint_name == "gripper" and "Overload" in str(exc):
                    gripper_faulted = True
                    print("\nWARNING: Gripper overload while setting base pose. Gripper commands disabled.")
                    continue
                raise
        time.sleep(0.25)

    apply_base_pose_for_selected()

    t0 = time.monotonic()
    running = True

    def handle_signal(_sig, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="") as csv_file:
        fieldnames = [
            "t_s",
            "selected_joint",
            "deadman",
            "left_stick_x",
            "cmd_vel_deg_s",
            "selected_goal_deg",
            "selected_goal_raw",
            "selected_pos_raw",
            "selected_pos_deg",
            "selected_vel_raw",
            "selected_load_raw",
            "selected_current_raw",
            "selected_temp_c",
            "selected_voltage_v",
            "sweep_active",
            "sweep_direction",
            "gripper_trigger",
        ] + [f"{name}_pos_deg" for name in JOINT_NAMES_WITH_GRIPPER]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        print("\nDirect Joint Diagnostic Running")
        print(f"Logging: {log_path}")
        print("Controls:")
        print("  LB hold: enable motion (deadman)")
        print("  Left stick X: selected joint velocity")
        print("    - right (+X): toward upper joint limit")
        print("    - left  (-X): toward lower joint limit")
        print("  D-pad left/right: select previous/next joint")
        print("  Y press: toggle auto-sweep for selected joint")
        print("  A press: reset selected joint to its base pose")
        print("  Right trigger: optional gripper control when using --gripper-trigger-control")

        while running:
            loop_start = time.monotonic()
            state = controller.read()

            edge = dpad_edge(state.dpad_x, prev_dpad_x)
            prev_dpad_x = state.dpad_x
            if edge != 0:
                selected_idx = (selected_idx + edge) % len(JOINT_NAMES_WITH_GRIPPER)
                selected_joint = JOINT_NAMES_WITH_GRIPPER[selected_idx]
                sweep.active = False
                apply_base_pose_for_selected()

            if state.y_button_pressed:
                sweep.active = not sweep.active
                sweep.direction = 1

            if state.a_button_pressed:
                goals_deg.update(build_base_positions(selected_joint, use_test_positions=use_test_positions))
                sweep.active = False

            cmd_vel = 0.0
            if state.left_bumper:
                if sweep.active:
                    lower, upper = JOINT_LIMITS_DEG[selected_joint]
                    at_upper = goals_deg[selected_joint] >= (upper - 0.5)
                    at_lower = goals_deg[selected_joint] <= (lower + 0.5)
                    if sweep.direction > 0 and at_upper:
                        sweep.direction = -1
                    elif sweep.direction < 0 and at_lower:
                        sweep.direction = 1
                    cmd_vel = sweep.direction * sweep_vel
                else:
                    cmd_vel = state.left_stick_x * max_vel

                lower, upper = JOINT_LIMITS_DEG[selected_joint]
                goals_deg[selected_joint] = advance_goal(
                    goals_deg[selected_joint],
                    cmd_vel,
                    loop_dt,
                    lower,
                    upper,
                )

            if args.gripper_trigger_control and selected_joint == "gripper" and not gripper_faulted:
                g_lower, g_upper = JOINT_LIMITS_DEG["gripper"]
                goals_deg["gripper"] = map_trigger_to_gripper_deg(state.right_trigger, g_lower, g_upper)

            try:
                selected_goal_raw = write_goal_deg(bus, selected_joint, goals_deg[selected_joint])
            except RuntimeError as exc:
                if selected_joint == "gripper" and "Overload" in str(exc):
                    gripper_faulted = True
                    sweep.active = False
                    selected_goal_raw = deg_to_raw(goals_deg[selected_joint])
                    print("\nWARNING: Gripper overload detected. Gripper commands disabled for this run.")
                else:
                    raise

            selected_key = str(MOTOR_IDS[selected_joint])
            selected_pos_raw = int(bus.read("Present_Position", selected_key, normalize=False))
            selected_vel_raw = int(bus.read("Present_Velocity", selected_key, normalize=False))
            selected_load_raw = int(bus.read("Present_Load", selected_key, normalize=False))
            selected_current_raw = int(bus.read("Present_Current", selected_key, normalize=False))
            selected_temp_c = int(bus.read("Present_Temperature", selected_key, normalize=False))
            selected_voltage_v = read_voltage_v(bus, selected_key)
            selected_pos_deg = float(raw_to_deg(selected_pos_raw))

            all_positions_deg: dict[str, float] = {}
            for joint_name in JOINT_NAMES_WITH_GRIPPER:
                try:
                    all_positions_deg[joint_name] = read_position_deg(bus, joint_name)
                except Exception:
                    all_positions_deg[joint_name] = float("nan")

            row = {
                "t_s": f"{time.monotonic() - t0:.3f}",
                "selected_joint": selected_joint,
                "deadman": int(state.left_bumper),
                "left_stick_x": f"{state.left_stick_x:.4f}",
                "cmd_vel_deg_s": f"{cmd_vel:.3f}",
                "selected_goal_deg": f"{goals_deg[selected_joint]:.3f}",
                "selected_goal_raw": selected_goal_raw,
                "selected_pos_raw": selected_pos_raw,
                "selected_pos_deg": f"{selected_pos_deg:.3f}",
                "selected_vel_raw": selected_vel_raw,
                "selected_load_raw": selected_load_raw,
                "selected_current_raw": selected_current_raw,
                "selected_temp_c": selected_temp_c,
                "selected_voltage_v": ""
                if selected_voltage_v is None
                else f"{selected_voltage_v:.2f}",
                "sweep_active": int(sweep.active),
                "sweep_direction": sweep.direction,
                "gripper_trigger": f"{state.right_trigger:.3f}",
            }
            for joint_name in JOINT_NAMES_WITH_GRIPPER:
                row[f"{joint_name}_pos_deg"] = f"{all_positions_deg[joint_name]:.3f}"
            writer.writerow(row)

            if cmd_vel > 0.1:
                direction_label = "UPPER"
            elif cmd_vel < -0.1:
                direction_label = "LOWER"
            else:
                direction_label = "HOLD"

            print(
                f"Joint:{selected_joint:13s} LB:{int(state.left_bumper)} "
                f"Cmd:{cmd_vel:+6.1f} ({direction_label:5s}) Goal:{goals_deg[selected_joint]:+7.2f} "
                f"Pos:{selected_pos_deg:+7.2f} VelRaw:{selected_vel_raw:5d} "
                f"Load:{selected_load_raw:5d} Sweep:{int(sweep.active)}   ",
                end="\r",
                flush=True,
            )

            elapsed = time.monotonic() - loop_start
            if elapsed < loop_dt:
                time.sleep(loop_dt - elapsed)

    print("\nStopping diagnostic...")
    controller.disconnect()
    bus.disconnect()
    print(f"Log written: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
