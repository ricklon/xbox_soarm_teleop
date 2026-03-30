#!/usr/bin/env python3
"""Xbox controller teleoperation for real SO-ARM101.

This example connects to a real SO-ARM101 robot and controls it
via Xbox controller using inverse kinematics or Jacobian-based control.

Usage:
    uv run python examples/teleoperate_real.py --port /dev/ttyUSB0
    uv run python examples/teleoperate_real.py --jacobian      # Use Jacobian control
    uv run python examples/teleoperate_real.py --recalibrate  # Fresh calibration
    uv run python examples/teleoperate_real.py --no-calibrate # Skip calibration
    uv run python examples/teleoperate_real.py --motion-routine
    uv run python examples/teleoperate_real.py --motion-routine --routine-pattern square-xyz

Controls:
    - Hold LB (left bumper) to enable arm movement
    - Left stick X: Left/right (Y axis)
    - Left stick Y: Forward/back (X axis)
    - Right stick Y: Up/down (Z axis)
    - Right stick X: Wrist roll rotation (direct, not IK)
    - Right trigger: Gripper (released=open, pulled=closed)
    - A button: Return to home position
    - Ctrl+C: Exit
"""

import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    IK_JOINT_NAMES,
    JOINT_LIMITS_DEG,
    JOINT_NAMES_WITH_GRIPPER,
)
from xbox_soarm_teleop.config.workspace import load_workspace_limits
from xbox_soarm_teleop.control.cartesian import (
    advance_cartesian_target,
    apply_ik_solution,
    full_joint_positions,
    make_cartesian_state,
    step_cartesian_home,
    step_gripper_toward,
    step_wrist_roll,
    sync_cartesian_state,
)
from xbox_soarm_teleop.control.routines import plane_offset
from xbox_soarm_teleop.control.safety import apply_strict_safety, clip_workspace
from xbox_soarm_teleop.control.units import deg_to_normalized, normalized_to_deg
from xbox_soarm_teleop.runtime import build_control_runtime, print_controls

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = PROJECT_ROOT / "assets" / "so101_abs.urdf"

# Default calibration directory (project-local)
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "calibration"

# Joint names (order matters - matches URDF and robot)
JOINT_NAMES = JOINT_NAMES_WITH_GRIPPER

# Control loop rate
CONTROL_RATE = 30  # Hz
LOOP_PERIOD = 1.0 / CONTROL_RATE


def find_serial_port() -> str | None:
    """Find available serial port for the robot."""
    import glob

    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def resolve_robot_id(calibration_dir: Path, robot_id: str | None) -> str | None:
    """Resolve robot calibration id, preferring known non-None calibration files."""
    if robot_id:
        return robot_id

    # Prefer explicit project calibration ids before falling back to None.json behavior.
    for candidate in ("so101_calibration", "lerobot_calibration_real", "test_arm"):
        if (calibration_dir / f"{candidate}.json").exists():
            return candidate
    return None


def warn_if_suspicious_pan_calibration(calibration_dir: Path, robot_id: str | None) -> None:
    """Warn when shoulder_pan calibration span is abnormally small."""
    calib_path = calibration_dir / f"{robot_id}.json"
    if not calib_path.exists():
        print(f"Calibration file not found yet: {calib_path.name}", flush=True)
        return
    try:
        data = json.loads(calib_path.read_text())
        pan = data.get("shoulder_pan")
        if not isinstance(pan, dict):
            return
        rmin = int(pan.get("range_min"))
        rmax = int(pan.get("range_max"))
    except Exception:
        return

    span = rmax - rmin
    print(f"Calibration file: {calib_path.name} (shoulder_pan raw span={span})", flush=True)
    if span < 500:
        print(
            "WARNING: shoulder_pan calibration span is very small. "
            "This can cause almost no base rotation.",
            flush=True,
        )
        print(
            "  Use --robot-id with the correct calibration file (e.g. --robot-id so101_calibration).",
            flush=True,
        )


def run_teleoperation(
    port: str,
    calibration_dir: Path | None = None,
    robot_id: str | None = None,
    recalibrate: bool = False,
    no_calibrate: bool = False,
    deadzone: float = 0.15,
    linear_scale: float | None = None,
    controller_type: str = "xbox",
    keyboard_grab: bool = False,
    keyboard_record: str | None = None,
    keyboard_playback: str | None = None,
    mode: str = "crane",
    debug_ik: bool = False,
    debug_ik_every: int = 10,
    motion_routine: bool = False,
    routine_duration: float = 15.0,
    routine_scale: float = 1.0,
    routine_pattern: str = "lissajous",
    routine_plane: str = "xy",
    routine_square_size: float = 0.06,
    routine_square_speed: float = 0.03,
    routine_center_x: float = 0.0,
    routine_center_y: float = 0.0,
    routine_center_z: float = 0.0,
    ik_log_path: str | None = None,
    ik_max_err_mm: float = 30.0,
    ik_mean_err_mm: float = 10.0,
    ik_vel_scale: float = 1.0,
    swap_xy: bool = False,
    debug_limiters: bool = False,
    debug_limiters_every: int = 30,
    use_jacobian: bool = False,
    jacobian_damping: float = 0.05,
    strict_safety: bool = True,
    strict_joint_margin_deg: float = 15.0,
    strict_danger_margin_deg: float = 8.0,
    strict_max_linear_speed: float = 0.02,
    strict_max_angular_speed: float = 0.25,
    strict_allow_orientation: bool = False,
    strict_wrist_min_deg: float = -60.0,
    strict_wrist_max_deg: float = 45.0,
    ik_seed_from_feedback: bool = True,
    ik_seed_every: int = 5,
    benchmark: bool = False,
    rerun_mode: str | None = None,
    rerun_addr: str = "0.0.0.0:9876",
    rerun_save: str = "session.rrd",
):
    """Run teleoperation with real robot.

    Args:
        port: Serial port for robot.
        calibration_dir: Directory for calibration files. Default: ./calibration/
        recalibrate: If True, delete existing calibration and run fresh.
        no_calibrate: If True, skip calibration (use existing only).
        deadzone: Controller deadzone (0.0-1.0).
        linear_scale: Linear velocity scale (m/s), or None for config default.
        debug_ik: If True, print IK debug output periodically.
        debug_ik_every: Print IK debug every N control loops.
        motion_routine: If True, run an automatic motion routine (no controller).
        routine_duration: Duration in seconds for motion routine.
        routine_scale: Scale for routine motion amplitudes.
        routine_pattern: Motion routine pattern (lissajous, square, square-xyz).
        routine_plane: Plane for square routine (xy, xz, yz).
        routine_square_size: Square side length in meters.
        routine_square_speed: Max tracking speed for square (m/s).
        routine_center_x: Offset to add to routine center (meters).
        routine_center_y: Offset to add to routine center (meters).
        routine_center_z: Offset to add to routine center (meters).
        ik_log_path: Optional CSV path for IK error logging.
        ik_max_err_mm: Warn if max IK position error exceeds this (mm).
        ik_mean_err_mm: Warn if mean IK position error exceeds this (mm).
        ik_vel_scale: Scale factor for IK joint velocity limits.
        swap_xy: If True, swap Cartesian X/Y commands for real-robot frame alignment.
        debug_limiters: If True, print live limiter diagnostics periodically.
        debug_limiters_every: Print limiter diagnostics every N control loops.
        use_jacobian: If True, use Jacobian-based control instead of IK.
        jacobian_damping: Damping factor for Jacobian pseudo-inverse.
        strict_safety: If True, apply conservative safety limits and motion gating.
        strict_joint_margin_deg: Keep IK joints this far from hard limits.
        strict_danger_margin_deg: Reject command steps below this margin.
        strict_max_linear_speed: Cap Cartesian linear speed (m/s) in strict mode.
        strict_max_angular_speed: Cap angular command speed (rad/s) in strict mode.
        strict_allow_orientation: If False, disable pitch/yaw in strict mode.
        strict_wrist_min_deg: Extra strict lower bound for wrist_flex in strict mode.
        strict_wrist_max_deg: Extra strict upper bound for wrist_flex in strict mode.
        ik_seed_from_feedback: If True, seed IK with measured joint positions.
        ik_seed_every: Seed IK every N control loops (cartesian mode only).
    """
    import shutil

    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    from xbox_soarm_teleop.config.modes import ControlMode
    from xbox_soarm_teleop.diagnostics.benchmark import LoopTimer
    from xbox_soarm_teleop.processors.xbox_to_ee import EEDelta, apply_axis_mapping

    control_mode = ControlMode(mode)
    print(f"Control mode: {control_mode.value.upper()}", flush=True)

    # Use project-local calibration directory by default
    if calibration_dir is None:
        calibration_dir = DEFAULT_CALIBRATION_DIR
    calibration_dir = Path(calibration_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    print(f"Calibration directory: {calibration_dir}", flush=True)

    # Handle recalibration - delete existing calibration
    if recalibrate and calibration_dir.exists():
        print("Deleting existing calibration for recalibration...", flush=True)
        shutil.rmtree(calibration_dir)
        calibration_dir.mkdir(parents=True, exist_ok=True)
        print("Calibration cleared.", flush=True)

    runtime = build_control_runtime(
        controller_type=controller_type,
        mode=mode,
        deadzone=deadzone,
        linear_scale=linear_scale,
        keyboard_grab=keyboard_grab,
        keyboard_record=keyboard_record,
        keyboard_playback=keyboard_playback,
        loop_dt=LOOP_PERIOD,
        urdf_path=str(URDF_PATH),
        use_jacobian=use_jacobian,
        jacobian_damping=jacobian_damping,
        announce_kinematics=True,
        keyboard_focus_target="this terminal",
        enable_controller=not motion_routine,
    )
    kinematics = runtime.kinematics
    jacobian_ctrl = runtime.jacobian_controller
    controller = runtime.controller
    processor = runtime.processor
    mapper = runtime.mapper
    _proc_cfg = runtime.processor_config
    gripper_rate = runtime.gripper_rate
    controller_cfg = runtime.controller_config

    if kinematics is not None:
        if use_jacobian:
            print(f"Kinematics: JACOBIAN (damping={jacobian_damping})", flush=True)
        else:
            print("Kinematics: IK", flush=True)

    # Joint velocity limits for IK joints (4 joints, no wrist_roll)
    ik_joint_vel_limits = np.array([120.0, 90.0, 90.0, 90.0]) * max(ik_vel_scale, 0.1)
    ik_joint_lower_limits = np.array([JOINT_LIMITS_DEG[name][0] for name in IK_JOINT_NAMES], dtype=float)
    ik_joint_upper_limits = np.array([JOINT_LIMITS_DEG[name][1] for name in IK_JOINT_NAMES], dtype=float)
    base_limits, strict_limits = load_workspace_limits()
    workspace_limits = strict_limits if strict_safety else base_limits
    safe_lower_limits = ik_joint_lower_limits + max(strict_joint_margin_deg, 0.0)
    safe_upper_limits = ik_joint_upper_limits - max(strict_joint_margin_deg, 0.0)
    # Hard override for wrist_flex envelope in strict mode (self-collision guard).
    if strict_safety:
        wrist_idx = IK_JOINT_NAMES.index("wrist_flex")
        safe_lower_limits[wrist_idx] = max(safe_lower_limits[wrist_idx], strict_wrist_min_deg)
        safe_upper_limits[wrist_idx] = min(safe_upper_limits[wrist_idx], strict_wrist_max_deg)
    invalid = safe_lower_limits >= safe_upper_limits
    if np.any(invalid):
        mids = 0.5 * (ik_joint_lower_limits + ik_joint_upper_limits)
        safe_lower_limits[invalid] = mids[invalid] - 1.0
        safe_upper_limits[invalid] = mids[invalid] + 1.0
    pitch_limit_rad = np.deg2rad(25.0 if strict_safety else 90.0)
    yaw_limit_rad = np.deg2rad(45.0 if strict_safety else 180.0)

    print(f"Controller deadzone: {getattr(controller_cfg, 'deadzone', 'n/a')}", flush=True)
    print(f"Linear scale: {_proc_cfg.linear_scale} m/s", flush=True)
    orientation_enabled = bool(getattr(mapper, "enable_pitch", False) or getattr(mapper, "enable_yaw", False))
    if orientation_enabled:
        print(f"Orientation scale: {_proc_cfg.orientation_scale} rad/s", flush=True)
    else:
        print("Orientation controls: OFF (touch mode)", flush=True)
    print(f"XY axis swap: {'ON' if swap_xy else 'OFF'}", flush=True)
    if strict_safety:
        print(
            "Strict safety: ON "
            f"(joint margin {strict_joint_margin_deg:.1f}deg, "
            f"danger margin {strict_danger_margin_deg:.1f}deg, "
            f"max v={strict_max_linear_speed:.3f}m/s, "
            f"wrist=[{strict_wrist_min_deg:.1f},{strict_wrist_max_deg:.1f}]deg)",
            flush=True,
        )
    else:
        print("Strict safety: OFF", flush=True)
    if control_mode == ControlMode.CARTESIAN and ik_seed_from_feedback:
        print(f"IK feedback seeding: ON (every {max(1, ik_seed_every)} frames)", flush=True)
    elif control_mode == ControlMode.CARTESIAN:
        print("IK feedback seeding: OFF", flush=True)

    if not motion_routine:
        if not controller.connect():
            print(f"ERROR: Failed to connect to {runtime.controller_label}")
            if controller_type == "keyboard":
                print("  Check 'input' group: groups $USER | grep input")
                print("  Add with: sudo usermod -aG input $USER  (then re-login)")
            elif controller_type == "joycon":
                print("  Joy-Con setup: bluetooth connect + press SL+SR for single-controller mode")
                print("  joycond must be running: systemctl is-active joycond")
            else:
                print("  - Check that controller is connected")
            sys.exit(1)
        print(f"{runtime.controller_label} connected", flush=True)
        print_controls(
            controller_type,
            mode,
            use_jacobian=use_jacobian,
            exit_hint="Ctrl+C                    exit",
        )
    else:
        print("Motion routine enabled (no controller required).", flush=True)
        print("Ensure the arm is in home position and the area is clear.", flush=True)

    # Initialize robot
    print(f"Connecting to robot on {port}...", flush=True)
    resolved_robot_id = resolve_robot_id(calibration_dir, robot_id)
    print(f"Robot calibration id: {resolved_robot_id}", flush=True)
    warn_if_suspicious_pan_calibration(calibration_dir, resolved_robot_id)
    robot_config = SOFollowerRobotConfig(
        port=port,
        calibration_dir=calibration_dir,
        id=resolved_robot_id,
    )
    robot = SOFollower(robot_config)

    # Determine calibration mode
    calibrate = not no_calibrate

    try:
        robot.connect(calibrate=calibrate)
        # When --no-calibrate is used, we still need to write calibration to the motor bus
        # lerobot skips this when calibrate=False, so we do it manually
        if no_calibrate and robot.calibration and not robot.bus.is_calibrated:
            print("Loading existing calibration...", flush=True)
            robot.bus.write_calibration(robot.calibration)
    except Exception as e:
        print(f"ERROR: Failed to connect to robot: {e}")
        if no_calibrate and "calibration" in str(e).lower():
            print("  No existing calibration found. Run without --no-calibrate first.")
        if controller is not None:
            controller.disconnect()
        sys.exit(1)

    if motion_routine and control_mode == ControlMode.JOINT:
        print("ERROR: --motion-routine is not supported with --mode joint.", flush=True)
        if controller is not None:
            controller.disconnect()
        robot.disconnect()
        sys.exit(1)

    print("Robot connected!", flush=True)
    if motion_routine:
        print("\nMotion routine:", flush=True)
        print(
            f"  Duration: {routine_duration:.1f}s | Scale: {routine_scale:.2f} | "
            f"Pattern: {routine_pattern}",
            flush=True,
        )
        print("  Ctrl+C: Stop\n", flush=True)
    else:
        print("Teleoperation active. Ctrl+C to exit.\n", flush=True)

    # IK joint positions (4 joints: base, shoulder_lift, elbow_flex, wrist_flex)
    home_ik_joint_pos_deg = np.array([HOME_POSITION_DEG[name] for name in IK_JOINT_NAMES], dtype=float)
    if strict_safety:
        clipped_home = np.clip(home_ik_joint_pos_deg, safe_lower_limits, safe_upper_limits)
        if np.any(clipped_home != home_ik_joint_pos_deg):
            print(
                "Strict safety adjusted home IK pose to fit safety envelope."
                f" from={np.array2string(home_ik_joint_pos_deg, precision=1)}"
                f" to={np.array2string(clipped_home, precision=1)}",
                flush=True,
            )
        home_ik_joint_pos_deg = clipped_home
    cartesian_state = (
        make_cartesian_state(
            kinematics,
            home_ik_joint_pos_deg,
            wrist_roll_deg=float(HOME_POSITION_DEG["wrist_roll"]),
        )
        if kinematics is not None
        else None
    )
    routine_center = (
        cartesian_state.ee_pose[:3, 3].copy() if cartesian_state is not None else np.zeros(3, dtype=float)
    )
    routine_center += np.array([routine_center_x, routine_center_y, routine_center_z], dtype=float)

    running = True
    loop_counter = 0
    routine_start = time.monotonic()
    error_count = 0
    error_sum = 0.0
    error_max = 0.0
    clip_count = 0
    clip_joints = 0
    workspace_clip_x = 0
    workspace_clip_y = 0
    workspace_clip_z = 0
    pitch_sat_count = 0
    yaw_sat_count = 0
    homing_active = False
    min_joint_margin_deg = float("inf")
    min_margin_joint = ""
    safety_reject_count = 0
    safety_clip_joint_count = 0
    safety_clip_speed_count = 0
    safety_clip_orient_count = 0

    csv_file = None
    csv_writer = None
    if ik_log_path:
        csv_file = open(ik_log_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "t_s",
                "target_x",
                "target_y",
                "target_z",
                "actual_x",
                "actual_y",
                "actual_z",
                "pos_err_mm",
                "clipped",
                "ws_clip_x",
                "ws_clip_y",
                "ws_clip_z",
                "safety_speed_clip",
                "safety_orient_clip",
                "safety_joint_clip",
                "safety_reject",
            ]
        )

    def signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...", flush=True)
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bm_timer = LoopTimer(mode=mode) if benchmark else None

    from xbox_soarm_teleop.diagnostics.rerun_logger import RerunLogger

    rerun_logger: RerunLogger | None = None
    if rerun_mode is not None:
        rerun_logger = RerunLogger(
            app_id="xbox_soarm_teleop_real",
            mode=rerun_mode,
            addr=rerun_addr,
            rrd_path=rerun_save,
        )

    try:
        while running:
            loop_start = time.monotonic()
            if bm_timer is not None:
                bm_timer.start_frame()
            controller_ms = ik_ms = servo_ms = 0.0
            clipped_this = False
            frame_ws_clip_x = 0
            frame_ws_clip_y = 0
            frame_ws_clip_z = 0
            frame_safety_speed_clip = 0
            frame_safety_orient_clip = 0
            frame_safety_joint_clip = 0
            frame_safety_reject = 0

            if motion_routine:
                assert cartesian_state is not None
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
                    delta = desired - cartesian_state.ee_pose[:3, 3]
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
                if routine_pattern == "lissajous":
                    droll = routine_scale * 0.10 * np.sin(t * 0.4)
                    gripper = 0.5 + 0.2 * np.sin(t * 0.2)
                    gripper = float(np.clip(gripper, 0.0, 1.0))
                else:
                    droll = 0.0
                    gripper = cartesian_state.gripper_pos
                ee_delta = EEDelta(dx=dx, dy=dy, dz=dz, droll=droll, gripper=gripper)
            elif control_mode in (ControlMode.JOINT, ControlMode.CRANE, ControlMode.PUPPET):
                _t0 = time.perf_counter()
                state = controller.read()
                joint_cmd = processor(state)
                controller_ms = (time.perf_counter() - _t0) * 1000.0
                action = {}
                for name in JOINT_NAMES[:-1]:
                    action[f"{name}.pos"] = deg_to_normalized(joint_cmd.goals_deg[name], name)
                # gripper in degrees → normalize to 0-100 range
                g_lower, g_upper = JOINT_LIMITS_DEG["gripper"]
                g_deg = joint_cmd.goals_deg["gripper"]
                action["gripper.pos"] = float(
                    np.clip((g_deg - g_lower) / max(g_upper - g_lower, 1e-6) * 100.0, 0.0, 100.0)
                )
                _t0 = time.perf_counter()
                robot.send_action(action)
                servo_ms = (time.perf_counter() - _t0) * 1000.0
                if bm_timer is not None:
                    bm_timer.record(loop_counter, controller_ms, 0.0, servo_ms)
                if rerun_logger is not None:
                    rerun_logger.log_frame(
                        loop_counter,
                        time.monotonic() - routine_start,
                        joint_cmd.goals_deg,
                        mode=mode,
                    )
                elapsed = time.monotonic() - loop_start
                if elapsed < LOOP_PERIOD:
                    time.sleep(LOOP_PERIOD - elapsed)
                loop_counter += 1
                continue
            else:
                _t0 = time.perf_counter()
                state = controller.read()
                controller_ms = (time.perf_counter() - _t0) * 1000.0

                if ik_seed_from_feedback and (loop_counter % max(1, ik_seed_every) == 0):
                    try:
                        obs = robot.get_observation()
                        for idx, name in enumerate(IK_JOINT_NAMES):
                            key = f"{name}.pos"
                            if key in obs:
                                cartesian_state.ik_joint_pos_deg[idx] = normalized_to_deg(
                                    float(obs[key]), name
                                )
                        wr_key = "wrist_roll.pos"
                        if wr_key in obs:
                            cartesian_state.wrist_roll_deg = normalized_to_deg(
                                float(obs[wr_key]), "wrist_roll"
                            )
                        sync_cartesian_state(
                            cartesian_state,
                            kinematics,
                            cartesian_state.ik_joint_pos_deg,
                            wrist_roll_deg=cartesian_state.wrist_roll_deg,
                        )
                    except Exception:
                        pass

                if state.a_button_pressed:
                    print("\nGoing home...", flush=True)
                    homing_active = True

                ee_delta = mapper(state)
                ee_delta = apply_axis_mapping(ee_delta, swap_xy=swap_xy)
                if strict_safety:
                    ee_delta, safety_flags = apply_strict_safety(
                        ee_delta,
                        max_linear_speed=strict_max_linear_speed,
                        max_angular_speed=strict_max_angular_speed,
                        allow_orientation=strict_allow_orientation,
                    )
                    if safety_flags["speed_clip"]:
                        safety_clip_speed_count += 1
                        frame_safety_speed_clip = 1
                    if safety_flags["orient_clip"]:
                        safety_clip_orient_count += 1
                        frame_safety_orient_clip = 1
                if use_jacobian and abs(ee_delta.dyaw) > 1e-6:
                    ee_delta = EEDelta(
                        dx=ee_delta.dx,
                        dy=ee_delta.dy,
                        dz=ee_delta.dz,
                        droll=ee_delta.droll,
                        dpitch=ee_delta.dpitch,
                        dyaw=0.0,
                        gripper=ee_delta.gripper,
                    )

            assert cartesian_state is not None
            if homing_active:
                homing_active = not step_cartesian_home(
                    cartesian_state,
                    kinematics,
                    home_ik_joint_pos_deg,
                    home_wrist_roll_deg=float(HOME_POSITION_DEG["wrist_roll"]),
                    home_gripper_pos=0.0,
                    ik_joint_max_step_deg=ik_joint_vel_limits,
                    wrist_roll_vel_deg_s=90.0,
                    gripper_rate=gripper_rate,
                    dt=LOOP_PERIOD,
                )
                ee_delta = EEDelta(gripper=cartesian_state.gripper_pos)
            else:
                cartesian_state.gripper_pos = step_gripper_toward(
                    cartesian_state.gripper_pos,
                    ee_delta.gripper,
                    gripper_rate=gripper_rate,
                    dt=LOOP_PERIOD,
                )

            if not homing_active and not ee_delta.is_zero_motion():
                target_pose, target_pos, target_flags = advance_cartesian_target(
                    cartesian_state,
                    ee_delta,
                    dt=LOOP_PERIOD,
                    clip_position=lambda pos: clip_workspace(pos, workspace_limits),
                    pitch_limit_rad=pitch_limit_rad,
                    yaw_limit_rad=yaw_limit_rad,
                )
                ws_flags = {
                    "ws_clip_x": bool(target_flags.get("ws_clip_x", False)),
                    "ws_clip_y": bool(target_flags.get("ws_clip_y", False)),
                    "ws_clip_z": bool(target_flags.get("ws_clip_z", False)),
                }
                if ws_flags["ws_clip_x"]:
                    workspace_clip_x += 1
                    frame_ws_clip_x = 1
                if ws_flags["ws_clip_y"]:
                    workspace_clip_y += 1
                    frame_ws_clip_y = 1
                if ws_flags["ws_clip_z"]:
                    workspace_clip_z += 1
                    frame_ws_clip_z = 1

                if target_flags["pitch_clipped"]:
                    pitch_sat_count += 1
                if target_flags["yaw_clipped"]:
                    yaw_sat_count += 1
                orientation_weight = float(target_flags["orientation_weight"])

                if use_jacobian:
                    # Jacobian-based control: EE velocity -> joint velocity -> integrate
                    # Use position + pitch mode (4 DOF) for optimal control with 4 joints
                    ee_vel = np.array([ee_delta.dx, ee_delta.dy, ee_delta.dz, ee_delta.dpitch])
                    joint_vel_deg = jacobian_ctrl.ee_vel_to_joint_vel(
                        ee_vel, cartesian_state.ik_joint_pos_deg, mode="position_pitch"
                    )

                    # Apply velocity limits
                    joint_vel_deg = np.clip(joint_vel_deg, -ik_joint_vel_limits, ik_joint_vel_limits)

                    # Integrate to get new joint positions
                    joint_delta = joint_vel_deg * LOOP_PERIOD
                    clipped_delta = joint_delta.copy()
                    candidate_joints = cartesian_state.ik_joint_pos_deg + joint_delta

                    # Apply conservative joint limits in strict mode.
                    if strict_safety:
                        clipped_next = np.clip(candidate_joints, safe_lower_limits, safe_upper_limits)
                        if np.any(clipped_next != candidate_joints):
                            safety_clip_joint_count += 1
                            frame_safety_joint_clip = 1
                        candidate_joints = clipped_next
                    else:
                        joint_limits_deg = np.array([180.0, 90.0, 90.0, 90.0])
                        candidate_joints = np.clip(
                            candidate_joints, -joint_limits_deg, joint_limits_deg
                        )
                else:
                    # IK-based control: solve IK for target pose
                    _t0 = time.perf_counter()
                    new_joints = kinematics.inverse_kinematics(
                        cartesian_state.ik_joint_pos_deg,
                        target_pose,
                        position_weight=1.0,
                        orientation_weight=orientation_weight,
                    )
                    ik_ms = (time.perf_counter() - _t0) * 1000.0
                    ik_result = new_joints[:4]

                    # Apply joint velocity limiting to smooth IK output
                    max_delta = ik_joint_vel_limits * LOOP_PERIOD
                    joint_delta = ik_result - cartesian_state.ik_joint_pos_deg
                    clipped_delta = np.clip(joint_delta, -max_delta, max_delta)
                    if np.any(np.abs(joint_delta) > max_delta):
                        clip_count += 1
                        clip_joints += int(np.sum(np.abs(joint_delta) > max_delta))
                        clipped_this = True
                    candidate_joints = cartesian_state.ik_joint_pos_deg + clipped_delta
                    if strict_safety:
                        candidate_joints = np.clip(candidate_joints, safe_lower_limits, safe_upper_limits)
                        if np.any(candidate_joints != cartesian_state.ik_joint_pos_deg + clipped_delta):
                            safety_clip_joint_count += 1
                            frame_safety_joint_clip = 1
                        cand_margins = np.minimum(
                            candidate_joints - ik_joint_lower_limits,
                            ik_joint_upper_limits - candidate_joints,
                        )
                        current_margins = np.minimum(
                            cartesian_state.ik_joint_pos_deg - ik_joint_lower_limits,
                            ik_joint_upper_limits - cartesian_state.ik_joint_pos_deg,
                        )
                        cand_min_margin = float(np.min(cand_margins))
                        current_min_margin_local = float(np.min(current_margins))
                        if (
                            cand_min_margin < strict_danger_margin_deg
                            and cand_min_margin < current_min_margin_local - 1e-6
                        ):
                            safety_reject_count += 1
                            frame_safety_reject = 1
                            clipped_this = True
                            candidate_joints = cartesian_state.ik_joint_pos_deg.copy()

                next_wrist_roll_deg = step_wrist_roll(
                    cartesian_state.wrist_roll_deg,
                    ee_delta.droll,
                    dt=LOOP_PERIOD,
                )
                apply_ik_solution(
                    cartesian_state,
                    kinematics,
                    candidate_joints,
                    wrist_roll_deg=next_wrist_roll_deg,
                    target_pose=target_pose,
                )

                if debug_ik and (loop_counter % max(debug_ik_every, 1) == 0):
                    pos_error = np.linalg.norm(target_pose[:3, 3] - cartesian_state.ee_pose[:3, 3])
                    raw = np.array2string(joint_delta, precision=2, separator=",")
                    clipped = np.array2string(clipped_delta, precision=2, separator=",")
                    print(
                        f"\nIK: target=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, "
                        f"{target_pos[2]:.3f}] actual=[{cartesian_state.ee_pose[0, 3]:.3f}, "
                        f"{cartesian_state.ee_pose[1, 3]:.3f}, {cartesian_state.ee_pose[2, 3]:.3f}] "
                        f"err={pos_error * 1000.0:.1f}mm "
                        f"raw_delta={raw} clipped={clipped}",
                        flush=True,
                    )

            # Combine base + IK joints for full 5-joint position
            full_joint_pos_deg = full_joint_positions(
                cartesian_state.ik_joint_pos_deg,
                cartesian_state.wrist_roll_deg,
            )
            joint_margins = np.minimum(
                cartesian_state.ik_joint_pos_deg - ik_joint_lower_limits,
                ik_joint_upper_limits - cartesian_state.ik_joint_pos_deg,
            )
            current_min_margin = float(np.min(joint_margins))
            if current_min_margin < min_joint_margin_deg:
                min_joint_margin_deg = current_min_margin
                min_idx = int(np.argmin(joint_margins))
                min_margin_joint = IK_JOINT_NAMES[min_idx]

            # Send to robot (convert to normalized values)
            action = {}
            for i, name in enumerate(JOINT_NAMES[:-1]):
                action[f"{name}.pos"] = deg_to_normalized(full_joint_pos_deg[i], name)
            # Gripper: 0-1 maps to 0-100
            action["gripper.pos"] = cartesian_state.gripper_pos * 100.0

            _t0 = time.perf_counter()
            robot.send_action(action)
            servo_ms = (time.perf_counter() - _t0) * 1000.0
            if bm_timer is not None:
                bm_timer.record(loop_counter, controller_ms, ik_ms, servo_ms)
            if rerun_logger is not None:
                arm_joints = dict(zip(JOINT_NAMES, full_joint_pos_deg))
                rerun_logger.log_frame(
                    loop_counter,
                    time.monotonic() - routine_start,
                    arm_joints,
                    ee_pos=cartesian_state.ee_pose[:3, 3],
                    mode=mode,
                )

            # Status - show position and orientation
            pos = cartesian_state.ee_pose[:3, 3]
            pos_err = float(
                np.linalg.norm(
                    cartesian_state.last_ee_pose[:3, 3] - cartesian_state.last_target_pose[:3, 3]
                )
            )
            error_count += 1
            error_sum += pos_err
            error_max = max(error_max, pos_err)
            if csv_writer:
                t_s = time.monotonic() - routine_start
                csv_writer.writerow(
                    [
                        f"{t_s:.3f}",
                        f"{cartesian_state.last_target_pose[0, 3]:.6f}",
                        f"{cartesian_state.last_target_pose[1, 3]:.6f}",
                        f"{cartesian_state.last_target_pose[2, 3]:.6f}",
                        f"{cartesian_state.last_ee_pose[0, 3]:.6f}",
                        f"{cartesian_state.last_ee_pose[1, 3]:.6f}",
                        f"{cartesian_state.last_ee_pose[2, 3]:.6f}",
                        f"{pos_err * 1000.0:.3f}",
                        "1" if clipped_this else "0",
                        str(frame_ws_clip_x),
                        str(frame_ws_clip_y),
                        str(frame_ws_clip_z),
                        str(frame_safety_speed_clip),
                        str(frame_safety_orient_clip),
                        str(frame_safety_joint_clip),
                        str(frame_safety_reject),
                    ]
                )
            pitch_deg = np.rad2deg(cartesian_state.target_pitch)
            yaw_deg = np.rad2deg(cartesian_state.target_yaw)
            print(
                f"EE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                f"P:{pitch_deg:+5.1f}° Y:{yaw_deg:+5.1f}° | Grip: {cartesian_state.gripper_pos:.2f}   ",
                end="\r",
                flush=True,
            )
            if debug_limiters and (loop_counter % max(debug_limiters_every, 1) == 0) and error_count > 0:
                clip_rate = (clip_count / error_count) * 100.0
                wx_rate = (workspace_clip_x / error_count) * 100.0
                wy_rate = (workspace_clip_y / error_count) * 100.0
                wz_rate = (workspace_clip_z / error_count) * 100.0
                print(
                    f"\nLIMITERS: clip={clip_rate:.1f}% "
                    f"ws[x={wx_rate:.1f}%, y={wy_rate:.1f}%, z={wz_rate:.1f}%] "
                    f"sat[pitch={pitch_sat_count}, yaw={yaw_sat_count}]",
                    flush=True,
                )
                print(
                    "JOINTS: "
                    f"pan={cartesian_state.ik_joint_pos_deg[0]:+6.1f} "
                    f"lift={cartesian_state.ik_joint_pos_deg[1]:+6.1f} "
                    f"elbow={cartesian_state.ik_joint_pos_deg[2]:+6.1f} "
                    f"wrist={cartesian_state.ik_joint_pos_deg[3]:+6.1f} "
                    f"| min_margin={current_min_margin:.1f}deg",
                    flush=True,
                )
                if strict_safety:
                    print(
                        "SAFETY: "
                        f"reject={safety_reject_count} "
                        f"joint_clip={safety_clip_joint_count} "
                        f"speed_clip={safety_clip_speed_count} "
                        f"orient_clip={safety_clip_orient_count}",
                        flush=True,
                    )

            loop_counter += 1
            elapsed = time.monotonic() - loop_start
            if elapsed < LOOP_PERIOD:
                time.sleep(LOOP_PERIOD - elapsed)

    finally:
        print("\nReturning to home position...", flush=True)
        try:
            # Send home position command
            home_action = {}
            home_deg = {name: HOME_POSITION_DEG[name] for name in JOINT_NAMES[:-1]}
            for idx, name in enumerate(IK_JOINT_NAMES):
                home_deg[name] = float(home_ik_joint_pos_deg[idx])
            for name in JOINT_NAMES[:-1]:  # All except gripper
                home_action[f"{name}.pos"] = deg_to_normalized(home_deg[name], name)
            home_action["gripper.pos"] = 0.0  # Gripper closed
            robot.send_action(home_action)
            time.sleep(2.0)  # Wait for movement
        except Exception as e:
            print(f"Warning: Could not return home: {e}")

        if controller is not None:
            controller.disconnect()
        robot.disconnect()
        if rerun_logger is not None:
            rerun_logger.close()
        if csv_file:
            csv_file.close()
        if bm_timer is not None and bm_timer.frame_count > 0:
            bm_path = LoopTimer.default_path("benchmark")
            bm_timer.write_csv(bm_path)
            print(f"\nBenchmark data written to: {bm_path}", flush=True)
            print(bm_timer.summary(), flush=True)
        if error_count > 0:
            mean_err_mm = (error_sum / error_count) * 1000.0
            max_err_mm = error_max * 1000.0
            clip_rate = (clip_count / error_count) * 100.0
            print("\nIK routine summary")
            print(f"  samples: {error_count}")
            print(f"  max position error: {max_err_mm:.1f} mm")
            print(f"  mean position error: {mean_err_mm:.1f} mm")
            print(f"  velocity clipping: {clip_rate:.1f}% (joints clipped: {clip_joints})")
            print(
                "  workspace clipping:"
                f" x={workspace_clip_x} y={workspace_clip_y} z={workspace_clip_z}"
            )
            print(
                f"  orientation saturation: pitch={pitch_sat_count} yaw={yaw_sat_count}"
            )
            if min_margin_joint:
                print(
                    f"  closest joint-to-limit margin: {min_joint_margin_deg:.1f} deg "
                    f"({min_margin_joint})"
                )
            if strict_safety:
                print(
                    "  safety interventions:"
                    f" reject={safety_reject_count}"
                    f" joint_clip={safety_clip_joint_count}"
                    f" speed_clip={safety_clip_speed_count}"
                    f" orient_clip={safety_clip_orient_count}"
                )
            if max_err_mm > ik_max_err_mm or mean_err_mm > ik_mean_err_mm:
                print("WARNING: IK error exceeded thresholds.")
        print("\nDisconnected.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Xbox teleoperation for real SO-ARM101")
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port for robot (e.g., /dev/ttyUSB0). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help=f"Calibration directory. Default: {DEFAULT_CALIBRATION_DIR}",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default=None,
        help="Calibration robot id (loads <calibration-dir>/<id>.json). Auto-selects known ids if omitted.",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Delete existing calibration and run fresh calibration.",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip calibration (use existing). Fails if no calibration exists.",
    )
    parser.add_argument(
        "--controller",
        choices=["xbox", "joycon", "keyboard"],
        default="xbox",
        help="Controller type: xbox, joycon, or keyboard. Default: xbox.",
    )
    parser.add_argument(
        "--keyboard-grab",
        action="store_true",
        help="Grab keyboard device exclusively (keyboard mode only). "
        "Prevents keypresses from reaching other windows when focus changes.",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        default=None,
        help="Save keystroke recording to PATH when Tab is pressed (keyboard mode only). "
        "Defaults to recording_<timestamp>.json if Tab is pressed without this flag.",
    )
    parser.add_argument(
        "--playback",
        metavar="PATH",
        default=None,
        help="Replay a saved keystroke recording instead of live keyboard input.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.15,
        help="Controller deadzone (0.0-1.0). Default: 0.15. Increase if sticks drift.",
    )
    parser.add_argument(
        "--linear-scale",
        type=float,
        default=None,
        help="Linear velocity scale (m/s at full stick). Default from config.",
    )
    parser.add_argument(
        "--debug-ik",
        action="store_true",
        help="Print IK target/achieved error and joint deltas periodically.",
    )
    parser.add_argument(
        "--debug-ik-every",
        type=int,
        default=10,
        help="Print IK debug every N control loops. Default: 10.",
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
        "--routine-center-x",
        type=float,
        default=0.0,
        help="Additive offset to routine center X (meters). Default: 0.",
    )
    parser.add_argument(
        "--routine-center-y",
        type=float,
        default=0.0,
        help="Additive offset to routine center Y (meters). Default: 0.",
    )
    parser.add_argument(
        "--routine-center-z",
        type=float,
        default=0.0,
        help="Additive offset to routine center Z (meters). Default: 0.",
    )
    parser.add_argument(
        "--ik-log",
        type=str,
        default=None,
        help="Write IK position error CSV to this path.",
    )
    parser.add_argument(
        "--ik-max-err-mm",
        type=float,
        default=30.0,
        help="Warn if max position error exceeds this (mm). Default: 30.",
    )
    parser.add_argument(
        "--ik-mean-err-mm",
        type=float,
        default=10.0,
        help="Warn if mean position error exceeds this (mm). Default: 10.",
    )
    parser.add_argument(
        "--ik-vel-scale",
        type=float,
        default=1.0,
        help="Scale IK joint velocity limits. Default: 1.0.",
    )
    parser.add_argument(
        "--swap-xy",
        action="store_true",
        dest="swap_xy",
        help="Enable legacy XY swap for Cartesian mapping.",
    )
    parser.add_argument(
        "--no-swap-xy",
        action="store_false",
        dest="swap_xy",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug-limiters",
        action="store_true",
        help="Print live limiter diagnostics (workspace/orientation/velocity clipping).",
    )
    parser.add_argument(
        "--debug-limiters-every",
        type=int,
        default=30,
        help="Print limiter diagnostics every N control loops. Default: 30.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cartesian", "joint", "crane", "puppet"],
        default="crane",
        help="Control mode: crane (cylindrical, default), joint (direct per-joint), cartesian (touch-point IK), puppet (IMU wrist).",
    )
    parser.add_argument(
        "--jacobian",
        action="store_true",
        help="Use Jacobian-based control instead of IK (cartesian mode only).",
    )
    parser.add_argument(
        "--jacobian-damping",
        type=float,
        default=0.05,
        help="Damping factor for Jacobian pseudo-inverse. Default: 0.05.",
    )
    parser.add_argument(
        "--no-strict-safety",
        action="store_true",
        help="Disable strict safety mode (not recommended on real hardware).",
    )
    parser.add_argument(
        "--strict-joint-margin-deg",
        type=float,
        default=15.0,
        help="Joint-limit margin (deg) used by strict safety. Default: 15.",
    )
    parser.add_argument(
        "--strict-danger-margin-deg",
        type=float,
        default=8.0,
        help="Reject IK updates that move deeper below this margin (deg). Default: 8.",
    )
    parser.add_argument(
        "--strict-max-linear-speed",
        type=float,
        default=0.02,
        help="Cartesian linear speed cap (m/s) in strict safety. Default: 0.02.",
    )
    parser.add_argument(
        "--strict-max-angular-speed",
        type=float,
        default=0.25,
        help="Angular speed cap (rad/s) in strict safety. Default: 0.25.",
    )
    parser.add_argument(
        "--strict-allow-orientation",
        action="store_true",
        help="Allow pitch/yaw commands in strict safety mode.",
    )
    parser.add_argument(
        "--strict-wrist-min-deg",
        type=float,
        default=-60.0,
        help="Strict lower wrist_flex bound (deg). Default: -60.",
    )
    parser.add_argument(
        "--strict-wrist-max-deg",
        type=float,
        default=45.0,
        help="Strict upper wrist_flex bound (deg). Default: 45.",
    )
    parser.add_argument(
        "--ik-seed-from-feedback",
        action="store_true",
        help="Seed IK from measured joint positions (cartesian mode only).",
    )
    parser.add_argument(
        "--no-ik-seed-from-feedback",
        action="store_true",
        help="Disable IK seeding from measured joints.",
    )
    parser.add_argument(
        "--ik-seed-every",
        type=int,
        default=5,
        help="Seed IK every N control loops (cartesian mode only). Default: 5.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Log per-frame timing data (controller read, IK solve, servo write). "
            "Writes benchmark_<timestamp>.csv on exit."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Stream joint angles and EE position to a local Rerun viewer (requires rerun-sdk).",
    )
    parser.add_argument(
        "--rerun-mode",
        choices=["spawn", "serve", "connect", "save"],
        default=None,
        help="Rerun connection mode. --rerun is shorthand for --rerun-mode spawn.",
    )
    parser.add_argument(
        "--rerun-addr",
        default="0.0.0.0:9876",
        help="gRPC address for --rerun-mode serve/connect. Default: 0.0.0.0:9876.",
    )
    parser.add_argument(
        "--rerun-save",
        default="session.rrd",
        help="Output .rrd file path for --rerun-mode save. Default: session.rrd.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.jacobian and args.mode != "cartesian":
        print("ERROR: --jacobian can only be used with --mode cartesian")
        sys.exit(1)

    if args.recalibrate and args.no_calibrate:
        print("ERROR: Cannot use both --recalibrate and --no-calibrate")
        sys.exit(1)

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

    run_teleoperation(
        port,
        calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
        robot_id=args.robot_id,
        recalibrate=args.recalibrate,
        no_calibrate=args.no_calibrate,
        deadzone=args.deadzone,
        linear_scale=args.linear_scale,
        controller_type=args.controller,
        keyboard_grab=args.keyboard_grab,
        keyboard_record=args.record,
        keyboard_playback=args.playback,
        mode=args.mode,
        debug_ik=args.debug_ik,
        debug_ik_every=args.debug_ik_every,
        motion_routine=args.motion_routine,
        routine_duration=args.routine_duration,
        routine_scale=args.routine_scale,
        routine_pattern=args.routine_pattern,
        routine_plane=args.routine_plane,
        routine_square_size=args.routine_square_size,
        routine_square_speed=args.routine_square_speed,
        routine_center_x=args.routine_center_x,
        routine_center_y=args.routine_center_y,
        routine_center_z=args.routine_center_z,
        ik_log_path=args.ik_log,
        ik_max_err_mm=args.ik_max_err_mm,
        ik_mean_err_mm=args.ik_mean_err_mm,
        ik_vel_scale=args.ik_vel_scale,
        swap_xy=args.swap_xy,
        debug_limiters=args.debug_limiters,
        debug_limiters_every=args.debug_limiters_every,
        use_jacobian=args.jacobian,
        jacobian_damping=args.jacobian_damping,
        strict_safety=not args.no_strict_safety,
        strict_joint_margin_deg=args.strict_joint_margin_deg,
        strict_danger_margin_deg=args.strict_danger_margin_deg,
        strict_max_linear_speed=args.strict_max_linear_speed,
        strict_max_angular_speed=args.strict_max_angular_speed,
        strict_allow_orientation=args.strict_allow_orientation,
        strict_wrist_min_deg=args.strict_wrist_min_deg,
        strict_wrist_max_deg=args.strict_wrist_max_deg,
        ik_seed_from_feedback=(
            args.ik_seed_from_feedback or not args.no_ik_seed_from_feedback
        ),
        ik_seed_every=args.ik_seed_every,
        benchmark=args.benchmark,
        rerun_mode=args.rerun_mode or ("spawn" if args.rerun else None),
        rerun_addr=args.rerun_addr,
        rerun_save=args.rerun_save,
    )


if __name__ == "__main__":
    main()
