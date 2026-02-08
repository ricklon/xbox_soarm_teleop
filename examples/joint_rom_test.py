#!/usr/bin/env python3
"""Joint range-of-motion (ROM) test and calibration tool.

Sweeps each joint through its full range, verifies positioning accuracy,
collects servo telemetry, and outputs calibration data.

Per-joint test positions:
    Some joints cannot be swept from the home (folded/parked) position without
    collisions.  For example, sweeping shoulder_lift from home drives the arm
    into the table, and sweeping wrist_flex with the elbow folded blocks full
    travel.  The SWEEP_TEST_POSITIONS config (in config/joints.py) defines
    per-joint base positions that provide clearance.  Use --no-test-positions
    to revert to the old behavior of always sweeping from home.

Usage:
    # Sweep (default subcommand, backward compatible)
    uv run python examples/joint_rom_test.py --sim
    uv run python examples/joint_rom_test.py sweep --sim --output /tmp/cal.json
    uv run python examples/joint_rom_test.py --port /dev/ttyACM0 --skip-gripper
    uv run python examples/joint_rom_test.py sweep --sim --no-telemetry
    uv run python examples/joint_rom_test.py sweep --sim --no-test-positions

    # Convert extended JSON to LeRobot calibration format
    uv run python examples/joint_rom_test.py convert --input /tmp/cal.json --output /tmp/lerobot.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    HOME_POSITION_RAW,
    JOINT_LIMITS_DEG,
    JOINT_NAMES_WITH_GRIPPER,
    MOTOR_IDS,
    SWEEP_TEST_POSITIONS,
    deg_to_raw,
    raw_to_deg,
)

# Path to URDF
URDF_PATH = Path(__file__).parent.parent / "assets" / "so101_abs.urdf"

# Control rate for ramping
RAMP_RATE_HZ = 50


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


# ---------------------------------------------------------------------------
# Telemetry dataclass
# ---------------------------------------------------------------------------


@dataclass
class TelemetrySample:
    """Single telemetry reading from a servo during a sweep."""

    t: float  # seconds since sweep start
    position_raw: int  # 0-4095
    position_deg: float
    velocity: int  # raw sign-magnitude
    load: int  # raw sign-magnitude (torque %)
    current: int  # raw current
    voltage_v: float  # volts (raw / 10)
    temperature_c: int  # Celsius


# ---------------------------------------------------------------------------
# Backend protocol and implementations
# ---------------------------------------------------------------------------


class ArmBackend(Protocol):
    """Hardware abstraction for joint ROM testing."""

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def go_to_positions(self, positions_deg: dict[str, float]) -> None: ...

    def read_positions(self) -> dict[str, float]: ...

    def read_telemetry(self, joint_name: str, t_ref: float) -> TelemetrySample | None: ...


class SimArmBackend:
    """MuJoCo simulation backend."""

    def __init__(self, urdf_path: str | Path):
        self.urdf_path = str(urdf_path)
        self.model = None
        self.data = None
        self.joint_ids: dict[str, int] = {}

    def connect(self) -> None:
        import mujoco

        self.model = mujoco.MjModel.from_xml_path(self.urdf_path)
        self.data = mujoco.MjData(self.model)
        for name in JOINT_NAMES_WITH_GRIPPER:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                self.joint_ids[name] = jnt_id
        self._forward()

    def disconnect(self) -> None:
        self.model = None
        self.data = None
        self.joint_ids.clear()

    def go_to_positions(self, positions_deg: dict[str, float]) -> None:
        for name, deg in positions_deg.items():
            if name in self.joint_ids:
                jnt_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[qpos_adr] = np.deg2rad(deg)
        self._forward()

    def read_positions(self) -> dict[str, float]:
        positions: dict[str, float] = {}
        for name, jnt_id in self.joint_ids.items():
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            positions[name] = float(np.rad2deg(self.data.qpos[qpos_adr]))
        return positions

    def read_telemetry(self, joint_name: str, t_ref: float) -> TelemetrySample | None:
        jnt_id = self.joint_ids.get(joint_name)
        if jnt_id is None:
            return None
        qpos_adr = self.model.jnt_qposadr[jnt_id]
        dof_adr = self.model.jnt_dofadr[jnt_id]
        pos_deg = float(np.rad2deg(self.data.qpos[qpos_adr]))
        vel = int(np.rad2deg(self.data.qvel[dof_adr]))
        return TelemetrySample(
            t=time.monotonic() - t_ref,
            position_raw=deg_to_raw(pos_deg),
            position_deg=pos_deg,
            velocity=vel,
            load=0,
            current=0,
            voltage_v=0.0,
            temperature_c=0,
        )

    def _forward(self) -> None:
        import mujoco

        mujoco.mj_forward(self.model, self.data)


class RealArmBackend:
    """Real hardware backend using Feetech motor bus."""

    def __init__(self, port: str):
        self.port = port
        self.bus = None

    def connect(self) -> None:
        from lerobot.motors.feetech.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode

        motors = {
            str(mid): Motor(id=mid, model="sts3215", norm_mode=MotorNormMode.DEGREES)
            for mid in MOTOR_IDS.values()
        }
        self.bus = FeetechMotorsBus(port=self.port, motors=motors)
        self.bus.connect()

    def disconnect(self) -> None:
        if self.bus is not None:
            self.bus.disconnect()
            self.bus = None

    def go_to_positions(self, positions_deg: dict[str, float]) -> None:
        for name, deg in positions_deg.items():
            motor_id = MOTOR_IDS.get(name)
            if motor_id is None:
                continue
            motor_key = str(motor_id)
            raw = deg_to_raw(deg)
            raw = max(0, min(4095, raw))
            self.bus.write("Torque_Enable", motor_key, 1, normalize=False)
            self.bus.write("Goal_Position", motor_key, raw, normalize=False)

    def read_positions(self) -> dict[str, float]:
        positions: dict[str, float] = {}
        for name, motor_id in MOTOR_IDS.items():
            motor_key = str(motor_id)
            try:
                raw = self.bus.read("Present_Position", motor_key, normalize=False)
                positions[name] = raw_to_deg(raw)
            except Exception as e:
                print(f"  WARNING: Failed to read {name}: {e}")
        return positions

    def read_telemetry(self, joint_name: str, t_ref: float) -> TelemetrySample | None:
        motor_id = MOTOR_IDS.get(joint_name)
        if motor_id is None:
            return None
        motor_key = str(motor_id)
        try:
            pos_raw = self.bus.read("Present_Position", motor_key, normalize=False)
            velocity = self.bus.read("Present_Velocity", motor_key, normalize=False)
            load = self.bus.read("Present_Load", motor_key, normalize=False)
            current = self.bus.read("Present_Current", motor_key, normalize=False)
            voltage_raw = self.bus.read("Present_Voltage", motor_key, normalize=False)
            temperature = self.bus.read("Present_Temperature", motor_key, normalize=False)
            return TelemetrySample(
                t=time.monotonic() - t_ref,
                position_raw=int(pos_raw),
                position_deg=raw_to_deg(int(pos_raw)),
                velocity=int(velocity),
                load=int(load),
                current=int(current),
                voltage_v=float(voltage_raw) / 10.0,
                temperature_c=int(temperature),
            )
        except Exception as e:
            print(f"  WARNING: Failed to read telemetry for {joint_name}: {e}")
            return None


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------


def ramp_to(
    backend: ArmBackend,
    target_deg: dict[str, float],
    moving_joint: str,
    speed_deg_s: float,
    telemetry_log: list[dict] | None = None,
    t_ref: float = 0.0,
) -> None:
    """Ramp a single joint to target at a controlled speed.

    Other joints are held at their target positions throughout.

    Args:
        telemetry_log: When provided, telemetry samples are appended each step.
        t_ref: Monotonic reference time for telemetry timestamps.
    """
    current = backend.read_positions()
    start_deg = current.get(moving_joint, 0.0)
    target = target_deg[moving_joint]
    distance = abs(target - start_deg)

    if distance < 0.01:
        backend.go_to_positions(target_deg)
        return

    duration = distance / max(speed_deg_s, 0.1)
    steps = max(int(duration * RAMP_RATE_HZ), 1)
    dt = duration / steps

    for i in range(1, steps + 1):
        frac = i / steps
        interp = start_deg + frac * (target - start_deg)
        pos = dict(target_deg)
        pos[moving_joint] = interp
        backend.go_to_positions(pos)
        if telemetry_log is not None:
            sample = backend.read_telemetry(moving_joint, t_ref)
            if sample is not None:
                telemetry_log.append(asdict(sample))
        time.sleep(dt)


def compute_diagnostics(telemetry: dict[str, list[dict]]) -> dict:
    """Compute diagnostic summary from telemetry phase arrays."""
    all_samples = []
    for phase_samples in telemetry.values():
        all_samples.extend(phase_samples)

    if not all_samples:
        return {
            "peak_load": 0,
            "peak_current": 0,
            "max_temperature_c": 0,
            "min_voltage_v": 0.0,
            "max_velocity": 0,
            "lower_reached_raw": 0,
            "upper_reached_raw": 0,
        }

    loads = [abs(s["load"]) for s in all_samples]
    currents = [abs(s["current"]) for s in all_samples]
    temps = [s["temperature_c"] for s in all_samples]
    voltages = [s["voltage_v"] for s in all_samples if s["voltage_v"] > 0]
    velocities = [abs(s["velocity"]) for s in all_samples]
    raws = [s["position_raw"] for s in all_samples]

    return {
        "peak_load": max(loads),
        "peak_current": max(currents),
        "max_temperature_c": max(temps),
        "min_voltage_v": min(voltages) if voltages else 0.0,
        "max_velocity": max(velocities),
        "lower_reached_raw": min(raws),
        "upper_reached_raw": max(raws),
    }


def sweep_joint(
    backend: ArmBackend,
    joint_name: str,
    base_positions: dict[str, float],
    lower_deg: float,
    upper_deg: float,
    speed_deg_s: float,
    settle_time: float,
    tolerance_deg: float,
    collect_telemetry: bool = False,
) -> dict:
    """Sweep a single joint through its range and report results.

    Args:
        base_positions: Positions for all joints while sweeping.  The target
            joint is overridden during the sweep; other joints are held here.
            This may differ from HOME_POSITION_DEG when the joint needs
            clearance (see SWEEP_TEST_POSITIONS).

    Returns dict with keys: joint, lower_target, upper_target,
    lower_actual, upper_actual, home_actual, max_error, passed.
    When collect_telemetry is True, also includes telemetry and diagnostics.
    """
    result: dict = {
        "joint": joint_name,
        "lower_target": lower_deg,
        "upper_target": upper_deg,
    }

    t_ref = time.monotonic() if collect_telemetry else 0.0
    telem_lower: list[dict] = []
    telem_upper: list[dict] = []
    telem_home: list[dict] = []

    # 1. Start at base position
    backend.go_to_positions(base_positions)
    time.sleep(settle_time)

    # 2. Ramp to lower limit
    lower_target = dict(base_positions)
    lower_target[joint_name] = lower_deg
    ramp_to(
        backend,
        lower_target,
        joint_name,
        speed_deg_s,
        telemetry_log=telem_lower if collect_telemetry else None,
        t_ref=t_ref,
    )
    time.sleep(settle_time)
    pos = backend.read_positions()
    result["lower_actual"] = pos.get(joint_name, float("nan"))

    # 3. Ramp to upper limit
    upper_target = dict(base_positions)
    upper_target[joint_name] = upper_deg
    ramp_to(
        backend,
        upper_target,
        joint_name,
        speed_deg_s,
        telemetry_log=telem_upper if collect_telemetry else None,
        t_ref=t_ref,
    )
    time.sleep(settle_time)
    pos = backend.read_positions()
    result["upper_actual"] = pos.get(joint_name, float("nan"))

    # 4. Ramp back to base position
    ramp_to(
        backend,
        base_positions,
        joint_name,
        speed_deg_s,
        telemetry_log=telem_home if collect_telemetry else None,
        t_ref=t_ref,
    )
    time.sleep(settle_time)
    pos = backend.read_positions()
    result["home_actual"] = pos.get(joint_name, float("nan"))

    # 5. Compute errors
    base_target = base_positions[joint_name]
    errors = [
        abs(result["lower_actual"] - lower_deg),
        abs(result["upper_actual"] - upper_deg),
        abs(result["home_actual"] - base_target),
    ]
    result["max_error"] = float(max(errors))
    result["passed"] = bool(result["max_error"] <= tolerance_deg)

    # 6. Attach telemetry and diagnostics
    if collect_telemetry:
        telemetry = {
            "sweep_to_lower": telem_lower,
            "sweep_to_upper": telem_upper,
            "sweep_to_home": telem_home,
        }
        result["telemetry"] = telemetry
        result["diagnostics"] = compute_diagnostics(telemetry)

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[dict]) -> int:
    """Print per-joint report and return exit code (0=all pass)."""
    print()
    print("=" * 74)
    print(f"{'Joint':<16} {'Range':>20}  {'Low Err':>8}  {'High Err':>8}  {'Home Err':>8}  {'Result'}")
    print("-" * 74)

    all_passed = True
    for r in results:
        joint = r["joint"]
        rng = f"[{r['lower_target']:+6.1f}, {r['upper_target']:+6.1f}]"
        low_err = abs(r["lower_actual"] - r["lower_target"])
        high_err = abs(r["upper_actual"] - r["upper_target"])
        home_err = abs(r["home_actual"] - HOME_POSITION_DEG.get(joint, 0.0))
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        print(f"{joint:<16} {rng:>20}  {low_err:7.2f}°  {high_err:7.2f}°  {home_err:7.2f}°  {status}")

    print("=" * 74)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"Result: {passed}/{total} joints PASS")
    print()

    return 0 if all_passed else 1


def print_diagnostic_summary(results: list[dict]) -> None:
    """Print diagnostic telemetry summary after the pass/fail table."""
    has_diag = any("diagnostics" in r for r in results)
    if not has_diag:
        return

    print()
    print("DIAGNOSTIC SUMMARY")
    print("=" * 74)
    print(
        f"{'Joint':<16} {'Peak Load':>10} {'Peak Curr':>10} "
        f"{'Max Temp':>9} {'Min Volt':>9} {'Max Vel':>8}"
    )
    print("-" * 74)

    for r in results:
        diag = r.get("diagnostics")
        if diag is None:
            continue
        joint = r["joint"]
        print(
            f"{joint:<16} {diag['peak_load']:>10} {diag['peak_current']:>10} "
            f"{diag['max_temperature_c']:>8}C {diag['min_voltage_v']:>8.1f}V "
            f"{diag['max_velocity']:>8}"
        )

    print("=" * 74)
    print()


# ---------------------------------------------------------------------------
# Extended calibration JSON
# ---------------------------------------------------------------------------


def compute_lerobot_fields(joint_name: str, result: dict) -> dict:
    """Compute the 5 LeRobot calibration fields for a joint.

    Args:
        joint_name: Name of the joint.
        result: Sweep result dict (must include diagnostics with raw extremes).

    Returns:
        Dict with id, drive_mode, homing_offset, range_min, range_max.
    """
    motor_id = MOTOR_IDS[joint_name]
    homing_offset = HOME_POSITION_RAW[joint_name] - 2047

    diag = result.get("diagnostics")
    if diag:
        range_min = diag["lower_reached_raw"]
        range_max = diag["upper_reached_raw"]
    else:
        range_min = deg_to_raw(result["lower_actual"])
        range_max = deg_to_raw(result["upper_actual"])

    return {
        "id": motor_id,
        "drive_mode": 0,
        "homing_offset": homing_offset,
        "range_min": range_min,
        "range_max": range_max,
    }


def build_extended_json(
    results: list[dict],
    backend_type: str,
    port: str | None,
    sweep_speed: float,
    settle_time: float,
    tolerance: float,
) -> dict:
    """Build the extended calibration JSON from sweep results."""
    joints_data: dict[str, dict] = {}
    for r in results:
        joint_name = r["joint"]
        motor_id = MOTOR_IDS[joint_name]
        lerobot_fields = compute_lerobot_fields(joint_name, r)

        joint_entry: dict = {
            "id": motor_id,
            "drive_mode": 0,
            "homing_offset": lerobot_fields["homing_offset"],
            "range_min": lerobot_fields["range_min"],
            "range_max": lerobot_fields["range_max"],
            "home_position_raw": HOME_POSITION_RAW[joint_name],
            "lower_limit_deg": r["lower_target"],
            "upper_limit_deg": r["upper_target"],
            "lower_actual_deg": r["lower_actual"],
            "upper_actual_deg": r["upper_actual"],
            "home_actual_deg": r["home_actual"],
            "max_error_deg": r["max_error"],
            "passed": r["passed"],
        }

        if "telemetry" in r:
            joint_entry["telemetry"] = r["telemetry"]
        if "diagnostics" in r:
            joint_entry["diagnostics"] = r["diagnostics"]

        joints_data[joint_name] = joint_entry

    return {
        "format_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend_type,
        "port": port,
        "sweep_params": {
            "speed_deg_s": sweep_speed,
            "settle_time_s": settle_time,
            "tolerance_deg": tolerance,
        },
        "joints": joints_data,
    }


def convert_to_lerobot(extended: dict) -> dict:
    """Extract LeRobot-compatible calibration from extended JSON.

    Args:
        extended: Extended calibration dict (as produced by build_extended_json).

    Returns:
        Dict matching LeRobot's calibration schema (5 fields per joint).
    """
    lerobot: dict[str, dict] = {}
    for joint_name, joint_data in extended["joints"].items():
        lerobot[joint_name] = {
            "id": joint_data["id"],
            "drive_mode": joint_data["drive_mode"],
            "homing_offset": joint_data["homing_offset"],
            "range_min": joint_data["range_min"],
            "range_max": joint_data["range_max"],
        }
    return lerobot


# ---------------------------------------------------------------------------
# CLI: subcommand helpers
# ---------------------------------------------------------------------------


def _add_sweep_args(parser: argparse.ArgumentParser) -> None:
    """Add common sweep arguments to a parser."""
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Use MuJoCo simulation instead of real hardware.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port for real hardware (auto-detect if omitted).",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=None,
        help="Test only these joints (default: all).",
    )
    parser.add_argument(
        "--skip-gripper",
        action="store_true",
        help="Skip gripper joint.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Pass/fail tolerance in degrees (default: 0.5 sim, 5.0 real).",
    )
    parser.add_argument(
        "--sweep-speed",
        type=float,
        default=20.0,
        help="Sweep speed in degrees/second. Default: 20.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.5,
        help="Settle time after each move in seconds. Default: 0.5.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save extended calibration JSON to this path.",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Skip telemetry collection (position-only, faster).",
    )
    parser.add_argument(
        "--no-test-positions",
        action="store_true",
        help="Always sweep from home position (ignore per-joint test positions).",
    )


def _resolve_backend(args: argparse.Namespace) -> tuple[ArmBackend, str, str | None]:
    """Create the appropriate backend from CLI args.

    Returns (backend, mode_str, port_str).
    """
    if args.sim:
        if not URDF_PATH.exists():
            print(f"ERROR: URDF not found at {URDF_PATH}")
            sys.exit(1)
        return SimArmBackend(URDF_PATH), "SIM (MuJoCo)", None
    else:
        port = args.port
        if port is None:
            import glob

            ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
            port = ports[0] if ports else None
        if port is None:
            print("ERROR: No serial port found. Use --port or --sim.")
            sys.exit(1)
        return RealArmBackend(port), f"REAL ({port})", port


def _resolve_joints(args: argparse.Namespace) -> list[str]:
    """Resolve which joints to test from CLI args."""
    if args.joints:
        joints = list(args.joints)
        for j in joints:
            if j not in JOINT_NAMES_WITH_GRIPPER:
                print(f"ERROR: Unknown joint '{j}'")
                print(f"  Valid joints: {JOINT_NAMES_WITH_GRIPPER}")
                sys.exit(1)
    else:
        joints = list(JOINT_NAMES_WITH_GRIPPER)

    if args.skip_gripper and "gripper" in joints:
        joints.remove("gripper")

    return joints


def build_base_positions(joint_name: str, use_test_positions: bool = True) -> dict[str, float]:
    """Build the base positions for sweeping a given joint.

    Starts with HOME_POSITION_DEG, then overlays any per-joint test positions
    from SWEEP_TEST_POSITIONS when *use_test_positions* is True.

    Args:
        joint_name: The joint about to be swept.
        use_test_positions: If False, always returns home positions.

    Returns:
        Dict of all joint positions to hold during the sweep.
    """
    base = {name: HOME_POSITION_DEG.get(name, 0.0) for name in JOINT_NAMES_WITH_GRIPPER}
    if use_test_positions and joint_name in SWEEP_TEST_POSITIONS:
        base.update(SWEEP_TEST_POSITIONS[joint_name])
    return base


def cmd_sweep(args: argparse.Namespace) -> None:
    """Execute the sweep subcommand."""
    tolerance = args.tolerance if args.tolerance is not None else (0.5 if args.sim else 5.0)
    collect_telemetry = not args.no_telemetry
    use_test_positions = not args.no_test_positions

    joints = _resolve_joints(args)
    if not joints:
        print("No joints to test.")
        sys.exit(0)

    backend, mode_str, port_str = _resolve_backend(args)
    backend_type = "sim" if args.sim else "real"

    print()
    print("=" * 50)
    print("JOINT ROM TEST")
    print("=" * 50)
    print(f"  Backend:     {mode_str}")
    print(f"  Joints:      {', '.join(joints)}")
    print(f"  Tolerance:   {tolerance:.1f} deg")
    print(f"  Sweep speed: {args.sweep_speed:.1f} deg/s")
    print(f"  Settle time: {args.settle_time:.2f} s")
    print(f"  Telemetry:   {'ON' if collect_telemetry else 'OFF'}")
    print(f"  Test pos:    {'ON' if use_test_positions else 'OFF'}")
    print()

    home_positions: dict[str, float] = {}
    for name in JOINT_NAMES_WITH_GRIPPER:
        home_positions[name] = HOME_POSITION_DEG.get(name, 0.0)

    backend.connect()
    try:
        print("Moving to home position...")
        backend.go_to_positions(home_positions)
        time.sleep(args.settle_time * 2)

        results = []
        for joint in joints:
            limits = JOINT_LIMITS_DEG.get(joint)
            if limits is None:
                print(f"  WARNING: No limits defined for {joint}, skipping.")
                continue
            lower, upper = limits

            # Build per-joint base positions (may differ from home)
            base_positions = build_base_positions(joint, use_test_positions)
            if use_test_positions and joint in SWEEP_TEST_POSITIONS:
                overrides = SWEEP_TEST_POSITIONS[joint]
                print(f"Testing {joint} [{lower:+.1f}, {upper:+.1f}] deg  (test pos: {overrides})")
            else:
                print(f"Testing {joint} [{lower:+.1f}, {upper:+.1f}] deg ...")

            # Move to base position before sweep
            backend.go_to_positions(base_positions)
            time.sleep(args.settle_time)

            result = sweep_joint(
                backend,
                joint,
                base_positions,
                lower,
                upper,
                args.sweep_speed,
                args.settle_time,
                tolerance,
                collect_telemetry=collect_telemetry,
            )
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  -> max error: {result['max_error']:.2f} deg  {status}")
            results.append(result)

            # Return to home between joints for safety
            backend.go_to_positions(home_positions)
            time.sleep(args.settle_time)

        # Final return to home
        print("Returning to home position...")
        backend.go_to_positions(home_positions)
        time.sleep(args.settle_time)

        exit_code = print_report(results)
        print_diagnostic_summary(results)

        # Save extended JSON if requested
        if args.output:
            output_path = Path(args.output)
        elif collect_telemetry:
            cal_dir = Path(__file__).parent.parent / "calibration"
            cal_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = cal_dir / f"rom_{ts}.json"
        else:
            output_path = None

        if output_path is not None:
            extended = build_extended_json(
                results,
                backend_type,
                port_str,
                args.sweep_speed,
                args.settle_time,
                tolerance,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(extended, indent=2, cls=_NumpyEncoder) + "\n")
            print(f"Extended calibration saved to: {output_path}")

    finally:
        backend.disconnect()

    sys.exit(exit_code)


def cmd_convert(args: argparse.Namespace) -> None:
    """Execute the convert subcommand."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    extended = json.loads(input_path.read_text())
    lerobot = convert_to_lerobot(extended)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "lerobot_calibration.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(lerobot, indent=4, cls=_NumpyEncoder) + "\n")
    print(f"LeRobot calibration saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Joint range-of-motion (ROM) test and calibration tool"
    )
    subparsers = parser.add_subparsers(dest="command")

    # sweep subcommand
    sweep_parser = subparsers.add_parser("sweep", help="Run ROM sweep test (default if no subcommand).")
    _add_sweep_args(sweep_parser)

    # convert subcommand
    convert_parser = subparsers.add_parser(
        "convert", help="Convert extended JSON to LeRobot calibration format."
    )
    convert_parser.add_argument(
        "--input", type=str, required=True, help="Path to extended calibration JSON."
    )
    convert_parser.add_argument(
        "--output", type=str, default=None, help="Output path for LeRobot JSON."
    )

    # Also add sweep args to the top-level parser for backward compatibility
    _add_sweep_args(parser)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "convert":
        cmd_convert(args)
    else:
        # Default to sweep (covers both explicit "sweep" and no subcommand)
        cmd_sweep(args)


if __name__ == "__main__":
    main()
