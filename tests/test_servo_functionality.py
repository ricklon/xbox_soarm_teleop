"""Test servo functionality - checks each servo responds and reports position.

These tests verify that all 6 servos (IDs 1-6) on the SO-ARM101 are:
- Connected and communicating
- Reporting valid position values
- Within reasonable temperature ranges

Run with: uv run pytest tests/test_servo_functionality.py -v
"""

import glob
from typing import Generator

import pytest

try:
    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode
except ImportError:
    pytest.skip("LeRobot not installed", allow_module_level=True)


MOTOR_IDS = [1, 2, 3, 4, 5, 6]
MOTOR_NAMES = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}
MAX_TEMPERATURE = 60  # Celsius - warn if above this


def find_port() -> str | None:
    """Find available serial port."""
    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


@pytest.fixture(scope="module")
def serial_port() -> str:
    """Find and return serial port for robot connection."""
    port = find_port()
    if port is None:
        pytest.skip("No serial port found - connect robot or specify --port")
    return port


@pytest.fixture(scope="module")
def motor_bus(serial_port: str) -> Generator[FeetechMotorsBus, None, None]:
    """Initialize motor bus, connect, and yield for tests."""
    motors = {str(i): Motor(id=i, model="sts3215", norm_mode=MotorNormMode.DEGREES) for i in MOTOR_IDS}
    bus = FeetechMotorsBus(port=serial_port, motors=motors)

    try:
        bus.connect()
        yield bus
    finally:
        bus.disconnect()


class TestServoConnectivity:
    """Test that all servos are connected and responsive."""

    @pytest.mark.parametrize("motor_id", MOTOR_IDS)
    def test_servo_responds(self, motor_bus: FeetechMotorsBus, motor_id: int) -> None:
        """Verify servo responds to position read request."""
        name = MOTOR_NAMES[motor_id]

        try:
            position = motor_bus.read("Present_Position", str(motor_id), normalize=False)
            # Raw position should be 0-4095 for STS3215 servos
            assert 0 <= position <= 4095, f"Motor {motor_id} ({name}) position {position} out of range"
        except Exception as e:
            pytest.fail(f"Motor {motor_id} ({name}) failed to respond: {e}")

    @pytest.mark.parametrize("motor_id", MOTOR_IDS)
    def test_servo_position_readable(self, motor_bus: FeetechMotorsBus, motor_id: int) -> None:
        """Verify servo position can be read."""
        name = MOTOR_NAMES[motor_id]

        # Read raw position
        pos_raw = motor_bus.read("Present_Position", str(motor_id), normalize=False)
        assert isinstance(pos_raw, int), f"Motor {motor_id} ({name}) raw position not an int"

    @pytest.mark.parametrize("motor_id", MOTOR_IDS)
    def test_servo_velocity_readable(self, motor_bus: FeetechMotorsBus, motor_id: int) -> None:
        """Verify servo velocity can be read."""
        name = MOTOR_NAMES[motor_id]

        try:
            velocity = motor_bus.read("Present_Velocity", str(motor_id), normalize=False)
            assert isinstance(velocity, int), f"Motor {motor_id} ({name}) velocity not an int"
        except Exception as e:
            pytest.fail(f"Motor {motor_id} ({name}) failed to read velocity: {e}")


class TestServoHealth:
    """Test servo health metrics."""

    @pytest.mark.parametrize("motor_id", MOTOR_IDS)
    def test_servo_temperature_readable(self, motor_bus: FeetechMotorsBus, motor_id: int) -> None:
        """Verify servo temperature can be read."""
        name = MOTOR_NAMES[motor_id]

        try:
            temp = motor_bus.read("Present_Temperature", str(motor_id), normalize=False)
            assert isinstance(temp, int), f"Motor {motor_id} ({name}) temperature not an int"
            assert 0 <= temp <= 100, f"Motor {motor_id} ({name}) temperature {temp} out of valid range"
        except Exception as e:
            pytest.fail(f"Motor {motor_id} ({name}) failed to read temperature: {e}")

    @pytest.mark.parametrize("motor_id", MOTOR_IDS)
    def test_servo_temperature_warning(self, motor_bus: FeetechMotorsBus, motor_id: int) -> None:
        """Warn if servo temperature is high (but don't fail)."""
        name = MOTOR_NAMES[motor_id]

        try:
            temp = motor_bus.read("Present_Temperature", str(motor_id), normalize=False)
            if temp > MAX_TEMPERATURE:
                pytest.warns(
                    UserWarning,
                    match=f"Motor {motor_id} ({name}) temperature {temp}°C exceeds {MAX_TEMPERATURE}°C",
                )
        except Exception:
            pytest.skip(f"Cannot read temperature for motor {motor_id}")


class TestServoPositions:
    """Test servo position readings."""

    def test_all_positions_reported(self, motor_bus: FeetechMotorsBus) -> None:
        """Verify all 6 servos report positions."""
        positions = {}

        for motor_id in MOTOR_IDS:
            name = MOTOR_NAMES[motor_id]
            try:
                pos = motor_bus.read("Present_Position", str(motor_id), normalize=False)
                positions[motor_id] = (name, pos)
            except Exception as e:
                pytest.fail(f"Failed to read motor {motor_id} ({name}): {e}")

        assert len(positions) == 6, f"Only {len(positions)}/6 servos reported positions"

        # Print positions for debugging
        print("\nServo Positions (raw 0-4095):")
        for motor_id, (name, pos) in positions.items():
            print(f"  ID {motor_id} ({name:12s}): {pos:7d}")

    def test_positions_in_valid_range(self, motor_bus: FeetechMotorsBus) -> None:
        """Verify all servo positions are within valid range."""
        for motor_id in MOTOR_IDS:
            name = MOTOR_NAMES[motor_id]
            pos = motor_bus.read("Present_Position", str(motor_id), normalize=False)

            # STS3215 servos have 0-4095 raw range
            assert 0 <= pos <= 4095, (
                f"Motor {motor_id} ({name}) position {pos} out of valid range [0, 4095]"
            )

    def test_position_consistency(self, motor_bus: FeetechMotorsBus) -> None:
        """Verify positions are stable (don't change wildly between reads)."""
        # Read twice with small delay
        first_read = {}
        second_read = {}

        for motor_id in MOTOR_IDS:
            first_read[motor_id] = motor_bus.read("Present_Position", str(motor_id), normalize=False)

        # Small delay
        import time

        time.sleep(0.1)

        for motor_id in MOTOR_IDS:
            second_read[motor_id] = motor_bus.read("Present_Position", str(motor_id), normalize=False)

        # Check positions are similar (shouldn't change more than 50 raw units unless moving)
        for motor_id in MOTOR_IDS:
            name = MOTOR_NAMES[motor_id]
            diff = abs(second_read[motor_id] - first_read[motor_id])
            assert diff < 50, (
                f"Motor {motor_id} ({name}) position unstable: "
                f"{first_read[motor_id]} -> {second_read[motor_id]}"
            )


def test_summary_report(motor_bus: FeetechMotorsBus) -> None:
    """Generate a summary report of all servo statuses."""
    print("\n" + "=" * 70)
    print("SERVO FUNCTIONALITY TEST SUMMARY")
    print("=" * 70)
    print(f"{'ID':<4} {'Name':<15} {'Position':<12} {'Velocity':<12} {'Temp':<10} {'Status':<10}")
    print("-" * 70)

    results = []

    for motor_id in MOTOR_IDS:
        name = MOTOR_NAMES[motor_id]

        try:
            pos = motor_bus.read("Present_Position", str(motor_id), normalize=False)

            try:
                vel = motor_bus.read("Present_Velocity", str(motor_id), normalize=False)
            except Exception:
                vel = None

            try:
                temp = motor_bus.read("Present_Temperature", str(motor_id), normalize=False)
            except Exception:
                temp = None

            results.append(
                {
                    "id": motor_id,
                    "name": name,
                    "position": pos,
                    "velocity": vel,
                    "temperature": temp,
                    "status": "OK",
                }
            )

        except Exception as e:
            results.append(
                {
                    "id": motor_id,
                    "name": name,
                    "position": None,
                    "velocity": None,
                    "temperature": None,
                    "status": f"ERROR: {e}",
                }
            )

    # Print results
    for r in results:
        pos_str = f"{r['position']:.2f}°" if r["position"] is not None else "N/A"
        vel_str = f"{r['velocity']:.2f}" if r["velocity"] is not None else "N/A"
        temp_str = f"{r['temperature']}°C" if r["temperature"] is not None else "N/A"

        status = "✓ OK" if r["status"] == "OK" else "✗ FAIL"

        print(f"{r['id']:<4} {r['name']:<15} {pos_str:<12} {vel_str:<12} {temp_str:<10} {status:<10}")

    print("=" * 70)

    # Assert all passed
    failures = [r for r in results if r["status"] != "OK"]
    if failures:
        fail_list = ", ".join(f"ID{r['id']} ({r['name']})" for r in failures)
        pytest.fail(f"Servo test failures: {fail_list}")
