#!/usr/bin/env python3
"""Emergency home position recovery.

Moves all servos to home position safely.
"""

import glob
import sys
import time

from xbox_soarm_teleop.config.joints import (
    HOME_POSITION_DEG,
    JOINT_NAMES_WITH_GRIPPER,
    MOTOR_IDS,
    deg_to_raw,
)


def find_port():
    ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    return ports[0] if ports else None


def go_home():
    port = find_port()
    if not port:
        print("ERROR: No serial port found")
        sys.exit(1)

    print(f"Connecting to {port}...")

    # Use low-level Feetech SDK directly
    import scservo_sdk as scs
    from scservo_sdk import PortHandler, sms_sts

    port_handler = PortHandler(port)
    packet_handler = sms_sts(port_handler)

    if not port_handler.openPort():
        print("ERROR: Failed to open port")
        sys.exit(1)

    port_handler.setBaudRate(1000000)
    print("Port opened at 1Mbps")

    # Enable torque and move to home
    print("\nMoving to home position...")
    print("Joint positions (degrees):")

    for name in JOINT_NAMES_WITH_GRIPPER:
        motor_id = MOTOR_IDS[name]
        deg = HOME_POSITION_DEG[name]
        raw = deg_to_raw(deg)

        # Enable torque
        packet_handler.write1ByteTxRx(motor_id, scs.SMS_STS_TORQUE_ENABLE, 1)

        # Write goal position (2 bytes, low/high)
        packet_handler.write2ByteTxRx(motor_id, scs.SMS_STS_GOAL_POSITION_L, raw)

        print(f"  {name:15s}: {deg:7.1f}° (raw={raw})")

    print("\nWaiting 3 seconds for movement...")
    time.sleep(3.0)

    # Read final positions
    print("\nFinal positions:")
    for name in JOINT_NAMES_WITH_GRIPPER:
        motor_id = MOTOR_IDS[name]
        pos_raw, _, _ = packet_handler.read2ByteTxRx(motor_id, scs.SMS_STS_PRESENT_POSITION_L)
        vel_raw, _, _ = packet_handler.read2ByteTxRx(motor_id, scs.SMS_STS_PRESENT_SPEED_L)
        temp_raw, _, _ = packet_handler.read1ByteTxRx(motor_id, scs.SMS_STS_PRESENT_TEMPERATURE)
        volt_raw, _, _ = packet_handler.read1ByteTxRx(motor_id, scs.SMS_STS_PRESENT_VOLTAGE)

        from xbox_soarm_teleop.config.joints import raw_to_deg

        pos_deg = raw_to_deg(pos_raw)
        voltage = volt_raw / 10.0

        status = "OK" if abs(pos_deg - HOME_POSITION_DEG[name]) < 5 else "CHECK"
        print(f"  {name:15s}: {pos_deg:7.1f}° temp={temp_raw:3}°C volt={voltage:.1f}V [{status}]")

    port_handler.closePort()
    print("\nDone!")


if __name__ == "__main__":
    try:
        go_home()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
