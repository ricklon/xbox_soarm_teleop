"""Read-only probe of the SO-101 follower: no writes, torque left as found."""

import sys

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

port = sys.argv[1]
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
bus = SerialMotorsBus(
    port, {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(names, 1)}
)
bus.connect(handshake=True)
try:
    rows = {}
    for field in ("Torque_Enable", "Present_Position", "Present_Voltage", "Present_Temperature"):
        try:
            rows[field] = bus.sync_read(field, normalize=False)
        except Exception as exc:  # noqa: BLE001
            rows[field] = {n: f"err {type(exc).__name__}" for n in names}
    print(f"{'joint':14s} {'torque':>6s} {'raw pos':>8s} {'deg*':>7s} {'volts':>6s} {'temp C':>6s}")
    for n in names:
        pos = rows["Present_Position"][n]
        deg = (pos - 2048) * 360 / 4096 if isinstance(pos, int) else float("nan")
        volts = rows["Present_Voltage"][n]
        volts = volts / 10 if isinstance(volts, int) else volts
        print(
            f"{n:14s} {rows['Torque_Enable'][n]!s:>6s} {pos!s:>8s} {deg:7.1f} {volts!s:>6s} {rows['Present_Temperature'][n]!s:>6s}"
        )
    print("* degrees from the encoder's middle (2048); not calibrated joint angles")
finally:
    bus.disconnect(disable_torque=False)
