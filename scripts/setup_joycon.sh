#!/usr/bin/env bash
# Setup Right Joy-Con for use with xbox_soarm_teleop on Ubuntu 22.04+/kernel 6.8+
# Run once as root: sudo bash scripts/setup_joycon.sh

set -e

echo "=== Joy-Con Linux Setup ==="

# 1. Load hid-nintendo driver
echo "[1/5] Loading hid-nintendo kernel module..."
modprobe hid_nintendo
echo "hid_nintendo" >> /etc/modules-load.d/hid-nintendo.conf 2>/dev/null || true
echo "  OK"

# 2. BlueZ input config
echo "[2/5] Configuring BlueZ input plugin..."
cat > /etc/bluetooth/input.conf << 'EOF'
[General]
ClassicBondedOnly=false
UserspaceHID=true
EOF
echo "  OK"

# 3. udev rules for Joy-Con permissions
echo "[3/5] Installing udev rules..."
cat > /etc/udev/rules.d/70-joycon.rules << 'EOF'
SUBSYSTEM=="input", KERNEL=="event*", ATTRS{name}=="Joy-Con*", RUN+="/bin/chmod 0660 /dev/%k"
KERNEL=="hidraw*", KERNELS=="*057E:2007*", RUN+="/bin/chmod 0660 /dev/%k", RUN+="/bin/chgrp input /dev/%k"
EOF
udevadm control --reload-rules
echo "  OK"

# 4. Build and install joycond
echo "[4/5] Installing joycond..."
if ! systemctl is-active --quiet joycond 2>/dev/null; then
    apt-get install -y --quiet cmake libevdev-dev libudev-dev build-essential git
    TMP=$(mktemp -d)
    git clone --quiet https://github.com/DanielOgorchock/joycond "$TMP/joycond"
    cd "$TMP/joycond" && cmake -DCMAKE_BUILD_TYPE=Release . -Wno-dev && make -j"$(nproc)" && make install
    systemctl enable --now joycond
    cd - > /dev/null
else
    echo "  joycond already running, skipping"
fi
echo "  OK"

# 5. Restart bluetooth
echo "[5/5] Restarting bluetooth..."
systemctl restart bluetooth
echo "  OK"

echo ""
echo "=== Done ==="
echo ""
echo "To connect the Joy-Con:"
echo "  1. Hold the sync button (rail side) for 3 seconds until LEDs chase"
echo "  2. bluetoothctl connect 98:B6:AF:5E:5D:39"
echo "  3. Press SL+SR on the Joy-Con to activate single-controller mode"
echo "  4. uv run python examples/probe_controller.py"
