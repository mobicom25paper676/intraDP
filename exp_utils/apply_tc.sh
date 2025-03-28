#!/bin/bash

# Network Interface on which traffic control is applied
INTERFACE="eno1"
IFB_INTERFACE="ifb0"
TRACE_FILE="$1"  # e.g., bandwidth_trace.txt
PKT_LIMIT=64     # Packet limit for netem queue

# Cleanup function for removing traffic control settings
cleanup() {
    echo -e "\nCleaning up traffic control and IFB configurations..."
    sudo tc qdisc del dev $INTERFACE root 2>/dev/null
    sudo tc qdisc del dev $INTERFACE ingress 2>/dev/null
    sudo tc qdisc del dev $IFB_INTERFACE root 2>/dev/null
    sudo ip link set dev $IFB_INTERFACE down 2>/dev/null
    exit 0
}

# Ensure cleanup runs on script exit (CTRL+C, kill, exit)
trap cleanup EXIT SIGINT SIGTERM

# === 1. Enable IFB for ingress traffic mirroring ===
sudo ip link add name $IFB_INTERFACE type ifb 2>/dev/null
sudo ip link set dev $IFB_INTERFACE up  

# === 2. Clear existing tc settings ===
sudo tc qdisc del dev $INTERFACE root 2>/dev/null
sudo tc qdisc del dev $INTERFACE ingress 2>/dev/null
sudo tc qdisc del dev $IFB_INTERFACE root 2>/dev/null

# === 3. Configure EGRESS (Upload) Traffic Control (Using NetEm) ===
sudo tc qdisc replace dev $INTERFACE root netem rate 20mbit limit $PKT_LIMIT

# === 4. Configure INGRESS (Download) Traffic Control using IFB ===
sudo tc qdisc add dev $INTERFACE handle ffff: ingress  
sudo tc filter replace dev $INTERFACE parent ffff: protocol all u32 match u32 0 0 \
    action mirred egress redirect dev $IFB_INTERFACE

# Apply shaping on IFB (controls ingress traffic for INTERFACE)
sudo tc qdisc replace dev $IFB_INTERFACE root netem rate 20mbit limit $PKT_LIMIT

echo "✅ Traffic control initialized on $INTERFACE using $IFB_INTERFACE."

# Track previous timestamp for sleep timing
prev_timestamp=""

# === 5. Process bandwidth trace file ===
while read -r timestamp bandwidth_mbps; do
    # Convert Mbps to Kbit/s
    rate_kbit=$(echo "$bandwidth_mbps * 1000" | bc -l | awk '{printf "%.0f", $1}')
    
    # Apply updated bandwidth limit to both EGRESS (INTERFACE) and INGRESS (IFB)
    sudo tc qdisc replace dev $INTERFACE root netem rate ${rate_kbit}kbit limit $PKT_LIMIT
    sudo tc qdisc replace dev $IFB_INTERFACE root netem rate ${rate_kbit}kbit limit $PKT_LIMIT

    echo "[Time: $timestamp] Applied rate ${rate_kbit} Kbit/s"

    # Uncomment below to vary sleep time based on timestamps in trace file
    if [[ -n "$prev_timestamp" ]]; then
        time_diff=$(echo "$timestamp - $prev_timestamp" | bc -l)
        sleep "$time_diff"
    fi
    prev_timestamp="$timestamp"

done < "$TRACE_FILE"

# Cleanup when done
cleanup
