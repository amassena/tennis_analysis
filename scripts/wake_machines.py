#!/usr/bin/env python3
"""Wake up GPU machines using Wake-on-LAN.

Note: WoL only works when:
1. Machine is on same local network, OR
2. Router is configured for WoL forwarding, OR
3. Another machine on the target LAN can relay the packet

For cross-network wake via Tailscale, the machines need to stay
connected to Tailscale even when sleeping (usually requires BIOS setting).
"""

import argparse
import socket
import struct
import subprocess
import sys
import time

# Machine configurations
MACHINES = {
    "windows": {
        "mac": "34:5A:60:BB:6D:E7",
        "tailscale_ip": "100.81.64.103",
    },
    "tmassena": {
        "mac": "74:56:3C:4D:73:C8",
        "tailscale_ip": "100.98.226.93",
    },
}

# Broadcast addresses to try
BROADCAST_ADDRS = [
    "255.255.255.255",      # Local broadcast
    "192.168.1.255",        # Common home network
    "192.168.0.255",        # Common home network
    "192.168.7.255",        # From your network config
    "172.20.10.15",         # From your network config
]


def create_magic_packet(mac_address: str) -> bytes:
    """Create a Wake-on-LAN magic packet."""
    # Normalize MAC address
    mac = mac_address.replace(":", "").replace("-", "")
    if len(mac) != 12:
        raise ValueError(f"Invalid MAC address: {mac_address}")

    # Convert to bytes
    mac_bytes = bytes.fromhex(mac)

    # Magic packet: 6 bytes of 0xFF followed by MAC repeated 16 times
    return b'\xff' * 6 + mac_bytes * 16


def send_wol_packet(mac_address: str, broadcast_addr: str = "255.255.255.255", port: int = 9):
    """Send a Wake-on-LAN packet."""
    packet = create_magic_packet(mac_address)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    try:
        sock.sendto(packet, (broadcast_addr, port))
        return True
    except Exception as e:
        print(f"  Error sending to {broadcast_addr}: {e}")
        return False
    finally:
        sock.close()


def check_machine_awake(hostname: str, timeout: int = 5) -> bool:
    """Check if machine is reachable via SSH."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout={}".format(timeout), "-o", "BatchMode=yes",
             hostname, "echo ok"],
            capture_output=True,
            text=True,
            timeout=timeout + 2
        )
        return result.returncode == 0
    except:
        return False


def send_wol_via_relay(target_mac: str, relay_host: str) -> bool:
    """Send WoL packet via another machine on the same LAN."""
    # PowerShell command to send WoL
    ps_cmd = f'''$mac = '{target_mac}' -replace ':',''; $bytes = [byte[]](@(0xFF)*6 + (@([convert]::ToByte($mac.Substring(0,2),16), [convert]::ToByte($mac.Substring(2,2),16), [convert]::ToByte($mac.Substring(4,2),16), [convert]::ToByte($mac.Substring(6,2),16), [convert]::ToByte($mac.Substring(8,2),16), [convert]::ToByte($mac.Substring(10,2),16))*16)); $udp = New-Object System.Net.Sockets.UdpClient; $udp.Connect('255.255.255.255', 9); $udp.Send($bytes, $bytes.Length); $udp.Close()'''

    try:
        result = subprocess.run(
            ["ssh", relay_host, f'powershell -Command "{ps_cmd}"'],
            capture_output=True,
            text=True,
            timeout=15
        )
        return result.returncode == 0
    except:
        return False


def wake_machine(name: str, wait: bool = True, timeout: int = 60) -> bool:
    """Wake a machine and optionally wait for it to come online."""
    if name not in MACHINES:
        print(f"Unknown machine: {name}")
        print(f"Available: {', '.join(MACHINES.keys())}")
        return False

    config = MACHINES[name]
    mac = config["mac"]

    print(f"Waking {name} ({mac})...")

    # First check if already awake
    if check_machine_awake(name, timeout=3):
        print(f"  {name} is already awake!")
        return True

    # Send WoL packets to multiple broadcast addresses
    sent = False
    for addr in BROADCAST_ADDRS:
        try:
            if send_wol_packet(mac, addr):
                print(f"  Sent magic packet to {addr}")
                sent = True
        except:
            pass

    # Also try sending directly to Tailscale IP (might work if machine maintains connection)
    try:
        send_wol_packet(mac, config.get("tailscale_ip", "255.255.255.255"))
    except:
        pass

    # Try relay via other awake machines on same LAN
    for relay_name, relay_config in MACHINES.items():
        if relay_name != name and check_machine_awake(relay_name, timeout=3):
            print(f"  Trying relay via {relay_name}...")
            if send_wol_via_relay(mac, relay_name):
                print(f"  Sent magic packet via {relay_name}")
                sent = True
                break

    if not sent:
        print("  Warning: Could not send any WoL packets")
        return False

    if not wait:
        print("  Magic packet sent. Not waiting for machine to wake.")
        return True

    # Wait for machine to come online
    print(f"  Waiting for {name} to wake up (timeout: {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        if check_machine_awake(name, timeout=3):
            elapsed = int(time.time() - start)
            print(f"  {name} is awake! (took {elapsed}s)")
            return True
        time.sleep(5)

    print(f"  Timeout: {name} did not respond within {timeout}s")
    return False


def wake_all(wait: bool = True) -> dict:
    """Wake all machines."""
    results = {}
    for name in MACHINES:
        results[name] = wake_machine(name, wait=wait)
    return results


def main():
    parser = argparse.ArgumentParser(description="Wake GPU machines using Wake-on-LAN")
    parser.add_argument("machine", nargs="?", choices=list(MACHINES.keys()) + ["all"],
                        default="all", help="Machine to wake (default: all)")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for machine to come online")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds (default: 60)")
    parser.add_argument("--status", action="store_true", help="Just check if machines are awake")
    args = parser.parse_args()

    if args.status:
        print("Checking machine status...")
        for name in MACHINES:
            awake = check_machine_awake(name, timeout=5)
            status = "AWAKE" if awake else "SLEEPING/UNREACHABLE"
            print(f"  {name}: {status}")
        return

    if args.machine == "all":
        results = wake_all(wait=not args.no_wait)
        success = all(results.values())
    else:
        success = wake_machine(args.machine, wait=not args.no_wait, timeout=args.timeout)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
