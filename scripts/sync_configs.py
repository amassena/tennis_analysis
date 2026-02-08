#!/usr/bin/env python3
"""Sync and verify configs across all GPU machines.

Ensures all machines have identical configs before processing.
Automatically wakes sleeping machines before syncing.
Run this before starting workers to prevent false starts.
"""

import hashlib
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Import WoL functions
try:
    from scripts.wake_machines import wake_machine, check_machine_awake
    WOL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: WoL not available ({e})")
    WOL_AVAILABLE = False

# Files that must be synced
CONFIG_FILES = [
    ".env",
    "gpu_worker/worker.py",
    "gpu_worker/start_worker_loop.bat",
    "scripts/upload.py",
    "config/settings.py",
    "preprocess_nvenc.py",
]

# GPU machines - host is Tailscale, local_host is for fixing Tailscale when it's down
MACHINES = [
    {
        "name": "windows",
        "host": "windows",  # Tailscale SSH config entry
        "local_host": "amass@192.168.1.170",  # Local network fallback (update IP as needed)
        "path": "C:/Users/amass/tennis_analysis",
    },
    {
        "name": "tmassena",
        "host": "tmassena",
        "local_host": "amass@192.168.1.171",  # Local network fallback (update IP as needed)
        "path": "C:/Users/amass/tennis_analysis",
    },
    {
        "name": "desktop3090",
        "host": "desktop3090",
        "local_host": "Andrew@desktop-3olaa45.home.local",  # Discovered via arp
        "path": "C:/Users/Andrew/tennis_analysis",
    },
]


def try_ssh(host, timeout=5):
    """Try SSH to a host. Returns True if successful."""
    cmd = f"ssh -o ConnectTimeout={timeout} -o BatchMode=yes {host} echo ok"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout + 2)
        return result.returncode == 0
    except:
        return False


def fix_tailscale_via_local(machine):
    """SSH via local network and restart Tailscale. Returns True if Tailscale now works."""
    local_host = machine.get("local_host")
    if not local_host:
        return False

    print(f"trying local SSH to fix Tailscale...", end=" ", flush=True)

    # First check if local SSH works
    if not try_ssh(local_host, timeout=5):
        print("local SSH failed")
        return False

    # Check Tailscale status and restart if needed
    cmd = f'ssh -o ConnectTimeout=5 {local_host} "powershell -Command \\"& {{Start-Process tailscale -ArgumentList \'up\' -Verb RunAs -Wait; tailscale status}}\\""'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("Tailscale restarted...", end=" ", flush=True)
            # Wait a moment for Tailscale to connect
            time.sleep(3)
            # Now check if Tailscale SSH works
            if try_ssh(machine["host"], timeout=5):
                print("Tailscale restored!")
                return True
            else:
                print("Tailscale still down")
    except Exception as e:
        print(f"error: {e}")

    return False


def get_local_hash(filepath):
    """Get MD5 hash of local file."""
    full_path = PROJECT_ROOT / filepath
    if not full_path.exists():
        return None
    return hashlib.md5(full_path.read_bytes()).hexdigest()[:8]


def get_remote_hash(machine, remote_path):
    """Get MD5 hash of remote file via SSH (always uses Tailscale)."""
    host = machine["host"]
    # Use certutil on Windows to get hash
    cmd = f'ssh {host} "certutil -hashfile \\"{remote_path}\\" MD5 2>nul | findstr /v MD5 | findstr /v CertUtil"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip().replace(" ", "")[:8]
    except:
        pass
    return None


def sync_file(filepath, machine):
    """Sync a file to remote machine (always uses Tailscale)."""
    host = machine["host"]
    local_path = PROJECT_ROOT / filepath

    cmd = f"scp '{local_path}' {host}:'{machine['path']}/{filepath}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
    return result.returncode == 0


def check_machine_online(machine):
    """Check if machine is reachable via Tailscale. If not, try to fix Tailscale via local SSH."""
    host = machine["host"]

    # Try Tailscale first
    if try_ssh(host, timeout=5):
        return True

    # Tailscale failed - try to fix it via local network
    if fix_tailscale_via_local(machine):
        return True

    return False


def ensure_machine_online(machine, max_attempts=2):
    """Wake a machine that is offline. Returns True if now reachable via Tailscale."""
    if not WOL_AVAILABLE:
        print("OFFLINE (WoL unavailable)")
        return False

    print("OFFLINE - waking...", end=" ", flush=True)

    for attempt in range(1, max_attempts + 1):
        try:
            wake_machine(machine["host"], wait=False)
        except Exception as e:
            print(f"WoL error: {e}")
            return False

        # Poll for ~30 seconds
        for _ in range(6):
            time.sleep(5)
            # Try Tailscale first
            if try_ssh(machine["host"], timeout=3):
                print("AWAKE")
                return True
            # Try local SSH to fix Tailscale
            if fix_tailscale_via_local(machine):
                return True

        if attempt < max_attempts:
            print(f"retrying...", end=" ", flush=True)

    print("FAILED (timed out)")
    return False


def main():
    print("=" * 50)
    print("Config Sync & Verify")
    print("(auto-wakes machines, fixes Tailscale if needed)")
    print("=" * 50)

    # Check which machines are online (wake and fix Tailscale if needed)
    online_machines = []
    for machine in MACHINES:
        print(f"\nChecking {machine['name']}...", end=" ", flush=True)

        if check_machine_online(machine):
            print("ONLINE")
            online_machines.append(machine)
        elif ensure_machine_online(machine):
            online_machines.append(machine)
        # else ensure_machine_online already printed failure status

    if not online_machines:
        print("\nNo machines online!")
        sys.exit(1)

    # Check and sync each file
    all_synced = True
    for filepath in CONFIG_FILES:
        print(f"\n{filepath}:")
        local_hash = get_local_hash(filepath)
        if not local_hash:
            print(f"  LOCAL: MISSING")
            continue
        print(f"  LOCAL: {local_hash}")

        for machine in online_machines:
            remote_path = f"{machine['path']}/{filepath}".replace("/", "\\\\")
            remote_hash = get_remote_hash(machine, remote_path)

            if remote_hash == local_hash:
                print(f"  {machine['name']}: {remote_hash} ✓")
            elif remote_hash:
                print(f"  {machine['name']}: {remote_hash} ✗ MISMATCH - syncing...")
                if sync_file(filepath, machine):
                    print(f"  {machine['name']}: SYNCED ✓")
                else:
                    print(f"  {machine['name']}: SYNC FAILED ✗")
                    all_synced = False
            else:
                print(f"  {machine['name']}: MISSING - syncing...")
                if sync_file(filepath, machine):
                    print(f"  {machine['name']}: SYNCED ✓")
                else:
                    print(f"  {machine['name']}: SYNC FAILED ✗")
                    all_synced = False

    print("\n" + "=" * 50)
    if all_synced:
        print("All configs synced and verified!")
        print("=" * 50)
        return 0
    else:
        print("WARNING: Some configs failed to sync!")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
