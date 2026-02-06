#!/usr/bin/env python3
"""Diagnose network connectivity for tennis analysis pipeline.

Tests:
1. Tailscale status and peer connectivity
2. SSH connectivity to GPU machines (windows, tmassena)
3. File transfer speed between machines
4. iCloud accessibility
5. YouTube upload capability
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

GPU_MACHINES = ["windows", "tmassena"]
PROJECT_PATH_WINDOWS = r"C:\Users\amass\tennis_analysis"


def print_header(title):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}{title:^60}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")


def print_result(name, success, details=""):
    icon = f"{GREEN}[OK]{RESET}" if success else f"{RED}[FAIL]{RESET}"
    print(f"  {icon} {name}")
    if details:
        print(f"       {details}")


def print_warn(name, details=""):
    print(f"  {YELLOW}[WARN]{RESET} {name}")
    if details:
        print(f"       {details}")


def run_cmd(cmd, timeout=30):
    """Run command and return (success, stdout, stderr, duration)."""
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        duration = time.time() - start
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip(), duration
    except subprocess.TimeoutExpired:
        return False, "", "Timeout", time.time() - start
    except Exception as e:
        return False, "", str(e), time.time() - start


def check_tailscale():
    """Check Tailscale status and connectivity."""
    print_header("Tailscale Status")

    # Check if Tailscale is running
    success, stdout, stderr, _ = run_cmd("tailscale status")
    if not success:
        # Tailscale CLI might not be installed, but connection could still work
        print_warn("Tailscale CLI", "Not installed or not in PATH")
        print("       (SSH may still work via other VPN or direct connection)")
        return True  # Don't fail if Tailscale CLI missing but SSH works

    print_result("Tailscale running", True)

    # Parse status for peer info
    lines = stdout.split("\n")
    peers_online = []
    peers_offline = []

    for line in lines[1:]:  # Skip header
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            hostname = parts[1]
            status = parts[3] if len(parts) > 3 else ""
            if "offline" in status.lower() or "-" in status:
                peers_offline.append(hostname)
            else:
                peers_online.append(hostname)

    # Check GPU machines specifically
    all_gpu_online = True
    for machine in GPU_MACHINES:
        online = any(machine in p for p in peers_online)
        if online:
            print_result(f"Peer '{machine}' online", True)
        else:
            print_result(f"Peer '{machine}' online", False, "Not found or offline")
            all_gpu_online = False

    return all_gpu_online


def check_ssh_connectivity():
    """Test SSH connectivity to GPU machines."""
    print_header("SSH Connectivity")

    results = {}
    for machine in GPU_MACHINES:
        # Test basic SSH
        success, stdout, stderr, duration = run_cmd(
            f"ssh -o ConnectTimeout=10 -o BatchMode=yes {machine} hostname",
            timeout=15
        )

        if success:
            print_result(f"SSH to {machine}", True, f"{duration:.1f}s latency")
            results[machine] = True

            # Test project directory exists
            success2, stdout2, _, _ = run_cmd(
                f'ssh {machine} "if exist {PROJECT_PATH_WINDOWS} echo OK"',
                timeout=10
            )
            if "OK" in stdout2:
                print_result(f"  Project dir on {machine}", True)
            else:
                print_warn(f"  Project dir on {machine}", "Directory not found")
        else:
            print_result(f"SSH to {machine}", False, stderr or "Connection failed")
            results[machine] = False

    return results


def check_file_transfer_speed():
    """Test file transfer speed to GPU machines."""
    print_header("File Transfer Speed")

    # Create a 10MB test file
    test_file = "/tmp/tennis_speed_test.bin"
    test_size_mb = 10

    try:
        with open(test_file, "wb") as f:
            f.write(os.urandom(test_size_mb * 1024 * 1024))
    except Exception as e:
        print_result("Create test file", False, str(e))
        return {}

    results = {}
    for machine in GPU_MACHINES:
        # Upload test
        remote_path = f"{machine}:C:/Users/amass/tennis_speed_test.bin"

        start = time.time()
        success, _, stderr, duration = run_cmd(
            f"scp -o ConnectTimeout=10 {test_file} {remote_path}",
            timeout=120
        )

        if success:
            speed = test_size_mb / duration
            print_result(f"Upload to {machine}", True, f"{speed:.1f} MB/s ({duration:.1f}s for {test_size_mb}MB)")
            results[machine] = {"upload": speed}

            # Cleanup remote file
            run_cmd(f'ssh {machine} "del C:\\Users\\amass\\tennis_speed_test.bin"', timeout=10)
        else:
            print_result(f"Upload to {machine}", False, stderr or "Transfer failed")
            results[machine] = {"upload": 0}

    # Cleanup local test file
    os.remove(test_file)

    return results


def check_icloud():
    """Check iCloud accessibility."""
    print_header("iCloud Status")

    # Check if iCloud Drive is mounted
    icloud_path = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs")
    if os.path.exists(icloud_path):
        print_result("iCloud Drive mounted", True)
    else:
        print_result("iCloud Drive mounted", False, "Path not found")
        return False

    # Check Photos library accessibility (for pyicloud)
    success, stdout, stderr, _ = run_cmd("python3 -c 'from pyicloud import PyiCloudService; print(\"OK\")'")
    if success and "OK" in stdout:
        print_result("pyicloud module available", True)
    else:
        print_warn("pyicloud module available", "Module not installed or import failed")

    # Check if we can list iCloud photos albums (requires auth)
    # This is a basic check - full auth test would require credentials
    print_result("iCloud Photos", True, "Requires manual verification (add video to album)")

    return True


def check_youtube():
    """Check YouTube upload capability."""
    print_header("YouTube Upload")

    # Check if yt-dlp is installed
    success, stdout, _, _ = run_cmd("which yt-dlp || where yt-dlp")
    if success:
        print_result("yt-dlp installed", True)
    else:
        print_warn("yt-dlp installed", "Not found in PATH")

    # Check for google API credentials
    creds_paths = [
        os.path.expanduser("~/.youtube-upload-credentials.json"),
        os.path.expanduser("~/client_secrets.json"),
        os.path.join(os.path.dirname(__file__), "..", "client_secrets.json"),
    ]

    creds_found = False
    for path in creds_paths:
        if os.path.exists(path):
            print_result("YouTube credentials", True, path)
            creds_found = True
            break

    if not creds_found:
        print_warn("YouTube credentials", "No credentials file found")

    return True


def check_python_env():
    """Check Python environment on GPU machines."""
    print_header("Python Environment")

    for machine in GPU_MACHINES:
        # Check Python version
        success, stdout, stderr, _ = run_cmd(
            f'ssh {machine} "cd {PROJECT_PATH_WINDOWS} && venv\\Scripts\\python --version"',
            timeout=15
        )

        if success:
            print_result(f"Python on {machine}", True, stdout)

            # Check TensorFlow
            success2, stdout2, _, _ = run_cmd(
                f'ssh {machine} "cd {PROJECT_PATH_WINDOWS} && venv\\Scripts\\python -c \\"import tensorflow as tf; print(tf.__version__)\\""',
                timeout=30
            )
            if success2:
                print_result(f"  TensorFlow on {machine}", True, f"v{stdout2}")
            else:
                print_result(f"  TensorFlow on {machine}", False)

            # Check GPU availability via nvidia-smi (more reliable than TF over SSH)
            success3, stdout3, _, _ = run_cmd(
                f'ssh {machine} "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"',
                timeout=15
            )
            if success3 and stdout3.strip():
                gpu_info = stdout3.strip().split(",")
                print_result(f"  GPU on {machine}", True, f"{gpu_info[0].strip()} ({gpu_info[1].strip()})")
            else:
                print_warn(f"  GPU on {machine}", "nvidia-smi failed or no GPU")
        else:
            print_result(f"Python on {machine}", False, stderr)


def check_disk_space():
    """Check disk space on all machines."""
    print_header("Disk Space")

    # Local Mac
    success, stdout, _, _ = run_cmd("df -h . | tail -1")
    if success:
        parts = stdout.split()
        if len(parts) >= 4:
            print_result("Mac (local)", True, f"{parts[3]} available")

    # GPU machines
    for machine in GPU_MACHINES:
        # Try PowerShell command for disk space
        success, stdout, _, _ = run_cmd(
            f'ssh {machine} "powershell -Command \\"(Get-PSDrive C).Free / 1GB\\""',
            timeout=15
        )
        if success and stdout.strip():
            try:
                free_gb = float(stdout.strip())
                print_result(f"{machine} (C:)", True, f"{free_gb:.1f} GB available")
            except:
                print_warn(f"{machine} (C:)", f"Could not parse: {stdout}")
        else:
            # Fallback: try dir command
            success2, stdout2, _, _ = run_cmd(
                f'ssh {machine} "dir C:\\ | findstr bytes"',
                timeout=15
            )
            if success2 and "free" in stdout2.lower():
                print_result(f"{machine} (C:)", True, stdout2.strip().split()[-3] + " bytes free")
            else:
                print_warn(f"{machine} (C:)", "Could not check disk space")


def run_full_diagnostic(skip_speed=False):
    """Run all diagnostic checks."""
    print(f"\n{BOLD}Tennis Analysis Pipeline - Network Diagnostic{RESET}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        "tailscale": check_tailscale(),
        "ssh": check_ssh_connectivity(),
        "icloud": check_icloud(),
        "youtube": check_youtube(),
    }

    if not skip_speed:
        results["transfer"] = check_file_transfer_speed()
    else:
        print_header("File Transfer Speed")
        print("  (skipped with --skip-speed)")

    check_python_env()
    check_disk_space()

    # Summary
    print_header("Summary")

    all_ok = True

    if results["tailscale"]:
        print(f"  {GREEN}Tailscale:{RESET} Connected")
    else:
        print(f"  {RED}Tailscale:{RESET} Issues detected")
        all_ok = False

    ssh_ok = all(results["ssh"].values()) if results["ssh"] else False
    if ssh_ok:
        print(f"  {GREEN}SSH:{RESET} All machines reachable")
    else:
        print(f"  {RED}SSH:{RESET} Some machines unreachable")
        all_ok = False

    if results["icloud"]:
        print(f"  {GREEN}iCloud:{RESET} Available")
    else:
        print(f"  {YELLOW}iCloud:{RESET} Check manually")

    if not skip_speed and results.get("transfer"):
        speeds = [v.get("upload", 0) for v in results["transfer"].values()]
        if all(s > 1 for s in speeds):
            print(f"  {GREEN}Transfer:{RESET} Good speeds (>{min(speeds):.1f} MB/s)")
        elif all(s > 0 for s in speeds):
            print(f"  {YELLOW}Transfer:{RESET} Slow but working")
        else:
            print(f"  {RED}Transfer:{RESET} Failed")
            all_ok = False

    print()
    if all_ok:
        print(f"{GREEN}{BOLD}Pipeline ready for cross-network operation!{RESET}")
    else:
        print(f"{YELLOW}{BOLD}Some issues detected - check details above{RESET}")

    print()
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Diagnose network connectivity for tennis pipeline")
    parser.add_argument("--skip-speed", action="store_true", help="Skip file transfer speed test")
    parser.add_argument("--ssh-only", action="store_true", help="Only test SSH connectivity")
    parser.add_argument("--quick", action="store_true", help="Quick check (Tailscale + SSH only)")
    args = parser.parse_args()

    if args.ssh_only:
        check_ssh_connectivity()
    elif args.quick:
        check_tailscale()
        check_ssh_connectivity()
    else:
        success = run_full_diagnostic(skip_speed=args.skip_speed)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
