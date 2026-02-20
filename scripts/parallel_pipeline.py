#!/usr/bin/env python3
"""Parallel video processing across multiple GPU machines.

Splits large videos at keyframe boundaries and processes chunks in parallel
across available GPU machines for ~2x speedup with 2 machines.

Usage:
    python scripts/parallel_pipeline.py video.mov
    python scripts/parallel_pipeline.py video.mov --machines windows tmassena
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    RAW_DIR, PREPROCESSED_DIR, POSES_DIR, PROJECT_ROOT, AUTO_PIPELINE,
)


# ── Keyframe Detection ─────────────────────────────────────────


def find_keyframes(video_path, min_interval=5.0):
    """Find I-frame (keyframe) positions in video using ffprobe.

    Args:
        video_path: Path to video file
        min_interval: Minimum seconds between keyframes to report

    Returns:
        List of (frame_number, timestamp) tuples for each keyframe
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type,pts_time,pkt_pts",
        "-of", "json",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"[ERROR] ffprobe failed: {result.stderr}")
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("[ERROR] Could not parse ffprobe output")
        return []

    keyframes = []
    last_time = -min_interval
    frame_num = 0

    for frame in data.get("frames", []):
        pict_type = frame.get("pict_type", "")
        pts_time = float(frame.get("pts_time", 0))

        if pict_type == "I" and pts_time - last_time >= min_interval:
            keyframes.append((frame_num, pts_time))
            last_time = pts_time

        frame_num += 1

    return keyframes


def get_video_info(video_path):
    """Get video duration, fps, and frame count using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames:format=duration",
        "-of", "json",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        fps_str = stream.get("r_frame_rate", "60/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        duration = float(fmt.get("duration", 0))
        nb_frames = int(stream.get("nb_frames", 0)) or int(duration * fps)

        return {
            "duration": duration,
            "fps": fps,
            "total_frames": nb_frames,
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


# ── Video Splitting ────────────────────────────────────────────


def split_at_keyframes(video_path, num_chunks, output_dir):
    """Split video into chunks at keyframe boundaries.

    Args:
        video_path: Path to source video
        num_chunks: Number of chunks to create
        output_dir: Directory to write chunk files

    Returns:
        List of (chunk_path, start_time, end_time) tuples
    """
    info = get_video_info(video_path)
    if not info:
        print("[ERROR] Could not get video info")
        return []

    duration = info["duration"]
    chunk_duration = duration / num_chunks

    # Find keyframes near ideal split points
    keyframes = find_keyframes(video_path, min_interval=1.0)
    if not keyframes:
        print("[WARN] No keyframes found, using time-based split")
        # Fall back to time-based splits
        split_times = [i * chunk_duration for i in range(num_chunks)]
        split_times.append(duration)
    else:
        # Find keyframes closest to ideal split points
        split_times = [0]
        for i in range(1, num_chunks):
            target_time = i * chunk_duration
            # Find closest keyframe
            closest = min(keyframes, key=lambda kf: abs(kf[1] - target_time))
            if closest[1] > split_times[-1] + 1.0:  # Ensure minimum 1s between splits
                split_times.append(closest[1])
        split_times.append(duration)

    # Create chunks
    base = os.path.splitext(os.path.basename(video_path))[0]
    chunks = []

    for i in range(len(split_times) - 1):
        start = split_times[i]
        end = split_times[i + 1]
        chunk_path = os.path.join(output_dir, f"{base}_chunk{i:02d}.mov")

        # Use stream copy for fast splitting (no re-encode)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", video_path,
            "-t", f"{end - start:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            chunk_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(chunk_path):
            chunks.append((chunk_path, start, end))
            size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            print(f"  Chunk {i}: {start:.1f}s - {end:.1f}s ({size_mb:.1f} MB)")
        else:
            print(f"  [ERROR] Failed to create chunk {i}: {result.stderr[:200]}")

    return chunks


# ── SSH/SCP Helpers ────────────────────────────────────────────


def run_ssh(host, cmd, timeout=3600):
    """Run a command on a remote machine via SSH."""
    full_cmd = ["ssh", "-o", "ConnectTimeout=10", host, cmd]
    print(f"  SSH [{host}]: {cmd[:80]}...")

    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"  [ERROR] SSH failed (rc={result.returncode}): {result.stderr[:200]}")
        return False, result.stderr
    return True, result.stdout


def scp_to(host, project, local_path, remote_relative, timeout=1800):
    """Copy a file to a remote machine via SCP."""
    remote_path = f"{host}:{project}/{remote_relative}"
    print(f"  SCP to {host}: {os.path.basename(local_path)}")

    result = subprocess.run(
        ["scp", local_path, remote_path],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode == 0


def scp_from(host, project, remote_relative, local_path, timeout=1800):
    """Copy a file from a remote machine via SCP."""
    remote_path = f"{host}:{project}/{remote_relative}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"  SCP from {host}: {remote_relative}")

    result = subprocess.run(
        ["scp", remote_path, local_path],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode == 0


# ── Parallel Processing ────────────────────────────────────────


def process_chunk_on_gpu(chunk_info, machine, video_base, chunk_idx):
    """Process a single video chunk on a GPU machine.

    Steps:
    1. Transfer chunk to GPU machine
    2. Preprocess (NVENC)
    3. Extract poses
    4. Transfer poses back

    Returns:
        (chunk_idx, poses_path, start_time) or (chunk_idx, None, start_time) on failure
    """
    chunk_path, start_time, end_time = chunk_info
    host = machine["host"]
    project = machine["project"]
    py = f"{project}/venv/Scripts/python.exe"

    chunk_name = os.path.basename(chunk_path)
    chunk_base = os.path.splitext(chunk_name)[0]

    print(f"\n[Chunk {chunk_idx}] Processing on {host} ({start_time:.1f}s - {end_time:.1f}s)")

    # 1. Transfer chunk to GPU machine
    if not scp_to(host, project, chunk_path, f"raw/{chunk_name}"):
        print(f"  [ERROR] Failed to transfer chunk {chunk_idx} to {host}")
        return (chunk_idx, None, start_time)

    # 2. Preprocess (NVENC, 60fps CFR)
    ok, _ = run_ssh(
        host,
        f"cd {project} && {py} preprocess_nvenc.py \"{chunk_name}\"",
        timeout=600,
    )
    if not ok:
        print(f"  [ERROR] Preprocess failed for chunk {chunk_idx} on {host}")
        return (chunk_idx, None, start_time)

    # 3. Extract poses
    preprocessed_name = f"{chunk_base}.mp4"
    poses_name = f"{chunk_base}.json"

    ok, _ = run_ssh(
        host,
        f"cd {project} && {py} scripts/extract_poses.py \"preprocessed/{preprocessed_name}\"",
        timeout=7200,  # 2 hours for large chunks
    )
    if not ok:
        print(f"  [ERROR] Pose extraction failed for chunk {chunk_idx} on {host}")
        return (chunk_idx, None, start_time)

    # 4. Transfer poses back
    local_poses = os.path.join(POSES_DIR, f"_{video_base}_chunk{chunk_idx:02d}.json")
    if not scp_from(host, project, f"poses/{poses_name}", local_poses):
        print(f"  [ERROR] Failed to retrieve poses for chunk {chunk_idx}")
        return (chunk_idx, None, start_time)

    print(f"[Chunk {chunk_idx}] Complete on {host}")
    return (chunk_idx, local_poses, start_time)


def merge_pose_files(pose_files_with_offsets, output_path, fps=60):
    """Merge multiple pose JSON files with frame offset correction.

    Args:
        pose_files_with_offsets: List of (pose_path, start_time_seconds) tuples
        output_path: Path for merged output JSON
        fps: Video framerate for offset calculation
    """
    merged_frames = []
    base_data = None
    total_detected = 0
    total_processed = 0
    total_time = 0

    # Sort by start time
    sorted_files = sorted(pose_files_with_offsets, key=lambda x: x[1])

    for pose_path, start_time in sorted_files:
        if not os.path.exists(pose_path):
            print(f"  [WARN] Pose file not found: {pose_path}")
            continue

        with open(pose_path) as f:
            data = json.load(f)

        if base_data is None:
            base_data = data.copy()
            base_data["frames"] = []

        # Calculate frame offset based on start time
        frame_offset = int(start_time * fps)

        # Adjust frame indices and timestamps
        for frame in data.get("frames", []):
            adjusted_frame = frame.copy()
            adjusted_frame["frame_idx"] = frame["frame_idx"] + frame_offset
            adjusted_frame["timestamp"] = frame["timestamp"] + start_time
            merged_frames.append(adjusted_frame)

        stats = data.get("detection_stats", {})
        total_detected += stats.get("frames_detected", 0)
        total_processed += stats.get("frames_processed", 0)
        total_time += stats.get("processing_time_seconds", 0)

    if base_data is None:
        print("[ERROR] No pose files to merge")
        return False

    # Sort merged frames by frame_idx (handles any overlap)
    merged_frames.sort(key=lambda f: f["frame_idx"])

    # Remove duplicates at chunk boundaries (keep first occurrence)
    seen_frames = set()
    unique_frames = []
    for frame in merged_frames:
        if frame["frame_idx"] not in seen_frames:
            seen_frames.add(frame["frame_idx"])
            unique_frames.append(frame)

    base_data["frames"] = unique_frames
    base_data["detection_stats"] = {
        "frames_processed": len(unique_frames),
        "frames_detected": sum(1 for f in unique_frames if f.get("detected")),
        "detection_rate": round(
            sum(1 for f in unique_frames if f.get("detected")) / len(unique_frames), 4
        ) if unique_frames else 0,
        "processing_time_seconds": round(total_time, 1),
        "parallel_chunks": len(sorted_files),
    }

    # Write merged file
    with open(output_path, "w") as f:
        json.dump(base_data, f, separators=(",", ":"))

    det_pct = base_data["detection_stats"]["detection_rate"] * 100
    print(f"\nMerged {len(sorted_files)} chunks -> {os.path.basename(output_path)}")
    print(f"  {len(unique_frames)} frames, {det_pct:.1f}% detected")

    return True


def get_available_machines(machines):
    """Filter machines to only those currently reachable via SSH."""
    available = []
    for machine in machines:
        host = machine["host"]
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, "echo ok"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and "ok" in result.stdout:
                available.append(machine)
                print(f"  Machine {host}: available")
            else:
                print(f"  Machine {host}: OFFLINE")
        except Exception as e:
            print(f"  Machine {host}: ERROR ({e})")
    return available


# ── Main Pipeline ──────────────────────────────────────────────


def process_video_parallel(video_name, machines=None, temp_dir=None):
    """Process a video in parallel across multiple GPU machines.

    Args:
        video_name: Name of video file in raw/
        machines: List of machine dicts (default: from AUTO_PIPELINE config)
        temp_dir: Directory for temporary chunk files

    Returns:
        True on success, False on failure
    """
    if machines is None:
        machines = AUTO_PIPELINE.get("gpu_machines", [])

    video_path = os.path.join(RAW_DIR, video_name)
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return False

    base = os.path.splitext(video_name)[0]

    print(f"\n{'='*60}")
    print(f"Parallel Processing: {video_name}")
    print(f"{'='*60}")

    # Get video info
    info = get_video_info(video_path)
    if info:
        duration_str = f"{info['duration'] / 60:.1f} min"
        print(f"Duration: {duration_str}, {info['total_frames']} frames @ {info['fps']:.1f} fps")

    # Check available machines
    print("\nChecking GPU machines...")
    available = get_available_machines(machines)

    if not available:
        print("[ERROR] No GPU machines available")
        return False

    num_machines = len(available)
    print(f"\n{num_machines} machine(s) available for parallel processing")

    # Create temp directory for chunks
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="tennis_parallel_")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Split video into chunks
        print(f"\nSplitting video into {num_machines} chunks...")
        chunks = split_at_keyframes(video_path, num_machines, temp_dir)

        if len(chunks) < num_machines:
            print(f"[WARN] Only created {len(chunks)} chunks (video may be short)")

        if not chunks:
            print("[ERROR] Failed to split video")
            return False

        # Process chunks in parallel
        print(f"\nProcessing {len(chunks)} chunks in parallel...")
        start_time = time.time()

        results = []
        with ThreadPoolExecutor(max_workers=len(available)) as pool:
            futures = {}
            for i, chunk in enumerate(chunks):
                machine = available[i % len(available)]
                fut = pool.submit(process_chunk_on_gpu, chunk, machine, base, i)
                futures[fut] = i

            for fut in as_completed(futures):
                chunk_idx = futures[fut]
                try:
                    result = fut.result()
                    results.append(result)
                except Exception as e:
                    print(f"[ERROR] Chunk {chunk_idx} failed: {e}")
                    results.append((chunk_idx, None, chunks[chunk_idx][1]))

        elapsed = time.time() - start_time

        # Check for failures
        failed = [r for r in results if r[1] is None]
        if failed:
            print(f"\n[ERROR] {len(failed)} chunk(s) failed to process")
            return False

        # Merge pose files
        print(f"\nMerging pose files...")
        os.makedirs(POSES_DIR, exist_ok=True)
        output_path = os.path.join(POSES_DIR, f"{base}.json")

        pose_files = [(r[1], r[2]) for r in results if r[1]]
        if not merge_pose_files(pose_files, output_path, fps=info["fps"] if info else 60):
            return False

        # Cleanup chunk pose files
        for pose_path, _ in pose_files:
            if os.path.exists(pose_path):
                os.remove(pose_path)

        print(f"\nParallel processing complete in {elapsed / 60:.1f} min")
        print(f"Output: {output_path}")

        return True

    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[WARN] Could not clean up temp dir: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process video in parallel across multiple GPU machines"
    )
    parser.add_argument("video", help="Video filename in raw/ directory")
    parser.add_argument("--machines", nargs="+", default=None,
                        help="SSH hostnames to use (default: from config)")
    args = parser.parse_args()

    # Build machine list from hostnames if provided
    machines = None
    if args.machines:
        default_machines = AUTO_PIPELINE.get("gpu_machines", [])
        machines = []
        for host in args.machines:
            # Find matching machine config or create minimal one
            match = next((m for m in default_machines if m["host"] == host), None)
            if match:
                machines.append(match)
            else:
                # Assume Windows default project path
                machines.append({
                    "host": host,
                    "project": r"C:\Users\amass\tennis_analysis",
                })

    success = process_video_parallel(args.video, machines=machines)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
