#!/usr/bin/env python3
"""Batch-convert raw VFR .MOV files to 60fps CFR .mp4 for ML processing."""

import json
import os
import subprocess
import sys

# Add project root to path so config is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import RAW_DIR, PREPROCESSED_DIR, VIDEO


# ── Helpers ──────────────────────────────────────────────────


def format_size(num_bytes):
    """Return human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def format_duration(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""
    seconds = int(seconds)
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"


# ── Probing ──────────────────────────────────────────────────


def probe_video(filepath):
    """Run ffprobe and return dict of video properties.

    Returns dict with keys: codec, width, height, fps, duration, vfr.
    Returns None on failure.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  [ERROR] ffprobe failed for {os.path.basename(filepath)}")
            return None
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  [ERROR] ffprobe: {e}")
        return None

    # Find the first video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        print(f"  [ERROR] No video stream found in {os.path.basename(filepath)}")
        return None

    # Parse frame rate
    fps = 0.0
    r_frame_rate = video_stream.get("r_frame_rate", "0/1")
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        if int(den) > 0:
            fps = int(num) / int(den)
    else:
        fps = float(r_frame_rate)

    # Parse duration (prefer format duration, fall back to stream)
    duration = float(data.get("format", {}).get("duration", 0))
    if duration == 0:
        duration = float(video_stream.get("duration", 0))

    # Detect VFR: avg_frame_rate != r_frame_rate suggests VFR
    avg_frame_rate = video_stream.get("avg_frame_rate", "0/1")
    vfr = avg_frame_rate != r_frame_rate

    return {
        "codec": video_stream.get("codec_name", "unknown"),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "duration": duration,
        "vfr": vfr,
    }


def display_source_info(filepath, info):
    """Print source video properties before conversion."""
    name = os.path.basename(filepath)
    size = os.path.getsize(filepath)
    vfr_str = "VFR" if info["vfr"] else "CFR"
    print(f"  Source:  {name} ({format_size(size)})")
    print(
        f"  Video:   {info['codec']} {info['width']}x{info['height']} "
        f"{info['fps']:.1f}fps ({vfr_str})"
    )
    print(f"  Length:  {format_duration(info['duration'])}")


def display_output_info(filepath):
    """Probe output file, display properties, and verify 60fps CFR.

    Returns True if output looks correct, False otherwise.
    """
    info = probe_video(filepath)
    if not info:
        return False

    size = os.path.getsize(filepath)
    vfr_str = "VFR" if info["vfr"] else "CFR"
    target = VIDEO["target_fps"]

    print(f"  Output:  {os.path.basename(filepath)} ({format_size(size)})")
    print(
        f"  Video:   {info['codec']} {info['width']}x{info['height']} "
        f"{info['fps']:.0f}fps ({vfr_str})"
    )
    print(f"  Length:  {format_duration(info['duration'])}")

    # Verify target fps
    ok = True
    if abs(info["fps"] - target) > 1:
        print(f"  [WARN] Expected {target}fps, got {info['fps']:.1f}fps")
        ok = False
    if info["vfr"]:
        print(f"  [WARN] Output is VFR, expected CFR")
        ok = False

    return ok


# ── Conversion ───────────────────────────────────────────────


def convert_video(src_path, dst_path, video_config):
    """Convert a single video using FFmpeg with progress monitoring.

    Writes to a .part temp file, renames on success, deletes on failure.
    Returns True on success, False on failure.
    """
    part_path = dst_path + ".part"

    # Get source duration for progress calculation
    src_info = probe_video(src_path)
    total_duration = src_info["duration"] if src_info else 0

    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-r", str(video_config["target_fps"]),
        "-c:v", video_config["codec"],
        "-crf", str(video_config["crf"]),
        "-preset", video_config["preset"],
        "-pix_fmt", video_config["pixel_format"],
        "-c:a", video_config["audio_codec"],
        "-f", "mp4",
        "-progress", "pipe:1",
        "-nostats",
        part_path,
    ]

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Capture stderr to a temp file to avoid pipe buffer deadlock
    stderr_path = part_path + ".log"

    try:
        stderr_file = open(stderr_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
        )

        out_time = 0.0
        speed = ""

        for line in proc.stdout:
            line = line.strip()

            # Parse out_time_us (microseconds)
            if line.startswith("out_time_us="):
                try:
                    out_time = int(line.split("=", 1)[1]) / 1_000_000
                except (ValueError, IndexError):
                    pass
            elif line.startswith("speed="):
                speed = line.split("=", 1)[1].strip()

            # Update progress line
            if total_duration > 0 and out_time > 0:
                pct = min(out_time / total_duration * 100, 100)
                speed_str = f" ({speed})" if speed and speed != "N/A" else ""
                print(
                    f"\r  Converting: {pct:5.1f}% "
                    f"[{format_duration(out_time)}/{format_duration(total_duration)}]"
                    f"{speed_str}",
                    end="",
                    flush=True,
                )

        proc.wait()
        stderr_file.close()
        print()  # newline after progress

        if proc.returncode != 0:
            print(f"  [ERROR] FFmpeg exited with code {proc.returncode}")
            # Show last few lines of stderr for diagnosis
            with open(stderr_path) as f:
                err_lines = f.read().strip().split("\n")
            for line in err_lines[-3:]:
                print(f"    {line}")
            if os.path.exists(part_path):
                os.remove(part_path)
            return False

        # Rename .part to final
        os.replace(part_path, dst_path)
        return True

    except FileNotFoundError:
        print("  [ERROR] ffmpeg not found. Install FFmpeg and try again.")
        if os.path.exists(part_path):
            os.remove(part_path)
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        if os.path.exists(part_path):
            os.remove(part_path)
        return False
    finally:
        if os.path.exists(stderr_path):
            os.remove(stderr_path)


# ── Discovery ────────────────────────────────────────────────


def find_raw_videos(raw_dir):
    """List all .MOV/.mov files in raw/, sorted by name."""
    if not os.path.isdir(raw_dir):
        return []
    videos = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(".mov")
    ]
    videos.sort()
    return videos


def is_already_processed(src_path, preprocessed_dir):
    """Check if matching .mp4 exists in preprocessed/ with nonzero size."""
    name = os.path.splitext(os.path.basename(src_path))[0]
    dst_path = os.path.join(preprocessed_dir, name + ".mp4")
    return os.path.exists(dst_path) and os.path.getsize(dst_path) > 0


# ── Main ─────────────────────────────────────────────────────


def main():
    # Check ffmpeg/ffprobe availability before processing
    for tool in ("ffmpeg", "ffprobe"):
        try:
            result = subprocess.run(
                [tool, "-version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                print(f"[ERROR] {tool} returned non-zero exit code.")
                sys.exit(1)
        except FileNotFoundError:
            print(f"[ERROR] {tool} not found. Install FFmpeg and try again.")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print(f"[ERROR] {tool} timed out.")
            sys.exit(1)

    # Find raw videos
    videos = find_raw_videos(RAW_DIR)
    if not videos:
        print(f"No .MOV files found in {RAW_DIR}/")
        return

    print(f"Found {len(videos)} video(s) in {RAW_DIR}/\n")

    results = {"ok": [], "skip": [], "fail": []}

    try:
        for i, filename in enumerate(videos, 1):
            src_path = os.path.join(RAW_DIR, filename)
            name = os.path.splitext(filename)[0]
            dst_path = os.path.join(PREPROCESSED_DIR, name + ".mp4")

            print(f"[{i}/{len(videos)}] {filename}")

            # Skip if already processed
            if is_already_processed(src_path, PREPROCESSED_DIR):
                size = os.path.getsize(dst_path)
                print(f"  [SKIP] {name}.mp4 already exists ({format_size(size)})\n")
                results["skip"].append(filename)
                continue

            # Probe source
            info = probe_video(src_path)
            if not info:
                results["fail"].append(filename)
                print()
                continue

            display_source_info(src_path, info)

            # Convert
            print()
            success = convert_video(src_path, dst_path, VIDEO)

            if success:
                # Verify output
                verified = display_output_info(dst_path)
                if verified:
                    print(f"  [OK] Conversion complete.")
                else:
                    print(f"  [WARN] Conversion complete but verification had warnings.")
                results["ok"].append(filename)
            else:
                print(f"  [FAILED] {filename}")
                results["fail"].append(filename)

            print()

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
        # Clean up any .part file in progress
        for filename in videos:
            name = os.path.splitext(filename)[0]
            part_path = os.path.join(PREPROCESSED_DIR, name + ".mp4.part")
            if os.path.exists(part_path):
                os.remove(part_path)
                print(f"  Cleaned up {name}.mp4.part")

    # Summary
    print("=" * 50)
    print("Preprocessing Summary")
    print("=" * 50)
    print(f"  Converted: {len(results['ok'])}")
    print(f"  Skipped:   {len(results['skip'])}")
    print(f"  Failed:    {len(results['fail'])}")
    if results["fail"]:
        print("  Failed files:")
        for name in results["fail"]:
            print(f"    - {name}")
    print()


if __name__ == "__main__":
    main()
