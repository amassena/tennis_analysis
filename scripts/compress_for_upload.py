#!/usr/bin/env python3
"""Compress videos for smaller iCloud uploads using HEVC.

Reduces file size by 40-50% with minimal quality loss. Useful for slow
upload connections before syncing to iCloud.

Usage:
    python scripts/compress_for_upload.py video.mov
    python scripts/compress_for_upload.py video.mov -o compressed/
    python scripts/compress_for_upload.py *.mov --crf 24  # more compression
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_video_info(video_path):
    """Get video file size and duration."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration,size",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None

    try:
        import json
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        return {
            "duration": float(fmt.get("duration", 0)),
            "size": int(fmt.get("size", 0)),
        }
    except Exception:
        return None


def format_size(num_bytes):
    """Format bytes as human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def compress_video(input_path, output_path, crf=22, keep_audio=False, hevc=True):
    """Compress video with HEVC (H.265) for smaller file size.

    Args:
        input_path: Source video path
        output_path: Destination path for compressed video
        crf: Constant Rate Factor (18-28, lower = better quality, bigger file)
        keep_audio: If False, strips audio track (saves 5-10%)
        hevc: If True, uses H.265/HEVC; if False, uses H.264

    Returns:
        dict with compression stats or None on failure
    """
    input_info = get_video_info(input_path)
    if not input_info:
        print(f"[ERROR] Could not read {input_path}")
        return None

    input_size = input_info["size"]
    duration = input_info["duration"]

    print(f"Input: {os.path.basename(input_path)}")
    print(f"  Size: {format_size(input_size)}, Duration: {duration/60:.1f} min")

    # Build ffmpeg command
    part_path = output_path + ".part"

    if hevc:
        # HEVC with hardware acceleration if available (VideoToolbox on Mac)
        # -tag:v hvc1 ensures QuickTime/iOS compatibility
        codec_args = [
            "-c:v", "libx265",
            "-crf", str(crf),
            "-preset", "medium",
            "-tag:v", "hvc1",  # Required for Apple devices
        ]
        codec_name = "HEVC"
    else:
        codec_args = [
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "medium",
        ]
        codec_name = "H.264"

    audio_args = ["-c:a", "aac", "-b:a", "128k"] if keep_audio else ["-an"]

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        *codec_args,
        "-pix_fmt", "yuv420p",
        *audio_args,
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        "-nostats",
        part_path,
    ]

    print(f"  Compressing with {codec_name} CRF {crf}...")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us="):
            try:
                t = int(line.split("=", 1)[1]) / 1_000_000
                if duration > 0 and t > 0:
                    pct = min(t / duration * 100, 100)
                    print(f"\r  Progress: {pct:5.1f}%", end="", flush=True)
            except (ValueError, IndexError):
                pass

    proc.wait()
    print()

    if proc.returncode != 0:
        print(f"[ERROR] FFmpeg failed: {proc.stderr.read()[:500]}")
        if os.path.exists(part_path):
            os.remove(part_path)
        return None

    os.replace(part_path, output_path)

    output_size = os.path.getsize(output_path)
    reduction = (1 - output_size / input_size) * 100
    ratio = input_size / output_size

    print(f"Output: {os.path.basename(output_path)}")
    print(f"  Size: {format_size(output_size)}")
    print(f"  Reduction: {reduction:.1f}% ({ratio:.2f}x smaller)")

    return {
        "input_size": input_size,
        "output_size": output_size,
        "reduction_pct": round(reduction, 1),
        "compression_ratio": round(ratio, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compress videos for smaller iCloud uploads"
    )
    parser.add_argument("videos", nargs="+", help="Video file(s) to compress")
    parser.add_argument("-o", "--output-dir",
                        help="Output directory (default: same as input)")
    parser.add_argument("--crf", type=int, default=22,
                        help="Quality (18-28, lower=better, default: 22)")
    parser.add_argument("--keep-audio", action="store_true",
                        help="Keep audio track (default: strip for smaller size)")
    parser.add_argument("--h264", action="store_true",
                        help="Use H.264 instead of HEVC (larger but faster decode)")
    parser.add_argument("--suffix", default="_compressed",
                        help="Suffix for output filename (default: _compressed)")
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    total_input = 0
    total_output = 0
    results = []

    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"[SKIP] File not found: {video_path}")
            continue

        # Build output path
        base = os.path.splitext(os.path.basename(video_path))[0]
        ext = ".mp4"  # Always output MP4 for compatibility
        output_name = f"{base}{args.suffix}{ext}"

        if args.output_dir:
            output_path = os.path.join(args.output_dir, output_name)
        else:
            output_path = os.path.join(os.path.dirname(video_path), output_name)

        print()
        result = compress_video(
            video_path,
            output_path,
            crf=args.crf,
            keep_audio=args.keep_audio,
            hevc=not args.h264,
        )

        if result:
            results.append((video_path, output_path, result))
            total_input += result["input_size"]
            total_output += result["output_size"]

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 50)
        print("Compression Summary")
        print("=" * 50)
        for video_path, output_path, result in results:
            print(f"  {os.path.basename(video_path)}: "
                  f"{result['reduction_pct']:.0f}% smaller")
        print()
        reduction = (1 - total_output / total_input) * 100 if total_input > 0 else 0
        print(f"Total: {format_size(total_input)} -> {format_size(total_output)} "
              f"({reduction:.1f}% reduction)")

    elif len(results) == 1:
        print()
        print("Compression complete. Upload the compressed file to iCloud for faster sync.")


if __name__ == "__main__":
    main()
