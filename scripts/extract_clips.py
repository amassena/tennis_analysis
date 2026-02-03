#!/usr/bin/env python3
"""Extract per-shot clips from preprocessed video using detection results.

Usage:
    python scripts/extract_clips.py                           # uses shots_detected.json
    python scripts/extract_clips.py -i my_detections.json     # custom input
    python scripts/extract_clips.py --min-confidence 0.8      # filter low confidence
    python scripts/extract_clips.py --highlights              # also compile highlights
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PREPROCESSED_DIR, CLIPS_DIR, HIGHLIGHTS_DIR, PROJECT_ROOT


def find_video(video_name):
    """Find the preprocessed video file."""
    for ext in [".mp4", ".mov"]:
        path = os.path.join(PREPROCESSED_DIR, video_name + ext)
        if os.path.exists(path):
            return path
    return None


def has_nvenc():
    """Check if NVENC hardware encoder is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


_NVENC_AVAILABLE = None


def get_nvenc():
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        _NVENC_AVAILABLE = has_nvenc()
    return _NVENC_AVAILABLE


def extract_clip(video_path, start_time, end_time, output_path, pad=0.5):
    """Extract a clip using FFmpeg with padding. Uses NVENC if available."""
    padded_start = max(0, start_time - pad)
    padded_end = end_time + pad
    duration = padded_end - padded_start

    if get_nvenc():
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
    else:
        codec_args = ["-c:v", "libx264", "-crf", "20", "-preset", "fast"]

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{padded_start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        *codec_args,
        "-an",  # no audio for individual clips
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def compile_highlights(clip_paths, output_path):
    """Concatenate clips into a highlight reel using FFmpeg concat demuxer."""
    if not clip_paths:
        return False

    # Write concat file
    concat_file = output_path + ".txt"
    with open(concat_file, "w") as f:
        for path in clip_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    if get_nvenc():
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18"]
    else:
        codec_args = ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        *codec_args,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(concat_file)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Extract shot clips and compile highlights")
    parser.add_argument("-i", "--input", help="Detection JSON (default: shots_detected.json)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum avg confidence to extract (default: 0.7)")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Minimum segment duration in seconds (default: 0.5)")
    parser.add_argument("--max-duration", type=float, default=10.0,
                        help="Maximum segment duration in seconds (default: 10.0)")
    parser.add_argument("--pad", type=float, default=0.5,
                        help="Padding before/after each clip in seconds (default: 0.5)")
    parser.add_argument("--skip-neutral", action="store_true", default=True,
                        help="Skip neutral segments (default: True)")
    parser.add_argument("--include-neutral", action="store_true",
                        help="Include neutral segments")
    parser.add_argument("--highlights", action="store_true",
                        help="Also compile per-type highlight reels")
    args = parser.parse_args()

    if args.include_neutral:
        args.skip_neutral = False

    # Load detection results
    input_path = args.input or os.path.join(PROJECT_ROOT, "shots_detected.json")
    if not os.path.exists(input_path):
        print(f"[ERROR] Detection file not found: {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    video_name = data["source_video"]
    segments = data["segments"]
    fps = data["fps"]

    # Find source video
    video_path = find_video(video_name)
    if not video_path:
        print(f"[ERROR] Video not found: {video_name} in {PREPROCESSED_DIR}")
        sys.exit(1)

    print(f"Source: {os.path.basename(video_path)}")
    print(f"Total segments: {len(segments)}")
    print(f"Filters: confidence >= {args.min_confidence}, "
          f"duration {args.min_duration}-{args.max_duration}s, "
          f"skip_neutral={args.skip_neutral}")
    print()

    # Filter segments
    filtered = []
    for seg in segments:
        duration = seg["end_time"] - seg["start_time"]
        if args.skip_neutral and seg["shot_type"] == "neutral":
            continue
        if seg["avg_confidence"] < args.min_confidence:
            continue
        if duration < args.min_duration:
            continue
        if duration > args.max_duration:
            continue
        filtered.append(seg)

    print(f"After filtering: {len(filtered)} clips to extract")

    # Count by type
    type_counts = {}
    for seg in filtered:
        st = seg["shot_type"]
        type_counts[st] = type_counts.get(st, 0) + 1
    for st, c in sorted(type_counts.items()):
        print(f"  {st}: {c}")
    print()

    # Create output directories
    os.makedirs(CLIPS_DIR, exist_ok=True)
    type_dirs = {}
    for st in type_counts:
        d = os.path.join(CLIPS_DIR, st)
        os.makedirs(d, exist_ok=True)
        type_dirs[st] = d

    # Extract clips
    extracted = {st: [] for st in type_counts}
    counters = {st: 0 for st in type_counts}

    for i, seg in enumerate(filtered):
        st = seg["shot_type"]
        counters[st] += 1
        filename = f"{video_name}_{st}_{counters[st]:03d}.mp4"
        output_path = os.path.join(type_dirs[st], filename)

        duration = seg["end_time"] - seg["start_time"]
        conf = seg["avg_confidence"]
        print(f"  [{i+1}/{len(filtered)}] {st} {seg['start_time']:.1f}-{seg['end_time']:.1f}s "
              f"({duration:.1f}s, {conf:.0%}) -> {filename}", end="")

        if extract_clip(video_path, seg["start_time"], seg["end_time"], output_path, args.pad):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  [{size_mb:.1f} MB]")
            extracted[st].append(output_path)
        else:
            print("  [FAILED]")

    print()
    print("=" * 50)
    print("Clip Extraction Summary")
    print("=" * 50)
    total_extracted = sum(len(v) for v in extracted.values())
    for st in sorted(extracted):
        print(f"  {st}: {len(extracted[st])} clips")
    print(f"  Total: {total_extracted} clips extracted")

    # Compile highlights
    if args.highlights and total_extracted > 0:
        print()
        print("Compiling highlight reels...")
        os.makedirs(HIGHLIGHTS_DIR, exist_ok=True)

        for st, clips in sorted(extracted.items()):
            if not clips:
                continue
            highlight_path = os.path.join(HIGHLIGHTS_DIR, f"{video_name}_{st}_highlights.mp4")
            print(f"  {st}: {len(clips)} clips -> {os.path.basename(highlight_path)}", end="")

            if compile_highlights(clips, highlight_path):
                size_mb = os.path.getsize(highlight_path) / (1024 * 1024)
                print(f"  [{size_mb:.1f} MB]")
            else:
                print("  [FAILED]")

        # All shots combined highlight
        all_clips = []
        for st in ["serve", "forehand", "backhand"]:
            all_clips.extend(extracted.get(st, []))

        if all_clips:
            combined_path = os.path.join(HIGHLIGHTS_DIR, f"{video_name}_all_highlights.mp4")
            print(f"  combined: {len(all_clips)} clips -> {os.path.basename(combined_path)}", end="")
            if compile_highlights(all_clips, combined_path):
                size_mb = os.path.getsize(combined_path) / (1024 * 1024)
                print(f"  [{size_mb:.1f} MB]")
            else:
                print("  [FAILED]")

    print()
    print(f"Clips saved to: {CLIPS_DIR}/")
    if args.highlights:
        print(f"Highlights saved to: {HIGHLIGHTS_DIR}/")


if __name__ == "__main__":
    main()
