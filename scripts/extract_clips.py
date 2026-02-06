#!/usr/bin/env python3
"""Extract per-shot clips from preprocessed video using detection results.

Supports two input formats:
  - JSON from detect_shots.py (segments with start/end times)
  - CSV from visual_label.py (single-frame contact points)

For CSV input, applies shot-type-specific duration buffers around each contact.

Usage:
    python scripts/extract_clips.py -i labels.csv -v video.mp4   # CSV labels
    python scripts/extract_clips.py -i detections.json           # JSON detections
    python scripts/extract_clips.py --highlights                 # compile highlights
"""

import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PREPROCESSED_DIR, CLIPS_DIR, HIGHLIGHTS_DIR, PROJECT_ROOT

# Duration buffers (seconds) before/after contact point per shot type
# Format: (before, after)
SHOT_BUFFERS = {
    "forehand": (4.5, 3.5),    # 8s total
    "backhand": (4.5, 3.5),    # 8s total
    "serve": (5.0, 3.0),       # 8s total, more wind-up
    "neutral": (4.0, 4.0),     # 8s total
}
DEFAULT_BUFFER = (4.5, 3.5)


def load_csv_labels(csv_path, fps):
    """Load single-frame contact points from CSV and convert to segments.

    Returns list of dicts with shot_type, start_time, end_time, avg_confidence.
    """
    segments = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue

            shot_type = row[0].strip().lower()
            if len(row) == 2:
                # New v2 format: shot_type, frame
                try:
                    frame = int(row[1].strip())
                except ValueError:
                    continue
            elif len(row) >= 3:
                # Old v1 format: shot_type, start_frame, end_frame - use midpoint
                try:
                    start = int(row[1].strip())
                    end = int(row[2].strip())
                    frame = (start + end) // 2
                except ValueError:
                    continue
            else:
                continue

            # Get shot-specific buffer
            before, after = SHOT_BUFFERS.get(shot_type, DEFAULT_BUFFER)

            contact_time = frame / fps
            start_time = max(0, contact_time - before)
            end_time = contact_time + after

            segments.append({
                "shot_type": shot_type,
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": max(0, int((contact_time - before) * fps)),
                "end_frame": int((contact_time + after) * fps),
                "contact_frame": frame,
                "avg_confidence": 1.0,  # Manual labels are high confidence
            })

    return segments


def get_video_fps(video_path):
    """Get video FPS using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    fps_str = result.stdout.strip()
    if "/" in fps_str:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    return float(fps_str) if fps_str else 60.0


def merge_close_segments(segments, max_gap=1.0):
    """Merge segments separated by less than max_gap seconds.

    Prevents padding overlap when clips are concatenated into highlights.
    The merged segment uses the shot_type of the longest sub-segment.
    """
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        gap = seg["start_time"] - merged[-1]["end_time"]
        if gap < max_gap:
            prev_dur = merged[-1]["end_time"] - merged[-1]["start_time"]
            curr_dur = seg["end_time"] - seg["start_time"]
            if curr_dur > prev_dur:
                merged[-1]["shot_type"] = seg["shot_type"]
            merged[-1]["end_time"] = seg["end_time"]
            total_dur = prev_dur + curr_dur
            merged[-1]["avg_confidence"] = (
                merged[-1]["avg_confidence"] * prev_dur +
                seg["avg_confidence"] * curr_dur
            ) / total_dur
        else:
            merged.append(seg.copy())

    return merged


def find_video(video_name):
    """Find the preprocessed video file."""
    for ext in [".mp4", ".mov"]:
        path = os.path.join(PREPROCESSED_DIR, video_name + ext)
        if os.path.exists(path):
            return path
    return None


def has_nvenc():
    """Check if NVENC hardware encoder is actually usable (not just listed)."""
    try:
        # Test real encoding — catches driver version mismatches
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-y",
             "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
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


def compile_slowmo_highlights(raw_video_path, segments, output_path,
                              slowmo_factor=4.0, output_fps=60,
                              min_confidence=0.7, min_duration=0.5,
                              max_duration=10.0, merge_gap=3.0,
                              group_by_type=False):
    """Extract non-neutral segments from 240fps raw video at slow-mo speed.

    Filters segments by confidence/duration, merges close segments to
    eliminate duplicates, then extracts with setpts=N*PTS to produce
    smooth slow-motion at the specified output fps.

    When group_by_type=True, orders clips by shot type (serve, forehand,
    backhand). When False, keeps chronological order.
    """
    non_neutral = [s for s in segments if s["shot_type"] != "neutral"]
    if not non_neutral:
        print("  No non-neutral segments for slow-mo.")
        return False

    # Apply same filters as normal highlights
    filtered = [s for s in non_neutral
                if s["avg_confidence"] >= min_confidence
                and min_duration <= (s["end_time"] - s["start_time"]) <= max_duration]
    if not filtered:
        print("  No segments passed filters for slow-mo.")
        return False

    merged = merge_close_segments(filtered, max_gap=merge_gap)

    if group_by_type:
        # Group by type: serve, forehand, backhand
        by_type = {}
        for s in merged:
            by_type.setdefault(s["shot_type"], []).append(s)
        ordered = []
        for shot_type in ["serve", "forehand", "backhand"]:
            ordered.extend(by_type.get(shot_type, []))
        order_label = "grouped by type"
    else:
        # Chronological order (already sorted by time from merge)
        ordered = [s for s in merged if s["shot_type"] != "neutral"]
        order_label = "chronological"

    print(f"  Slow-mo: {len(non_neutral)} raw -> {len(filtered)} filtered "
          f"-> {len(ordered)} merged ({order_label})")

    if get_nvenc():
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
    else:
        codec_args = ["-c:v", "libx264", "-crf", "20", "-preset", "fast"]

    temp_clips = []
    temp_dir = os.path.dirname(output_path)

    for i, seg in enumerate(ordered):
        start = seg["start_time"]
        duration = seg["end_time"] - seg["start_time"]
        # Add padding
        padded_start = max(0, start - 0.5)
        padded_duration = duration + 1.0

        temp_path = os.path.join(temp_dir, f"_slowmo_temp_{i:04d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{padded_start:.3f}",
            "-i", raw_video_path,
            "-t", f"{padded_duration * slowmo_factor:.3f}",
            "-vf", f"setpts={slowmo_factor}*(PTS-STARTPTS)",
            "-r", str(output_fps),
            "-fps_mode", "cfr",
            *codec_args,
            "-an", "-pix_fmt", "yuv420p",
            temp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(temp_path):
            temp_clips.append(temp_path)
        else:
            print(f"  [WARN] Slow-mo clip {i} failed: {seg['shot_type']} "
                  f"{start:.1f}-{seg['end_time']:.1f}s")

    if not temp_clips:
        return False

    # Concatenate all slow-mo clips
    success = compile_highlights(temp_clips, output_path)

    # Clean up temp clips
    for p in temp_clips:
        if os.path.exists(p):
            os.remove(p)

    return success


def compile_combined_video(normal_path, slowmo_path, output_path):
    """Concatenate normal-speed and slow-mo highlights into one video."""
    if not os.path.exists(normal_path):
        print(f"  [ERROR] Normal highlights not found: {normal_path}")
        return False
    if not os.path.exists(slowmo_path):
        print(f"  [ERROR] Slow-mo highlights not found: {slowmo_path}")
        return False

    if get_nvenc():
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18"]
    else:
        codec_args = ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]

    concat_file = output_path + ".txt"
    with open(concat_file, "w") as f:
        f.write(f"file '{os.path.abspath(normal_path)}'\n")
        f.write(f"file '{os.path.abspath(slowmo_path)}'\n")

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
    if os.path.exists(concat_file):
        os.remove(concat_file)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Extract shot clips and compile highlights")
    parser.add_argument("-i", "--input", help="Detection JSON or labels CSV")
    parser.add_argument("-v", "--video", help="Video path (required for CSV input)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum avg confidence to extract (default: 0.7)")
    parser.add_argument("--min-duration", type=float, default=0.3,
                        help="Minimum segment duration in seconds (default: 0.3)")
    parser.add_argument("--max-duration", type=float, default=10.0,
                        help="Maximum segment duration in seconds (default: 10.0)")
    parser.add_argument("--pad", type=float, default=0.3,
                        help="Extra padding before/after each clip (default: 0.3)")
    parser.add_argument("--skip-neutral", action="store_true", default=True,
                        help="Skip neutral segments (default: True)")
    parser.add_argument("--include-neutral", action="store_true",
                        help="Include neutral segments")
    parser.add_argument("--highlights", action="store_true",
                        help="Also compile per-type highlight reels")
    parser.add_argument("--group-by-type", action="store_true",
                        help="Group clips by shot type instead of chronological order")
    args = parser.parse_args()

    if args.include_neutral:
        args.skip_neutral = False

    # Determine input file
    input_path = args.input or os.path.join(PROJECT_ROOT, "shots_detected.json")
    if not os.path.exists(input_path):
        # Try looking for a labels CSV
        csvs = [f for f in os.listdir(PROJECT_ROOT) if f.endswith("_labels.csv")]
        if csvs:
            input_path = os.path.join(PROJECT_ROOT, sorted(csvs)[-1])
            print(f"Auto-selected: {input_path}")
        else:
            print(f"[ERROR] No input file found. Specify with -i")
            sys.exit(1)

    is_csv = input_path.endswith(".csv")

    if is_csv:
        # CSV input - need video path
        if args.video:
            video_path = args.video
        else:
            # Try to find video from CSV name (e.g., IMG_6713_labels.csv -> IMG_6713.mp4)
            base = os.path.basename(input_path).replace("_labels.csv", "").replace("_audio_labels.csv", "")
            video_path = find_video(base)
            if not video_path:
                print(f"[ERROR] Could not find video for {base}. Specify with -v")
                sys.exit(1)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        fps = get_video_fps(video_path)
        segments = load_csv_labels(input_path, fps)
        print(f"Loaded {len(segments)} labels from CSV")
        print(f"Shot buffers: {SHOT_BUFFERS}")
    else:
        # JSON input
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

    # Merge close segments to eliminate near-duplicate clips in highlights
    pre_merge = len(filtered)
    filtered = merge_close_segments(filtered, max_gap=3.0)
    if len(filtered) < pre_merge:
        print(f"After filtering: {pre_merge} clips, merged to {len(filtered)}")
    else:
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
    all_extracted_in_order = []  # (shot_type, path) in chronological order
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
            all_extracted_in_order.append((st, output_path))
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
        if args.group_by_type:
            # Group by type: serve, forehand, backhand
            all_clips = []
            for st in ["serve", "forehand", "backhand"]:
                all_clips.extend(extracted.get(st, []))
        else:
            # Chronological order (already in time order from extraction)
            all_clips = [p for st, p in all_extracted_in_order
                         if st != "neutral"]

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
