#!/usr/bin/env python3
"""Export tennis analysis videos in 4 formats:
1. Full video with shot timeline bar at the bottom
2. Highlights in chronological order
3. Highlights grouped by shot type with title cards
4. Rally mode — full points kept, dead time between points removed

Usage:
    python scripts/export_videos.py preprocessed/IMG_6878.mp4
    python scripts/export_videos.py preprocessed/IMG_6878.mp4 --types rally
    python scripts/export_videos.py preprocessed/IMG_6878.mp4 --types rally --slow-motion --upload
"""

import json
import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Shot type colors (RGB)
SHOT_COLORS = {
    'serve':        (255, 140, 0),      # orange
    'forehand':     (155, 89, 182),     # purple
    'backhand':     (39, 174, 96),      # green
    'unknown_shot': (149, 165, 166),    # gray
    'practice':     (241, 196, 15),     # yellow
    'offscreen':    (127, 140, 141),    # dark gray
    'not_shot':     (127, 140, 141),    # dark gray
}

SHOT_LABELS = {
    'serve': 'S',
    'forehand': 'FH',
    'backhand': 'BH',
    'unknown_shot': '?',
    'practice': 'P',
    'offscreen': 'X',
}

# Hex colors for ffmpeg filters
SHOT_COLORS_HEX = {
    'serve':        '0xFF8C00',
    'forehand':     '0x9B59B6',
    'backhand':     '0x27AE60',
    'unknown_shot': '0x95A5A6',
    'practice':     '0xF1C40F',
    'offscreen':    '0x7F8C8D',
    'not_shot':     '0x7F8C8D',
}

SHOT_ORDER = ['forehand', 'backhand', 'serve', 'unknown_shot']

# Shot types to exclude from highlight exports
EXCLUDED_FROM_HIGHLIGHTS = {'practice', 'offscreen', 'not_shot'}

GROUP_TITLES = {
    'forehand': 'FOREHANDS',
    'backhand': 'BACKHANDS',
    'serve': 'SERVES',
    'unknown_shot': 'OTHER SHOTS',
    'practice': 'PRACTICE',
}


def load_detections(video_name):
    """Load detections, preferring GT file over detection file."""
    gt_path = f"detections/{video_name}_fused.json"
    det_path = f"detections/{video_name}_fused_detections.json"

    path = gt_path if os.path.exists(gt_path) else det_path
    if not os.path.exists(path):
        print(f"[ERROR] No detection file found for {video_name}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    print(f"  Loaded: {path} ({len(data.get('detections', []))} shots)")
    return data


def get_video_info(video_path):
    """Get video width, height, duration, fps using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-show_format', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)

    video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    duration = float(info['format']['duration'])
    fps_parts = video_stream.get('r_frame_rate', '60/1').split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 60.0

    return width, height, duration, fps


def create_timeline_png(detections, duration, width, height, bar_height=56):
    """Create a PNG timeline bar overlay using PIL."""
    if not HAS_PIL:
        return None

    img = Image.new('RGBA', (width, bar_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark background bar
    draw.rectangle([0, 0, width, bar_height], fill=(20, 20, 20, 210))

    # Thin top border
    draw.rectangle([0, 0, width, 2], fill=(60, 60, 60, 255))

    # Load font
    font = None
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, 15)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    # Calculate marker positions
    markers = []
    for det in detections:
        ts = det['timestamp']
        shot_type = det.get('shot_type', 'unknown_shot')
        label = SHOT_LABELS.get(shot_type, '?')
        color = SHOT_COLORS.get(shot_type, (149, 165, 166))

        x_center = int((ts / duration) * width)

        # Label box dimensions
        try:
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w = len(label) * 10
            text_h = 14

        label_w = text_w + 12
        label_h = text_h + 8
        label_x = x_center - label_w // 2
        label_x = max(2, min(label_x, width - label_w - 2))

        markers.append({
            'x': label_x,
            'w': label_w,
            'h': label_h,
            'text_w': text_w,
            'text_h': text_h,
            'label': label,
            'color': color,
            'row': 0,
            'ts': ts,
        })

    # Resolve overlapping labels (up to 2 rows)
    for i in range(1, len(markers)):
        for j in range(max(0, i - 5), i):  # check recent markers
            if markers[j]['row'] == markers[i]['row']:
                if markers[i]['x'] < markers[j]['x'] + markers[j]['w'] + 3:
                    markers[i]['row'] = 1
                    break

    # Draw tick marks for each shot (thin colored line)
    for m in markers:
        x_tick = m['x'] + m['w'] // 2
        draw.rectangle([x_tick, 3, x_tick + 1, 8], fill=(*m['color'], 255))

    # Draw label boxes
    for m in markers:
        row_y = 12 + m['row'] * (m['h'] + 4)

        # Rounded rectangle box
        box_coords = [m['x'], row_y, m['x'] + m['w'], row_y + m['h']]
        draw.rounded_rectangle(box_coords, radius=4, fill=(*m['color'], 240))

        # Text (centered in box)
        text_x = m['x'] + (m['w'] - m['text_w']) // 2
        text_y = row_y + (m['h'] - m['text_h']) // 2
        draw.text((text_x, text_y), m['label'], fill=(255, 255, 255, 255), font=font)

    return img


def create_label_pngs(tmpdir):
    """Create PNG images for each shot type label. Returns dict of paths."""
    if not HAS_PIL:
        return {}

    font = None
    for fp in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try:
            font = ImageFont.truetype(fp, 28)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    label_pngs = {}
    for shot_type, label in SHOT_LABELS.items():
        color = SHOT_COLORS.get(shot_type, (149, 165, 166))

        try:
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w = len(label) * 16
            text_h = 20

        pad_x, pad_y = 14, 10
        img_w = text_w + pad_x * 2
        img_h = text_h + pad_y * 2

        img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, img_w - 1, img_h - 1], radius=6,
                               fill=(*color, 220))
        text_x = (img_w - text_w) // 2
        text_y = (img_h - text_h) // 2
        draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)

        path = os.path.join(tmpdir, f"label_{shot_type}.png")
        img.save(path)
        label_pngs[shot_type] = path

    return label_pngs


def export_full_timeline(video_path, det_data, output_path, width, height, duration):
    """Export full video with shot labels overlaid on the video content.

    Uses drawbox for timeline strip + colored boxes, and overlay filter
    with PIL-generated label PNGs for text (since drawtext is unavailable).
    """
    print("\n[1/3] Exporting full video with shot overlays...")

    detections = det_data['detections']
    tmpdir = tempfile.mkdtemp(prefix='tennis_timeline_')

    try:
        # Create label PNGs for each shot type
        label_pngs = create_label_pngs(tmpdir)
        if not label_pngs:
            print("  [SKIP] PIL not available")
            return

        # Collect unique shot types used in detections
        used_types = list(dict.fromkeys(
            det.get('shot_type', 'unknown_shot') for det in detections
        ))

        # Build input list: [0]=video, [1..N]=label PNGs
        inputs = ['-i', video_path]
        type_to_input = {}
        for i, st in enumerate(used_types):
            if st in label_pngs:
                inputs.extend(['-i', label_pngs[st]])
                type_to_input[st] = i + 1  # input index (0 is video)

        # Build filter_complex
        filter_parts = []
        current_tag = '0:v'

        # 1. Timeline strip at top (drawbox — no text needed)
        strip_h = 8
        strip_filters = [
            f"drawbox=x=0:y=0:w=iw:h={strip_h}:color=black@0.6:t=fill"
        ]
        # Colored tick marks
        for det in detections:
            ts = det['timestamp']
            frac = ts / duration
            color = SHOT_COLORS_HEX.get(det.get('shot_type', 'unknown_shot'), '0x95A5A6')
            strip_filters.append(
                f"drawbox=x='{frac}*iw-1':y=0:w=3:h={strip_h}:color={color}:t=fill"
            )
        # Moving red progress
        strip_filters.append(
            f"drawbox=x=0:y={strip_h - 2}:w='(t/{duration})*iw':h=2:color=0xFF4444:t=fill"
        )

        # Combine strip filters on the video
        strip_str = ','.join(strip_filters)
        filter_parts.append(f"[{current_tag}]{strip_str}[strip]")
        current_tag = 'strip'

        # 2. Overlay shot labels at each detection time
        show_before = 0.3
        show_after = 1.7
        label_x = 30
        label_y = 30

        for i, det in enumerate(detections):
            ts = det['timestamp']
            shot_type = det.get('shot_type', 'unknown_shot')
            if shot_type not in type_to_input:
                continue

            inp_idx = type_to_input[shot_type]
            t_start = max(0, ts - show_before)
            t_end = min(duration, ts + show_after)

            out_tag = f"v{i}"
            filter_parts.append(
                f"[{current_tag}][{inp_idx}:v]overlay=x={label_x}:y={label_y}"
                f":enable='between(t,{t_start:.2f},{t_end:.2f})'"
                f"[{out_tag}]"
            )
            current_tag = out_tag

        filter_complex = ';'.join(filter_parts)

        cmd = [
            'ffmpeg', '-y',
            *inputs,
            '-filter_complex', filter_complex,
            '-map', f'[{current_tag}]', '-map', '0:a',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart',
            output_path
        ]

        print(f"  {len(detections)} shot overlays + timeline strip")
        print(f"  Encoding ({duration:.0f}s)...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [ERROR] ffmpeg failed:\n{result.stderr[-500:]}")
            return

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved: {output_path} ({size_mb:.0f}MB)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def compute_segments(detections, duration, before=2.0, after=2.0, min_gap=0.5):
    """Compute highlight segments, merging overlaps."""
    if not detections:
        return []

    raw_segments = []
    for det in detections:
        ts = det['timestamp']
        start = max(0, ts - before)
        end = min(duration, ts + after)
        raw_segments.append((start, end, det))

    raw_segments.sort(key=lambda s: s[0])

    # Merge overlapping/adjacent segments
    merged = []
    for start, end, det in raw_segments:
        if merged and start <= merged[-1]['end'] + min_gap:
            merged[-1]['end'] = max(merged[-1]['end'], end)
            merged[-1]['shots'].append(det)
        else:
            merged.append({
                'start': start,
                'end': end,
                'shots': [det],
            })

    return merged


def compute_rally_segments(detections, duration, point_gap=8.0, before=3.5, after=4.5):
    """Group shots into rally points by gap threshold, removing dead time between points.

    Shots within `point_gap` seconds of each other belong to the same point.
    Each point gets `before` seconds of lead-in and `after` seconds of trail.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d['timestamp'])

    # Split into points by gap threshold
    points = []
    current_point = [sorted_dets[0]]

    for det in sorted_dets[1:]:
        gap = det['timestamp'] - current_point[-1]['timestamp']
        if gap > point_gap:
            points.append(current_point)
            current_point = [det]
        else:
            current_point.append(det)
    points.append(current_point)

    # Build segments with padding
    segments = []
    for i, point_shots in enumerate(points):
        first_ts = point_shots[0]['timestamp']
        last_ts = point_shots[-1]['timestamp']
        start = max(0, first_ts - before)
        end = min(duration, last_ts + after)
        segments.append({
            'start': start,
            'end': end,
            'shots': point_shots,
            'point_num': i + 1,
        })

    return segments


def extract_segment(video_path, start, duration, output_path, fps=60, speed=1.0):
    """Extract a video segment with re-encoding for precise cuts.

    speed: playback speed multiplier. 0.25 = 4x slow motion.
    """
    vf_filters = []
    drop_audio = False

    if speed != 1.0:
        # setpts: divide by speed (0.25x speed → multiply PTS by 4)
        vf_filters.append(f"setpts=PTS/{speed:.4f}")
        # Drop audio for slow-mo — sounds garbled
        drop_audio = True

    vf_arg = ['-vf', ','.join(vf_filters)] if vf_filters else []

    # -t is output duration: at 0.25x speed, output needs to be duration/speed
    # to show the full segment content
    output_duration = duration / speed if speed != 1.0 else duration

    audio_args = ['-an'] if drop_audio else [
        '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',
    ]

    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-i', video_path,
        '-t', str(output_duration),
        *vf_arg,
        '-r', str(fps),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        *audio_args,
        '-movflags', '+faststart',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def create_title_card(text, width, height, duration_secs, output_path, fps=60, no_audio=False):
    """Create a title card video clip with text on black background."""
    # Create title image with PIL
    img = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load large font for title
    font = None
    for fp in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try:
            font = ImageFont.truetype(fp, 64)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    # Center text
    try:
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w = len(text) * 30
        text_h = 50

    x = (width - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    # Decorative line under text
    line_w = min(text_w + 40, width - 100)
    line_x = (width - line_w) // 2
    draw.rectangle([line_x, y + text_h + 20, line_x + line_w, y + text_h + 23],
                   fill=(255, 140, 0))

    # Save temp image
    img_path = output_path.replace('.mp4', '.png')
    img.save(img_path)

    # Generate video from image
    if no_audio:
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1', '-framerate', str(fps), '-i', img_path,
            '-t', str(duration_secs),
            '-r', str(fps),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-an',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ]
    else:
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1', '-framerate', str(fps), '-i', img_path,
            '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo',
            '-t', str(duration_secs),
            '-r', str(fps),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(img_path)
    return result.returncode == 0


def concat_videos(segment_paths, output_path, reencode=False):
    """Concatenate video segments using ffmpeg concat demuxer.

    reencode: If True, re-encode output to fix audio drift from mixed sources
              (e.g., title cards with synthetic audio + real video segments).
    """
    # Write concat file
    concat_path = output_path.replace('.mp4', '_concat.txt')
    with open(concat_path, 'w') as f:
        for path in segment_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    if reencode:
        codec_args = [
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',
        ]
    else:
        codec_args = ['-c', 'copy']

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', concat_path,
        *codec_args,
        '-movflags', '+faststart',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(concat_path)
    return result.returncode == 0


def export_highlights_chronological(video_path, det_data, output_path,
                                     width, height, duration, fps=60, before=2.0, after=2.0,
                                     speed=1.0):
    """Export highlight clips in chronological order."""
    label = " (slow motion)" if speed < 1.0 else ""
    print(f"\nExporting highlights (chronological{label})...")

    detections = [d for d in det_data['detections']
                  if d.get('shot_type', 'unknown_shot') not in EXCLUDED_FROM_HIGHLIGHTS]
    segments = compute_segments(detections, duration, before, after)

    print(f"  {len(detections)} shots -> {len(segments)} segments (merged overlaps)")

    tmpdir = tempfile.mkdtemp(prefix='tennis_export_')
    segment_paths = []

    try:
        for i, seg in enumerate(segments):
            seg_path = os.path.join(tmpdir, f"seg_{i:03d}.mp4")
            seg_dur = seg['end'] - seg['start']
            shot_labels = ', '.join(SHOT_LABELS.get(s['shot_type'], '?')
                                    for s in seg['shots'])
            print(f"  Segment {i+1}/{len(segments)}: "
                  f"{seg['start']:.1f}s-{seg['end']:.1f}s ({seg_dur:.1f}s) [{shot_labels}]")

            if not extract_segment(video_path, seg['start'], seg_dur, seg_path, fps=fps,
                                   speed=speed):
                print(f"    [ERROR] Failed to extract segment {i+1}")
                continue
            segment_paths.append(seg_path)

        if not segment_paths:
            print("  [ERROR] No segments extracted")
            return

        print(f"  Concatenating {len(segment_paths)} segments...")
        if concat_videos(segment_paths, output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Saved: {output_path} ({size_mb:.0f}MB)")
        else:
            print("  [ERROR] Concat failed")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def export_highlights_grouped(video_path, det_data, output_path,
                               width, height, duration, fps=60, before=2.0, after=2.0,
                               speed=1.0):
    """Export highlights grouped by shot type with title cards."""
    label = " (slow motion)" if speed < 1.0 else ""
    print(f"\nExporting highlights (grouped by type{label})...")

    detections = [d for d in det_data['detections']
                  if d.get('shot_type', 'unknown_shot') not in EXCLUDED_FROM_HIGHLIGHTS]

    # Group detections by type
    groups = {}
    for det in detections:
        st = det.get('shot_type', 'unknown_shot')
        groups.setdefault(st, []).append(det)

    tmpdir = tempfile.mkdtemp(prefix='tennis_grouped_')
    all_paths = []

    try:
        for shot_type in SHOT_ORDER:
            if shot_type not in groups:
                continue

            group_dets = groups[shot_type]
            title = GROUP_TITLES.get(shot_type, shot_type.upper())
            count = len(group_dets)
            print(f"  Group: {title} ({count} shots)")

            # Title card
            title_path = os.path.join(tmpdir, f"title_{shot_type}.mp4")
            title_text = f"{title}  ({count})"
            if create_title_card(title_text, width, height, 2.0, title_path, fps=fps,
                                no_audio=(speed != 1.0)):
                all_paths.append(title_path)

            # Compute segments for this group
            segments = compute_segments(group_dets, duration, before, after)

            for i, seg in enumerate(segments):
                seg_path = os.path.join(tmpdir, f"{shot_type}_{i:03d}.mp4")
                seg_dur = seg['end'] - seg['start']

                if not extract_segment(video_path, seg['start'], seg_dur, seg_path, fps=fps,
                                       speed=speed):
                    print(f"    [ERROR] Failed to extract {shot_type} segment {i+1}")
                    continue
                all_paths.append(seg_path)

        if not all_paths:
            print("  [ERROR] No segments generated")
            return

        print(f"  Concatenating {len(all_paths)} clips...")
        # Re-encode to fix audio drift from title cards with synthetic audio
        if concat_videos(all_paths, output_path, reencode=True):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Saved: {output_path} ({size_mb:.0f}MB)")
        else:
            print("  [ERROR] Concat failed")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def export_rally_points(video_path, det_data, output_path,
                         width, height, duration, fps=60,
                         point_gap=8.0, before=3.5, after=4.5, speed=1.0):
    """Export video with dead time between points removed, keeping full rally points."""
    label = " (slow motion)" if speed < 1.0 else ""
    print(f"\nExporting rally points{label}...")

    detections = [d for d in det_data['detections']
                  if d.get('shot_type', 'unknown_shot') not in EXCLUDED_FROM_HIGHLIGHTS]
    segments = compute_rally_segments(detections, duration, point_gap, before, after)

    total_kept = sum(s['end'] - s['start'] for s in segments)
    dead_removed = duration - total_kept
    print(f"  {len(detections)} shots -> {len(segments)} points "
          f"(gap threshold: {point_gap}s)")
    print(f"  Keeping {total_kept:.0f}s, removing {dead_removed:.0f}s dead time "
          f"({dead_removed/duration*100:.0f}%)")

    tmpdir = tempfile.mkdtemp(prefix='tennis_rally_')
    segment_paths = []

    try:
        for i, seg in enumerate(segments):
            seg_path = os.path.join(tmpdir, f"pt_{i:03d}.mp4")
            seg_dur = seg['end'] - seg['start']
            shot_types = {}
            for s in seg['shots']:
                st = SHOT_LABELS.get(s.get('shot_type', 'unknown_shot'), '?')
                shot_types[st] = shot_types.get(st, 0) + 1
            breakdown = ' '.join(f"{v}{k}" for k, v in shot_types.items())
            print(f"  Point {seg['point_num']}: {seg['start']:.1f}s-{seg['end']:.1f}s "
                  f"({seg_dur:.1f}s, {len(seg['shots'])} shots: {breakdown})")

            if not extract_segment(video_path, seg['start'], seg_dur, seg_path, fps=fps,
                                   speed=speed):
                print(f"    [ERROR] Failed to extract point {seg['point_num']}")
                continue
            segment_paths.append(seg_path)

        if not segment_paths:
            print("  [ERROR] No segments extracted")
            return

        print(f"  Concatenating {len(segment_paths)} points...")
        if concat_videos(segment_paths, output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Saved: {output_path} ({size_mb:.0f}MB)")
        else:
            print("  [ERROR] Concat failed")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def upload_to_r2(local_path, remote_key):
    """Upload a file to Cloudflare R2 storage."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

    # Reload settings after env vars are set
    import importlib
    import config.settings
    importlib.reload(config.settings)

    from storage.r2_client import R2Client
    client = R2Client()
    client.upload(local_path, remote_key)
    public_url = f"https://media.playfullife.com/{remote_key}"
    print(f"  Public URL: {public_url}")
    return public_url


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export tennis analysis videos')
    parser.add_argument('video', nargs='+', help='Path(s) to preprocessed video(s)')
    parser.add_argument('--before', type=float, default=2.0,
                        help='Seconds before each shot for highlights (default: 2.0)')
    parser.add_argument('--after', type=float, default=2.0,
                        help='Seconds after each shot for highlights (default: 2.0)')
    parser.add_argument('--types', nargs='+', default=['all'],
                        choices=['all', 'timeline', 'highlights', 'grouped', 'rally', 'comparison'],
                        help='Which export types to generate')
    parser.add_argument('--point-gap', type=float, default=8.0,
                        help='Max inter-shot gap within a point for rally mode (default: 8.0)')
    parser.add_argument('--rally-before', type=float, default=3.5,
                        help='Seconds before first shot in a point (default: 3.5)')
    parser.add_argument('--rally-after', type=float, default=4.5,
                        help='Seconds after last shot in a point (default: 4.5)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed (0.25 = 4x slow motion, default: 1.0)')
    parser.add_argument('--slow-motion', action='store_true',
                        help='Generate both normal and slow motion (0.25x) versions')
    parser.add_argument('--upload', action='store_true',
                        help='Upload generated files to R2 (playfullife.com)')
    args = parser.parse_args()

    for video_path in args.video:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if not os.path.exists(video_path):
            print(f"[ERROR] Video not found: {video_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Export Videos: {video_name}")
        print(f"  Video: {video_path}")

        # Load detections
        det_data = load_detections(video_name)
        detections = det_data['detections']
        det_duration = det_data.get('duration', 0)

        # Get video info
        width, height, vid_duration, fps = get_video_info(video_path)
        duration = vid_duration if vid_duration > 0 else det_duration
        print(f"  Resolution: {width}x{height}, Duration: {duration:.1f}s, FPS: {fps:.0f}")
        print(f"  Shots: {len(detections)}")

        # Create output directory
        export_dir = f"exports/{video_name}"
        os.makedirs(export_dir, exist_ok=True)

        export_types = args.types
        if 'all' in export_types:
            export_types = ['timeline', 'highlights', 'grouped', 'rally']
            # 'comparison' excluded from 'all' — must be requested explicitly

        # Determine speed variants to generate
        speeds = []
        if args.slow_motion:
            speeds = [(1.0, ''), (0.25, '_slowmo')]
        else:
            speeds = [(args.speed, '_slowmo' if args.speed < 1.0 else '')]

        exported_files = []

        if 'timeline' in export_types:
            out = f"{export_dir}/{video_name}_timeline.mp4"
            export_full_timeline(
                video_path, det_data, out,
                width, height, duration
            )
            if os.path.exists(out):
                exported_files.append(out)

        for speed, suffix in speeds:
            if 'highlights' in export_types:
                out = f"{export_dir}/{video_name}_highlights{suffix}.mp4"
                export_highlights_chronological(
                    video_path, det_data, out,
                    width, height, duration, fps=fps,
                    before=args.before, after=args.after, speed=speed
                )
                if os.path.exists(out):
                    exported_files.append(out)

            if 'grouped' in export_types:
                out = f"{export_dir}/{video_name}_grouped{suffix}.mp4"
                export_highlights_grouped(
                    video_path, det_data, out,
                    width, height, duration, fps=fps,
                    before=args.before, after=args.after, speed=speed
                )
                if os.path.exists(out):
                    exported_files.append(out)

            if 'rally' in export_types:
                out = f"{export_dir}/{video_name}_rally{suffix}.mp4"
                export_rally_points(
                    video_path, det_data, out,
                    width, height, duration, fps=fps,
                    point_gap=args.point_gap,
                    before=args.rally_before, after=args.rally_after,
                    speed=speed
                )
                if os.path.exists(out):
                    exported_files.append(out)

        # Pro comparison (if requested)
        if 'comparison' in export_types:
            try:
                from scripts.pro_comparison import generate_comparisons
                comparison_files = generate_comparisons(
                    video_path, output_dir=export_dir, upload=False
                )
                exported_files.extend(comparison_files)
            except Exception as e:
                print(f"  [WARN] Comparison generation failed: {e}")

        # Upload to R2 if requested
        if args.upload and exported_files:
            print(f"\nUploading {len(exported_files)} files to R2...")
            for filepath in exported_files:
                filename = os.path.basename(filepath)
                remote_key = f"highlights/{video_name}/{filename}"
                upload_to_r2(filepath, remote_key)

    print("\nDone!")


if __name__ == '__main__':
    main()
