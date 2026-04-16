#!/usr/bin/env python3
"""Backfill highlights/{vid}/shots.json on R2 for existing videos.

Reads each video's detection JSON locally and computes per-shot positions
in every output variant that's actually present on R2. Uploads shots.json
to R2 alongside the videos. Safe to run repeatedly.

Usage:
    python scripts/backfill_shots_json.py                 # all videos with detections
    python scripts/backfill_shots_json.py IMG_1141        # single video
    python scripts/backfill_shots_json.py --dry-run       # print, don't upload
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Reuse the pure math from export_videos
from scripts.export_videos import (
    compute_segments, compute_rally_segments,
    _positions_for_segments, EXCLUDED_FROM_HIGHLIGHTS,
)

PROJECT_ROOT = Path(__file__).parent.parent
DETECTIONS_DIR = PROJECT_ROOT / "detections"

# Default export knobs (match export_videos.py defaults)
DEFAULTS = dict(before=2.0, after=2.0, point_gap=8.0, rally_before=3.5, rally_after=4.5)


def positions_for_video(det_data, duration, present_variants):
    """Compute positions only for variants that are present in present_variants."""
    from argparse import Namespace
    all_detections = det_data['detections']
    filtered = [d for d in all_detections
                if d.get('shot_type', 'unknown_shot') not in EXCLUDED_FROM_HIGHLIGHTS]
    ts_list = [d['timestamp'] for d in filtered]
    ts_to_idx = {d['timestamp']: i for i, d in enumerate(all_detections)}

    positions = {}

    if 'timeline' in present_variants:
        positions['timeline'] = {
            i: round(d['timestamp'], 3) for i, d in enumerate(all_detections)
        }

    if 'rally' in present_variants or 'rally_slowmo' in present_variants:
        rally_segs = compute_rally_segments(
            filtered, duration,
            point_gap=DEFAULTS['point_gap'],
            before=DEFAULTS['rally_before'], after=DEFAULTS['rally_after'],
        )
        for speed, key in [(1.0, 'rally'), (0.25, 'rally_slowmo')]:
            if key not in present_variants:
                continue
            pos = _positions_for_segments(ts_list, rally_segs, 0.0, speed)
            positions[key] = {ts_to_idx[ts]: round(v, 3) for ts, v in pos.items()}

    # bytype variants
    bytype_groups = defaultdict(list)
    for det in filtered:
        st = det.get('shot_type', 'unknown_shot')
        if st in ('forehand_volley', 'backhand_volley', 'overhead'):
            bytype_groups['volleys'].append(det)
        elif st == 'forehand':
            bytype_groups['forehands'].append(det)
        elif st == 'backhand':
            bytype_groups['backhands'].append(det)
        elif st == 'serve':
            bytype_groups['serves'].append(det)
        else:
            bytype_groups['other'].append(det)

    title_s = 2.0
    for bucket, dets in bytype_groups.items():
        if len(dets) < 2:
            continue
        segs = compute_segments(dets, duration, DEFAULTS['before'], DEFAULTS['after'])
        group_ts = [d['timestamp'] for d in dets]
        for speed, suffix in [(1.0, ''), (0.25, '_slowmo')]:
            key = bucket + suffix
            if key not in present_variants:
                continue
            pos = _positions_for_segments(group_ts, segs, title_s, speed)
            positions[key] = {ts_to_idx[ts]: round(v, 3) for ts, v in pos.items()}

    # grouped (legacy)
    if 'grouped' in present_variants or 'grouped_slowmo' in present_variants:
        grouped = defaultdict(list)
        for det in filtered:
            grouped[det.get('shot_type', 'unknown_shot')].append(det)
        for speed, suffix in [(1.0, ''), (0.25, '_slowmo')]:
            key = 'grouped' + suffix
            if key not in present_variants:
                continue
            cursor = 0.0
            out_map = {}
            for st in ['forehand', 'backhand', 'serve', 'unknown_shot']:
                if st not in grouped:
                    continue
                gds = grouped[st]
                cursor += 2.0
                segs = compute_segments(gds, duration, DEFAULTS['before'], DEFAULTS['after'])
                group_ts = [d['timestamp'] for d in gds]
                pos = _positions_for_segments(group_ts, segs, 0.0, speed)
                for ts, v in pos.items():
                    out_map[ts_to_idx[ts]] = round(cursor + v, 3)
                cursor += sum((s['end'] - s['start']) / speed for s in segs)
            positions[key] = out_map

    return positions


def build_shots_json(video_name, det_data, positions):
    shots = []
    for idx, d in enumerate(det_data['detections']):
        entry = {
            'idx': idx,
            't': round(d['timestamp'], 3),
            'type': d.get('shot_type', 'unknown'),
            'confidence': round(d.get('confidence', 0), 3),
            'positions': {v: pos_map[idx]
                          for v, pos_map in positions.items()
                          if idx in pos_map},
        }
        shots.append(entry)
    return {'video': video_name, 'shots': shots}


def load_det(vid):
    for name in [f"{vid}_fused.json", f"{vid}_fused_detections.json"]:
        p = DETECTIONS_DIR / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def list_variants_on_r2(r2_client, vid):
    """Return set of variant keys that exist on R2 for this video."""
    prefix = f'highlights/{vid}/'
    keys = r2_client.list(prefix=prefix, max_keys=100)
    variants = set()
    for k in keys:
        name = k.split('/')[-1]
        if not name.endswith('.mp4'):
            continue
        stem = name[:-4]  # remove .mp4
        if stem.startswith(f"{vid}_"):
            variant = stem[len(vid) + 1:]
            variants.add(variant)
    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('videos', nargs='*', help='Video IDs (default: all with detection files)')
    ap.add_argument('--dry-run', action='store_true', help='Print, do not upload')
    args = ap.parse_args()

    from storage.r2_client import R2Client
    c = R2Client()

    if args.videos:
        targets = args.videos
    else:
        targets = sorted({
            p.stem.replace('_fused_detections', '').replace('_fused', '')
            for p in DETECTIONS_DIR.glob("*_fused*.json")
        })

    print(f"Backfilling shots.json for {len(targets)} videos")
    success = fail = 0
    for vid in targets:
        det = load_det(vid)
        if not det:
            print(f"  {vid}: no detection JSON, skip")
            fail += 1
            continue
        variants = list_variants_on_r2(c, vid)
        if not variants:
            print(f"  {vid}: no MP4 variants on R2, skip")
            continue
        duration = det.get('duration', 0)
        positions = positions_for_video(det, duration, variants)
        blob = build_shots_json(vid, det, positions)
        n_with_pos = sum(1 for s in blob['shots'] if s['positions'])

        if args.dry_run:
            print(f"  {vid}: {len(blob['shots'])} shots, {n_with_pos} with positions, "
                  f"variants={sorted(positions)}")
            continue

        tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w')
        json.dump(blob, tmp, separators=(',', ':'))
        tmp.close()
        c.upload(tmp.name, f'highlights/{vid}/shots.json', content_type='application/json')
        os.unlink(tmp.name)
        print(f"  {vid}: {len(blob['shots'])} shots, {n_with_pos} with positions "
              f"({len(positions)} variants)")
        success += 1

    print(f"\nDone: {success} uploaded, {fail} failed")


if __name__ == '__main__':
    main()
