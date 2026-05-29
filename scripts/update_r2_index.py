#!/usr/bin/env python3
"""Regenerate the browsable index.html on R2 with thumbnails and metadata.

Usage:
    .venv/bin/python scripts/update_r2_index.py
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_thumbnail(vid, user_hash=None):
    """Generate a thumbnail for a video if it doesn't exist. Returns True if available.

    user_hash: when set, also probe the per-user R2 location
    highlights/<user_hash>/thumbs/<vid>.jpg as a fallback — that's where
    backfilled thumbs live post per-user refactor.
    """
    thumb_dir = os.path.join(PROJECT_ROOT, 'exports', 'thumbs')
    os.makedirs(thumb_dir, exist_ok=True)
    thumb = os.path.join(thumb_dir, f'{vid}.jpg')

    if os.path.exists(thumb):
        return True

    # Try downloading from R2. Probe the per-user location first because
    # that's where the post-refactor backfill puts thumbs; legacy flat
    # paths are kept as a second-pass fallback for any pre-refactor video.
    import urllib.request
    candidate_urls = []
    if user_hash:
        candidate_urls.append(
            f'https://tennis.playfullife.com/highlights/{user_hash}/thumbs/{vid}.jpg'
        )
    candidate_urls += [
        f'https://tennis.playfullife.com/thumbs/{vid}.jpg',
        f'https://tennis.playfullife.com/highlights/thumbs/{vid}.jpg',
    ]
    for r2_url in candidate_urls:
        try:
            req = urllib.request.Request(r2_url, headers={'User-Agent': 'tennis-index/1.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                if len(data) > 100:
                    with open(thumb, 'wb') as f:
                        f.write(data)
                    return True
        except Exception:
            pass

    # Try generating from local preprocessed video — use thumbnail filter to
    # pick a representative (non-black) frame automatically
    for pp in [
        os.path.join(PROJECT_ROOT, 'preprocessed', f'{vid}.mp4'),
        os.path.join(PROJECT_ROOT, 'preprocessed', f'{vid}_240fps.mp4'),
    ]:
        if os.path.exists(pp):
            # thumbnail filter picks the most representative frame from the first ~100 frames
            # after seeking to 25% into the video (avoids title cards / dark intros)
            subprocess.run(
                ['ffmpeg', '-y', '-ss', '30', '-i', pp, '-vf',
                 'thumbnail=100,scale=480:-1', '-frames:v', '1',
                 '-q:v', '6', thumb],
                capture_output=True, timeout=30
            )
            if os.path.exists(thumb):
                return True

    return False


def upload_thumbnail(client, vid, user_hash=None):
    """Upload thumbnail to R2 if it exists locally.

    user_hash: if set, upload under highlights/<user_hash>/thumbs/ so the
    per-user gallery URL can resolve it. Otherwise (legacy) upload flat.
    """
    thumb = os.path.join(PROJECT_ROOT, 'exports', 'thumbs', f'{vid}.jpg')
    if os.path.exists(thumb):
        key = (f'highlights/{user_hash}/thumbs/{vid}.jpg'
               if user_hash else f'highlights/thumbs/{vid}.jpg')
        client.upload(thumb, key, content_type='image/jpeg')


def get_video_metadata(vid, r2_client=None, user_hash=None):
    """Gather metadata for a video from detection JSON, R2 meta.json, or raw MOV.

    user_hash: when set, look up the meta.json at the per-user path
    (highlights/<user_hash>/<vid>/meta.json) first, then fall back to
    the legacy flat path for any video that predates the per-user refactor.
    Without this, every backfilled video falls through to "Unknown Date"
    because the legacy flat meta.json was moved during backfill.
    """
    info = {}

    # 1. Local detection JSON
    for det_name in [f'{vid}_fused.json', f'{vid}_fused_detections.json']:
        det_path = os.path.join(PROJECT_ROOT, 'detections', det_name)
        if os.path.exists(det_path):
            with open(det_path) as f:
                d = json.load(f)
            info['duration'] = d.get('duration', 0)
            dets = d.get('detections', [])
            info['shots'] = len(dets)
            types = {}
            for det in dets:
                st = det.get('shot_type', 'unknown')
                types[st] = types.get(st, 0) + 1
            info['breakdown'] = types
            if d.get('created'):
                info['created'] = d['created']
            break

    # 2. R2 meta.json (uploaded by GPU worker — has metadata even when local files missing)
    if r2_client:
        meta_candidates = []
        if user_hash:
            meta_candidates.append(f'highlights/{user_hash}/{vid}/meta.json')
        meta_candidates.append(f'highlights/{vid}/meta.json')  # legacy flat
        for meta_key in meta_candidates:
            try:
                obj = r2_client.client.get_object(
                    Bucket=r2_client.bucket_name, Key=meta_key)
                meta = json.loads(obj['Body'].read())
                if not info.get('shots'):
                    info['duration'] = meta.get('duration', 0)
                    info['shots'] = meta.get('shots', 0)
                    info['breakdown'] = meta.get('breakdown', {})
                if meta.get('created') and 'created' not in info:
                    info['created'] = meta['created']
                if meta.get('display_name'):
                    info['display_name'] = meta['display_name']
                for bk in ('ball_avg_speed', 'ball_max_speed', 'ball_detection_rate',
                            'avg_speed_mph', 'max_speed_mph', 'in_count', 'out_count'):
                    if meta.get(bk) is not None:
                        info[bk] = meta[bk]
                break  # found one — don't probe the legacy path
            except Exception:
                continue

    # 3. Creation date from raw MOV
    if 'created' not in info:
        raw_path = os.path.join(PROJECT_ROOT, 'raw', f'{vid}.MOV')
        if os.path.exists(raw_path):
            try:
                r = subprocess.run(
                    ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', raw_path],
                    capture_output=True, text=True, timeout=10
                )
                tags = json.loads(r.stdout).get('format', {}).get('tags', {})
                cd = tags.get('com.apple.quicktime.creationdate', tags.get('creation_time', ''))
                if cd:
                    info['created'] = cd
            except Exception:
                pass

    # 4. Fallback: preprocessed file mod time
    if 'created' not in info:
        pp = os.path.join(PROJECT_ROOT, 'preprocessed', f'{vid}.mp4')
        if os.path.exists(pp):
            mtime = os.path.getmtime(pp)
            info['created'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')

    return info


def format_date_time(created_str):
    """Parse creation date string into (date_str, time_str)."""
    if not created_str:
        return '?', ''
    if 'T' not in created_str:
        return created_str[:10], ''
    try:
        dt = created_str.split('T')
        d = datetime.strptime(dt[0], '%Y-%m-%d')
        date_str = d.strftime('%b %d, %Y')
        time_clean = dt[1].split('-')[0].split('+')[0].split('.')[0]
        h, mn, s = time_clean.split(':')
        h = int(h)
        ampm = 'AM' if h < 12 else 'PM'
        h12 = h % 12 or 12
        time_str = f'{h12}:{mn} {ampm}'
        return date_str, time_str
    except Exception:
        return created_str[:10], ''


def build_index_html(videos_meta):
    """Build the HTML index page with session grouping, search, and filters."""

    label_map = {
        'timeline': ('Timeline', '#FF8C00'),
        'rally': ('Rally', '#27AE60'),
        'rally_slowmo': ('Rally Slow-Mo', '#9B59B6'),
        'forehands': ('Forehands', '#E67E22'),
        'forehands_slowmo': ('Forehands Slow-Mo', '#D35400'),
        'backhands': ('Backhands', '#3498DB'),
        'backhands_slowmo': ('Backhands Slow-Mo', '#2980B9'),
        'serves': ('Serves', '#2ECC71'),
        'serves_slowmo': ('Serves Slow-Mo', '#27AE60'),
        'volleys': ('Volleys', '#9B59B6'),
        'volleys_slowmo': ('Volleys Slow-Mo', '#8E44AD'),
        'grouped': ('All by Type', '#5DADE2'),
        'grouped_slowmo': ('All by Type Slow-Mo', '#3498DB'),
        'highlights': ('Highlights', '#2ECC71'),
        'highlights_slowmo': ('Highlights Slow-Mo', '#8E44AD'),
        'comparisons': ('Pro Compare', '#E74C3C'),
        'tracked': ('Player Tracking', '#00BCD4'),
    }
    link_order = [
        'timeline', 'rally', 'rally_slowmo',
        'forehands', 'forehands_slowmo',
        'backhands', 'backhands_slowmo',
        'serves', 'serves_slowmo',
        'volleys', 'volleys_slowmo',
        'highlights', 'highlights_slowmo',
        'grouped', 'grouped_slowmo',  # legacy: shown only if no per-type files exist
        'tracked',
        'comparisons',
    ]

    # Build JSON data for client-side rendering
    video_data = []
    for vid, m in videos_meta.items():
        dur = m.get('duration', 0)
        bd = m.get('breakdown', {})
        files = m.get('files', [])
        file_keys = {}
        for f in files:
            key = f.replace(vid + '_', '').replace('.mp4', '')
            file_keys[key] = f

        # Hide legacy 'grouped' files if any per-type files exist (cleaner UX)
        has_per_type = any(k in file_keys for k in
                           ['forehands', 'backhands', 'serves', 'volleys'])
        skip_keys = set()
        if has_per_type:
            skip_keys.update(['grouped', 'grouped_slowmo', 'highlights', 'highlights_slowmo'])

        links = []
        for key in link_order:
            if key not in file_keys or key in skip_keys:
                continue
            f = file_keys[key]
            label, color = label_map.get(key, (key, '#5DADE2'))
            links.append({'key': key, 'file': f, 'label': label, 'color': color})
        for key, f in file_keys.items():
            if key not in link_order and key not in skip_keys:
                label, color = label_map.get(key, (key, '#5DADE2'))
                links.append({'key': key, 'file': f, 'label': label, 'color': color})

        # Hide non-tennis uploads: pipeline ran to export (has files) but the
        # shot detector (F1=96.5%) found 0 swings. Files stay in R2 — admin
        # can recover via direct API if a false negative ever shows up.
        if m.get('shots', 0) == 0 and links:
            print(f'  skip {vid}: 0 shots detected (probably not tennis)')
            continue

        video_data.append({
            'id': vid,
            'created': m.get('created', ''),
            'duration': dur,
            'shots': m.get('shots', 0),
            'breakdown': bd,
            'has_thumb': m.get('has_thumb', False),
            'links': links,
            'ball_avg_speed': m.get('ball_avg_speed'),
            'ball_detection_rate': m.get('ball_detection_rate'),
            'avg_speed_mph': m.get('avg_speed_mph'),
            'max_speed_mph': m.get('max_speed_mph'),
            'in_count': m.get('in_count'),
            'out_count': m.get('out_count'),
            'features': m.get('features', []),
        })

    video_json = json.dumps(video_data, separators=(',', ':'))

    return f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tennis Highlights</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ccircle cx='50' cy='50' r='45' fill='%23dbf757' stroke='%23a8c93f' stroke-width='2'/%3E%3Cpath d='M 8 35 Q 50 50 8 65' fill='none' stroke='%23ffffff' stroke-width='2.5'/%3E%3Cpath d='M 92 35 Q 50 50 92 65' fill='none' stroke='%23ffffff' stroke-width='2.5'/%3E%3C/svg%3E">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0a;color:#eee;overflow-x:hidden}}

/* ── Header ── */
.header{{position:sticky;top:0;z-index:100;background:#0a0a0a;border-bottom:1px solid #1a1a1a;padding:12px 20px}}
.header-inner{{max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
.logo{{color:#FF8C00;font-size:1.3em;font-weight:800;white-space:nowrap}}
.search-box{{flex:1;min-width:180px;position:relative}}
.search-box input{{width:100%;padding:8px 12px 8px 32px;background:#1a1a1a;border:1px solid #2a2a2a;
  border-radius:8px;color:#eee;font-size:0.9em;outline:none;transition:border-color .2s}}
.search-box input:focus{{border-color:#FF8C00}}
.search-box svg{{position:absolute;left:9px;top:50%;transform:translateY(-50%);width:14px;height:14px;fill:#666}}
.header-actions{{display:flex;gap:8px;align-items:center}}
.btn-upload{{padding:6px 16px;background:#FF8C00;color:#fff;border:none;border-radius:6px;
  font-size:0.85em;font-weight:600;cursor:pointer;white-space:nowrap}}
.btn-upload:hover{{background:#e07800}}
.stat-badge{{font-size:0.8em;color:#666;white-space:nowrap}}

/* ── Filters ── */
.filter-toggle{{max-width:1100px;margin:0 auto;padding:8px 20px;display:flex;align-items:center;
  gap:8px;cursor:pointer;color:#999;font-size:.85em;font-weight:600;transition:color .15s}}
.filter-toggle:hover{{color:#fff}}
.filter-badge{{background:#FF8C00;color:#fff;font-size:.7em;padding:1px 7px;border-radius:10px;
  display:none;font-weight:700}}
.filter-badge.show{{display:inline}}
.filter-arrow{{font-size:.6em;transition:transform .2s}}
.filter-arrow.open{{transform:rotate(180deg)}}
.filters{{max-width:1100px;margin:0 auto;padding:10px 20px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;
  overflow:hidden;transition:max-height .25s ease,opacity .2s,padding .2s}}
.filters.collapsed{{max-height:0;opacity:0;padding-top:0;padding-bottom:0}}
.filter-sep{{width:1px;height:20px;background:#333;margin:0 6px;flex-shrink:0}}
.filter-label{{font-size:.7em;color:#555;text-transform:uppercase;letter-spacing:.05em;margin-right:2px}}
.active-filter{{display:none;align-items:center;gap:6px;margin-left:auto;padding:4px 10px 4px 12px;
  background:#FF8C00;border-radius:20px;font-size:.78em;color:#fff;font-weight:600}}
.active-filter.show{{display:flex}}
.active-filter .clear{{background:none;border:none;color:rgba(255,255,255,.7);cursor:pointer;
  font-size:1.1em;line-height:1;padding:0 0 0 4px}}
.active-filter .clear:hover{{color:#fff}}
.sort-select{{margin-left:auto;padding:5px 10px;background:#1a1a1a;border:1px solid #2a2a2a;
  border-radius:8px;color:#999;font-size:.78em;cursor:pointer;outline:none}}
.sort-select:focus{{border-color:#FF8C00}}
.chip{{padding:5px 14px;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:20px;
  color:#999;font-size:0.8em;cursor:pointer;transition:all .15s;white-space:nowrap}}
.chip:hover{{border-color:#555;color:#ddd}}
.chip.active{{background:#FF8C00;border-color:#FF8C00;color:#fff}}

/* ── Dashboard / Processing Banner ── */
.proc-banner{{max-width:1100px;margin:8px auto;padding:0 20px;display:flex;flex-direction:column;gap:6px}}
.proc-bar{{display:flex;align-items:center;gap:10px;padding:8px 14px;background:#1a1a1a;
  border-radius:8px;border:1px solid #2a2a2a;font-size:0.85em;color:#aaa;flex-wrap:wrap}}
.proc-bar.proc-summary{{color:#bbb;font-size:0.82em}}
.proc-bar.proc-summary b{{color:#fff;font-weight:600}}
.proc-bar.proc-summary .sep{{color:#444;margin:0 6px}}
.proc-bar.proc-failed-bar{{border-color:#3a1a1a;background:#1a0f0f}}
.proc-label{{font-size:0.75em;text-transform:uppercase;letter-spacing:0.08em;color:#888;font-weight:600;flex-shrink:0}}
.proc-bar.proc-failed-bar .proc-label{{color:#E74C3C}}
.proc-dot{{width:8px;height:8px;border-radius:50%;background:#3498DB;animation:pulse 1.5s infinite;flex-shrink:0}}
.proc-items{{flex:1;display:flex;gap:12px;flex-wrap:wrap}}
.proc-item{{display:flex;align-items:center;gap:6px}}
.proc-item .stage{{color:#666;font-size:0.85em}}
.proc-item .err{{color:#E74C3C;font-size:0.82em;font-style:italic;max-width:380px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.proc-complete .proc-dot{{background:#27AE60;animation:none}}
.proc-failed .proc-dot{{background:#E74C3C;animation:none}}
.proc-pending .proc-dot{{background:#F1C40F;animation:none}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}

/* ── Content ── */
.content{{max-width:1100px;margin:0 auto;padding:16px 20px}}

/* ── Session Groups ── */
.session{{margin-bottom:28px}}
.session-header{{display:flex;align-items:baseline;gap:12px;margin-bottom:12px;padding-bottom:6px;
  border-bottom:1px solid #1a1a1a}}
.session-header{{cursor:pointer;transition:opacity .15s}}
.session-header:hover{{opacity:.85}}
.session-header:hover .share-icon{{opacity:1}}
.session-date{{font-size:1.1em;font-weight:700;color:#fff}}
.session-stats{{font-size:0.8em;color:#666}}
.share-icon{{font-size:0.8em;opacity:0.3;transition:opacity .15s}}
.session-tags{{display:flex;gap:4px;align-items:center;margin-left:auto}}
.tag{{padding:2px 8px;background:#2a2a2a;border-radius:12px;font-size:.72em;color:#ccc;
  display:inline-flex;align-items:center;gap:4px}}
.tag .rm{{cursor:pointer;opacity:.5;font-size:1.1em}}.tag .rm:hover{{opacity:1}}
.add-tag{{padding:2px 8px;background:transparent;border:1px dashed #333;border-radius:12px;
  font-size:.72em;color:#555;cursor:pointer;transition:all .15s}}
.add-tag:hover{{border-color:#FF8C00;color:#FF8C00}}
.tag-input{{width:100px;padding:2px 8px;background:#1a1a1a;border:1px solid #FF8C00;
  border-radius:12px;font-size:.72em;color:#eee;outline:none}}
.tag-suggest{{position:absolute;top:100%;left:0;background:#1a1a1a;border:1px solid #333;
  border-radius:8px;z-index:10;min-width:120px;max-height:150px;overflow-y:auto;display:none}}
.tag-suggest.show{{display:block}}
.tag-suggest div{{padding:6px 10px;font-size:.8em;color:#ccc;cursor:pointer}}
.tag-suggest div:hover{{background:#222;color:#fff}}
.session.highlighted{{animation:highlightFade 2s ease-out}}
@keyframes highlightFade{{0%{{background:rgba(255,140,0,.15)}}100%{{background:transparent}}}}

/* ── Card Grid ── */
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
.card{{background:#141414;border-radius:10px;border:1px solid #222;overflow:hidden;
  cursor:pointer;transition:border-color .2s,transform .15s}}
.card:hover{{border-color:#FF8C00;transform:translateY(-2px)}}
.card-thumb-wrap{{position:relative}}
.card-thumb{{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;background:#1a1a1a}}
.card-thumb-placeholder{{width:100%;aspect-ratio:16/9;background:#1a1a1a;display:flex;
  align-items:center;justify-content:center;color:#444;font-size:0.85em}}
.card-id{{position:absolute;bottom:4px;left:4px;background:rgba(0,0,0,.7);color:#aaa;
  font-size:.65em;padding:2px 6px;border-radius:4px;font-family:monospace}}
.card-body{{padding:10px 12px}}
.card-time{{font-size:0.95em;font-weight:600;color:#eee}}
.card-meta{{display:flex;gap:8px;margin-top:4px;font-size:0.78em;color:#777}}
.card-breakdown{{font-size:0.75em;color:#999;margin-top:3px}}
/* Compact coach summary on card (always visible when coaching exists) */
/* Compact coach pill — just a label that opens the full sheet/modal.
   Headline text is hidden by default; available via the modal only. */
.card-coach-summary{{margin-top:8px;padding:6px 10px;background:#161a17;
  border-left:3px solid #5ed694;border-radius:3px;cursor:pointer;
  transition:background .15s;display:none}}
.card-coach-summary.loaded{{display:flex;align-items:center;
  justify-content:space-between;gap:8px}}
.card-coach-summary:hover{{background:#1c211d}}
.coach-summary-label{{color:#5ed694;font-size:.66em;font-weight:700;
  text-transform:uppercase;letter-spacing:.08em;display:flex;align-items:center;gap:6px}}
.coach-summary-label .more{{color:#8ae6ae;font-size:.92em;opacity:.85;
  font-weight:600;text-transform:none;letter-spacing:0}}
/* Headline text is kept in the DOM (the click handler reads it) but
   visually hidden so the card stays compact. */
.coach-summary-text{{display:none}}

/* Sequences button on card */
.seq-btn{{display:inline-flex;align-items:center;gap:4px;padding:5px 10px;
  background:#1a1a2e;border:1px solid #2a2a4e;border-radius:5px;color:#a0a0ff;
  font-size:.72em;font-weight:600;cursor:pointer;transition:all .15s;margin-top:6px}}
.seq-btn:hover{{background:#2a2a4e;color:#fff;border-color:#5555aa}}

/* Sequences modal */
.seq-modal-overlay{{display:none;position:fixed;inset:0;z-index:900;
  background:rgba(0,0,0,.9);overflow-y:auto;padding:40px 20px;
  align-items:flex-start;justify-content:center}}
.seq-modal-overlay.open{{display:flex}}
.seq-modal{{max-width:1100px;width:100%;color:#eee}}
.seq-modal h2{{color:#a0a0ff;font-size:.85em;text-transform:uppercase;
  letter-spacing:.1em;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center}}
.seq-modal .close{{background:none;border:none;color:#999;font-size:1.8em;cursor:pointer}}
.seq-modal .close:hover{{color:#fff}}
.seq-grid{{display:flex;flex-direction:column;gap:12px}}
.seq-item{{border-radius:6px;overflow:hidden;background:#111;cursor:pointer;transition:transform .15s}}
.seq-item:hover{{transform:scale(1.01)}}
.seq-item img{{width:100%;display:block}}
.seq-item .seq-label{{padding:6px 10px;font-size:.75em;color:#aaa;
  display:flex;justify-content:space-between}}
.seq-fullscreen{{position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,.95);
  display:flex;align-items:center;justify-content:center;cursor:zoom-out}}
.seq-fullscreen img{{max-width:100vw;max-height:100vh;object-fit:contain}}
@media(max-width:600px){{
  /* Filmstrip used to overflow horizontally on mobile (180px tall, intrinsic
     wide aspect), which detached it from its label below. Fit to width
     instead so the label always sits directly under the image. */
  .seq-item .seq-img-wrap{{overflow:visible}}
  .seq-item .seq-img-wrap img{{width:100%;height:auto;min-width:0}}
  .seq-item .seq-label{{padding:8px 12px;font-size:.78em;color:#ccc;
    background:#1a1a1a;border-top:1px solid #222}}
}}

/* Coach modal filmstrip inline */
.coach-filmstrip{{margin-top:8px;border-radius:4px;overflow:hidden;display:none}}
.coach-filmstrip.loaded{{display:block}}
.coach-filmstrip img{{width:100%;display:block;border-radius:4px}}

/* Full coach modal */
.coach-modal-overlay{{display:none;position:fixed;inset:0;z-index:800;
  background:rgba(0,0,0,.85);align-items:flex-start;justify-content:center;
  overflow-y:auto;padding:40px 20px}}
.coach-modal-overlay.open{{display:flex}}
.coach-modal{{background:#181b19;border-radius:10px;max-width:720px;width:100%;
  padding:28px 32px;color:#e5e5e5;line-height:1.55;font-size:.95em;
  border:1px solid #2a332d;box-shadow:0 20px 60px rgba(0,0,0,.5)}}
.coach-modal h2{{color:#5ed694;font-size:.8em;text-transform:uppercase;
  letter-spacing:.1em;margin-bottom:8px;font-weight:700}}
.coach-modal .vid-label{{color:#888;font-size:.75em;margin-bottom:14px;
  font-family:monospace}}
.coach-modal .headline{{font-size:1.25em;font-weight:600;color:#ffffff;
  margin-bottom:22px;line-height:1.35}}
.coach-modal .section-title{{color:#8ae6ae;font-size:.72em;text-transform:uppercase;
  letter-spacing:.1em;font-weight:700;margin:22px 0 10px}}
.coach-modal .item{{margin:12px 0;padding:12px 14px;background:#1f2320;
  border-radius:6px;border-left:2px solid #334035}}
.coach-modal .item .pt{{color:#ffffff;font-weight:600;margin-bottom:4px;
  font-size:1.02em}}
.coach-modal .item .dt{{color:#d8d8d8;font-size:.92em}}
.coach-modal .ex{{margin-top:8px;display:flex;flex-wrap:wrap;gap:6px}}
.coach-modal .ex-btn{{background:#2a3a2f;color:#aee8c3;border:1px solid #3a5944;
  font-size:.78em;padding:4px 10px;border-radius:4px;cursor:pointer;
  transition:all .15s;display:inline-flex;align-items:center;gap:6px;font-family:inherit}}
.coach-modal .ex-btn:hover{{background:#3a5944;color:#fff;border-color:#5ed694}}
.coach-modal .ex-btn .ts{{font-family:monospace;font-weight:700;color:#5ed694}}
.coach-modal .ex-btn:hover .ts{{color:#fff}}
.coach-modal .drill{{margin-top:24px;padding:16px 18px;
  background:rgba(94,214,148,.08);border-left:3px solid #5ed694;border-radius:4px}}
.coach-modal .drill-label{{color:#5ed694;font-size:.72em;text-transform:uppercase;
  letter-spacing:.1em;font-weight:700;margin-bottom:6px}}
.coach-modal .drill-body{{color:#f0f0f0;line-height:1.5}}
.coach-modal .close{{position:absolute;top:20px;right:24px;background:none;
  border:none;color:#999;font-size:1.8em;cursor:pointer;line-height:1}}
.coach-modal .close:hover{{color:#fff}}
@media(max-width:600px){{
  .coach-modal-overlay{{padding:20px 12px}}
  .coach-modal{{padding:20px 18px;font-size:.92em}}
  .coach-modal .headline{{font-size:1.1em;margin-bottom:16px}}
  .coach-modal .close{{top:14px;right:16px}}
}}
/* Redesigned card actions — compact horizontal strip + slim footer */
.card-links{{margin-top:10px;padding-top:8px;border-top:1px solid #222;
  display:flex;flex-direction:column;gap:8px}}
.play-strip{{display:flex;flex-wrap:wrap;gap:4px}}
.play-chip{{color:#fff;text-decoration:none;font-size:0.72em;font-weight:600;
  padding:5px 10px;border-radius:999px;opacity:.92;transition:opacity .15s;
  display:inline-flex;align-items:center;gap:6px;white-space:nowrap}}
.play-chip:hover{{opacity:1}}
.play-chip .ch-ct{{font-weight:500;opacity:.85;font-size:.85em;
  padding:1px 6px;background:rgba(0,0,0,.22);border-radius:8px}}
.play-chip-primary{{width:100%;justify-content:center;background:#FF8C00;
  padding:9px 14px;font-size:.86em;letter-spacing:.02em;
  box-shadow:0 1px 0 rgba(0,0,0,0.25) inset}}
.play-chip-primary:hover{{background:#ff9b1f}}
.play-chip-primary .ch-ct{{background:rgba(0,0,0,0.32)}}
.play-chip-slow{{color:#bbb;text-decoration:none;font-size:.7em;font-weight:600;
  padding:5px 9px;border-radius:999px;background:#222;border:1px solid #2f2f2f;
  transition:all .15s}}
.play-chip-slow:hover{{color:#fff;background:#2c2c2c;border-color:#444}}
.card-footer{{display:flex;align-items:center;gap:0;border-top:1px solid #1c1c1c;
  margin-top:4px;padding-top:6px}}
.card-footer .foot-btn{{flex:1;text-align:center;padding:6px 4px;font-size:.78em;
  color:#888;cursor:pointer;opacity:.7;transition:all .15s;background:none;border:0;
  text-decoration:none}}
.card-footer .foot-btn:hover{{opacity:1;color:#fff}}
.card-footer .foot-btn.danger:hover{{color:#E74C3C}}
.card-footer .foot-btn.share:hover{{color:#C7FF00}}
/* Hide-but-keep for legacy CSS callers (so older referenced classes don't NPE) */
.del-btn{{}}.dl-btn{{}}.link-row{{}}

/* ── Upload Modal ── */
.modal-overlay{{display:none;position:fixed;inset:0;z-index:500;background:rgba(0,0,0,.7);
  align-items:center;justify-content:center}}
.modal-overlay.open{{display:flex}}
.modal{{background:#1a1a1a;border-radius:12px;border:1px solid #2a2a2a;padding:24px;
  width:90vw;max-width:450px}}
.modal h3{{color:#FF8C00;margin-bottom:16px}}
.modal input[type="file"],.modal input[type="password"]{{display:block;width:100%;margin-bottom:10px;
  padding:10px;background:#222;border:1px solid #333;border-radius:6px;color:#eee;font-size:.9em}}
.modal .btn-row{{display:flex;gap:8px;margin-top:12px}}
.modal .btn-row button{{flex:1;padding:8px;border:none;border-radius:6px;font-weight:600;cursor:pointer}}
.modal .btn-primary{{background:#FF8C00;color:#fff}}
.modal .btn-cancel{{background:#333;color:#aaa}}
.upload-progress{{display:none;margin-top:12px;background:#222;border-radius:6px;
  overflow:hidden;position:relative;height:20px}}
.upload-bar{{height:100%;background:#FF8C00;width:0;transition:width .3s}}
.upload-pct{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-size:.75em;color:#fff}}
.upload-status{{margin-top:8px;font-size:.85em;color:#999}}

/* ── Video Player Overlay ── */
#playerOverlay{{display:none;position:fixed;inset:0;z-index:1000;
  background:rgba(0,0,0,.95);align-items:center;justify-content:center}}
.player-wrap{{width:95vw;max-width:1200px;display:flex;flex-direction:column}}
.player-head{{display:flex;justify-content:space-between;align-items:center;padding:8px 4px;color:#ccc;font-size:.9em}}
.player-head button{{background:none;border:none;color:#999;font-size:1.8em;cursor:pointer;padding:0 8px;line-height:1}}
.player-head button:hover{{color:#fff}}
#vid{{width:100%;max-height:75vh;background:#000;border-radius:8px}}
.player-bar{{display:flex;flex-wrap:wrap;gap:8px;padding:10px 0;align-items:center}}
.player-bar button{{padding:6px 14px;background:#333;color:#ddd;border:none;border-radius:6px;font-size:.85em;cursor:pointer}}
.player-bar button:hover{{background:#444}}
.player-bar button.active{{background:#FF8C00;color:#fff}}

/* Shot-type filter row — Phase 1 of the playlist refactor.
   One chip per detected shot type; tapping plays only those segments
   back-to-back. Lives inside the player overlay; collapses the per-type
   buttons that used to spread across each gallery card. */
.type-filter-row{{display:none;gap:6px;flex-wrap:wrap;padding:8px 4px 4px;
  border-top:1px solid #222;margin-top:4px}}
.type-filter-row.show{{display:flex}}
.type-filter-chip{{padding:5px 11px;background:#1a1a1a;border:1px solid #2a2a2a;
  border-radius:14px;color:#aaa;font-size:.78em;font-weight:600;cursor:pointer;
  transition:all .15s;display:inline-flex;align-items:center;gap:5px;letter-spacing:.02em}}
.type-filter-chip:hover{{border-color:#555;color:#eee}}
.type-filter-chip.active{{background:#FF8C00;border-color:#FF8C00;color:#fff}}
.type-filter-chip .ct{{font-size:.85em;opacity:.7;font-variant-numeric:tabular-nums}}
.type-filter-chip.active .ct{{opacity:.95}}
.type-filter-chip.slo{{margin-left:auto;background:transparent;border-color:#444}}
.type-filter-chip.slo.active{{background:#9B59B6;border-color:#9B59B6;color:#fff}}

/* Shot strip */
.shot-strip{{display:none;gap:6px;overflow-x:auto;padding:8px 4px;margin-top:4px;
  border-top:1px solid #222;max-width:100%;scrollbar-width:thin}}
.shot-strip.show{{display:flex}}
.shot-chip{{flex-shrink:0;display:flex;flex-direction:column;align-items:center;
  padding:5px 9px;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:5px;
  cursor:pointer;transition:all .15s;min-width:52px}}
.shot-chip:hover{{border-color:#555;background:#222}}
.shot-chip.active{{border-color:#FF8C00;background:#2a1e10}}
.shot-chip .t{{font-size:.7em;color:#888;font-family:monospace}}
.shot-chip .ty{{font-size:.65em;font-weight:700;margin-top:2px;letter-spacing:.03em}}
.shot-chip.serve .ty{{color:#FF8C00}}
.shot-chip.forehand .ty{{color:#27AE60}}
.shot-chip.backhand .ty{{color:#5DADE2}}
.shot-chip.forehand_volley .ty,.shot-chip.backhand_volley .ty{{color:#9B59B6}}
.shot-chip.overhead .ty,.shot-chip.unknown_shot .ty{{color:#aaa}}
.shot-chip .vs{{font-size:.6em;color:#FF8C00;margin-top:3px;padding:0 5px;border:1px solid rgba(255,140,0,0.5);border-radius:6px;cursor:pointer;font-weight:600;letter-spacing:.04em;}}
.shot-chip .vs:hover{{background:rgba(255,140,0,0.2);color:#fff;}}
.compare-modal-overlay{{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.9);z-index:1000;justify-content:center;align-items:center;padding:20px;}}
.compare-modal-overlay.open{{display:flex;}}
.compare-modal{{background:#0f0f0f;border:1px solid #2a2a2a;border-radius:8px;max-width:96vw;max-height:96vh;padding:18px;overflow:auto;position:relative;}}
.compare-modal h2{{color:#ddd;font-size:1.05em;margin:0 0 12px 0;font-weight:600;}}
.compare-modal img{{display:block;max-width:100%;max-height:78vh;border-radius:4px;}}
.compare-modal .close{{position:absolute;top:6px;right:14px;background:none;border:none;color:#aaa;font-size:1.5em;cursor:pointer;}}
.compare-modal .err{{color:#E74C3C;padding:30px;text-align:center;font-size:.9em;}}
.shot-strip-hdr{{font-size:.7em;color:#666;text-transform:uppercase;letter-spacing:.08em;
  padding:4px 0 0;display:none}}
.shot-strip-hdr.show{{display:block}}
.speed-group,.frame-group{{display:flex;gap:4px}}
.share-btn{{margin-left:auto!important;background:#2a7!important}}
.share-btn:hover{{background:#3b8!important}}
.time-display{{color:#999;font-size:.85em;font-family:monospace;min-width:60px}}

/* ── No Results ── */
.no-results{{text-align:center;padding:60px 20px;color:#555;font-size:1.1em}}

/* ── Responsive ── */
@media(max-width:700px){{
  .header-inner{{gap:10px}}
  .logo{{font-size:1.1em}}
  /* min-width:0 lets the search box shrink instead of forcing the header
     wider than the viewport (caused page-level horizontal scroll + cut
     right column on phones). */
  .search-box{{min-width:0;flex-basis:100%}}
  /* Exactly two equal columns on phones — auto-fill with a px min could
     compute 3 cols at some widths and cut the right one. 1fr 1fr always
     fits the viewport regardless of exact width. */
  .grid{{grid-template-columns:1fr 1fr;gap:8px}}
  .card-body{{padding:8px 10px}}
  .filters{{padding:8px 12px}}
  .content{{padding:12px}}
  .player-bar{{justify-content:center}}
  .share-btn{{margin-left:0!important;width:100%}}
  /* keep the sort dropdown from overflowing the filter row */
  .sort-select{{max-width:100%}}
}}
</style>
</head><body>

<!-- Header -->
<div class="header">
  <div class="header-inner">
    <div class="logo">Tennis Highlights</div>
    <div class="search-box">
      <svg viewBox="0 0 20 20"><path d="M8 3a5 5 0 104.4 7.5l4.3 4.3a.7.7 0 001-1l-4.3-4.3A5 5 0 008 3zm0 1.4a3.6 3.6 0 110 7.2 3.6 3.6 0 010-7.2z"/></svg>
      <input type="text" id="searchInput" placeholder="Search videos..." autocomplete="off">
    </div>
    <div class="header-actions">
      <span class="stat-badge" id="statBadge"></span>
    </div>
  </div>
</div>

<!-- Filters -->
<div class="filter-toggle" id="filterToggle" onclick="toggleFilters()">
  <span>Filter &amp; Sort</span>
  <span class="filter-badge" id="filterBadge"></span>
  <span class="filter-arrow open" id="filterArrow">&#9660;</span>
</div>
<div class="filters" id="filters"></div>
<div class="filters" style="padding-top:0" id="filtersRow2">
  <div class="active-filter" id="activeFilter"><span id="activeFilterText"></span><button class="clear" onclick="clearFilter()">&times;</button></div>
  <select class="sort-select" id="sortSelect" onchange="changeSort(this.value)">
    <option value="recorded-desc">Date Recorded (newest)</option>
    <option value="recorded-asc">Date Recorded (oldest)</option>
    <option value="shots-desc">Most Shots</option>
    <option value="shots-asc">Fewest Shots</option>
    <option value="duration-desc">Longest</option>
    <option value="duration-asc">Shortest</option>
  </select>
</div>

<!-- Processing Banner (populated by JS) -->
<div class="proc-banner" id="procBanner" style="display:none"></div>

<!-- Main Content (rendered by JS) -->
<div class="content" id="content"></div>

<!-- Upload Modal -->
<div class="modal-overlay" id="uploadModal" onclick="if(event.target===this)this.classList.remove('open')">
  <div class="modal">
    <h3>Upload Video</h3>
    <input type="file" id="videoFile" accept=".mov,.mp4">
    <input type="password" id="uploadPwd" placeholder="Password">
    <div class="upload-progress" id="uploadProgress">
      <div class="upload-bar" id="uploadBar"></div>
      <div class="upload-pct" id="uploadPct"></div>
    </div>
    <div class="upload-status" id="uploadStatus"></div>
    <div class="btn-row">
      <button class="btn-cancel" onclick="document.getElementById('uploadModal').classList.remove('open')">Cancel</button>
      <button class="btn-primary" id="uploadBtn" onclick="startUpload()">Upload</button>
    </div>
  </div>
</div>

<!-- Video Player Overlay -->
<!-- Sequences Modal -->
<div class="seq-modal-overlay" id="seqModal" onclick="if(event.target===this)closeSeqModal()">
  <div class="seq-modal">
    <h2><span id="seqTitle">Swing Sequences</span><button class="close" onclick="closeSeqModal()">&times;</button></h2>
    <div class="seq-grid" id="seqGrid"></div>
  </div>
</div>

<!-- Compare Modal -->
<div class="compare-modal-overlay" id="compareModal" onclick="if(event.target===this)closeCompareModal()">
  <div class="compare-modal">
    <button class="close" onclick="closeCompareModal()">&times;</button>
    <h2 id="compareTitle">You vs Pro</h2>
    <video id="compareVid" controls playsinline style="display:none;max-width:100%;max-height:78vh;border-radius:4px;background:#000"></video>
    <div class="err" id="compareErr" style="display:none">No pro comparison available for this shot yet.</div>
  </div>
</div>

<!-- Coach Modal -->
<div class="coach-modal-overlay" id="coachModal" onclick="if(event.target===this)closeCoachModal()">
  <div class="coach-modal" style="position:relative">
    <button class="close" onclick="closeCoachModal()">&times;</button>
    <h2>Coach Analysis</h2>
    <div class="vid-label" id="coachVid"></div>
    <div class="headline" id="coachHeadline"></div>
    <div id="coachBody"></div>
  </div>
</div>

<div id="playerOverlay" onclick="if(event.target===this)closePlayer()">
  <div class="player-wrap">
    <div class="player-head">
      <span id="playerTitle"></span>
      <button onclick="closePlayer()">&times;</button>
    </div>
    <video id="vid" controls playsinline></video>
    <div class="player-bar">
      <div class="speed-group">
        <button onclick="setSpeed(0.25,this)">0.25x</button>
        <button onclick="setSpeed(0.5,this)">0.5x</button>
        <button onclick="setSpeed(1,this)" class="active">1x</button>
        <button onclick="setSpeed(2,this)">2x</button>
      </div>
      <div class="frame-group">
        <button onclick="stepFrame(-5)" title="-5s">&laquo;</button>
        <button onclick="stepFrame(-1)" title="-1 frame">&larr;</button>
        <button onclick="stepFrame(1)" title="+1 frame">&rarr;</button>
        <button onclick="stepFrame(5)" title="+5s">&raquo;</button>
      </div>
      <span class="time-display" id="timeDisplay">0:00.0</span>
      <button onclick="copyTimeLink()" class="share-btn" id="shareBtn">Copy link at time</button>
      <button onclick="downloadCurrent()" class="share-btn" style="background:#555!important">Download</button>
    </div>
    <div class="type-filter-row" id="typeFilter"></div>
    <div class="shot-strip-hdr" id="shotStripHdr">Jump to shot</div>
    <div class="shot-strip" id="shotStrip"></div>
  </div>
</div>

<script>
// ── Data ──
var VIDEOS = {video_json};

// ── Video Player ──
var vid = document.getElementById('vid');
var overlay = document.getElementById('playerOverlay');
var pendingTime = null;

function fmtTime(s) {{
  var m = Math.floor(s/60), sec = s - m*60;
  return m+':'+(sec<10?'0':'')+sec.toFixed(1);
}}
vid.addEventListener('timeupdate', function() {{
  document.getElementById('timeDisplay').textContent = fmtTime(vid.currentTime);
}});

var pausedOnOpen = false;  // set by jumpToExample so coach examples open paused
var currentShots = null;   // {{video, shots[]}} loaded from shots.json
var currentVariant = null; // e.g. "timeline" / "rally_slowmo" — derived from url

function openPlayer(url, title) {{
  // PR-J: if we're inside the iOS WebView wrapper, hand the URL off
  // to native AVPlayerViewController. We forward `pendingTime` (set
  // by jumpToExample / deep links) so the native player seeks to the
  // same starting point the HTML5 player would.
  try {{
    if (window.webkit && window.webkit.messageHandlers
        && window.webkit.messageHandlers.openVideo) {{
      var msg = {{url: url, title: title || ''}};
      if (pendingTime !== null && pendingTime > 0) {{
        msg.startTime = pendingTime;
      }}
      // Phase 3: extract <vid> + <variant> from the URL so the native
      // side can fetch shots.json and surface the same chip filter
      // that the web player has. Swift owns the fetch (no JS-to-Swift
      // data payload needed) — keeps the bridge surface area small.
      var nm = url.match(/\\/([A-Za-z0-9_]+)\\/\\1_(\\w+)\\.mp4$/);
      if (nm) {{ msg.videoId = nm[1]; msg.variant = nm[2]; }}
      pendingTime = null;
      window.webkit.messageHandlers.openVideo.postMessage(msg);
      return;
    }}
  }} catch (e) {{}}

  // Reset to a clean paused state before loading new source so native
  // controls don't flash the wrong play/pause icon.
  try {{ vid.pause(); }} catch (e) {{}}
  vid.removeAttribute('autoplay');
  vid.autoplay = false;
  vid.src = url;
  document.getElementById('playerTitle').textContent = title;
  overlay.style.display = 'flex';
  document.body.style.overflow = 'hidden';
  var wantPaused = pausedOnOpen; pausedOnOpen = false;
  var wantT = pendingTime; pendingTime = null;
  vid.addEventListener('loadedmetadata', function() {{
    if (wantT !== null) vid.currentTime = wantT;
    if (wantPaused) {{
      // Double pause defeats browser race where autoplay heuristic fires on seek
      try {{ vid.pause(); }} catch (e) {{}}
      setTimeout(function(){{ try {{ vid.pause(); }} catch(e){{}} }}, 50);
    }} else {{
      vid.play().catch(function(){{}});
    }}
  }}, {{once:true}});
  history.replaceState(null,'','?v='+encodeURIComponent(url.replace('https://tennis.playfullife.com/','')));
  // Parse "<vid>/<vid>_<variant>.mp4" from the URL and load shots.json
  var m = url.match(/\\/([A-Za-z0-9_]+)\\/\\1_(\\w+)\\.mp4$/);
  if (m) loadShotsStrip(m[1], m[2]);
  else {{ currentShots = null; currentVariant = null; document.getElementById('shotStrip').classList.remove('show'); document.getElementById('shotStripHdr').classList.remove('show'); document.getElementById('typeFilter').classList.remove('show'); }}
}}

function loadShotsStrip(vidId, variant) {{
  currentVariant = variant;
  var doRender = function() {{ renderShotStrip(); }};
  // Load the per-shot pro-comparison manifest (best-effort): the set of
  // global shot indices that have a {vid}_comparison_shot_NNN.mp4 clip.
  // Drives whether the "vs pro" pill shows on each shot chip.
  loadComparisonIndex(vidId);
  if (currentShots && currentShots.video === vidId) {{ doRender(); return; }}
  fetch('/' + vidId + '/shots.json', {{cache: 'no-store'}})
    .then(function(r) {{ if(!r.ok) throw new Error('404'); return r.json(); }})
    .then(function(d) {{ currentShots = d; doRender(); }})
    .catch(function() {{
      currentShots = null;
      document.getElementById('shotStrip').classList.remove('show');
      document.getElementById('shotStripHdr').classList.remove('show');
      document.getElementById('typeFilter').classList.remove('show');
    }});
}}

var comparisonShots = null;   // {{video, Set of global shot idxs with a clip}}
function loadComparisonIndex(vidId) {{
  if (comparisonShots && comparisonShots.video === vidId) {{ return; }}
  comparisonShots = null;
  fetch('/' + vidId + '/' + vidId + '_comparisons_index.json', {{cache:'no-store'}})
    .then(function(r) {{ if(!r.ok) throw new Error('404'); return r.json(); }})
    .then(function(d) {{
      comparisonShots = {{video: vidId, set: {{}}}};
      (d.shots || []).forEach(function(ix) {{ comparisonShots.set[ix] = true; }});
      renderShotStrip();  // re-render so pills appear once the manifest lands
    }})
    .catch(function() {{ comparisonShots = {{video: vidId, set: {{}}}}; }});
}}
function pad3(n) {{ n = String(n); while (n.length < 3) n = '0' + n; return n; }}

function renderShotStrip() {{
  var strip = document.getElementById('shotStrip');
  var hdr = document.getElementById('shotStripHdr');
  if (!currentShots || !currentVariant) return;
  // Only shots with a position in the current variant are clickable
  var shots = currentShots.shots.filter(function(s) {{
    return s.positions && s.positions[currentVariant] !== undefined;
  }});
  if (shots.length === 0) {{
    strip.classList.remove('show'); hdr.classList.remove('show');
    return;
  }}
  var html = '';
  shots.forEach(function(s, i) {{
    var pos = s.positions[currentVariant];
    var gidx = (s.idx !== undefined) ? s.idx : i;  // global shot index
    var label = ({{'serve':'S','forehand':'FH','backhand':'BH','forehand_volley':'FV','backhand_volley':'BV','overhead':'OH','unknown_shot':'?'}})[s.type] || '?';
    // Show the "vs pro" pill only for shots that actually have a comparison
    // clip (per the manifest); tapping plays {vid}_comparison_shot_NNN.mp4.
    var hasCompare = comparisonShots && comparisonShots.set && comparisonShots.set[gidx];
    html += '<div class="shot-chip ' + s.type + '" data-t="' + pos + '" data-idx="' + i + '" data-gidx="' + gidx + '">'
      + '<span class="t">' + fmtShotTime(pos) + '</span>'
      + '<span class="ty">' + label + '</span>'
      + (hasCompare ? '<span class="vs" title="Compare to pro">vs pro</span>' : '')
      + '</div>';
  }});
  strip.innerHTML = html;
  strip.classList.add('show');
  hdr.classList.add('show');
  updateActiveShotChip();
  renderTypeFilter();
}}

function fmtShotTime(s) {{
  s = Math.max(0, Math.round(s));
  var m = Math.floor(s / 60);
  var sec = s % 60;
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}}

function updateActiveShotChip() {{
  if (!currentShots || !currentVariant) return;
  var strip = document.getElementById('shotStrip');
  var chips = strip.querySelectorAll('.shot-chip');
  var now = vid.currentTime;
  var activeIdx = -1;
  var bestDelta = 999;
  chips.forEach(function(chip, i) {{
    var t = parseFloat(chip.dataset.t);
    if (now >= t - 0.5 && now <= t + 3.5) {{
      var delta = Math.abs(now - t);
      if (delta < bestDelta) {{ bestDelta = delta; activeIdx = i; }}
    }}
  }});
  chips.forEach(function(c, i) {{
    if (i === activeIdx) c.classList.add('active'); else c.classList.remove('active');
  }});
}}

vid.addEventListener('timeupdate', updateActiveShotChip);

// ── Phase 1 playlist filter ──
// Replaces the per-type chip-explosion on each gallery card. The card
// opens the full timeline; the player surfaces a chip row that filters
// to a single shot type, then auto-seeks the playhead from segment to
// segment so the user hears/sees only those shots back-to-back.
//
// SEGMENT_PRE/POST = 1.5s/2.5s window per detected swing. This matches
// roughly what export_videos.py's bytype path produces (it uses
// before=2.0, after=2.0), tightened slightly so the playlist feels
// snappy rather than padded.
//
// Named playerFilter (not currentFilter) because the gallery's older
// session-level filter chip code also uses `currentFilter` at top-scope
// and `var` would merge them.
var playerFilter = 'all';
var SEGMENT_PRE = 1.5;
var SEGMENT_POST = 2.5;
var FILTER_TYPE_MAP = {{
  'serve':    ['serve'],
  'forehand': ['forehand'],
  'backhand': ['backhand'],
  'volley':   ['forehand_volley','backhand_volley'],
  'overhead': ['overhead'],
}};

function buildRallySegments() {{
  // Rally = group shots into "points" (consecutive within 8s of each
  // other) and play each point as a single longer segment. Matches the
  // logic that used to produce rally.mp4 server-side (point_gap=8.0,
  // before=3.5, after=4.5).
  if (!currentShots || !currentVariant) return [];
  var ts = [];
  currentShots.shots.forEach(function(s) {{
    if (s.positions && s.positions[currentVariant] !== undefined) {{
      ts.push(s.positions[currentVariant]);
    }}
  }});
  ts.sort(function(a,b){{ return a-b; }});
  if (ts.length === 0) return [];
  var POINT_GAP = 8.0, BEFORE = 3.5, AFTER = 4.5;
  var points = [[ts[0]]];
  for (var i = 1; i < ts.length; i++) {{
    var prev = points[points.length-1];
    if (ts[i] - prev[prev.length-1] > POINT_GAP) points.push([ts[i]]);
    else prev.push(ts[i]);
  }}
  return points.map(function(p) {{
    return {{start: Math.max(0, p[0] - BEFORE), end: p[p.length-1] + AFTER}};
  }});
}}

function buildSegmentList(filter) {{
  if (filter === 'rally') return buildRallySegments();
  if (!currentShots || !currentVariant) return [];
  var types = FILTER_TYPE_MAP[filter];  // undefined → 'all' / no filter
  var segs = [];
  currentShots.shots.forEach(function(s) {{
    if (!s.positions || s.positions[currentVariant] === undefined) return;
    if (types && types.indexOf(s.type) < 0) return;
    var t = s.positions[currentVariant];
    segs.push({{start: Math.max(0, t - SEGMENT_PRE), end: t + SEGMENT_POST}});
  }});
  segs.sort(function(a,b){{ return a.start - b.start; }});
  // Merge any adjacent segments so consecutive same-type shots play
  // as one continuous run rather than micro-seeking between them.
  var merged = [];
  segs.forEach(function(s) {{
    var last = merged[merged.length - 1];
    if (last && s.start <= last.end + 0.3) {{
      last.end = Math.max(last.end, s.end);
    }} else {{
      merged.push({{start: s.start, end: s.end}});
    }}
  }});
  return merged;
}}

function renderTypeFilter() {{
  var row = document.getElementById('typeFilter');
  if (!currentShots || !currentVariant) {{ row.classList.remove('show'); return; }}
  // Filter chips only matter on full-session variants. Per-type files
  // (forehands.mp4 etc.) already filter their own content.
  var variantOK = currentVariant === 'timeline' || currentVariant === 'rally'
    || currentVariant === 'grouped' || currentVariant === 'highlights';
  if (!variantOK) {{ row.classList.remove('show'); return; }}
  var counts = {{}};
  var total = 0;
  currentShots.shots.forEach(function(s) {{
    if (!s.positions || s.positions[currentVariant] === undefined) return;
    total++;
    Object.keys(FILTER_TYPE_MAP).forEach(function(k) {{
      if (FILTER_TYPE_MAP[k].indexOf(s.type) >= 0) {{
        counts[k] = (counts[k] || 0) + 1;
      }}
    }});
  }});
  var html = '<span class="type-filter-chip ' + (playerFilter==='all'?'active':'')
    + '" data-f="all">All <span class="ct">'+total+'</span></span>';
  // Rally chip — group-by-point segment view. Count = number of points.
  var rallySegs = buildRallySegments();
  if (rallySegs.length > 0) {{
    html += '<span class="type-filter-chip ' + (playerFilter==='rally'?'active':'')
      + '" data-f="rally">Rally <span class="ct">'+rallySegs.length+'</span></span>';
  }}
  [['serve','Serve'],['forehand','FH'],['backhand','BH'],['volley','Volley'],['overhead','OH']].forEach(function(p) {{
    var key = p[0], label = p[1];
    var c = counts[key] || 0;
    if (c === 0) return;
    html += '<span class="type-filter-chip ' + (playerFilter===key?'active':'')
      + '" data-f="'+key+'">'+label+' <span class="ct">'+c+'</span></span>';
  }});
  var sloActive = vid.playbackRate <= 0.6;
  html += '<span class="type-filter-chip slo ' + (sloActive?'active':'')
    + '" data-f="slo">&#x1F422; Slo</span>';
  row.innerHTML = html;
  row.classList.add('show');
}}

function applyTypeFilter(f) {{
  if (f === 'slo') {{
    var newRate = vid.playbackRate <= 0.6 ? 1 : 0.5;
    vid.playbackRate = newRate;
    document.querySelectorAll('.speed-group button').forEach(function(b) {{
      b.classList.toggle('active', parseFloat(b.textContent) === newRate);
    }});
    renderTypeFilter();
    return;
  }}
  playerFilter = f;
  renderTypeFilter();
  var segs = buildSegmentList(f);
  if (segs.length > 0) {{
    // Wait for the seek to commit before play(), otherwise the browser
    // can drop the play() call mid-seek and the player ends up paused.
    // Reproduced on iphone_9ca0a615 Rally: needed 3 taps before
    // anything started.
    var onSeeked = function() {{
      vid.removeEventListener('seeked', onSeeked);
      vid.play().catch(function(){{}});
    }};
    vid.addEventListener('seeked', onSeeked);
    vid.currentTime = segs[0].start;
  }}
}}

document.getElementById('typeFilter').addEventListener('click', function(e) {{
  var chip = e.target.closest('.type-filter-chip');
  if (!chip) return;
  applyTypeFilter(chip.dataset.f);
}});

// Auto-seek playhead from segment to segment when a non-'all' filter
// is active. Cheap: runs on every timeupdate (~4Hz) and only mutates
// currentTime when we're actually in a gap.
function segmentAutoSeek() {{
  if (playerFilter === 'all') return;
  var segs = buildSegmentList(playerFilter);
  if (segs.length === 0) return;
  var now = vid.currentTime;
  for (var i = 0; i < segs.length; i++) {{
    if (now >= segs[i].start && now <= segs[i].end) return;  // inside
  }}
  for (var j = 0; j < segs.length; j++) {{
    if (segs[j].start > now) {{ vid.currentTime = segs[j].start; return; }}
  }}
  // Past the last segment — loop back so filter playback feels continuous.
  vid.currentTime = segs[0].start;
}}
vid.addEventListener('timeupdate', segmentAutoSeek);

document.getElementById('shotStrip').addEventListener('click', function(e) {{
  // Compare button — small "vs pro" pill inside the chip. Opens the
  // per-shot side-by-side comparison clip in an overlay ON TOP of the
  // timeline player, so closing returns you to the timeline where you
  // were (not out to the gallery).
  if (e.target.classList.contains('vs')) {{
    var chip = e.target.closest('.shot-chip');
    if (chip && currentShots) {{
      openCompareModal(currentShots.video, parseInt(chip.dataset.gidx));
    }}
    e.stopPropagation();
    return;
  }}
  var chip = e.target.closest('.shot-chip');
  if (!chip) return;
  var t = parseFloat(chip.dataset.t);
  if (!isNaN(t)) {{
    vid.currentTime = Math.max(0, t - 0.3);  // slight lead-in
    vid.play().catch(function(){{}});
  }}
}});

function openCompareModal(videoId, shotIdx) {{
  var cvid = document.getElementById('compareVid');
  var err = document.getElementById('compareErr');
  document.getElementById('compareTitle').textContent =
    'You vs Pro \\u2014 shot ' + (shotIdx + 1);
  // Pause the timeline underneath so two videos don't play at once; it
  // keeps its position so closing resumes exactly where you were.
  try {{ vid.pause(); }} catch(e) {{}}
  err.style.display = 'none';
  cvid.style.display = 'block';
  cvid.src = 'https://tennis.playfullife.com/' + videoId
    + '/' + videoId + '_comparison_shot_' + pad3(shotIdx) + '.mp4';
  cvid.onerror = function(){{ cvid.style.display='none'; err.style.display='block'; }};
  document.getElementById('compareModal').classList.add('open');
  cvid.play().catch(function(){{}});
}}
function closeCompareModal() {{
  var cvid = document.getElementById('compareVid');
  try {{ cvid.pause(); cvid.removeAttribute('src'); cvid.load(); }} catch(e) {{}}
  document.getElementById('compareModal').classList.remove('open');
  // Return to the timeline player (still open underneath) — resume play.
  vid.play().catch(function(){{}});
}}

function closePlayer() {{
  vid.pause(); vid.removeAttribute('src'); vid.load();
  overlay.style.display = 'none'; document.body.style.overflow = '';
  history.replaceState(null,'',location.pathname);
  playerFilter = 'all';  // reset so the next-opened player starts unfiltered
  document.getElementById('typeFilter').classList.remove('show');
}}

function setSpeed(s,btn) {{
  vid.playbackRate = s;
  document.querySelectorAll('.speed-group button').forEach(function(b){{b.classList.remove('active')}});
  if(btn)btn.classList.add('active');
}}
function stepFrame(dir) {{
  vid.pause();
  if(Math.abs(dir) >= 2) {{ vid.currentTime = Math.max(0, vid.currentTime + dir); }}
  else {{ vid.currentTime = Math.max(0, vid.currentTime + dir/60); }}
}}

function dlFile(url) {{
  var f = document.getElementById('dlframe');
  if(!f) {{ f = document.createElement('iframe'); f.id='dlframe'; f.style.display='none'; document.body.appendChild(f); }}
  f.src = url + (url.includes('?')?'&':'?') + 'dl=1';
}}

// PR-D — rename a video. Owner JWT (or admin) only; persists to meta.json
// `display_name`. Empty string clears the rename. Updates the local
// VIDEOS array so the gallery re-renders without a full reload.
function renameVideo(vid) {{
  var current = '';
  for (var i=0; i<VIDEOS.length; i++) {{
    if(VIDEOS[i].id === vid) {{ current = VIDEOS[i].display_name || ''; break; }}
  }}
  var name = prompt('New name (blank to reset to '+vid+'):', current);
  if (name === null) return;
  name = name.trim().slice(0, 80);
  fetch('/api/video/'+vid+'/rename', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    credentials:'include',
    body: JSON.stringify({{display_name: name}}),
  }}).then(function(r){{ return r.json().then(function(j){{ return {{status:r.status, json:j}}; }}); }})
  .then(function(res){{
    if(res.status !== 200) throw new Error(res.json.error || ('failed: '+res.status));
    for (var i=0; i<VIDEOS.length; i++) {{
      if(VIDEOS[i].id === vid) {{
        if (res.json.display_name) VIDEOS[i].display_name = res.json.display_name;
        else delete VIDEOS[i].display_name;
        break;
      }}
    }}
    renderGallery();
  }})
  .catch(function(e){{ alert('Rename failed: '+e.message); }});
}}

// PR-F — generate a public share link for one video. Worker stores
// `shares/<token>.json`, returns the URL. We try the iOS native share
// sheet first; otherwise we put the URL on the clipboard.
function createShareLink(vid) {{
  fetch('/api/video/'+vid+'/share', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    credentials:'include',
    body:'{{}}',
  }}).then(function(r){{
    return r.json().then(function(j){{ return {{status:r.status, json:j}}; }});
  }}).then(function(res){{
    if(res.status !== 200) throw new Error(res.json.error || ('failed: '+res.status));
    var url = res.json.url;
    // Native iOS share sheet via Web Share API where available.
    if (navigator.share) {{
      navigator.share({{title: vid, url: url}}).catch(function(){{}});
      return;
    }}
    if (navigator.clipboard) {{
      navigator.clipboard.writeText(url).then(function(){{
        alert('Share link copied:\\n' + url);
      }}, function(){{ prompt('Share link:', url); }});
      return;
    }}
    prompt('Share link:', url);
  }}).catch(function(e){{ alert('Error: '+e.message); }});
}}

function deleteVideo(vid) {{
  if(!confirm('Permanently delete '+vid+' and all its files?')) return;
  // Per-user gallery is cookie-authenticated. Worker validates the JWT
  // and accepts the delete from the owner or an admin. No fallback —
  // the legacy shared password is gone (every signed-in user has a JWT).
  fetch('/api/video/'+vid+'/delete', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    credentials:'include',
    body:'{{}}',
  }}).then(function(r){{
    return r.json().then(function(j){{ return {{status:r.status, json:j}}; }});
  }}).then(function(res){{
    if(res.status !== 200) throw new Error(res.json.error || ('delete failed: '+res.status));
    return res.json;
  }}).then(function(d){{
    alert('Deleted '+vid+' ('+d.deleted+' files removed)');
    VIDEOS = VIDEOS.filter(function(v){{ return v.id !== vid; }});
    buildFilters();
    renderGallery();
  }}).catch(function(e){{ alert('Error: '+e.message); }});
}}

function downloadCurrent() {{
  if(vid.src) dlFile(vid.src);
}}

function copyTimeLink() {{
  var vKey = vid.src.replace('https://tennis.playfullife.com/','');
  var t = Math.round(vid.currentTime*10)/10;
  var link = location.origin+location.pathname+'?v='+encodeURIComponent(vKey)+(t>0?'&t='+t:'');
  navigator.clipboard.writeText(link).then(function(){{
    var btn=document.getElementById('shareBtn');btn.textContent='Copied!';
    setTimeout(function(){{btn.textContent='Copy link at time'}},2000);
  }});
}}

document.addEventListener('keydown', function(e) {{
  if(e.key==='Escape' && document.getElementById('seqModal').classList.contains('open')) {{
    closeSeqModal(); return;
  }}
  if(e.key==='Escape' && document.getElementById('coachModal').classList.contains('open')) {{
    closeCoachModal(); return;
  }}
  if(overlay.style.display!=='flex')return;
  if(e.key==='Escape')closePlayer();
  if(e.key==='ArrowLeft'){{e.preventDefault();vid.currentTime=Math.max(0,vid.currentTime-5)}}
  if(e.key==='ArrowRight'){{e.preventDefault();vid.currentTime+=5}}
  if(e.key===' '){{e.preventDefault();vid.paused?vid.play():vid.pause()}}
}});

// Deep link
(function(){{
  var p=new URLSearchParams(location.search), v=p.get('v');
  if(v){{var t=parseFloat(p.get('t'));if(t>0)pendingTime=t;
    openPlayer('https://tennis.playfullife.com/'+v,v.split('/').pop().replace('.mp4','').replace(/_/g,' '));
  }}
}})();

function toggleCard(el) {{ el.classList.toggle('expanded'); }}

// Cache of coaching JSON keyed by video id
var coachCache = {{}};

function loadCoachingSummary(vid) {{
  if(coachCache[vid] !== undefined) {{ applySummary(vid, coachCache[vid]); return; }}
  fetch('/'+vid+'/coaching.json', {{cache:'no-store'}})
    .then(function(r){{ if(!r.ok) throw new Error('404'); return r.json(); }})
    .then(function(d){{ coachCache[vid] = d; applySummary(vid, d); }})
    .catch(function(){{ coachCache[vid] = null; }});
}}

function applySummary(vid, d) {{
  var box = document.getElementById('coachSum-'+vid);
  if(!box || !d || !d.headline) return;
  box.querySelector('.coach-summary-text').textContent = d.headline;
  box.classList.add('loaded');
}}

function openCoachModal(vid) {{
  var d = coachCache[vid];
  if(!d) return;
  // UX-3: if we're inside the iOS WebView wrapper, hand off to the
  // native SwiftUI sheet. JS posts the cached coaching JSON over the
  // openCoach message bridge instead of opening the inline HTML modal.
  try {{
    if (window.webkit && window.webkit.messageHandlers
        && window.webkit.messageHandlers.openCoach) {{
      window.webkit.messageHandlers.openCoach.postMessage(
        {{vid: vid, coaching: d}}
      );
      return;
    }}
  }} catch (e) {{}}
  document.getElementById('coachVid').textContent = vid;
  document.getElementById('coachHeadline').textContent = d.headline || '';
  var body = document.getElementById('coachBody');
  var html = '';
  function renderItems(items, title) {{
    if(!items || !items.length) return '';
    var h = '<div class="section-title">'+title+'</div>';
    items.forEach(function(s){{
      h += '<div class="item">';
      h += '<div class="pt">'+escapeHtml(s.point||'')+'</div>';
      h += '<div class="dt">'+escapeHtml(s.detail||'')+'</div>';
      if(s.examples && s.examples.length) {{
        h += '<div class="ex">';
        s.examples.forEach(function(ex, ei){{
          var ts = fmtTs(ex.t);
          var note = ex.note ? ' — '+escapeHtml(ex.note) : '';
          h += '<button class="ex-btn" onclick="jumpToExample(\\''+vid+'\\','+ex.t+')">'
            + '<span class="ts">'+ts+'</span> '+escapeHtml(ex.type||'')+note+'</button>';
        }});
        h += '</div>';
        // Filmstrip for first example (most illustrative)
        var firstEx = s.examples[0];
        if(firstEx && firstEx.type) {{
          var filmId = 'film_'+vid+'_'+Math.round(firstEx.t);
          h += '<div class="coach-filmstrip" id="'+filmId+'">'
            + '<img loading="lazy" onerror="this.parentNode.style.display=\\'none\\'" '
            + 'onload="this.parentNode.classList.add(\\'loaded\\')" '
            + 'src="https://tennis.playfullife.com/'+vid+'/sequences/shot_'+matchShotIdx(vid, firstEx.t, firstEx.type)+'_'+firstEx.type+'.jpg">'
            + '</div>';
        }}
      }}
      h += '</div>';
    }});
    return h;
  }}
  html += renderItems(d.strengths, 'Strengths');
  html += renderItems(d.work_on, 'Work On');
  if(d.drill) {{
    html += '<div class="drill"><div class="drill-label">Suggested Drill</div>'
      + '<div class="drill-body">'+escapeHtml(d.drill)+'</div></div>';
  }}
  body.innerHTML = html;
  document.getElementById('coachModal').classList.add('open');
  document.body.style.overflow = 'hidden';
}}

function closeCoachModal() {{
  document.getElementById('coachModal').classList.remove('open');
  document.body.style.overflow = '';
}}

// ── Sequences Modal ──
var seqCacheBust = Date.now();
function openSeqModal(vid) {{
  var modal = document.getElementById('seqModal');
  var grid = document.getElementById('seqGrid');
  document.getElementById('seqTitle').textContent = 'Swing Sequences — ' + vid;
  grid.innerHTML = '<div style="color:#888;padding:20px">Loading sequences...</div>';
  modal.classList.add('open');
  document.body.style.overflow = 'hidden';

  var showSkel = false;
  var showNoRacket = false;
  var vdata = VIDEOS.find(function(x){{ return x.id === vid; }});
  var hasRacketRemoved = vdata && vdata.features && vdata.features.indexOf('racket_removed') >= 0;
  var hasComparisons = vdata && vdata.features && vdata.features.indexOf('comparisons') >= 0;

  fetch('/' + vid + '/shots.json', {{cache:'no-store'}})
    .then(function(r){{ if(!r.ok) throw new Error('no shots.json'); return r.json(); }})
    .then(function(data) {{
      var shots = data.shots || [];
      if(shots.length === 0) {{ grid.innerHTML = '<div style="color:#888;padding:20px">No shots detected</div>'; return; }}

      function renderSeqs() {{
        var suffix = showSkel ? '_skel' : '';
        var noracket = showNoRacket ? '_noracket' : '';
        var html = '<div style="display:flex;gap:8px;margin-bottom:12px;align-items:center;flex-wrap:wrap">'
          + '<button onclick="toggleSeqSkel()" style="padding:6px 14px;background:'+(showSkel?'#5555aa':'#333')+';color:#eee;border:none;border-radius:6px;cursor:pointer;font-size:.8em">'
          + (showSkel ? 'Skeleton: ON' : 'Skeleton: OFF') + '</button>';
        if(hasRacketRemoved) {{
          html += '<button onclick="toggleSeqRacket()" style="padding:6px 14px;background:'+(showNoRacket?'#7B1FA2':'#333')+';color:#eee;border:none;border-radius:6px;cursor:pointer;font-size:.8em">'
            + (showNoRacket ? 'Racket: REMOVED' : 'Racket: NORMAL') + '</button>';
        }}
        html += '</div>';

        shots.forEach(function(s) {{
          if(s.type === 'practice' || s.type === 'offscreen' || s.type === 'not_shot') return;
          var idx = ('00' + s.idx).slice(-3);
          var base = 'shot_' + idx + '_' + s.type;
          var cb = '?v=' + seqCacheBust;
          var imgUrl = 'https://tennis.playfullife.com/' + vid + '/sequences/' + base + noracket + suffix + '.jpg' + cb;
          var fallback1 = 'https://tennis.playfullife.com/' + vid + '/sequences/' + base + suffix + '.jpg' + cb;
          var fallback2 = 'https://tennis.playfullife.com/' + vid + '/sequences/' + base + '.jpg' + cb;
          html += '<div class="seq-item">'
            + '<div class="seq-img-wrap" onclick="openSeqFullscreen(this.querySelector(\\'img\\').src)">'
            + '<img src="' + imgUrl + '" loading="lazy" onerror="if(this.src.indexOf(\\'noracket\\')>=0)this.src=\\''+fallback1+'\\';else if(this.src.indexOf(\\'_skel\\')>=0)this.src=\\''+fallback2+'\\';else this.parentNode.parentNode.style.display=\\'none\\'">'
            + '</div>'
            + '<div class="seq-label"><span>' + s.type.toUpperCase() + ' #' + (s.idx+1) + '</span>'
            + '<span>t=' + fmtTs(s.t) + '</span></div></div>';
        }});

        // Pro Comparisons section
        if(hasComparisons) {{
          html += '<div style="margin-top:24px;border-top:1px solid #333;padding-top:16px">'
            + '<h3 style="color:#E74C3C;font-size:.8em;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px">Pro Comparisons</h3>';
          shots.forEach(function(s) {{
            if(s.type === 'practice' || s.type === 'offscreen' || s.type === 'not_shot') return;
            var idx = ('00' + s.idx).slice(-3);
            var cmpUrl = 'https://tennis.playfullife.com/' + vid + '/comparisons/compare_' + idx + '_' + s.type + '.jpg';
            html += '<div class="seq-item">'
              + '<img src="' + cmpUrl + '" loading="lazy" onerror="this.parentNode.style.display=\\'none\\'">'
              + '<div class="seq-label"><span style="color:#E74C3C">' + s.type.toUpperCase() + ' #' + (s.idx+1) + ' vs PRO</span>'
              + '<span>t=' + fmtTs(s.t) + '</span></div></div>';
          }});
          html += '</div>';
        }}

        grid.innerHTML = html || '<div style="color:#888;padding:20px">No sequence images found.</div>';
      }}
      window.toggleSeqSkel = function() {{ showSkel = !showSkel; renderSeqs(); }};
      window.toggleSeqRacket = function() {{ showNoRacket = !showNoRacket; renderSeqs(); }};
      window._seqRender = renderSeqs;
      renderSeqs();
    }})
    .catch(function() {{
      grid.innerHTML = '<div style="color:#888;padding:20px">No sequence data available for this video.</div>';
    }});
}}

function closeSeqModal() {{
  document.getElementById('seqModal').classList.remove('open');
  document.body.style.overflow = '';
}}

function openSeqFullscreen(src) {{
  var overlay = document.createElement('div');
  overlay.className = 'seq-fullscreen';
  overlay.onclick = function() {{ overlay.remove(); }};
  var img = document.createElement('img');
  img.src = src;
  overlay.appendChild(img);
  document.body.appendChild(overlay);
}}

function matchShotIdx(vid, t, type) {{
  // Find the shot index closest to timestamp t for this type.
  // Uses cached shots data if available, otherwise guesses from VIDEOS data.
  if(currentShots && currentShots.video === vid) {{
    var best = null, bestDelta = 999;
    currentShots.shots.forEach(function(s){{
      if(s.type === type) {{
        var d = Math.abs(s.t - t);
        if(d < bestDelta) {{ bestDelta = d; best = s; }}
      }}
    }});
    if(best) return ('00' + best.idx).slice(-3);
  }}
  // Fallback: estimate from the VIDEOS array shot count
  return '000';
}}

function fmtTs(t) {{
  t = Math.max(0, Math.round(t));
  var m = Math.floor(t/60), s = t%60;
  return m+':'+(s<10?'0':'')+s;
}}

function jumpToExample(vid, t) {{
  // Prefer a slow-mo variant so the user can actually study the form.
  // Priority of variants to check: rally_slowmo > forehands_slowmo > backhands_slowmo
  // > serves_slowmo > volleys_slowmo > timeline (fallback).
  var v = VIDEOS.find(function(x){{ return x.id === vid; }});
  if(!v) return;

  var SLOWMO_PREF = ['rally_slowmo','forehands_slowmo','backhands_slowmo','serves_slowmo','volleys_slowmo','grouped_slowmo'];
  var haveLink = function(key){{ return v.links.find(function(l){{ return l.key === key; }}); }};

  closeCoachModal();

  // Fetch shots.json to map the example's original timestamp into the slow-mo variant.
  fetch('/' + vid + '/shots.json', {{cache:'no-store'}})
    .then(function(r){{ if(!r.ok) throw new Error('404'); return r.json(); }})
    .then(function(data){{
      // Find the shot whose original t is closest to the example's t
      var best = null, bestDelta = 999;
      (data.shots || []).forEach(function(s){{
        var d = Math.abs((s.t||0) - t);
        if (d < bestDelta) {{ bestDelta = d; best = s; }}
      }});
      if (!best || bestDelta > 3) {{ openTimeline(t); return; }}
      // Pick the first slow-mo variant this shot actually appears in
      for (var i = 0; i < SLOWMO_PREF.length; i++) {{
        var key = SLOWMO_PREF[i];
        if (best.positions && best.positions[key] !== undefined && haveLink(key)) {{
          var link = haveLink(key);
          var url = 'https://tennis.playfullife.com/'+vid+'/'+link.file;
          pendingTime = best.positions[key];
          pausedOnOpen = true;
          openPlayer(url, link.label+' — '+vid+' @ '+fmtTs(t));
          return;
        }}
      }}
      openTimeline(t);
    }})
    .catch(function(){{ openTimeline(t); }});

  function openTimeline(tt) {{
    var link = haveLink('timeline') || v.links[0];
    if (!link) return;
    var url = 'https://tennis.playfullife.com/'+vid+'/'+link.file;
    pendingTime = tt;
    pausedOnOpen = true;
    openPlayer(url, link.label+' — '+vid+' @ '+fmtTs(tt));
  }}
}}

function escapeHtml(s) {{
  if(!s) return '';
  return String(s).replace(/[&<>"']/g, function(c){{
    return {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c];
  }});
}}

function shareSession(dk) {{
  var link = location.origin + location.pathname + '#' + dk;
  navigator.clipboard.writeText(link).then(function() {{
    var icon = document.getElementById('share-'+dk);
    if(icon) {{ icon.textContent = 'Copied!'; setTimeout(function(){{ icon.innerHTML = '&#128279;'; }}, 2000); }}
  }});
}}

// ── Tags ──
var sessionTags = {{}};
function loadTags() {{
  fetch('/api/tags').then(function(r){{return r.json()}}).then(function(d) {{
    sessionTags = d || {{}};
    buildFilters();
    renderGallery();
  }}).catch(function(){{}});
}}

function saveTags(dateKey, tags) {{
  var pwd = localStorage.getItem('upload_pwd');
  if(!pwd) {{ pwd = prompt('Password:'); if(!pwd) return; localStorage.setItem('upload_pwd', pwd); }}
  sessionTags[dateKey] = tags;
  fetch('/api/tags', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{password:pwd, date:dateKey, tags:tags}})
  }}).then(function(r){{return r.json()}}).then(function(d){{
    if(d.error) {{ localStorage.removeItem('upload_pwd'); alert(d.error); return; }}
    if(d.tags) sessionTags = d.tags;
    buildFilters();
    renderGallery();
  }}).catch(function(){{}});
}}

function allPeople() {{
  var s = {{}};
  Object.values(sessionTags).forEach(function(arr){{
    arr.forEach(function(n){{ s[n] = true; }});
  }});
  return Object.keys(s).sort();
}}

function addTagToSession(dateKey) {{
  var el = document.getElementById('taginput-'+dateKey);
  if(el) return; // already open
  var container = document.getElementById('tags-'+dateKey);
  var wrap = document.createElement('span');
  wrap.style.position = 'relative';
  wrap.innerHTML = '<input class="tag-input" id="taginput-'+dateKey+'" placeholder="Name...">'
    + '<div class="tag-suggest" id="tagsuggest-'+dateKey+'"></div>';
  container.insertBefore(wrap, container.querySelector('.add-tag'));
  var inp = wrap.querySelector('input');
  var suggest = wrap.querySelector('.tag-suggest');
  inp.focus();
  inp.addEventListener('input', function() {{
    var q = inp.value.toLowerCase();
    var people = allPeople().filter(function(p){{ return p.toLowerCase().indexOf(q)>=0 && (sessionTags[dateKey]||[]).indexOf(p)<0; }});
    if(q.length>0 && people.length>0) {{
      suggest.innerHTML = people.map(function(p){{ return '<div data-name="'+p+'">'+p+'</div>'; }}).join('');
      suggest.classList.add('show');
    }} else {{ suggest.classList.remove('show'); }}
  }});
  suggest.addEventListener('click', function(e) {{
    var name = e.target.dataset.name;
    if(name) commitTag(dateKey, name, wrap);
  }});
  inp.addEventListener('keydown', function(e) {{
    if(e.key==='Enter' && inp.value.trim()) {{ commitTag(dateKey, inp.value.trim(), wrap); }}
    if(e.key==='Escape') {{ wrap.remove(); }}
  }});
  inp.addEventListener('blur', function() {{ setTimeout(function(){{ wrap.remove(); }}, 200); }});
}}

function commitTag(dateKey, name, wrap) {{
  var tags = (sessionTags[dateKey]||[]).slice();
  if(tags.indexOf(name)<0) tags.push(name);
  wrap.remove();
  saveTags(dateKey, tags);
}}

function removeTag(dateKey, name) {{
  var tags = (sessionTags[dateKey]||[]).filter(function(n){{ return n!==name; }});
  saveTags(dateKey, tags);
}}

// ── Rendering ──
var currentFilter = 'all';
var currentSort = 'recorded-desc';
var searchQuery = '';

function toggleFilters() {{
  var f1 = document.getElementById('filters');
  var f2 = document.getElementById('filtersRow2');
  var arrow = document.getElementById('filterArrow');
  var isOpen = !f1.classList.contains('collapsed');
  if(isOpen) {{
    f1.classList.add('collapsed');
    f2.classList.add('collapsed');
    arrow.classList.remove('open');
  }} else {{
    f1.classList.remove('collapsed');
    f2.classList.remove('collapsed');
    arrow.classList.add('open');
  }}
}}

// UX-4 — iOS native filter bridge. iOS posts the chosen {{filter, sort}}
// from a SwiftUI sheet; we set state, hide the desktop UI artefacts,
// and re-render. Called via webView.evaluateJavaScript from Swift.
function applyNativeFilter(opts) {{
  if(!opts || typeof opts !== 'object') return;
  if(typeof opts.filter === 'string') currentFilter = opts.filter;
  if(typeof opts.sort === 'string') currentSort = opts.sort;
  // Sync the inline UI so re-toggling the WebView's own panel reflects
  // what the native sheet set.
  document.querySelectorAll('.chip').forEach(function(c){{ c.classList.remove('active'); }});
  var matching = document.querySelector('.chip[data-filter="'+currentFilter+'"]');
  if(matching) matching.classList.add('active');
  var sortSel = document.getElementById('sortSelect');
  if(sortSel) sortSel.value = currentSort;
  updateActiveFilter();
  updateFilterBadge();
  renderGallery();
}}

// Hide the WebView's own Filter & Sort row when we detect we're
// running inside the iOS native shell (where the user gets a native
// sheet instead). The probe uses the openVideo message bridge as a
// proxy for "this WebView is our iOS app".
(function hideInlineFiltersOnIOS() {{
  try {{
    if (window.webkit && window.webkit.messageHandlers
        && window.webkit.messageHandlers.openVideo) {{
      var t = document.getElementById('filterToggle');
      var f1 = document.getElementById('filters');
      var f2 = document.getElementById('filtersRow2');
      if (t) t.style.display = 'none';
      if (f1) f1.style.display = 'none';
      if (f2) f2.style.display = 'none';
    }}
  }} catch (e) {{}}
}})();

function updateFilterBadge() {{
  var badge = document.getElementById('filterBadge');
  if(currentFilter === 'all') {{
    badge.classList.remove('show');
    badge.textContent = '';
  }} else {{
    var label = document.querySelector('.chip[data-filter="'+currentFilter+'"]');
    badge.textContent = label ? label.textContent : currentFilter;
    badge.classList.add('show');
  }}
}}

function clearFilter() {{
  currentFilter = 'all';
  document.querySelectorAll('.chip').forEach(function(c){{c.classList.remove('active')}});
  var allChip = document.querySelector('.chip[data-filter="all"]');
  if(allChip) allChip.classList.add('active');
  updateActiveFilter();
  updateFilterBadge();
  renderGallery();
}}

function changeSort(val) {{
  currentSort = val;
  renderGallery();
}}

function updateActiveFilter() {{
  var el = document.getElementById('activeFilter');
  var txt = document.getElementById('activeFilterText');
  if(currentFilter === 'all') {{
    el.classList.remove('show');
  }} else {{
    el.classList.add('show');
    var label = document.querySelector('.chip[data-filter="'+currentFilter+'"]');
    txt.textContent = label ? label.textContent : currentFilter;
  }}
}}

function parseDate(s) {{
  if(!s) return null;
  try {{ return new Date(s.includes('T') ? s : s+'T00:00:00'); }} catch(e){{ return null; }}
}}

function dateKey(s) {{
  if(!s) return 'Unknown';
  return s.split('T')[0];
}}

function formatSessionDate(key) {{
  if(key==='Unknown') return 'Unknown Date';
  try {{
    var d = new Date(key+'T12:00:00');
    return d.toLocaleDateString('en-US', {{weekday:'long', month:'long', day:'numeric', year:'numeric'}});
  }} catch(e) {{ return key; }}
}}

function formatTime(created) {{
  if(!created || !created.includes('T')) return '';
  try {{
    var t = created.split('T')[1].split(/[-+.Z]/)[0];
    var parts = t.split(':');
    var h = parseInt(parts[0]), m = parts[1];
    var ampm = h>=12?'PM':'AM';
    return (h%12||12)+':'+m+' '+ampm;
  }} catch(e) {{ return ''; }}
}}

function fmtDur(s) {{
  if(!s) return '';
  return Math.floor(s/60)+':'+(('0'+(Math.floor(s)%60)).slice(-2));
}}

function matchesFilter(v) {{
  if(currentFilter==='all') return true;
  // Shot type filter
  if(['serve','forehand','backhand'].indexOf(currentFilter) >= 0) {{
    return (v.breakdown[currentFilter]||0) > 0;
  }}
  // Month filter (format: "2026-04")
  if(currentFilter.match(/^\d{{4}}-\d{{2}}$/)) {{
    return (v.created||'').substring(0,7) === currentFilter;
  }}
  // People filter (format: "person:Name")
  if(currentFilter.indexOf('person:') === 0) {{
    var name = currentFilter.substring(7);
    var tags = sessionTags[dateKey(v.created)] || [];
    return tags.indexOf(name) >= 0;
  }}
  return true;
}}

function matchesSearch(v) {{
  if(!searchQuery) return true;
  var q = searchQuery.toLowerCase();
  if(v.id.toLowerCase().includes(q)) return true;
  if((v.created||'').toLowerCase().includes(q)) return true;
  var dk = formatSessionDate(dateKey(v.created)).toLowerCase();
  if(dk.includes(q)) return true;
  // Search by tagged people
  var tags = sessionTags[dateKey(v.created)] || [];
  for(var i=0; i<tags.length; i++) {{ if(tags[i].toLowerCase().includes(q)) return true; }}
  return false;
}}

function renderGallery() {{
  var filtered = VIDEOS.filter(function(v) {{ return matchesFilter(v) && matchesSearch(v); }});
  var sortFns = {{
    'recorded-desc': function(a,b){{ return (b.created||'').localeCompare(a.created||''); }},
    'recorded-asc': function(a,b){{ return (a.created||'').localeCompare(b.created||''); }},
    'shots-desc': function(a,b){{ return (b.shots||0)-(a.shots||0); }},
    'shots-asc': function(a,b){{ return (a.shots||0)-(b.shots||0); }},
    'duration-desc': function(a,b){{ return (b.duration||0)-(a.duration||0); }},
    'duration-asc': function(a,b){{ return (a.duration||0)-(b.duration||0); }},
  }};
  filtered.sort(sortFns[currentSort] || sortFns['recorded-desc']);

  // Group by date
  var sessions = {{}};
  var order = [];
  filtered.forEach(function(v) {{
    var dk = dateKey(v.created);
    if(!sessions[dk]) {{ sessions[dk] = []; order.push(dk); }}
    sessions[dk].push(v);
  }});

  var totalShots = 0;
  VIDEOS.forEach(function(v) {{ totalShots += v.shots||0; }});
  document.getElementById('statBadge').textContent = VIDEOS.length+' videos / '+totalShots+' shots';

  if(filtered.length === 0) {{
    document.getElementById('content').innerHTML = '<div class="no-results">No videos match your search</div>';
    return;
  }}

  var html = '';
  order.forEach(function(dk) {{
    var vids = sessions[dk];
    var sessionShots = 0;
    vids.forEach(function(v){{ sessionShots += v.shots||0; }});

    var dkTags = sessionTags[dk] || [];
    var tagsHtml = '<span class="session-tags" id="tags-'+dk+'">';
    dkTags.forEach(function(name) {{
      tagsHtml += '<span class="tag">'+name+' <span class="rm" data-action="rmtag" data-dk="'+dk+'" data-name="'+name+'">&times;</span></span>';
    }});
    tagsHtml += '<span class="add-tag" data-action="addtag" data-dk="'+dk+'">+ person</span>';
    tagsHtml += '</span>';

    html += '<div class="session" id="session-'+dk+'">';
    html += '<div class="session-header" data-dk="'+dk+'">';
    html += '<span class="session-date" data-action="share" data-dk="'+dk+'">'+formatSessionDate(dk)+'</span>';
    html += '<span class="session-stats">'+vids.length+' video'+(vids.length>1?'s':'')+' / '+sessionShots+' shots</span>';
    html += tagsHtml;
    html += '<span class="share-icon" id="share-'+dk+'" data-action="share" data-dk="'+dk+'">&#128279;</span>';
    html += '</div>';
    html += '<div class="grid">';

    vids.forEach(function(v) {{
      var time = formatTime(v.created);
      var dur = fmtDur(v.duration);
      var bd = v.breakdown || {{}};
      var abbrev = {{'serve':'S','forehand':'FH','backhand':'BH'}};
      var bdParts = [];
      ['serve','forehand','backhand'].forEach(function(st){{
        if(bd[st]) bdParts.push(bd[st]+' '+abbrev[st]);
      }});

      var thumbInner;
      if(v.has_thumb) {{
        thumbInner = '<img class="card-thumb" src="https://tennis.playfullife.com/thumbs/'+v.id+'.jpg" alt="'+v.id+'" loading="lazy">';
      }} else {{
        thumbInner = '<div class="card-thumb-placeholder">'+v.id+'</div>';
      }}
      // Clicking the thumbnail plays the first available video (timeline → rally → first link)
      var primaryPlay = v.links.length ? v.links[0] : null;
      var thumbAction = '';
      if(primaryPlay) {{
        var thumbUrl = 'https://tennis.playfullife.com/'+v.id+'/'+primaryPlay.file;
        thumbAction = ' data-action="play" data-url="'+thumbUrl+'" data-title="'+primaryPlay.label+' \\u2014 '+v.id+'" style="cursor:pointer"';
      }}
      // Card label = user-set display_name if present, else vid. The
      // raw vid stays available via a small subtitle for context.
      var titleLabel = v.display_name || v.id;
      var thumbHtml = '<div class="card-thumb-wrap"'+thumbAction+'>'+thumbInner
        + '<span class="card-id" title="Click to rename" data-action="rename" data-vid="'+v.id+'">'
        + escapeHtml(titleLabel)
        + (v.display_name ? ' <span style="opacity:0.6">&#9998;</span>' : '')
        + '</span></div>';

      // Group links by base type (e.g. "rally" + "rally_slowmo" → one row)
      var groups = {{}};
      var groupOrder = [];
      v.links.forEach(function(lk) {{
        var isSlow = lk.key.endsWith('_slowmo');
        var baseKey = isSlow ? lk.key.replace('_slowmo','') : lk.key;
        if(!groups[baseKey]) {{ groups[baseKey] = {{normal:null, slow:null, label:'', color:''}}; groupOrder.push(baseKey); }}
        if(isSlow) groups[baseKey].slow = lk;
        else {{ groups[baseKey].normal = lk; groups[baseKey].label = lk.label; groups[baseKey].color = lk.color; }}
        if(!groups[baseKey].label) {{ groups[baseKey].label = lk.label.replace(' Slow-Mo',''); groups[baseKey].color = lk.color; }}
      }});

      var bd = v.breakdown || {{}};
      var countFor = function(baseKey) {{
        if(baseKey === 'forehands') return bd.forehand || 0;
        if(baseKey === 'backhands') return bd.backhand || 0;
        if(baseKey === 'serves')    return bd.serve || 0;
        if(baseKey === 'volleys')   return (bd.forehand_volley||0) + (bd.backhand_volley||0) + (bd.overhead||0);
        if(baseKey === 'other')     return bd.unknown_shot || 0;
        // timeline / rally / grouped / highlights — show total shots
        return v.shots || 0;
      }};

      // Single Play action per card. The in-player chip row handles
      // shot-type filtering and slo-mo, so the card stays clean even
      // as we add more shot types (slice FH/BH, overhead, etc.). Falls
      // back through timeline → rally → highlights → grouped → first
      // available so older videos without a timeline still play.
      var preferKeys = ['timeline','rally','highlights','grouped'];
      var primary = null;
      for (var pi = 0; pi < preferKeys.length && !primary; pi++) {{
        var g = groups[preferKeys[pi]];
        if (g && (g.normal || g.slow)) primary = g.normal || g.slow;
      }}
      if (!primary) {{
        // Last resort: first available variant in groupOrder
        for (var gi = 0; gi < groupOrder.length && !primary; gi++) {{
          var g2 = groups[groupOrder[gi]];
          primary = g2.normal || g2.slow;
        }}
      }}
      var linksHtml = '';
      if (primary) {{
        var primaryUrl = 'https://tennis.playfullife.com/'+v.id+'/'+primary.file;
        // Compose a human-friendly player title from the recorded
        // date/time (falling back to the raw video id). Beats showing
        // `iphone_9ca0a615` as the modal header.
        var humanTitle = time ? (time + ' \\u2014 ' + v.id) : v.id;
        linksHtml = '<div class="play-strip">'
          + '<a href="'+primaryUrl+'" class="play-chip play-chip-primary" '
          + 'data-title="'+humanTitle+'" '
          + 'onclick="event.stopPropagation();openPlayer(this.href,this.dataset.title);return false">'
          + '<span class="ch-lbl">&#9654; Watch</span>'
          + (v.shots ? '<span class="ch-ct">'+v.shots+'</span>' : '')
          + '</a></div>';
      }}

      html += '<div class="card">';
      html += thumbHtml;
      html += '<div class="card-body">';
      html += '<div class="card-time">'+(time||v.id)+'</div>';
      html += '<div class="card-meta">';
      if(dur) html += '<span>'+dur+'</span>';
      if(v.shots) html += '<span>'+v.shots+' shots</span>';
      if(v.avg_speed_mph) html += '<span style="color:#FFD700">\u26be '+v.avg_speed_mph+' mph</span>';
      else if(v.ball_avg_speed) html += '<span style="color:#FFD700">\u26be '+v.ball_avg_speed.toFixed(0)+'</span>';
      if(v.in_count || v.out_count) html += '<span style="color:#8f8"><small>IN:'+
        (v.in_count||0)+'</small></span><span style="color:#f88"><small>OUT:'+(v.out_count||0)+'</small></span>';
      var ft = v.features || [];
      if(ft.indexOf('tracked')>=0) html += '<span style="color:#00BCD4;font-size:.7em" title="Dynamic player tracking">&#127909; Tracked</span>';
      if(ft.indexOf('comparisons')>=0) html += '<span style="color:#E74C3C;font-size:.7em" title="Pro comparison available">&#127941; vs Pro</span>';
      if(ft.indexOf('racket_removed')>=0) html += '<span style="color:#AB47BC;font-size:.7em" title="Racket-removed composites">&#9997; No Racket</span>';
      html += '</div>';
      if(bdParts.length) html += '<div class="card-breakdown">'+bdParts.join(', ')+'</div>';
      html += '<div class="card-coach-summary" id="coachSum-'+v.id+'" data-action="coach" data-vid="'+v.id+'">'
        +'<span class="coach-summary-label">Coach summary</span>'
        +'<span class="more">View &rsaquo;</span>'
        +'<span class="coach-summary-text" hidden></span></div>';
      html += '<div class="card-links">'+linksHtml
        +'<div class="card-footer">'
        +'<span class="foot-btn" data-action="sequences" data-vid="'+v.id+'" title="Swing sequences">&#127910; Sequences</span>'
        +'<span class="foot-btn share" data-action="share-link" data-vid="'+v.id+'" title="Get a share link">&#128279; Share</span>'
        +'<span class="foot-btn danger" data-action="delete" data-vid="'+v.id+'" title="Delete this video">&#128465;</span>'
        +'</div>'
        +'</div>';
      html += '</div></div>';
    }});

    html += '</div></div>';
  }});

  document.getElementById('content').innerHTML = html;
  // Trigger coaching summary loads for visible cards
  filtered.forEach(function(v){{ loadCoachingSummary(v.id); }});
}}

// ── Filters ──
function buildFilters() {{
  var months = {{}};
  var monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  VIDEOS.forEach(function(v) {{
    if(v.created) {{
      var ym = v.created.substring(0,7);
      if(!months[ym]) {{
        var parts = ym.split('-');
        months[ym] = monthNames[parseInt(parts[1])-1]+' '+parts[0];
      }}
    }}
  }});
  var sortedMonths = Object.keys(months).sort().reverse();

  var html = '<span class="chip active" data-filter="all">All</span>';
  html += '<span class="filter-sep"></span>';
  html += '<span class="filter-label">Type</span>';
  html += '<span class="chip" data-filter="serve">Serves</span>';
  html += '<span class="chip" data-filter="forehand">Forehands</span>';
  html += '<span class="chip" data-filter="backhand">Backhands</span>';
  if(sortedMonths.length > 0) {{
    html += '<span class="filter-sep"></span>';
    html += '<span class="filter-label">Month</span>';
    sortedMonths.forEach(function(ym) {{
      html += '<span class="chip" data-filter="'+ym+'">'+months[ym]+'</span>';
    }});
  }}
  var people = allPeople();
  if(people.length > 0) {{
    html += '<span class="filter-sep"></span>';
    html += '<span class="filter-label">People</span>';
    people.forEach(function(name) {{
      html += '<span class="chip" data-filter="person:'+name+'">'+name+'</span>';
    }});
  }}
  document.getElementById('filters').innerHTML = html;
  // Re-mark active chip if filter is still set
  if(currentFilter !== 'all') {{
    var active = document.querySelector('.chip[data-filter="'+currentFilter+'"]');
    if(active) active.classList.add('active');
    else {{ var all = document.querySelector('.chip[data-filter="all"]'); if(all) all.classList.add('active'); currentFilter='all'; }}
  }}
}}
buildFilters();

document.getElementById('filters').addEventListener('click', function(e) {{
  var chip = e.target.closest('.chip');
  if(!chip) return;
  currentFilter = chip.dataset.filter;
  document.querySelectorAll('.chip').forEach(function(c){{c.classList.remove('active')}});
  chip.classList.add('active');
  updateActiveFilter();
  updateFilterBadge();
  renderGallery();
}});

// ── Search ──
var searchTimeout;
document.getElementById('searchInput').addEventListener('input', function(e) {{
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(function() {{
    searchQuery = e.target.value.trim();
    renderGallery();
  }}, 200);
}});

// ── Processing Queue ──
(function() {{
  var stageLabels = {{
    'uploading':'Uploading','pending':'Queued','downloading':'Downloading',
    'preprocessing':'Preprocessing','extracting_poses':'Extracting Poses',
    'detecting_shots':'Detecting Shots','exporting':'Exporting',
    'uploading_results':'Uploading','processing':'Processing',
    'awaiting_coordinator':'Queued','coordinator_registered':'Queued',
    'complete':'Complete','failed':'Failed'
  }};
  var DAY_MS = 24*3600*1000;
  var FAIL_WINDOW_MS = 7*DAY_MS;

  function stripName(filename) {{
    return (filename||'').replace(/\.(MOV|mov|MP4|mp4)$/,'');
  }}

  function todaySummary() {{
    // Computed from VIDEOS (already loaded) — local day boundaries.
    var now = new Date();
    var startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    var processed = 0, shots = 0;
    VIDEOS.forEach(function(v) {{
      var t = v.created ? new Date(v.created).getTime() : 0;
      if(t >= startOfDay) {{ processed++; shots += (v.shots||0); }}
    }});
    return {{processed:processed, shots:shots}};
  }}

  function renderDashboard(queue) {{
    var banner = document.getElementById('procBanner');
    var now = Date.now();
    var inflight = [], failed = [];
    (queue||[]).forEach(function(i) {{
      if(i.status==='failed') {{
        var t = i.updated_at ? new Date(i.updated_at).getTime() : 0;
        if(now - t < FAIL_WINDOW_MS) failed.push(i);
      }} else if(i.status==='complete') {{
        // hide; the gallery card is the success surface
      }} else if(i.status==='not_found') {{
        // skip
      }} else {{
        inflight.push(i);
      }}
    }});

    var today = todaySummary();
    var failedToday = failed.filter(function(i) {{
      var t = i.updated_at ? new Date(i.updated_at).getTime() : 0;
      var s = new Date(); s.setHours(0,0,0,0);
      return t >= s.getTime();
    }}).length;

    var rows = [];
    if(today.processed>0 || today.shots>0 || failedToday>0) {{
      var sum = '<b>Today:</b> '+today.processed+' processed'+
                '<span class="sep">·</span><b>'+today.shots+'</b> shots';
      if(failedToday>0) sum += '<span class="sep">·</span><b style="color:#E74C3C">'+failedToday+'</b> failed';
      rows.push('<div class="proc-bar proc-summary">'+sum+'</div>');
    }}

    if(inflight.length>0) {{
      var inner = '';
      inflight.forEach(function(item) {{
        var name = stripName(item.filename);
        // Status may be missing on freshly-written markers (the worker
        // writes the marker before the coordinator stamps a status).
        // Show those as "Queued" instead of literal 'undefined'.
        var label = stageLabels[item.stage]||stageLabels[item.status]||
                    (item.status ? item.status : 'Queued');
        var pct = (item.progress!=null && item.progress!==0) ? ' '+item.progress+'%' : '';
        var cls = 'proc-item';
        if(!item.status||item.status==='pending'||item.status==='coordinator_registered'||item.status==='awaiting_coordinator') cls += ' proc-pending';
        inner += '<span class="'+cls+'"><span class="proc-dot"></span>'+name+' <span class="stage">'+label+pct+'</span></span>';
      }});
      rows.push('<div class="proc-bar"><span class="proc-label">Processing now ('+inflight.length+')</span><div class="proc-items">'+inner+'</div></div>');
    }}

    if(failed.length>0) {{
      var inner = '';
      failed.forEach(function(item) {{
        var name = stripName(item.filename);
        var err = item.error ? ' <span class="err">'+item.error+'</span>' : '';
        inner += '<span class="proc-item proc-failed"><span class="proc-dot"></span>'+name+err+'</span>';
      }});
      rows.push('<div class="proc-bar proc-failed-bar"><span class="proc-label">Recently failed ('+failed.length+')</span><div class="proc-items">'+inner+'</div></div>');
    }}

    if(rows.length===0) {{ banner.style.display='none'; return; }}
    banner.style.display='';
    banner.innerHTML = rows.join('');
  }}

  function loadQueue() {{
    fetch('/api/queue').then(function(r){{return r.json()}}).then(function(d) {{
      renderDashboard(d.queue);
    }}).catch(function(){{
      // Still render the today summary even if queue endpoint is down.
      renderDashboard([]);
    }});
  }}
  loadQueue();
  setInterval(loadQueue, 30000);
}})();

// ── Upload ──
var CHUNK_SIZE = 5*1024*1024;

async function startUpload() {{
  var password = document.getElementById('uploadPwd').value;
  if(!password) {{ alert('Enter password'); return; }}
  var file = document.getElementById('videoFile').files[0];
  if(!file) {{ alert('Select a file'); return; }}

  var btn = document.getElementById('uploadBtn');
  var prog = document.getElementById('uploadProgress');
  var bar = document.getElementById('uploadBar');
  var pct = document.getElementById('uploadPct');
  var st = document.getElementById('uploadStatus');

  btn.disabled = true; prog.style.display = 'block'; st.textContent = 'Initializing...';

  try {{
    var initRes = await fetch('/api/upload/init', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{password:password, filename:file.name}})
    }});
    if(!initRes.ok) {{ var e = await initRes.json(); throw new Error(e.error); }}
    var init = await initRes.json();

    st.textContent = 'Uploading...';
    var totalChunks = Math.ceil(file.size/CHUNK_SIZE);
    var parts = [];
    for(var i=0; i<totalChunks; i++) {{
      var start = i*CHUNK_SIZE, end = Math.min(start+CHUNK_SIZE, file.size);
      var chunk = file.slice(start,end);
      for(var attempt=0; attempt<4; attempt++) {{
        try {{
          var r = await fetch('/api/upload/'+init.id+'/'+(i+1), {{method:'PUT',body:chunk}});
          if(r.ok) {{ parts.push(await r.json()); break; }}
        }} catch(err) {{ if(attempt===3) throw err; await new Promise(function(ok){{setTimeout(ok,2000*(attempt+1))}}); }}
      }}
      var p = Math.round(end/file.size*100);
      bar.style.width = p+'%'; pct.textContent = p+'%';
    }}

    st.textContent = 'Finalizing...';
    await fetch('/api/upload/'+init.id+'/complete', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{parts:parts}})
    }});

    bar.style.width='100%'; bar.style.background='#27AE60';
    st.innerHTML = 'Upload complete! Processing will begin shortly.';
  }} catch(err) {{
    st.textContent = 'Error: '+err.message; bar.style.background='#E74C3C';
  }} finally {{ btn.disabled = false; }}
}}

// ── Event delegation for content area ──
document.getElementById('content').addEventListener('click', function(e) {{
  var el = e.target.closest('[data-action]');
  if(!el) return;
  var action = el.dataset.action;
  var dk = el.dataset.dk;
  e.stopPropagation();
  if(action === 'addtag') addTagToSession(dk);
  else if(action === 'rmtag') removeTag(dk, el.dataset.name);
  else if(action === 'share') shareSession(dk);
  else if(action === 'download') dlFile(el.dataset.url);
  else if(action === 'delete') deleteVideo(el.dataset.vid);
  else if(action === 'share-link') createShareLink(el.dataset.vid);
  else if(action === 'rename') renameVideo(el.dataset.vid);
  else if(action === 'play') openPlayer(el.dataset.url, el.dataset.title);
  else if(action === 'coach') openCoachModal(el.dataset.vid);
  else if(action === 'sequences') openSeqModal(el.dataset.vid);
}});

// ── Init ──
renderGallery();
loadTags();

// Deep link to session via hash (e.g. #2026-04-02)
(function() {{
  var hash = location.hash.replace('#','');
  if(!hash) return;
  var el = document.getElementById('session-'+hash);
  if(el) {{
    setTimeout(function() {{
      el.scrollIntoView({{behavior:'smooth', block:'start'}});
      el.classList.add('highlighted');
    }}, 100);
  }}
}})();
</script>
</body></html>'''


def get_branch_slug():
    """Best-effort current git branch slug for staging/preview names."""
    try:
        r = subprocess.run(['git', '-C', str(PROJECT_ROOT), 'rev-parse', '--abbrev-ref', 'HEAD'],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            br = r.stdout.strip().replace('/', '-').replace('_', '-')
            return br[:60] or 'unknown'
    except Exception:
        pass
    return 'unknown'


def update_index(mode='production', user_hash=None):
    """Main: gather metadata, build HTML, deploy.

    mode:
      'production' — upload to highlights/index.html (live URL)
      'staging'    — upload to staging/<branch>/highlights/index.html
      'preview'    — write to ~/whiteboards/preview-<branch>/index.html (no upload)

    user_hash: if set (e.g. 'u_666f1a02'), generate a PER-USER gallery:
      - Source keys filtered to highlights/<user_hash>/...
      - All in-HTML URLs rewritten with `/u/<user_hash>` prefix
      - Index uploaded to highlights/<user_hash>/index.html
      Otherwise behaves as the legacy flat root gallery.
    """
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    import importlib, config.settings
    importlib.reload(config.settings)
    from storage.r2_client import R2Client

    c = R2Client()
    if user_hash:
        list_prefix = f'highlights/{user_hash}/'
        strip_prefix = f'highlights/{user_hash}/'
    else:
        list_prefix = 'highlights/'
        strip_prefix = 'highlights/'
    keys = c.list(prefix=list_prefix, max_keys=10000)

    # Group files by video. The flat layout has parts = [highlights, vid, file]
    # while the per-user layout has [highlights, user_hash, vid, file]. We
    # normalize by stripping the highlights/<user_hash?>/ prefix and reparsing.
    videos = {}
    video_features = {}  # vid -> set of features detected from R2 keys
    for k in keys:
        if 'index.html' in k or 'thumbs/' in k:
            continue
        if not k.startswith(strip_prefix):
            continue
        rel = k[len(strip_prefix):]  # e.g. "IMG_1108/timeline.mp4"
        parts = rel.split('/')
        if len(parts) >= 2:
            vid = parts[0]
            fname = parts[-1]
            if len(parts) == 3:
                subfolder = parts[1]
                feats = video_features.setdefault(vid, set())
                if subfolder == 'sequences':
                    feats.add('sequences')
                    if '_noracket' in fname:
                        feats.add('racket_removed')
                elif subfolder == 'comparisons':
                    feats.add('comparisons')
            if len(parts) == 2 and fname.endswith('.mp4'):
                videos.setdefault(vid, []).append(fname)
                if '_tracked' in fname:
                    video_features.setdefault(vid, set()).add('tracked')

    # Gather metadata + ensure thumbnails
    all_meta = {}
    for vid in videos:
        meta = get_video_metadata(vid, r2_client=c, user_hash=user_hash)
        meta['files'] = sorted(videos[vid])
        meta['features'] = sorted(video_features.get(vid, set()))
        has_thumb = generate_thumbnail(vid, user_hash=user_hash)
        if has_thumb:
            upload_thumbnail(c, vid, user_hash=user_hash)
        meta['has_thumb'] = has_thumb
        all_meta[vid] = meta

    # Build and upload index
    html = build_index_html(all_meta)
    # Per-user mode: rewrite every absolute URL to live under /u/<user_hash>/.
    # Every URL in the generated HTML/JS is built as `https://tennis.playfullife.com/<...>`
    # so a single string-replace catches all of them, including the JS
    # `?v=...` deep-link path which strips and re-prepends this exact prefix.
    if user_hash:
        old = 'https://tennis.playfullife.com/'
        new = f'https://tennis.playfullife.com/u/{user_hash}/'
        html = html.replace(old, new)
    tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8')
    tmp.write(html)
    tmp.close()

    # Validate JS syntax before uploading (catches f-string escaping bugs)
    try:
        import re as _re
        scripts = _re.findall(r'<script>(.*?)</script>', html, _re.DOTALL)
        if scripts:
            js_tmp = tempfile.NamedTemporaryFile(suffix='.js', delete=False, mode='w', encoding='utf-8')
            js_tmp.write('(function(){\n')
            for s in scripts:
                js_tmp.write(s + '\n')
            js_tmp.write('});\n')
            js_tmp.close()
            r = subprocess.run(['node', '--check', js_tmp.name],
                             capture_output=True, text=True, timeout=5)
            os.unlink(js_tmp.name)
            if r.returncode != 0:
                print(f'ERROR: JS syntax error in generated HTML — aborting upload')
                print(r.stderr.strip()[:200])
                os.unlink(tmp.name)
                return
            print('JS syntax check: OK')
    except FileNotFoundError:
        pass  # node not installed, skip check

    if mode == 'preview':
        branch = get_branch_slug()
        out_dir = Path.home() / 'whiteboards' / f'preview-{branch}'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'index.html'
        out_path.write_text(html, encoding='utf-8')
        os.unlink(tmp.name)
        url = f'http://localhost:8088/preview-{branch}/'
        print(f'PREVIEW: {len(all_meta)} videos → {out_path}')
        print(f'         {url}')
        try:
            subprocess.run(['open', url], check=False, timeout=2)
        except Exception:
            pass
        return out_path

    if mode == 'staging':
        branch = get_branch_slug()
        key = f'staging/{branch}/highlights/index.html'
        c.upload(tmp.name, key, content_type='text/html')
        os.unlink(tmp.name)
        print(f'STAGING: {len(all_meta)} videos → r2://{key}')
        print(f'         https://tennis.playfullife.com/staging/{branch}/highlights/index.html')
        print(f'         (note: requires worker route /staging/<branch>/* → R2 staging/<branch>/*)')
        return None

    # production
    if user_hash:
        key = f'highlights/{user_hash}/index.html'
        c.upload(tmp.name, key, content_type='text/html')
        os.unlink(tmp.name)
        print(f'Updated per-user index: {len(all_meta)} videos → r2://{key}')
        print(f'https://tennis.playfullife.com/u/{user_hash}')
    else:
        c.upload(tmp.name, 'highlights/index.html', content_type='text/html')
        c.upload(tmp.name, 'highlights/', content_type='text/html')
        os.unlink(tmp.name)
        print(f'Updated index: {len(all_meta)} videos')
        print('https://tennis.playfullife.com/')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument('--preview', action='store_true',
                   help='Write HTML locally to ~/whiteboards/preview-<branch>/, no upload. Fast iteration.')
    g.add_argument('--staging', action='store_true',
                   help='Upload to r2://staging/<branch>/ — shareable URL, does not affect production.')
    p.add_argument('--user', dest='user_hash', default=None,
                   help='Generate per-user index (e.g. --user u_666f1a02). Sources keys '
                        'under highlights/<hash>/, rewrites URLs with /u/<hash> prefix, '
                        'uploads to highlights/<hash>/index.html.')
    args = p.parse_args()
    mode = 'preview' if args.preview else ('staging' if args.staging else 'production')
    update_index(mode, user_hash=args.user_hash)
