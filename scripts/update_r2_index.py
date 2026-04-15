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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_thumbnail(vid):
    """Generate a thumbnail for a video if it doesn't exist. Returns True if available."""
    thumb_dir = os.path.join(PROJECT_ROOT, 'exports', 'thumbs')
    os.makedirs(thumb_dir, exist_ok=True)
    thumb = os.path.join(thumb_dir, f'{vid}.jpg')

    if os.path.exists(thumb):
        return True

    # Try downloading from R2 (try both locations — worker uploads to thumbs/,
    # index uploads to highlights/thumbs/)
    import urllib.request
    for r2_url in [
        f'https://tennis.playfullife.com/thumbs/{vid}.jpg',
        f'https://tennis.playfullife.com/highlights/thumbs/{vid}.jpg',
    ]:
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


def upload_thumbnail(client, vid):
    """Upload thumbnail to R2 if it exists locally."""
    thumb = os.path.join(PROJECT_ROOT, 'exports', 'thumbs', f'{vid}.jpg')
    if os.path.exists(thumb):
        client.upload(thumb, f'highlights/thumbs/{vid}.jpg', content_type='image/jpeg')


def get_video_metadata(vid, r2_client=None):
    """Gather metadata for a video from detection JSON, R2 meta.json, or raw MOV."""
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
    if not info.get('shots') and r2_client:
        meta_key = f'highlights/{vid}/meta.json'
        try:
            obj = r2_client.client.get_object(
                Bucket=r2_client.bucket_name, Key=meta_key)
            meta = json.loads(obj['Body'].read())
            info['duration'] = meta.get('duration', 0)
            info['shots'] = meta.get('shots', 0)
            info['breakdown'] = meta.get('breakdown', {})
            if meta.get('created'):
                info['created'] = meta['created']
        except Exception:
            pass

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
    }
    link_order = [
        'timeline', 'rally', 'rally_slowmo',
        'forehands', 'forehands_slowmo',
        'backhands', 'backhands_slowmo',
        'serves', 'serves_slowmo',
        'volleys', 'volleys_slowmo',
        'highlights', 'highlights_slowmo',
        'grouped', 'grouped_slowmo',  # legacy: shown only if no per-type files exist
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

        video_data.append({
            'id': vid,
            'created': m.get('created', ''),
            'duration': dur,
            'shots': m.get('shots', 0),
            'breakdown': bd,
            'has_thumb': m.get('has_thumb', False),
            'links': links,
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
body{{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0a;color:#eee}}

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
.filters{{max-width:1100px;margin:0 auto;padding:10px 20px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}}
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

/* ── Processing Banner ── */
.proc-banner{{max-width:1100px;margin:8px auto;padding:0 20px}}
.proc-bar{{display:flex;align-items:center;gap:10px;padding:8px 14px;background:#1a1a1a;
  border-radius:8px;border:1px solid #2a2a2a;font-size:0.85em;color:#aaa;flex-wrap:wrap}}
.proc-dot{{width:8px;height:8px;border-radius:50%;background:#3498DB;animation:pulse 1.5s infinite;flex-shrink:0}}
.proc-items{{flex:1;display:flex;gap:12px;flex-wrap:wrap}}
.proc-item{{display:flex;align-items:center;gap:6px}}
.proc-item .stage{{color:#666;font-size:0.85em}}
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
.card-coach{{display:none;margin-top:8px;padding:10px 12px;background:#0f1a14;
  border:1px solid #1e3624;border-radius:6px;font-size:.78em;color:#ccc}}
.card.expanded .card-coach{{display:block}}
.coach-head{{color:#5ed694;font-weight:700;margin-bottom:4px;font-size:.9em}}
.coach-headline{{font-style:italic;color:#aaa;margin-bottom:8px}}
.coach-section{{margin-top:8px}}
.coach-section-title{{color:#8ae6ae;font-size:.78em;text-transform:uppercase;
  letter-spacing:.05em;margin-bottom:3px}}
.coach-item{{margin:3px 0 3px 12px;line-height:1.35}}
.coach-item b{{color:#e0e0e0}}
.coach-drill{{margin-top:8px;padding:8px;background:rgba(94,214,148,.08);
  border-left:2px solid #5ed694;border-radius:3px;color:#ddd}}
.coach-loading{{color:#666;font-style:italic}}
.coach-none{{color:#555;font-size:.75em;font-style:italic}}
.card-links{{display:none;flex-direction:column;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid #222}}
.card.expanded .card-links{{display:flex}}
.link-row{{display:flex;align-items:center;gap:6px}}
.link-row a.play-btn{{flex:1;color:#fff;text-decoration:none;font-size:0.74em;font-weight:600;
  padding:5px 10px;border-radius:5px;opacity:.9;transition:opacity .15s}}
.link-row a.play-btn:hover{{opacity:1}}
.link-row a.slow-btn{{color:#aaa;text-decoration:none;font-size:.65em;font-weight:600;
  padding:3px 8px;border-radius:4px;background:#2a2a2a;border:1px solid #333;
  text-transform:uppercase;letter-spacing:.05em;transition:all .15s}}
.link-row a.slow-btn:hover{{color:#fff;background:#3a3a3a;border-color:#555}}
.dl-btn{{display:inline-block;padding:4px 6px;font-size:.68em;color:#888;cursor:pointer;
  text-decoration:none;opacity:.6;transition:opacity .15s;vertical-align:middle}}
.dl-btn:hover{{opacity:1;color:#fff}}
.del-btn{{display:inline-block;padding:6px 10px;font-size:.78em;color:#666;cursor:pointer;
  text-align:center;border-top:1px solid #222;margin-top:4px;opacity:.5;transition:all .15s}}
.del-btn:hover{{opacity:1;color:#E74C3C}}

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
  .grid{{grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}}
  .card-body{{padding:8px 10px}}
  .filters{{padding:8px 12px}}
  .content{{padding:12px}}
  .player-bar{{justify-content:center}}
  .share-btn{{margin-left:0!important;width:100%}}
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
      <button class="btn-upload" onclick="document.getElementById('uploadModal').classList.add('open')">Upload</button>
    </div>
  </div>
</div>

<!-- Filters -->
<div class="filters" id="filters"></div>
<div class="filters" style="padding-top:0">
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

function openPlayer(url, title) {{
  vid.src = url;
  document.getElementById('playerTitle').textContent = title;
  overlay.style.display = 'flex';
  document.body.style.overflow = 'hidden';
  if (pendingTime !== null) {{
    vid.addEventListener('loadedmetadata', function() {{
      vid.currentTime = pendingTime; pendingTime = null;
    }}, {{once:true}});
  }}
  vid.play().catch(function(){{}});
  history.replaceState(null,'','?v='+encodeURIComponent(url.replace('https://tennis.playfullife.com/','')));
}}

function closePlayer() {{
  vid.pause(); vid.removeAttribute('src'); vid.load();
  overlay.style.display = 'none'; document.body.style.overflow = '';
  history.replaceState(null,'',location.pathname);
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

function deleteVideo(vid) {{
  if(!confirm('Permanently delete '+vid+' and all its files?')) return;
  var pwd = prompt('Enter delete password:');
  if(!pwd) return;
  fetch('/api/video/'+vid+'/delete', {{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{password:pwd}})
  }}).then(function(r){{return r.json()}}).then(function(d) {{
    if(d.error) {{ alert('Delete failed: '+d.error); return; }}
    alert('Deleted '+vid+' ('+d.deleted+' files removed)');
    // Remove from local data and re-render
    VIDEOS = VIDEOS.filter(function(v){{ return v.id !== vid; }});
    buildFilters();
    renderGallery();
  }}).catch(function(e){{ alert('Error: '+e); }});
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

function toggleCard(el) {{
  el.classList.toggle('expanded');
  if(el.classList.contains('expanded')) {{
    var coach = el.querySelector('.card-coach');
    if(coach && coach.dataset.loaded === '0') {{
      coach.dataset.loaded = '1';
      var vid = coach.id.replace('coach-','');
      loadCoaching(vid, coach);
    }}
  }}
}}

function loadCoaching(vid, container) {{
  fetch('/'+vid+'/coaching.json', {{cache:'no-store'}})
    .then(function(r){{ if(!r.ok) throw new Error('404'); return r.json(); }})
    .then(function(d){{ renderCoaching(d, container); }})
    .catch(function(){{
      container.innerHTML = '<div class="coach-head">Coach</div>'
        +'<div class="coach-none">No coaching summary available yet.</div>';
    }});
}}

function renderCoaching(d, container) {{
  var html = '<div class="coach-head">Coach</div>';
  if(d.headline) html += '<div class="coach-headline">'+escapeHtml(d.headline)+'</div>';
  if(d.strengths && d.strengths.length) {{
    html += '<div class="coach-section"><div class="coach-section-title">Strengths</div>';
    d.strengths.forEach(function(s){{
      html += '<div class="coach-item"><b>'+escapeHtml(s.point)+':</b> '+escapeHtml(s.detail)+'</div>';
    }});
    html += '</div>';
  }}
  if(d.work_on && d.work_on.length) {{
    html += '<div class="coach-section"><div class="coach-section-title">Work On</div>';
    d.work_on.forEach(function(s){{
      html += '<div class="coach-item"><b>'+escapeHtml(s.point)+':</b> '+escapeHtml(s.detail)+'</div>';
    }});
    html += '</div>';
  }}
  if(d.drill) html += '<div class="coach-drill"><b>Drill:</b> '+escapeHtml(d.drill)+'</div>';
  container.innerHTML = html;
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

function clearFilter() {{
  currentFilter = 'all';
  document.querySelectorAll('.chip').forEach(function(c){{c.classList.remove('active')}});
  var allChip = document.querySelector('.chip[data-filter="all"]');
  if(allChip) allChip.classList.add('active');
  updateActiveFilter();
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
      var thumbHtml = '<div class="card-thumb-wrap"'+thumbAction+'>'+thumbInner+'<span class="card-id">'+v.id+'</span></div>';

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

      var linksHtml = '';
      groupOrder.forEach(function(baseKey) {{
        var g = groups[baseKey];
        var primary = g.normal || g.slow;
        var primaryUrl = 'https://tennis.playfullife.com/'+v.id+'/'+primary.file;
        linksHtml += '<div class="link-row">';
        linksHtml += '<a href="'+primaryUrl+'" class="play-btn" data-title="'+g.label+' \\u2014 '+v.id+'" '
          +'onclick="event.stopPropagation();openPlayer(this.href,this.dataset.title);return false" '
          +'style="background:'+g.color+'">'+g.label+'</a>';
        if(g.normal) {{
          var nUrl = 'https://tennis.playfullife.com/'+v.id+'/'+g.normal.file;
          linksHtml += '<span class="dl-btn" data-action="download" data-url="'+nUrl+'" title="Download">&#8681;</span>';
        }}
        if(g.slow) {{
          var sUrl = 'https://tennis.playfullife.com/'+v.id+'/'+g.slow.file;
          linksHtml += '<a href="'+sUrl+'" class="slow-btn" data-title="'+g.label+' (Slow-Mo) \\u2014 '+v.id+'" '
            +'onclick="event.stopPropagation();openPlayer(this.href,this.dataset.title);return false" '
            +'title="Slow Motion">slow</a>';
          linksHtml += '<span class="dl-btn" data-action="download" data-url="'+sUrl+'" title="Download Slow-Mo">&#8681;</span>';
        }}
        linksHtml += '</div>';
      }});

      html += '<div class="card" onclick="toggleCard(this)">';
      html += thumbHtml;
      html += '<div class="card-body">';
      html += '<div class="card-time">'+(time||v.id)+'</div>';
      html += '<div class="card-meta">';
      if(dur) html += '<span>'+dur+'</span>';
      if(v.shots) html += '<span>'+v.shots+' shots</span>';
      html += '</div>';
      if(bdParts.length) html += '<div class="card-breakdown">'+bdParts.join(', ')+'</div>';
      html += '<div class="card-coach" id="coach-'+v.id+'" data-loaded="0">'
        +'<div class="coach-head">Coach</div>'
        +'<div class="coach-loading">Loading insights...</div></div>';
      html += '<div class="card-links">'+linksHtml
        +'<span class="del-btn" data-action="delete" data-vid="'+v.id+'" title="Delete this video">&#128465;</span>'
        +'</div>';
      html += '</div></div>';
    }});

    html += '</div></div>';
  }});

  document.getElementById('content').innerHTML = html;
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
    'complete':'Complete','failed':'Failed'
  }};

  function loadQueue() {{
    fetch('/api/queue').then(function(r){{return r.json()}}).then(function(d) {{
      var banner = document.getElementById('procBanner');
      if(!d.queue) return;
      var active = d.queue.filter(function(i) {{
        if(i.status==='complete') return Date.now()-new Date(i.updated_at).getTime() < 3600000;
        return i.status!=='not_found';
      }});
      if(active.length===0) {{ banner.style.display='none'; return; }}
      banner.style.display='';
      var html = '<div class="proc-bar"><div class="proc-items">';
      active.forEach(function(item) {{
        var name = (item.filename||'').replace('.MOV','').replace('.mov','').replace('.mp4','');
        var label = stageLabels[item.stage||item.status]||item.status;
        var pct = item.progress ? ' '+item.progress+'%' : '';
        var cls = 'proc-item';
        if(item.status==='complete') cls += ' proc-complete';
        else if(item.status==='failed') cls += ' proc-failed';
        else if(item.status==='pending') cls += ' proc-pending';
        html += '<span class="'+cls+'"><span class="proc-dot"></span>'+name+' <span class="stage">'+label+pct+'</span></span>';
      }});
      html += '</div></div>';
      banner.innerHTML = html;
    }}).catch(function(){{}});
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
  else if(action === 'play') openPlayer(el.dataset.url, el.dataset.title);
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


def update_index():
    """Main: gather metadata, build HTML, upload to R2."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    import importlib, config.settings
    importlib.reload(config.settings)
    from storage.r2_client import R2Client

    c = R2Client()
    keys = c.list(prefix='highlights/', max_keys=1000)

    # Group files by video
    videos = {}
    for k in keys:
        if 'index.html' in k or 'thumbs/' in k:
            continue
        parts = k.split('/')
        if len(parts) == 3:
            fname = parts[2]
            # Skip non-video files (meta.json, etc.)
            if not fname.endswith('.mp4'):
                continue
            videos.setdefault(parts[1], []).append(fname)

    # Gather metadata + ensure thumbnails
    all_meta = {}
    for vid in videos:
        meta = get_video_metadata(vid, r2_client=c)
        meta['files'] = sorted(videos[vid])
        has_thumb = generate_thumbnail(vid)
        if has_thumb:
            upload_thumbnail(c, vid)
        meta['has_thumb'] = has_thumb
        all_meta[vid] = meta

    # Build and upload index
    html = build_index_html(all_meta)
    tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w')
    tmp.write(html)
    tmp.close()

    # Validate JS syntax before uploading (catches f-string escaping bugs)
    try:
        import re as _re
        scripts = _re.findall(r'<script>(.*?)</script>', html, _re.DOTALL)
        if scripts:
            js_tmp = tempfile.NamedTemporaryFile(suffix='.js', delete=False, mode='w')
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

    c.upload(tmp.name, 'highlights/index.html', content_type='text/html')
    c.upload(tmp.name, 'highlights/', content_type='text/html')
    os.unlink(tmp.name)
    print(f'Updated index: {len(all_meta)} videos')
    print('https://tennis.playfullife.com/')


if __name__ == '__main__':
    update_index()
