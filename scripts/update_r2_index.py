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
        f'https://media.playfullife.com/thumbs/{vid}.jpg',
        f'https://media.playfullife.com/highlights/thumbs/{vid}.jpg',
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

    # Try generating from local preprocessed video
    for pp in [
        os.path.join(PROJECT_ROOT, 'preprocessed', f'{vid}.mp4'),
        os.path.join(PROJECT_ROOT, 'preprocessed', f'{vid}_240fps.mp4'),
    ]:
        if os.path.exists(pp):
            subprocess.run(
                ['ffmpeg', '-y', '-ss', '5', '-i', pp, '-vframes', '1',
                 '-q:v', '8', '-vf', 'scale=480:-1', thumb],
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


def get_video_metadata(vid):
    """Gather metadata for a video from detection JSON and raw MOV."""
    info = {}

    # Detection JSON
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
            # Check for embedded creation date in detection JSON
            if d.get('created'):
                info['created'] = d['created']
            break

    # Creation date from raw MOV
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

    # Fallback: preprocessed file mod time
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
        'grouped': ('By Shot Type', '#5DADE2'),
        'grouped_slowmo': ('By Shot Type Slow-Mo', '#3498DB'),
        'highlights': ('Highlights', '#2ECC71'),
        'highlights_slowmo': ('Highlights Slow-Mo', '#8E44AD'),
        'comparisons': ('Pro Compare', '#E74C3C'),
    }
    link_order = [
        'timeline', 'rally', 'rally_slowmo',
        'grouped', 'grouped_slowmo',
        'highlights', 'highlights_slowmo',
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

        links = []
        for key in link_order:
            if key not in file_keys:
                continue
            f = file_keys[key]
            label, color = label_map.get(key, (key, '#5DADE2'))
            links.append({'key': key, 'file': f, 'label': label, 'color': color})
        for key, f in file_keys.items():
            if key not in link_order:
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
.filters{{max-width:1100px;margin:0 auto;padding:10px 20px;display:flex;gap:6px;flex-wrap:wrap}}
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
.session-date{{font-size:1.1em;font-weight:700;color:#fff}}
.session-stats{{font-size:0.8em;color:#666}}

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
.card-links{{display:none;flex-wrap:wrap;gap:4px;margin-top:8px;padding-top:8px;border-top:1px solid #222}}
.card.expanded .card-links{{display:flex}}
.card-links a{{color:#fff;text-decoration:none;font-size:0.72em;font-weight:600;
  padding:4px 10px;border-radius:5px;opacity:.9;transition:opacity .15s}}
.card-links a:hover{{opacity:1}}

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
<div class="filters" id="filters">
  <span class="chip active" data-filter="all">All</span>
  <span class="chip" data-filter="serve">Serves</span>
  <span class="chip" data-filter="forehand">Forehands</span>
  <span class="chip" data-filter="backhand">Backhands</span>
  <span class="chip" data-filter="recent">This Week</span>
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
        <button onclick="stepFrame(-1)">&larr;</button>
        <button onclick="stepFrame(1)">&rarr;</button>
      </div>
      <span class="time-display" id="timeDisplay">0:00.0</span>
      <button onclick="copyTimeLink()" class="share-btn" id="shareBtn">Copy link at time</button>
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
  history.replaceState(null,'','?v='+encodeURIComponent(url.replace('https://media.playfullife.com/','')));
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
function stepFrame(dir) {{ vid.pause(); vid.currentTime = Math.max(0, vid.currentTime + dir/60); }}

function copyTimeLink() {{
  var vKey = vid.src.replace('https://media.playfullife.com/','');
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
    openPlayer('https://media.playfullife.com/'+v,v.split('/').pop().replace('.mp4','').replace(/_/g,' '));
  }}
}})();

function toggleCard(el) {{ el.classList.toggle('expanded'); }}

// ── Rendering ──
var currentFilter = 'all';
var searchQuery = '';

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
  if(currentFilter==='recent') {{
    var d = parseDate(v.created);
    return d && (Date.now()-d.getTime()) < 7*86400000;
  }}
  return (v.breakdown[currentFilter]||0) > 0;
}}

function matchesSearch(v) {{
  if(!searchQuery) return true;
  var q = searchQuery.toLowerCase();
  if(v.id.toLowerCase().includes(q)) return true;
  if((v.created||'').toLowerCase().includes(q)) return true;
  var dk = formatSessionDate(dateKey(v.created)).toLowerCase();
  if(dk.includes(q)) return true;
  return false;
}}

function renderGallery() {{
  var filtered = VIDEOS.filter(function(v) {{ return matchesFilter(v) && matchesSearch(v); }});
  filtered.sort(function(a,b) {{ return (b.created||'').localeCompare(a.created||''); }});

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

    html += '<div class="session">';
    html += '<div class="session-header">';
    html += '<span class="session-date">'+formatSessionDate(dk)+'</span>';
    html += '<span class="session-stats">'+vids.length+' video'+(vids.length>1?'s':'')+' / '+sessionShots+' shots</span>';
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
        thumbInner = '<img class="card-thumb" src="https://media.playfullife.com/thumbs/'+v.id+'.jpg" alt="'+v.id+'" loading="lazy">';
      }} else {{
        thumbInner = '<div class="card-thumb-placeholder">'+v.id+'</div>';
      }}
      var thumbHtml = '<div class="card-thumb-wrap">'+thumbInner+'<span class="card-id">'+v.id+'</span></div>';

      var linksHtml = '';
      v.links.forEach(function(lk) {{
        var url = 'https://media.playfullife.com/'+v.id+'/'+lk.file;
        linksHtml += '<a href="'+url+'" data-title="'+lk.label+' \\u2014 '+v.id+'" '
          +'onclick="event.stopPropagation();openPlayer(this.href,this.dataset.title);return false" '
          +'style="background:'+lk.color+'">'+lk.label+'</a>';
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
      html += '<div class="card-links">'+linksHtml+'</div>';
      html += '</div></div>';
    }});

    html += '</div></div>';
  }});

  document.getElementById('content').innerHTML = html;
}}

// ── Filters ──
document.getElementById('filters').addEventListener('click', function(e) {{
  var chip = e.target.closest('.chip');
  if(!chip) return;
  currentFilter = chip.dataset.filter;
  document.querySelectorAll('.chip').forEach(function(c){{c.classList.remove('active')}});
  chip.classList.add('active');
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

// ── Init ──
renderGallery();
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
            videos.setdefault(parts[1], []).append(parts[2])

    # Gather metadata + ensure thumbnails
    all_meta = {}
    for vid in videos:
        meta = get_video_metadata(vid)
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
    print('https://media.playfullife.com/')


if __name__ == '__main__':
    update_index()
