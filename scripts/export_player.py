#!/usr/bin/env python3
"""Generate a standalone read-only HTML player for sharing tennis analysis.

Creates a self-contained HTML file with embedded detection data that mirrors
the shot_review.py editor UI but without editing capabilities.

Usage:
    python scripts/export_player.py preprocessed/IMG_6878.mp4
    python scripts/export_player.py preprocessed/IMG_6878.mp4 --video-url https://example.com/video.mp4

The video is referenced by URL (default: relative path same directory).
Upload the HTML + video together to any static hosting.
"""

import json
import os
import sys


def load_detections(video_name):
    """Load detections, preferring GT file."""
    for path in [f"detections/{video_name}_fused.json",
                 f"detections/{video_name}_fused_detections.json"]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"  Loaded: {path} ({len(data.get('detections', []))} shots)")
            return data
    print(f"[ERROR] No detection file for {video_name}")
    sys.exit(1)


def get_video_info(video_path):
    """Get duration and fps via ffprobe."""
    import subprocess
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
           '-show_streams', '-show_format', video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    duration = float(info['format']['duration'])
    vs = next(s for s in info['streams'] if s['codec_type'] == 'video')
    fps_parts = vs.get('r_frame_rate', '60/1').split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 60.0
    return duration, fps


def build_viewer_html(video_name, det_data, video_url, duration, fps):
    """Build the standalone viewer HTML."""
    det_json = json.dumps(det_data, separators=(',', ':'))
    detections = det_data.get('detections', [])
    num_shots = len(detections)
    camera = det_data.get('camera_angle', '') or (det_data.get('video_metadata', {}) or {}).get('camera_angle', '')
    session = (det_data.get('video_metadata', {}) or {}).get('session_type', '')
    hand = det_data.get('dominant_hand', 'right')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tennis Analysis — {video_name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0a0a0f;
    color: #f1f1f1;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
}}

/* ── Stats bar ── */
#stats {{
    display: flex;
    gap: 18px;
    padding: 8px 16px;
    background: #12121a;
    border-bottom: 1px solid #2a2a35;
    font-size: 13px;
    flex-wrap: wrap;
    align-items: center;
}}
#stats .stat {{ color: #888; }}
#stats .stat b {{ color: #f1f1f1; margin-right: 2px; }}
.pill {{
    display: inline-block;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
}}
.pill-serve    {{ background: rgba(255,100,0,.25); color: #FF6400; }}
.pill-forehand {{ background: rgba(0,200,0,.20);   color: #00C800; }}
.pill-backhand {{ background: rgba(0,100,255,.25); color: #0064FF; }}
.pill-unknown  {{ background: rgba(180,180,180,.18); color: #B4B4B4; }}
.pill-practice {{ background: rgba(200,120,255,.20); color: #C878FF; }}
.pill-offscreen {{ background: rgba(120,120,120,.18); color: #777; }}

.opp-badge {{
    display: inline-block;
    font-size: 9px;
    font-weight: 700;
    color: #f59e0b;
    background: rgba(245,158,11,.15);
    padding: 0 4px;
    border-radius: 3px;
    margin-left: 4px;
    vertical-align: middle;
}}

/* ── Meta bar ── */
#meta-bar {{
    display: flex;
    gap: 12px;
    padding: 5px 16px;
    background: #12121a;
    border-bottom: 1px solid #2a2a35;
    font-size: 12px;
    align-items: center;
    color: #888;
}}
#meta-bar span {{ color: #ccc; font-weight: 600; }}

/* ── Video ── */
#video-wrap {{
    flex: 1 1 auto;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #000;
    min-height: 0;
    position: relative;
}}
video {{
    max-width: 100%;
    max-height: 100%;
    display: block;
}}

/* ── Scrub bar ── */
#scrub-wrap {{
    padding: 6px 12px;
    background: #12121a;
}}
#scrub {{
    position: relative;
    height: 18px;
    cursor: pointer;
}}
#scrub-track {{
    position: absolute;
    top: 7px;
    left: 0;
    right: 0;
    height: 4px;
    background: #2a2a35;
    border-radius: 2px;
}}
#scrub-fill {{
    height: 100%;
    width: 0%;
    background: #6366f1;
    border-radius: 2px;
}}
#scrub-handle {{
    position: absolute;
    top: 3px;
    left: 0%;
    width: 12px;
    height: 12px;
    background: #6366f1;
    border-radius: 50%;
    transform: translateX(-50%);
}}

/* ── Controls ── */
#controls {{
    display: flex;
    gap: 10px;
    padding: 5px 12px;
    background: #12121a;
    align-items: center;
    border-top: 1px solid #2a2a35;
}}
#controls button {{
    background: #2a2a35;
    color: #f1f1f1;
    border: none;
    padding: 3px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
}}
#controls button:hover {{ background: #3a3a45; }}
#play-btn {{ font-size: 16px; padding: 3px 12px; }}
#time-display {{ color: #aaa; font-size: 12px; font-family: monospace; min-width: 120px; }}
.speed-group {{ display: flex; gap: 3px; }}
.speed-group button.active {{ background: #6366f1; color: #fff; }}

/* ── Info bar ── */
#info-bar {{
    background: #1a1a2e;
    color: #888;
    font-size: 0.8em;
    padding: 4px 12px;
    font-family: monospace;
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}}

/* ── Minimap ── */
#minimap-wrap {{
    padding: 2px 12px;
    background: #12121a;
}}
#minimap {{
    position: relative;
    height: 8px;
    background: #1a1a2e;
    border-radius: 2px;
    overflow: hidden;
}}
#minimap-progress {{
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 0%;
    background: rgba(99,102,241,.3);
    pointer-events: none;
}}
#minimap-viewport {{
    position: absolute;
    top: 0;
    height: 100%;
    border: 1px solid rgba(99,102,241,.5);
    border-radius: 2px;
    background: rgba(99,102,241,.08);
    pointer-events: none;
}}
.mini-marker {{
    position: absolute;
    top: 0;
    width: 2px;
    height: 100%;
    pointer-events: none;
}}

/* ── Timeline ── */
#timeline-wrap {{
    display: flex;
    align-items: center;
    padding: 2px 4px;
    background: #12121a;
    gap: 2px;
}}
.zoom-btn {{
    background: #2a2a35;
    color: #aaa;
    border: none;
    width: 24px;
    height: 24px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    line-height: 1;
}}
.zoom-btn:hover {{ background: #3a3a45; color: #fff; }}
#timeline {{
    flex: 1;
    height: 36px;
    background: #1a1a2e;
    position: relative;
    border-radius: 4px;
    overflow: hidden;
    cursor: grab;
}}
#timeline-progress {{
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 0%;
    background: rgba(99,102,241,.12);
    pointer-events: none;
}}
.marker {{
    position: absolute;
    top: 4px;
    height: 28px;
    min-width: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    color: #fff;
    cursor: pointer;
    transition: transform .12s, box-shadow .12s;
    z-index: 1;
    user-select: none;
}}
.marker:hover {{ transform: scale(1.15); z-index: 5; }}
.marker.active {{
    transform: scale(1.2);
    box-shadow: 0 0 8px rgba(255,255,255,.4);
    z-index: 10;
}}
.marker.opponent {{ opacity: 0.5; }}
.marker-serve    {{ background: #FF6400; }}
.marker-forehand {{ background: #00C800; }}
.marker-backhand {{ background: #0064FF; }}
.marker-unknown  {{ background: #666; }}
.marker-practice {{ background: #C878FF; }}
.marker-offscreen {{ background: #555; }}

#zoom-label {{
    position: absolute;
    right: 6px;
    top: 2px;
    font-size: 9px;
    color: #555;
    pointer-events: none;
}}

/* ── Video loading overlay ── */
#video-loading {{
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: rgba(0,0,0,.85);
    z-index: 10;
    gap: 12px;
}}
#video-loading.hidden {{ display: none; }}
#load-bar-wrap {{
    width: 280px;
    height: 6px;
    background: #2a2a35;
    border-radius: 3px;
    overflow: hidden;
}}
#load-bar {{
    height: 100%;
    width: 0%;
    background: #6366f1;
    border-radius: 3px;
    transition: width .15s;
}}
#load-text {{
    color: #888;
    font-size: 13px;
}}
</style>
</head>
<body>

<div id="stats"></div>

<div id="meta-bar">
    <span>{video_name}</span>
    <span style="color:#888;">Camera: {camera or 'unset'}</span>
    <span style="color:#888;">Session: {session or 'unset'}</span>
    <span style="color:#888;">Hand: {hand}</span>
    <span style="color:#666;">{fps:.0f} fps</span>
    <span style="color:#666;">{num_shots} shots</span>
    <span style="color:#666;">{duration:.1f}s</span>
</div>

<div id="video-wrap">
    <div id="video-loading">
        <div id="load-bar-wrap"><div id="load-bar"></div></div>
        <div id="load-text">Loading video...</div>
    </div>
    <video id="player" preload="none"></video>
</div>

<div id="scrub-wrap">
    <div id="scrub">
        <div id="scrub-track"><div id="scrub-fill"></div></div>
        <div id="scrub-handle"></div>
    </div>
</div>

<div id="controls">
    <button id="play-btn" title="Play/Pause (Space)">&#9654;</button>
    <span id="time-display">0:00.0 / 0:00</span>
    <div class="speed-group">
        <button data-speed="0.25">0.25x</button>
        <button data-speed="0.5">0.5x</button>
        <button data-speed="1" class="active">1x</button>
        <button data-speed="2">2x</button>
        <button data-speed="4">4x</button>
        <button data-speed="8">8x</button>
    </div>
</div>

<div id="minimap-wrap">
    <div id="minimap">
        <div id="minimap-progress"></div>
        <div id="minimap-viewport"></div>
    </div>
</div>

<div id="timeline-wrap">
    <button class="zoom-btn" id="zoom-out" title="Zoom out">&minus;</button>
    <div id="timeline">
        <div id="timeline-progress"></div>
        <span id="zoom-label"></span>
    </div>
    <button class="zoom-btn" id="zoom-in" title="Zoom in">+</button>
</div>


<script>
const DETECTION_DATA = {det_json};

const player    = document.getElementById('player');
const timeline  = document.getElementById('timeline');
const progress  = document.getElementById('timeline-progress');
const statsEl   = document.getElementById('stats');
const timDisp   = document.getElementById('time-display');
const playBtn   = document.getElementById('play-btn');
const scrub     = document.getElementById('scrub');
const scrubFill = document.getElementById('scrub-fill');
const scrubHdl  = document.getElementById('scrub-handle');
const minimap      = document.getElementById('minimap');
const minimapVP    = document.getElementById('minimap-viewport');
const minimapProg  = document.getElementById('minimap-progress');
const zoomLabel    = document.getElementById('zoom-label');

let detections = DETECTION_DATA.detections || [];
let activeIdx  = -1;
let videoDur   = DETECTION_DATA.duration || {duration};
let videoReady = false;

// ── Pre-download entire video as blob for reliable seeking ──
const VIDEO_SRC = '{video_url}';
(async function preloadVideo() {{
    const loadEl = document.getElementById('video-loading');
    const loadBar = document.getElementById('load-bar');
    const loadText = document.getElementById('load-text');
    try {{
        const resp = await fetch(VIDEO_SRC);
        const total = parseInt(resp.headers.get('Content-Length') || '0', 10);
        const reader = resp.body.getReader();
        const chunks = [];
        let received = 0;
        while (true) {{
            const {{done, value}} = await reader.read();
            if (done) break;
            chunks.push(value);
            received += value.length;
            if (total > 0) {{
                const pct = Math.min(100, received / total * 100);
                loadBar.style.width = pct.toFixed(1) + '%';
                const mb = (received / 1048576).toFixed(0);
                const totalMb = (total / 1048576).toFixed(0);
                loadText.textContent = 'Loading video... ' + mb + ' / ' + totalMb + ' MB';
            }}
        }}
        const blob = new Blob(chunks, {{type: 'video/mp4'}});
        player.src = URL.createObjectURL(blob);
        player.load();
        videoReady = true;
        loadEl.classList.add('hidden');
    }} catch (e) {{
        loadText.textContent = 'Blob download failed, using stream...';
        player.src = VIDEO_SRC;
        player.load();
        videoReady = true;
        setTimeout(() => loadEl.classList.add('hidden'), 1000);
    }}
}})()

// ── Zoom state ──
let zoomLevel = 1;
let viewStart = 0;
let viewEnd   = 1;
const MIN_ZOOM = 1;
const MAX_ZOOM = 30;

const TYPE_LABELS = {{
    serve: 'S', forehand: 'FH', backhand: 'BH',
    forehand_volley: 'FV', unknown_shot: '?', neutral: 'N', practice: 'P', offscreen: 'X'
}};
const TYPE_CLASS = {{
    serve: 'serve', forehand: 'forehand', backhand: 'backhand',
    forehand_volley: 'forehand', unknown_shot: 'unknown', neutral: 'unknown', practice: 'practice', offscreen: 'offscreen'
}};

function fmtTime(s) {{
    const m = Math.floor(s / 60);
    const sec = s - m * 60;
    return m + ':' + sec.toFixed(1).padStart(4, '0');
}}
function fmtTimeShort(s) {{
    const m = Math.floor(s / 60);
    const sec = Math.floor(s - m * 60);
    return m + ':' + String(sec).padStart(2, '0');
}}

// ── Duration sync ──
player.addEventListener('loadedmetadata', () => {{
    const realDur = player.duration;
    if (realDur && isFinite(realDur) && Math.abs(realDur - videoDur) > 2) {{
        videoDur = realDur;
        renderTimeline(videoDur);
    }}
}});

// ── Play/Pause ──
playBtn.addEventListener('click', () => {{
    if (!videoReady) return;
    player.paused ? player.play() : player.pause();
}});
player.addEventListener('play',  () => {{ playBtn.innerHTML = '&#9646;&#9646;'; }});
player.addEventListener('pause', () => {{ playBtn.innerHTML = '&#9654;'; }});

// ── Speed buttons ──
document.querySelectorAll('[data-speed]').forEach(btn => {{
    btn.addEventListener('click', () => {{
        player.playbackRate = parseFloat(btn.dataset.speed);
        document.querySelectorAll('[data-speed]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }});
}});

// ── Scrub bar ──
let scrubbing = false;
function scrubSeek(e) {{
    const rect = scrub.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    player.currentTime = pct * (player.duration || videoDur);
}}
scrub.addEventListener('mousedown', (e) => {{ scrubbing = true; scrubSeek(e); }});
document.addEventListener('mousemove', (e) => {{ if (scrubbing) scrubSeek(e); }});
document.addEventListener('mouseup', () => {{ scrubbing = false; }});

// ── Stats ──
function renderStats() {{
    const counts = {{}};
    const tiers = {{}};
    detections.forEach(d => {{
        counts[d.shot_type] = (counts[d.shot_type] || 0) + 1;
        tiers[d.tier] = (tiers[d.tier] || 0) + 1;
    }});
    const fh = counts.forehand || 0;
    const bh = counts.backhand || 0;
    const ratio = bh > 0 ? (fh / bh).toFixed(1) : (fh > 0 ? '∞' : '–');
    statsEl.innerHTML = `
        <span class="stat"><b>${{detections.length}}</b> shots</span>
        <span class="pill pill-serve">S ${{counts.serve || 0}}</span>
        <span class="pill pill-forehand">FH ${{fh}}</span>
        <span class="pill pill-backhand">BH ${{bh}}</span>
        <span class="pill pill-unknown">? ${{counts.unknown_shot || 0}}</span>
        <span class="pill pill-practice">P ${{counts.practice || 0}}</span>
        <span class="pill pill-offscreen">X ${{counts.offscreen || 0}}</span>
        <span class="stat">FH:BH <b>${{ratio}}</b></span>
        <span class="stat">HIGH <b>${{tiers.high || 0}}</b></span>
        <span class="stat">MED <b>${{tiers.medium || 0}}</b></span>
        <span class="stat">LOW <b>${{tiers.low || 0}}</b></span>
    `;
}}

// ── Minimap ──
function renderMinimap(dur) {{
    minimap.querySelectorAll('.mini-marker').forEach(m => m.remove());
    detections.forEach(d => {{
        const el = document.createElement('div');
        el.className = 'mini-marker';
        const cls = TYPE_CLASS[d.shot_type] || 'unknown';
        const colors = {{serve:'#FF6400',forehand:'#00C800',backhand:'#0064FF',unknown:'#666',practice:'#C878FF',offscreen:'#555'}};
        el.style.left = ((d.timestamp / dur) * 100) + '%';
        el.style.background = colors[cls] || '#666';
        if (d.opponent) el.style.opacity = '0.4';
        minimap.appendChild(el);
    }});
}}

// ── Timeline ──
function renderTimeline(dur) {{
    timeline.querySelectorAll('.marker').forEach(m => m.remove());
    const range = viewEnd - viewStart;
    detections.forEach((d, i) => {{
        const frac = d.timestamp / dur;
        const visFrac = (frac - viewStart) / range;
        if (visFrac < -0.05 || visFrac > 1.05) return;
        const el = document.createElement('div');
        const cls = TYPE_CLASS[d.shot_type] || 'unknown';
        el.className = 'marker marker-' + cls + (i === activeIdx ? ' active' : '') + (d.opponent ? ' opponent' : '');
        el.style.left = 'calc(' + (visFrac * 100) + '% - 11px)';
        el.textContent = TYPE_LABELS[d.shot_type] || '?';
        el.addEventListener('click', () => jumpTo(i));
        timeline.appendChild(el);
    }});
    zoomLabel.textContent = zoomLevel > 1 ? zoomLevel.toFixed(1) + 'x' : '';
    // Update minimap viewport
    minimapVP.style.left = (viewStart * 100) + '%';
    minimapVP.style.width = (range * 100) + '%';
}}

// ── Zoom ──
function setZoom(newZoom, anchorFrac) {{
    newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));
    if (anchorFrac === undefined) anchorFrac = (viewStart + viewEnd) / 2;
    const range = 1 / newZoom;
    let newStart = anchorFrac - range * ((anchorFrac - viewStart) / (viewEnd - viewStart));
    newStart = Math.max(0, Math.min(1 - range, newStart));
    zoomLevel = newZoom;
    viewStart = newStart;
    viewEnd = newStart + range;
    renderTimeline(videoDur);
}}
document.getElementById('zoom-in').addEventListener('click', () => setZoom(zoomLevel * 1.5));
document.getElementById('zoom-out').addEventListener('click', () => setZoom(zoomLevel / 1.5));

// ── Pan ──
let panning = false, panStartX = 0, panStartVS = 0;
timeline.addEventListener('mousedown', (e) => {{
    if (e.target.classList.contains('marker')) return;
    panning = true;
    panStartX = e.clientX;
    panStartVS = viewStart;
    timeline.style.cursor = 'grabbing';
}});
document.addEventListener('mousemove', (e) => {{
    if (!panning) return;
    const dx = e.clientX - panStartX;
    const range = viewEnd - viewStart;
    const pxWidth = timeline.getBoundingClientRect().width;
    const shift = -(dx / pxWidth) * range;
    let ns = Math.max(0, Math.min(1 - range, panStartVS + shift));
    viewStart = ns;
    viewEnd = ns + range;
    renderTimeline(videoDur);
}});
document.addEventListener('mouseup', () => {{
    panning = false;
    timeline.style.cursor = 'grab';
}});

function jumpTo(idx) {{
    if (idx < 0 || idx >= detections.length) return;
    activeIdx = idx;
    player.currentTime = detections[idx].timestamp;
    // Auto-pan to show active marker
    const frac = detections[idx].timestamp / videoDur;
    if (frac < viewStart || frac > viewEnd) {{
        const range = viewEnd - viewStart;
        viewStart = Math.max(0, frac - range * 0.3);
        viewEnd = viewStart + range;
    }}
    renderTimeline(videoDur);
}}

// ── Playback update loop ──
function updateLoop() {{
    if (player.duration && isFinite(player.duration)) {{
        const t = player.currentTime;
        const dur = player.duration || videoDur;
        const pct = (t / dur) * 100;
        // Scrub bar
        scrubFill.style.width = pct + '%';
        scrubHdl.style.left = pct + '%';
        // Time display
        const rate = player.playbackRate;
        const rateStr = rate !== 1 ? ' @' + rate + 'x' : '';
        timDisp.textContent = fmtTime(t) + ' / ' + fmtTimeShort(dur) + rateStr;
        // Timeline progress
        const range = viewEnd - viewStart;
        const visPct = ((t / dur - viewStart) / range) * 100;
        progress.style.width = Math.max(0, Math.min(100, visPct)) + '%';
        // Minimap progress
        minimapProg.style.width = pct + '%';
    }}
    requestAnimationFrame(updateLoop);
}}
requestAnimationFrame(updateLoop);

// ── Keyboard shortcuts ──
document.addEventListener('keydown', (e) => {{
    if (e.key === ' ') {{
        e.preventDefault();
        player.paused ? player.play() : player.pause();
    }}
    if (e.key === 'ArrowRight') {{
        e.preventDefault();
        if (e.shiftKey) {{ player.currentTime = Math.min(player.duration, player.currentTime + 5); }}
        else {{
            // Next shot
            const t = player.currentTime;
            const next = detections.findIndex(d => d.timestamp > t + 0.1);
            if (next >= 0) jumpTo(next);
        }}
    }}
    if (e.key === 'ArrowLeft') {{
        e.preventDefault();
        if (e.shiftKey) {{ player.currentTime = Math.max(0, player.currentTime - 5); }}
        else {{
            // Previous shot
            const t = player.currentTime;
            let prev = -1;
            for (let i = detections.length - 1; i >= 0; i--) {{
                if (detections[i].timestamp < t - 0.5) {{ prev = i; break; }}
            }}
            if (prev >= 0) jumpTo(prev);
        }}
    }}
    // Zoom
    if (e.key === '=' || e.key === '+') setZoom(zoomLevel * 1.5);
    if (e.key === '-') setZoom(zoomLevel / 1.5);
}});

// ── Initialize ──
renderStats();
renderMinimap(videoDur);
renderTimeline(videoDur);
</script>
</body>
</html>"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate standalone tennis analysis HTML player')
    parser.add_argument('video', help='Path to preprocessed video')
    parser.add_argument('--video-url', default=None,
                        help='URL for the video in the HTML (default: same filename)')
    args = parser.parse_args()

    video_path = args.video
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"Export Player: {video_name}")

    det_data = load_detections(video_name)
    duration, fps = get_video_info(video_path)
    print(f"  Duration: {duration:.1f}s, FPS: {fps:.0f}")

    video_url = args.video_url or os.path.basename(video_path)

    html = build_viewer_html(video_name, det_data, video_url, duration, fps)

    os.makedirs(f"exports/{video_name}", exist_ok=True)
    output_path = f"exports/{video_name}/{video_name}_player.html"
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"  Saved: {output_path}")
    print(f"  Video ref: {video_url}")
    print(f"\n  To view locally: open {output_path} (with video in same dir)")
    print(f"  To share: upload HTML + video to same directory on your host")


if __name__ == '__main__':
    main()
