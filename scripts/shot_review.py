#!/usr/bin/env python3
"""Interactive shot review player — serves a video + detection timeline in the browser."""

import argparse
import json
import os
import re
import signal
import socketserver
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote


def find_detections(video_path: str) -> str | None:
    """Auto-discover detection JSON for a given video file.

    Prefers the newest versioned file (e.g. _fused_v5.json > _fused_v3.json > _fused.json).
    """
    stem = Path(video_path).stem
    search_dirs = [
        Path(video_path).parent.parent / "detections",
        Path(video_path).parent,
        Path("detections"),
        Path("."),
    ]
    # Collect all matching files across search dirs
    import glob as _glob
    all_matches = []
    for d in search_dirs:
        all_matches.extend(d.glob(f"{stem}_fused*.json"))
    if not all_matches:
        return None
    # Pick the most recently modified file
    best = max(all_matches, key=lambda p: p.stat().st_mtime)
    return str(best.resolve())


def build_html(video_filename: str, duration: float, video_url: str = "/video",
               video_path: str = "", det_path: str = "", fps: float = 0,
               num_shots: int = 0) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Shot Review — {video_filename}</title>
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

/* ── Opponent indicator ── */
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
.marker.opponent {{ opacity: 0.5; }}

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

/* ── Scrub bar ── */
#scrub-wrap {{
    padding: 0 16px;
    background: #12121a;
    border-top: 1px solid #2a2a35;
    user-select: none;
}}
#scrub {{
    position: relative;
    height: 18px;
    cursor: pointer;
    display: flex;
    align-items: center;
}}
#scrub-track {{
    position: absolute;
    top: 7px;
    left: 0; right: 0;
    height: 4px;
    background: #2a2a35;
    border-radius: 2px;
}}
#scrub-fill {{
    height: 100%;
    background: #6366f1;
    border-radius: 2px;
    width: 0%;
    pointer-events: none;
}}
#scrub-handle {{
    position: absolute;
    top: 3px;
    width: 12px;
    height: 12px;
    background: #f1f1f1;
    border-radius: 50%;
    transform: translateX(-50%);
    left: 0%;
    pointer-events: none;
    box-shadow: 0 0 4px rgba(0,0,0,.5);
}}

/* ── Controls bar ── */
#controls {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 16px;
    background: #12121a;
    font-size: 13px;
    user-select: none;
}}
#controls button {{
    background: none;
    border: 1px solid #333;
    color: #f1f1f1;
    padding: 4px 10px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    font-family: inherit;
    transition: background .12s, border-color .12s;
}}
#controls button:hover {{ background: #22222e; border-color: #555; }}
#controls button.active {{ background: #6366f1; border-color: #6366f1; }}
#play-btn {{ min-width: 32px; font-size: 16px; padding: 4px 8px; }}
#time-display {{
    font-variant-numeric: tabular-nums;
    color: #ccc;
    font-size: 13px;
    min-width: 110px;
}}
.speed-group {{ display: flex; gap: 2px; }}
.speed-group button {{ font-size: 11px; padding: 3px 7px; }}

/* ── Minimap (full timeline overview) ── */
#minimap-wrap {{
    padding: 4px 16px 0;
    background: #12121a;
    user-select: none;
}}
#minimap {{
    position: relative;
    height: 14px;
    background: #1a1a24;
    border-radius: 3px;
    cursor: pointer;
}}
#minimap .mini-marker {{
    position: absolute;
    top: 2px;
    width: 4px;
    height: 10px;
    transform: translateX(-50%);
    border-radius: 1px;
    pointer-events: none;
}}
#minimap .mini-marker.mm-serve    {{ background: #FF6400; }}
#minimap .mini-marker.mm-forehand {{ background: #00C800; }}
#minimap .mini-marker.mm-backhand {{ background: #0064FF; }}
#minimap .mini-marker.mm-unknown  {{ background: #666; }}
#minimap .mini-marker.mm-practice {{ background: #C878FF; }}
#minimap .mini-marker.mm-offscreen {{ background: #555; }}
#minimap-viewport {{
    position: absolute;
    top: 0; bottom: 0;
    background: rgba(99,102,241,.25);
    border: 1px solid rgba(99,102,241,.6);
    border-radius: 3px;
    pointer-events: none;
}}
#minimap-progress {{
    position: absolute;
    top: 0; left: 0; bottom: 0; width: 0%;
    background: rgba(99,102,241,.45);
    border-radius: 3px 0 0 3px;
    pointer-events: none;
}}

/* ── Shot marker timeline (zoomable) ── */
#timeline-wrap {{
    padding: 2px 16px 6px;
    background: #12121a;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.zoom-btn {{
    background: #2a2a3a;
    color: #ccc;
    border: 1px solid #444;
    border-radius: 4px;
    font-size: 16px;
    font-weight: bold;
    width: 28px;
    height: 28px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}
.zoom-btn:hover {{ background: #3a3a4a; }}
.zoom-btn:active {{ background: #4a4a5a; }}
#timeline {{
    position: relative;
    height: 32px;
    background: #1a1a24;
    border-radius: 4px;
    cursor: pointer;
    overflow: hidden;
    flex: 1;
}}
#timeline-progress {{
    position: absolute;
    top: 0; left: 0; bottom: 0;
    background: rgba(255,120,0,.25);
    border-right: 2px solid #ff8800;
    border-radius: 4px 0 0 4px;
    pointer-events: none;
}}
#zoom-label {{
    position: absolute;
    top: 1px; right: 4px;
    font-size: 9px;
    color: #555;
    pointer-events: none;
    z-index: 5;
}}
.marker {{
    position: absolute;
    top: 1px;
    width: 28px;
    height: 30px;
    transform: translateX(-50%);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    color: #fff;
    border-radius: 4px;
    z-index: 2;
    transition: filter .15s;
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    border: 1px solid rgba(0,0,0,0.25);
}}
.marker:hover {{ filter: brightness(1.3); z-index: 3; }}
.marker.active {{ box-shadow: 0 0 0 4px #ff6600, 0 0 20px 8px rgba(255,102,0,0.8); z-index: 4; transform: translateX(-50%) scale(1.25); }}
.marker-serve    {{ background: #FF6400; }}
.marker-forehand {{ background: #00C800; }}
.marker-backhand {{ background: #0064FF; }}
.marker-unknown  {{ background: #666; }}
.marker-practice {{ background: #C878FF; }}
.marker-offscreen {{ background: #555; }}

/* ── Confidence color coding ── */
.marker.low-conf {{ border: 2px solid #fbbf24; }}
.marker.very-low-conf {{ border: 2px solid #ef4444; }}
.conf-warn {{ color: #fbbf24; font-size: 10px; margin-left: 4px; }}
.conf-danger {{ color: #ef4444; font-size: 10px; margin-left: 4px; }}
.ml-info {{ color: #888; font-size: 11px; }}

/* ── Shot list ── */
#shot-list-wrap {{
    flex: 0 0 auto;
    max-height: 30vh;
    overflow-y: auto;
    background: #0a0a0f;
    border-top: 1px solid #2a2a35;
}}
#shot-list-wrap::-webkit-scrollbar {{ width: 6px; }}
#shot-list-wrap::-webkit-scrollbar-thumb {{ background: #333; border-radius: 3px; }}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
thead th {{
    position: sticky;
    top: 0;
    background: #12121a;
    padding: 6px 12px;
    text-align: left;
    color: #888;
    font-weight: 500;
    border-bottom: 1px solid #2a2a35;
    z-index: 1;
}}
tbody tr {{
    cursor: pointer;
    transition: background .12s;
}}
tbody tr:hover {{ background: #1a1a24; }}
tbody tr.active {{ background: #22222e; }}
tbody td {{
    padding: 5px 12px;
    border-bottom: 1px solid rgba(42,42,53,.5);
    white-space: nowrap;
}}
td.type-cell {{
    font-weight: 600;
}}
.type-serve    {{ color: #FF6400; }}
.type-forehand {{ color: #00C800; }}
.type-backhand {{ color: #0064FF; }}
.type-unknown  {{ color: #B4B4B4; }}
.type-practice {{ color: #C878FF; }}
.type-offscreen {{ color: #777; }}

.tier-high   {{ color: #22c55e; }}
.tier-medium {{ color: #f59e0b; }}
.tier-low    {{ color: #888; }}

/* ── Type selector dropdown ── */
select.type-select {{
    background: #1a1a24;
    color: inherit;
    border: 1px solid #333;
    border-radius: 3px;
    padding: 2px 4px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
}}
select.type-select:focus {{ outline: 1px solid #6366f1; }}

/* ── Delete button ── */
.del-btn {{
    background: none;
    border: 1px solid #333;
    color: #666;
    width: 22px;
    height: 22px;
    border-radius: 3px;
    cursor: pointer;
    font-size: 12px;
    line-height: 1;
    transition: all .12s;
}}
.del-btn:hover {{ background: #ef4444; border-color: #ef4444; color: #fff; }}

/* ── Nudge buttons ── */
.time-cell {{
    display: flex;
    align-items: center;
    gap: 3px;
    font-variant-numeric: tabular-nums;
}}
.nudge-btn {{
    background: none;
    border: 1px solid transparent;
    color: #555;
    width: 16px;
    height: 18px;
    border-radius: 2px;
    cursor: pointer;
    font-size: 14px;
    line-height: 1;
    padding: 0;
    transition: all .12s;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.nudge-btn:hover {{ background: #22222e; border-color: #555; color: #f1f1f1; }}
tr.active .nudge-btn {{ color: #888; }}

/* ── Save indicator ── */
#save-status {{
    font-size: 11px;
    color: #555;
    margin-left: auto;
    transition: color .3s;
}}
#save-status.saving {{ color: #f59e0b; }}
#save-status.saved  {{ color: #22c55e; }}
#save-status.error  {{ color: #ef4444; }}

/* ── Video metadata bar ── */
#meta-bar {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 16px;
    background: #12121a;
    border-bottom: 1px solid #2a2a35;
    font-size: 12px;
    color: #888;
}}
#meta-bar label {{ color: #666; }}
#meta-bar select {{
    background: #1a1a24;
    color: #f1f1f1;
    border: 1px solid #333;
    border-radius: 3px;
    padding: 2px 6px;
    font-size: 12px;
    font-family: inherit;
    cursor: pointer;
}}
#meta-bar select:focus {{ outline: 1px solid #6366f1; }}

/* ── Help toast ── */
#help {{
    position: fixed;
    bottom: 8px;
    left: 12px;
    font-size: 11px;
    color: #444;
}}
</style>
</head>
<body>

<div id="stats"></div>

<div id="meta-bar">
    <label>Camera:</label>
    <select id="camera-angle">
        <option value="">unset</option>
        <option value="back_court">back court (behind)</option>
        <option value="front_left">front left (ad side)</option>
        <option value="front_right">front right (deuce side)</option>
        <option value="side_left">side left</option>
        <option value="side_right">side right</option>
    </select>
    <label>Session:</label>
    <select id="session-type">
        <option value="">unset</option>
        <option value="serve_practice">serve practice</option>
        <option value="rally">rally / points</option>
        <option value="match">match</option>
        <option value="warmup">warmup</option>
        <option value="mixed">mixed</option>
    </select>
    <label>Hand:</label>
    <select id="dominant-hand">
        <option value="right">right</option>
        <option value="left">left</option>
    </select>
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
    <span id="save-status"></span>
</div>

<div id="info-bar" style="background:#1a1a2e; color:#888; font-size:0.8em; padding:4px 12px; font-family:monospace; display:flex; gap:20px; flex-wrap:wrap;">
    <span style="color:#e0e0e0; font-weight:bold; font-size:1.1em;">{video_filename}</span>
    <span>{video_path}</span>
    <span>det: {det_path}</span>
    <span>{fps:.1f} fps</span>
    <span>{num_shots} shots</span>
    <span>{duration:.1f}s</span>
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

<div id="shot-list-wrap">
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Time</th>
                <th>Type</th>
                <th>Tier</th>
                <th>Conf</th>
                <th>Source</th>
                <th></th>
            </tr>
        </thead>
        <tbody id="shot-tbody"></tbody>
    </table>
</div>

<div id="help">Space: play/pause &nbsp; &larr;/&rarr;: prev/next shot &nbsp; Shift+arrows: &plusmn;5s &nbsp; S/F/B/U/P/X: reclassify &nbsp; O: opponent &nbsp; M: mark shot &nbsp; D: delete &nbsp; [/]: nudge &plusmn;0.1s (shift &plusmn;0.5s) &nbsp; Ctrl+Z: undo &nbsp; +/&minus;: zoom timeline &nbsp; Drag: pan</div>

<script>
const player    = document.getElementById('player');
const timeline  = document.getElementById('timeline');
const progress  = document.getElementById('timeline-progress');
const tbody     = document.getElementById('shot-tbody');
const statsEl   = document.getElementById('stats');
const timDisp   = document.getElementById('time-display');
const playBtn   = document.getElementById('play-btn');
const scrub     = document.getElementById('scrub');
const scrubFill = document.getElementById('scrub-fill');
const scrubHdl  = document.getElementById('scrub-handle');
const saveStatus = document.getElementById('save-status');
const cameraEl   = document.getElementById('camera-angle');
const sessionEl  = document.getElementById('session-type');
const handEl     = document.getElementById('dominant-hand');
const minimap      = document.getElementById('minimap');
const minimapVP    = document.getElementById('minimap-viewport');
const minimapProg  = document.getElementById('minimap-progress');
const zoomLabel    = document.getElementById('zoom-label');

let detections  = [];
let fullData    = null;
let activeIdx   = -1;
let videoDur    = {duration};
let dirty       = false;
let undoStack   = [];  // {{ action, idx, detection }} for undo

// ── Zoom state ──
let zoomLevel = 1;       // 1 = full view, higher = more zoomed
let viewStart = 0;       // start of visible range as fraction (0-1)
let autoPanCooldown = 0; // timestamp until which auto-pan is suppressed
let viewEnd   = 1;       // end of visible range as fraction (0-1)
const MIN_ZOOM = 1;
const MAX_ZOOM = 30;

const SHOT_TYPES = ['serve', 'forehand', 'backhand', 'forehand_volley', 'backhand_volley', 'forehand_slice', 'backhand_slice', 'overhead', 'unknown_shot', 'practice', 'offscreen'];
const TYPE_LABELS = {{
    serve: 'S', forehand: 'FH', backhand: 'BH',
    forehand_volley: 'FV', backhand_volley: 'BV',
    forehand_slice: 'FS', backhand_slice: 'BS', overhead: 'OH',
    unknown_shot: '?', neutral: 'N', practice: 'P', offscreen: 'X'
}};
const TYPE_CLASS = {{
    serve: 'serve', forehand: 'forehand', backhand: 'backhand',
    forehand_volley: 'forehand', backhand_volley: 'backhand',
    forehand_slice: 'forehand', backhand_slice: 'backhand', overhead: 'serve',
    unknown_shot: 'unknown', neutral: 'unknown', practice: 'practice', offscreen: 'offscreen'
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

// ── Pre-download entire video as blob to eliminate streaming speed issues ──
const VIDEO_SRC = '{video_url}';
let videoReady = false;
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
        // Fallback to streaming if blob download fails
        loadText.textContent = 'Blob download failed, using stream...';
        player.src = VIDEO_SRC;
        player.load();
        videoReady = true;
        setTimeout(() => loadEl.classList.add('hidden'), 1000);
    }}
}})();

// ── Sync videoDur from actual video metadata (fixes bad JSON duration) ──
player.addEventListener('loadedmetadata', () => {{
    const realDur = player.duration;
    if (realDur && isFinite(realDur) && Math.abs(realDur - videoDur) > 2) {{
        console.warn('Duration mismatch: JSON=' + videoDur.toFixed(1) + 's, video=' + realDur.toFixed(1) + 's. Using video duration.');
        videoDur = realDur;
        // Also fix the in-memory data so saves write the correct duration
        if (fullData) fullData.duration = Math.round(realDur * 100) / 100;
        renderTimeline(videoDur);
    }}
}});

// ── Play/Pause button ──
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

// ── Scrub bar (click + drag to seek anywhere) ──
let scrubbing = false;
function scrubSeek(e) {{
    const rect = scrub.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    player.currentTime = pct * (player.duration || videoDur);
}}
scrub.addEventListener('mousedown', (e) => {{
    scrubbing = true;
    scrubSeek(e);
}});
document.addEventListener('mousemove', (e) => {{
    if (scrubbing) scrubSeek(e);
}});
document.addEventListener('mouseup', () => {{ scrubbing = false; }});

// ── Load detections ──
fetch('/detections.json').then(r => r.json()).then(data => {{
    fullData = data;
    detections = data.detections || [];
    videoDur = data.duration || {duration};
    // Sanity check: if max detection timestamp exceeds stated duration, use frames/fps or max timestamp
    const maxTs = detections.reduce((mx, d) => Math.max(mx, d.timestamp || 0), 0);
    if (maxTs > videoDur * 1.05) {{
        const frameDur = data.total_frames && data.fps ? data.total_frames / data.fps : 0;
        const corrected = frameDur > maxTs ? frameDur : maxTs * 1.1;
        console.warn('JSON duration (' + videoDur.toFixed(1) + 's) < max timestamp (' + maxTs.toFixed(1) + 's). Correcting to ' + corrected.toFixed(1) + 's');
        videoDur = corrected;
    }}
    // Also prefer player.duration if already available and differs significantly
    if (player.duration && isFinite(player.duration) && Math.abs(player.duration - videoDur) > 2) {{
        videoDur = player.duration;
    }}
    // load video metadata (check top-level camera_angle from auto-detect, then video_metadata)
    const meta = data.video_metadata || {{}};
    cameraEl.value = data.camera_angle || meta.camera_angle || '';
    sessionEl.value = meta.session_type || '';
    handEl.value = data.dominant_hand || 'right';
    renderStats();
    renderTimeline(videoDur);
    renderTable();
}});

// ── Video metadata changes ──
[cameraEl, sessionEl, handEl].forEach(el => {{
    el.addEventListener('change', () => {{
        fullData.camera_angle = cameraEl.value || null;
        if (!fullData.video_metadata) fullData.video_metadata = {{}};
        fullData.video_metadata.session_type = sessionEl.value || null;
        fullData.dominant_hand = handEl.value;
        dirty = true;
        saveChanges();
    }});
}});

function renderStats() {{
    // recompute from current detections (reflects edits)
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
        <span class="stat">HIGH <b class="tier-high">${{tiers.high || 0}}</b></span>
        <span class="stat">MED <b class="tier-medium">${{tiers.medium || 0}}</b></span>
        <span class="stat">LOW <b class="tier-low">${{tiers.low || 0}}</b></span>
    `;
}}

function renderMinimap(dur) {{
    minimap.querySelectorAll('.mini-marker').forEach(m => m.remove());
    detections.forEach((d, i) => {{
        const pct = (d.timestamp / dur) * 100;
        const cls = TYPE_CLASS[d.shot_type] || 'unknown';
        const el = document.createElement('div');
        el.className = 'mini-marker mm-' + cls;
        el.style.left = pct + '%';
        minimap.appendChild(el);
    }});
}}

function updateMinimapViewport() {{
    minimapVP.style.left = (viewStart * 100) + '%';
    minimapVP.style.width = ((viewEnd - viewStart) * 100) + '%';
    // hide viewport indicator when fully zoomed out
    minimapVP.style.display = zoomLevel <= 1.05 ? 'none' : 'block';
    zoomLabel.textContent = zoomLevel > 1.05 ? zoomLevel.toFixed(1) + 'x' : '';
}}

function setZoom(newZoom, anchorFrac) {{
    // anchorFrac = position in the visible range (0-1) that should stay fixed
    newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));
    const viewWidth = 1 / newZoom;
    // keep anchor point stable
    const oldViewWidth = viewEnd - viewStart;
    const anchorAbs = viewStart + anchorFrac * oldViewWidth;
    let newStart = anchorAbs - anchorFrac * viewWidth;
    let newEnd = newStart + viewWidth;
    // clamp
    if (newStart < 0) {{ newStart = 0; newEnd = viewWidth; }}
    if (newEnd > 1) {{ newEnd = 1; newStart = Math.max(0, 1 - viewWidth); }}
    zoomLevel = newZoom;
    viewStart = newStart;
    viewEnd = newEnd;
    updateMinimapViewport();
    renderZoomedTimeline(videoDur);
}}

function panView(deltaFrac) {{
    const width = viewEnd - viewStart;
    let newStart = viewStart + deltaFrac;
    let newEnd = viewEnd + deltaFrac;
    if (newStart < 0) {{ newStart = 0; newEnd = width; }}
    if (newEnd > 1) {{ newEnd = 1; newStart = Math.max(0, 1 - width); }}
    viewStart = newStart;
    viewEnd = newEnd;
    updateMinimapViewport();
    renderZoomedTimeline(videoDur);
}}

function renderZoomedTimeline(dur) {{
    // clear old markers (keep progress bar and zoom label)
    timeline.querySelectorAll('.marker').forEach(m => m.remove());
    const vw = viewEnd - viewStart;
    detections.forEach((d, i) => {{
        const frac = d.timestamp / dur;
        // skip markers outside visible range
        if (frac < viewStart - 0.01 || frac > viewEnd + 0.01) return;
        const pct = ((frac - viewStart) / vw) * 100;
        const cls = TYPE_CLASS[d.shot_type] || 'unknown';
        const lbl = TYPE_LABELS[d.shot_type] || '?';
        const el = document.createElement('div');
        let confClass = '';
        const conf = d.confidence || 0;
        if (conf < 0.4) confClass = ' very-low-conf';
        else if (conf < 0.6) confClass = ' low-conf';
        el.className = 'marker marker-' + cls + (d.opponent ? ' opponent' : '') + confClass;
        el.style.left = pct + '%';
        el.textContent = lbl;
        const mlInfo = d.ml_confidence != null ? ` ml:${{d.ml_confidence.toFixed(2)}}` : '';
        const nsInfo = d.not_shot_prob != null && d.not_shot_prob > 0.05 ? ` ns:${{d.not_shot_prob.toFixed(2)}}` : '';
        el.title = '#' + (i+1) + ' ' + (d.opponent ? 'OPP ' : '') + (d.shot_type||'?') + ' @ ' + fmtTime(d.timestamp) + mlInfo + nsInfo;
        el.dataset.idx = i;
        if (i === activeIdx) el.classList.add('active');
        el.addEventListener('click', (e) => {{ e.stopPropagation(); jumpTo(i); }});
        timeline.appendChild(el);
    }});
}}

function renderTimeline(dur) {{
    renderMinimap(dur);
    updateMinimapViewport();
    renderZoomedTimeline(dur);
}}

// ── Zoom buttons ──
document.getElementById('zoom-in').addEventListener('click', () => {{
    setZoom(zoomLevel * 1.5, 0.5);
}});
document.getElementById('zoom-out').addEventListener('click', () => {{
    setZoom(zoomLevel / 1.5, 0.5);
}});

// ── Pan timeline by click+drag ──
let tlDragging = false;
let tlDragStartX = 0;
let tlDragStartViewStart = 0;
timeline.addEventListener('mousedown', (e) => {{
    if (e.target.classList.contains('marker')) return;
    if (zoomLevel <= 1.05) {{
        // not zoomed — click to seek
        const rect = timeline.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        player.currentTime = pct * videoDur;
        return;
    }}
    tlDragging = true;
    tlDragStartX = e.clientX;
    tlDragStartViewStart = viewStart;
    e.preventDefault();
}});
document.addEventListener('mousemove', (e) => {{
    if (!tlDragging) return;
    const rect = timeline.getBoundingClientRect();
    const dx = e.clientX - tlDragStartX;
    const fracDx = dx / rect.width * (viewEnd - viewStart);
    // invert: drag right = pan left (move view to earlier time)
    const newStart = tlDragStartViewStart - fracDx;
    const width = viewEnd - viewStart;
    viewStart = Math.max(0, Math.min(1 - width, newStart));
    viewEnd = viewStart + width;
    updateMinimapViewport();
    renderZoomedTimeline(videoDur);
}});
document.addEventListener('mouseup', () => {{
    if (tlDragging) {{ tlDragging = false; }}
}});

// ── Minimap click to pan ──
minimap.addEventListener('click', (e) => {{
    const rect = minimap.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    // Seek video to clicked position
    player.currentTime = frac * videoDur;
    // Also pan viewport to center on click
    const width = viewEnd - viewStart;
    let newStart = frac - width / 2;
    if (newStart < 0) newStart = 0;
    if (newStart + width > 1) newStart = 1 - width;
    viewStart = newStart;
    viewEnd = newStart + width;
    updateMinimapViewport();
    renderZoomedTimeline(videoDur);
}});

// (scroll-wheel zoom removed — use +/- buttons instead)

function renderTable() {{
    tbody.innerHTML = '';
    detections.forEach((d, i) => {{
        const cls = TYPE_CLASS[d.shot_type] || 'unknown';
        const tr = document.createElement('tr');
        tr.dataset.idx = i;

        // build type <select>
        const opts = SHOT_TYPES.map(t =>
            `<option value="${{t}}" ${{t === d.shot_type ? 'selected' : ''}}>${{t}}</option>`
        ).join('');

        tr.innerHTML = `
            <td>${{i + 1}}</td>
            <td class="time-cell">
                <button class="nudge-btn" data-idx="${{i}}" data-delta="-0.1" title="Earlier (-0.1s)">&lsaquo;</button>
                <span class="time-val">${{fmtTime(d.timestamp)}}</span>
                <button class="nudge-btn" data-idx="${{i}}" data-delta="0.1" title="Later (+0.1s)">&rsaquo;</button>
            </td>
            <td class="type-cell">
                <select class="type-select type-${{cls}}" data-idx="${{i}}">
                    ${{opts}}
                </select>${{d.opponent ? '<span class="opp-badge">OPP</span>' : ''}}
            </td>
            <td class="tier-${{d.tier}}">${{(d.tier||'').toUpperCase()}}</td>
            <td>${{(d.confidence||0).toFixed(2)}}${{
                d.ml_confidence != null
                    ? '<span class="ml-info"> ml:' + d.ml_confidence.toFixed(2) + '</span>'
                    : ''
            }}${{
                d.not_shot_prob != null && d.not_shot_prob > 0.05
                    ? '<span class="' + (d.not_shot_prob > 0.3 ? 'conf-danger' : 'conf-warn') + '"> ns:' + d.not_shot_prob.toFixed(2) + '</span>'
                    : ''
            }}</td>
            <td style="color:#666">${{d.source || ''}}</td>
            <td><button class="del-btn" data-idx="${{i}}" title="Not a shot (N)">x</button></td>
        `;

        // click row (but not interactive elements) to jump
        tr.addEventListener('click', (e) => {{
            if (e.target.tagName === 'SELECT' || e.target.classList.contains('del-btn')
                || e.target.classList.contains('nudge-btn')) return;
            jumpTo(i);
        }});

        // nudge button handlers
        tr.querySelectorAll('.nudge-btn').forEach(btn => {{
            btn.addEventListener('click', (e) => {{
                e.stopPropagation();
                nudgeTime(parseInt(btn.dataset.idx), parseFloat(btn.dataset.delta));
            }});
        }});

        // type change handler
        const sel = tr.querySelector('select');
        sel.addEventListener('change', (e) => {{
            e.stopPropagation();
            changeType(i, sel.value);
        }});
        sel.addEventListener('click', (e) => e.stopPropagation());

        // delete button handler
        const delBtn = tr.querySelector('.del-btn');
        delBtn.addEventListener('click', (e) => {{
            e.stopPropagation();
            deleteShot(parseInt(e.target.dataset.idx));
        }});

        tbody.appendChild(tr);
    }});
}}

function changeType(idx, newType) {{
    undoStack.push({{ action: 'retype', idx, oldType: detections[idx].shot_type }});
    detections[idx].shot_type = newType;
    dirty = true;
    refreshAll();
    saveChanges();
}}

function deleteShot(idx) {{
    const removed = detections.splice(idx, 1)[0];
    undoStack.push({{ action: 'delete', idx, detection: removed }});
    dirty = true;
    // adjust activeIdx
    if (activeIdx >= detections.length) activeIdx = detections.length - 1;
    if (activeIdx < 0) activeIdx = -1;
    refreshAll();
    saveChanges();
}}

function undoLast() {{
    if (undoStack.length === 0) return;
    const op = undoStack.pop();
    if (op.action === 'add') {{
        detections.splice(op.idx, 1);
        activeIdx = Math.min(op.idx, detections.length - 1);
    }} else if (op.action === 'delete') {{
        detections.splice(op.idx, 0, op.detection);
        activeIdx = op.idx;
    }} else if (op.action === 'retype') {{
        detections[op.idx].shot_type = op.oldType;
        activeIdx = op.idx;
    }} else if (op.action === 'nudge') {{
        detections[op.idx].timestamp = op.oldTime;
        detections[op.idx].frame = Math.round(op.oldTime * (fullData.fps || 60));
        activeIdx = op.idx;
    }}
    dirty = true;
    refreshAll();
    saveChanges();
}}

function toggleOpponent(idx) {{
    detections[idx].opponent = !detections[idx].opponent;
    dirty = true;
    refreshAll();
    saveChanges();
}}

function addShot(timestamp) {{
    const fps = fullData.fps || 60;
    const newDet = {{
        timestamp: +timestamp.toFixed(3),
        frame: Math.round(timestamp * fps),
        shot_type: 'unknown_shot',
        confidence: 1.0,
        tier: 'manual',
        source: 'manual',
        audio_peak_time: null,
        audio_amplitude: 0.0,
        heuristic_frame: null,
        pattern_confidence: 0.0,
        trigger: 'manually added',
        velocity: 0.0
    }};
    // insert sorted by timestamp
    let insertIdx = detections.findIndex(d => d.timestamp > timestamp);
    if (insertIdx < 0) insertIdx = detections.length;
    detections.splice(insertIdx, 0, newDet);
    undoStack.push({{ action: 'add', idx: insertIdx }});
    activeIdx = insertIdx;
    dirty = true;
    refreshAll();
    saveChanges();
}}

function nudgeTime(idx, delta) {{
    const oldTime = detections[idx].timestamp;
    const newTime = Math.max(0, +(oldTime + delta).toFixed(3));
    undoStack.push({{ action: 'nudge', idx, oldTime }});
    detections[idx].timestamp = newTime;
    // update frame estimate
    const fps = fullData.fps || 60;
    detections[idx].frame = Math.round(newTime * fps);
    dirty = true;
    // seek video to new position
    player.currentTime = Math.max(0, newTime - 0.5);
    refreshAll();
    saveChanges();
}}

function refreshAll() {{
    renderStats();
    renderMinimap(videoDur);
    renderZoomedTimeline(videoDur);
    renderTable();
    if (activeIdx >= 0) setActive(activeIdx, true);
}}

function saveChanges() {{
    saveStatus.textContent = 'saving...';
    saveStatus.className = 'saving';
    fullData.detections = detections;
    // recompute summary
    const counts = {{}};
    const tiers = {{}};
    detections.forEach(d => {{
        counts[d.shot_type] = (counts[d.shot_type] || 0) + 1;
        tiers[d.tier] = (tiers[d.tier] || 0) + 1;
    }});
    fullData.summary.total_detections = detections.length;
    fullData.summary.by_type = counts;
    fullData.summary.by_tier = tiers;

    fetch('/save', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(fullData)
    }}).then(r => {{
        if (r.ok) {{
            saveStatus.textContent = 'saved';
            saveStatus.className = 'saved';
            dirty = false;
            setTimeout(() => {{ if (!dirty) saveStatus.textContent = ''; }}, 2000);
        }} else {{
            saveStatus.textContent = 'save failed';
            saveStatus.className = 'error';
        }}
    }}).catch(() => {{
        saveStatus.textContent = 'save failed';
        saveStatus.className = 'error';
    }});
}}

function jumpTo(idx) {{
    if (idx < 0 || idx >= detections.length) return;
    const targetTime = Math.max(0, detections[idx].timestamp - 0.5);
    const dur = player.duration || videoDur || 1;

    // Set seek guard BEFORE triggering seek — blocks stale timeupdate rendering
    seekGuardTarget = targetTime;
    seekGuardUntil = Date.now() + 600;

    // Immediately render progress bars at target position (don't wait for timeupdate)
    const globalPct = targetTime / dur * 100;
    scrubFill.style.width = globalPct + '%';
    scrubHdl.style.left = globalPct + '%';
    timDisp.textContent = fmtTime(targetTime) + ' / ' + fmtTimeShort(dur);
    minimapProg.style.width = globalPct + '%';
    const vw = viewEnd - viewStart;
    const frac = targetTime / dur;
    const zoomedPct = ((frac - viewStart) / vw) * 100;
    progress.style.width = Math.max(0, Math.min(100, zoomedPct)) + '%';

    const wasPaused = player.paused;
    player.currentTime = targetTime;
    if (!wasPaused) player.play();

    // only pan if shot is outside visible range
    if (zoomLevel > 1.05) {{
        const shotFrac = detections[idx].timestamp / videoDur;
        if (shotFrac < viewStart || shotFrac > viewEnd) {{
            const width = viewEnd - viewStart;
            let newStart = shotFrac - width * 0.10;
            if (newStart < 0) newStart = 0;
            if (newStart + width > 1) newStart = 1 - width;
            viewStart = newStart;
            viewEnd = newStart + width;
            updateMinimapViewport();
            renderZoomedTimeline(videoDur);
            // re-render progress bar with updated view
            const newZoomedPct = ((shotFrac - viewStart) / (viewEnd - viewStart)) * 100;
            progress.style.width = Math.max(0, Math.min(100, newZoomedPct)) + '%';
        }}
    }}
    // suppress auto-pan briefly to prevent flicker from pending timeupdate
    autoPanCooldown = Date.now() + 600;
    setActive(idx, true);
}}

function setActive(idx, force) {{
    if (idx === activeIdx && !force) return;
    activeIdx = idx;
    // markers — match by data-idx since zoomed view may not have all markers
    document.querySelectorAll('.marker').forEach(m => {{
        m.classList.toggle('active', +m.dataset.idx === idx);
    }});
    // table rows
    document.querySelectorAll('#shot-tbody tr').forEach((r) =>
        r.classList.toggle('active', +r.dataset.idx === idx));
    // scroll row into view
    const row = tbody.children[idx];
    if (row) row.scrollIntoView({{ block: 'nearest' }});
}}

// ── Playback tracking ──
// Seek guard: blocks stale timeupdate events during programmatic seeks.
// Uses time+position check instead of seeking/seeked events (which fire unreliably).
let seekGuardTarget = -1;   // expected currentTime after seek
let seekGuardUntil = 0;     // timestamp (ms) when guard expires

player.addEventListener('timeupdate', () => {{
    const t = player.currentTime;
    // Block stale timeupdate events during a seek: if currentTime is far from
    // the seek target, skip rendering to prevent the progress bar overshoot.
    if (seekGuardUntil > 0) {{
        if (Date.now() < seekGuardUntil && Math.abs(t - seekGuardTarget) > 1.0) return;
        // Either guard expired or currentTime reached target — clear it
        seekGuardUntil = 0;
        seekGuardTarget = -1;
    }}
    const dur = player.duration || videoDur || 1;
    const globalPct = t / dur * 100;
    // scrub bar (always full width)
    scrubFill.style.width = globalPct + '%';
    scrubHdl.style.left = globalPct + '%';
    // Speed monitor: measure actual playback speed from currentTime
    const now = performance.now();
    if (typeof _lastT !== 'undefined' && !player.paused && (now - _lastWall) > 200) {{
        const dtVideo = t - _lastT;
        const dtWall = (now - _lastWall) / 1000;
        const actualSpeed = dtWall > 0 ? (dtVideo / dtWall) : 0;
        const setSpeed = player.playbackRate;
        const ratio = setSpeed > 0 ? actualSpeed / setSpeed : 0;
        const speedStr = ratio.toFixed(2) + 'x';
        const color = (ratio > 0.85 && ratio < 1.15) ? '#4f4' : '#f44';
        timDisp.textContent = fmtTime(t) + ' / ' + fmtTimeShort(dur) + '  [' + speedStr + ']';
        timDisp.style.color = color;
    }} else {{
        timDisp.textContent = fmtTime(t) + ' / ' + fmtTimeShort(dur);
        timDisp.style.color = '#ccc';
    }}
    _lastT = t; _lastWall = now;
    // minimap progress
    minimapProg.style.width = globalPct + '%';
    // zoomed timeline progress
    const vw = viewEnd - viewStart;
    const frac = t / dur;
    const zoomedPct = ((frac - viewStart) / vw) * 100;
    progress.style.width = Math.max(0, Math.min(100, zoomedPct)) + '%';
    // auto-pan: only scroll when playhead reaches right edge of visible range
    if (!tlDragging && zoomLevel > 1.05 && Date.now() > autoPanCooldown) {{
        const playheadPos = (frac - viewStart) / vw;
        if (playheadPos > 0.90 || playheadPos < -0.05) {{
            // snap so playhead is at 10% from left edge
            const targetStart = frac - vw * 0.10;
            const clampedStart = Math.max(0, Math.min(1 - vw, targetStart));
            viewStart = clampedStart;
            viewEnd = clampedStart + vw;
            updateMinimapViewport();
            renderZoomedTimeline(dur);
        }}
    }}
    // find nearest shot within 2s
    let best = -1, bestDist = Infinity;
    detections.forEach((d, i) => {{
        const dist = Math.abs(d.timestamp - t);
        if (dist < bestDist && dist < 2) {{ bestDist = dist; best = i; }}
    }});
    if (best >= 0) setActive(best);
}});

// ── Keyboard shortcuts ──
document.addEventListener('keydown', (e) => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    switch (e.key) {{
        case ' ':
            e.preventDefault();
            player.paused ? player.play() : player.pause();
            break;
        case 'ArrowRight':
            e.preventDefault();
            if (e.shiftKey) {{ player.currentTime += 5; }}
            else {{ jumpTo(Math.min(detections.length - 1, activeIdx + 1)); }}
            break;
        case 'ArrowLeft':
            e.preventDefault();
            if (e.shiftKey) {{ player.currentTime -= 5; }}
            else {{ jumpTo(Math.max(0, activeIdx - 1)); }}
            break;
        // quick reclassify keys
        case 's': if (activeIdx >= 0) changeType(activeIdx, 'serve'); break;
        case 'f': if (activeIdx >= 0) changeType(activeIdx, 'forehand'); break;
        case 'b': if (activeIdx >= 0) changeType(activeIdx, 'backhand'); break;
        case 'u': if (activeIdx >= 0) changeType(activeIdx, 'unknown_shot'); break;
        case 'v': if (activeIdx >= 0) changeType(activeIdx, 'forehand_volley'); break;
        case 'g': if (activeIdx >= 0) changeType(activeIdx, 'backhand_volley'); break;
        case 'h': if (activeIdx >= 0) changeType(activeIdx, 'overhead'); break;
        case '1': if (activeIdx >= 0) changeType(activeIdx, 'forehand_slice'); break;
        case '2': if (activeIdx >= 0) changeType(activeIdx, 'backhand_slice'); break;
        case 'p': if (activeIdx >= 0) changeType(activeIdx, 'practice'); break;
        case 'm':
            e.preventDefault();
            addShot(player.currentTime);
            break;
        case 'o':
            if (activeIdx >= 0) toggleOpponent(activeIdx);
            break;
        case 'x':
            if (activeIdx >= 0) changeType(activeIdx, 'offscreen');
            break;
        case 'd': case 'Delete': case 'Backspace':
            e.preventDefault();
            if (activeIdx >= 0) deleteShot(activeIdx);
            break;
        case 'z':
            if (e.metaKey || e.ctrlKey) {{ e.preventDefault(); undoLast(); }}
            break;
        case '[':
            e.preventDefault();
            if (activeIdx >= 0) nudgeTime(activeIdx, e.shiftKey ? -0.5 : -0.1);
            break;
        case ']':
            e.preventDefault();
            if (activeIdx >= 0) nudgeTime(activeIdx, e.shiftKey ? 0.5 : 0.1);
            break;
        case '+': case '=':
            e.preventDefault();
            setZoom(zoomLevel * 1.5, 0.5);
            break;
        case '-': case '_':
            e.preventDefault();
            setZoom(zoomLevel / 1.5, 0.5);
            break;
    }}
}});

// warn before leaving with unsaved changes
window.addEventListener('beforeunload', (e) => {{
    if (dirty) {{ e.preventDefault(); e.returnValue = ''; }}
}});
</script>
</body>
</html>"""


class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler with Range request support for video streaming."""

    def __init__(self, *args, video_path=None, detections_json=None,
                 detections_json_ref=None, det_path=None,
                 html_content=None, **kwargs):
        self.video_path = video_path
        self.detections_json = detections_json
        self.detections_json_ref = detections_json_ref
        self.det_path = det_path
        self.html_content = html_content
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = unquote(self.path).split("?")[0]

        if path == "/":
            self._serve_bytes(self.html_content.encode(), "text/html")
        elif path == "/detections.json":
            self._serve_bytes(self.detections_json_ref[0].encode(), "application/json")
        elif path == "/video":
            self._serve_video()
        else:
            self.send_error(404)

    def do_POST(self):
        path = unquote(self.path).split("?")[0]
        if path == "/save":
            self._handle_save()
        else:
            self.send_error(404)

    def _handle_save(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            # Stamp save time so we know when edits happened
            from datetime import datetime, timezone
            data["last_saved"] = datetime.now(timezone.utc).isoformat()
            det_path = self.det_path
            with open(det_path, "w") as f:
                json.dump(data, f, indent=2)
            # update in-memory copy so future GETs reflect the save
            self.detections_json_ref[0] = json.dumps(data)
            print(f"  Saved {len(data.get('detections', []))} detections to {os.path.basename(det_path)}")
            self._serve_bytes(b'{"ok":true}', "application/json")
        except Exception as e:
            print(f"  Save error: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _serve_bytes(self, data: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_video(self):
        fpath = self.video_path
        fsize = os.path.getsize(fpath)
        range_header = self.headers.get("Range")

        if range_header:
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if not m:
                self.send_error(416)
                return
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else fsize - 1
            end = min(end, fsize - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{fsize}")
            self.send_header("Content-Length", str(length))
        else:
            start = 0
            length = fsize
            self.send_response(200)
            self.send_header("Content-Length", str(fsize))

        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(fpath, "rb") as f:
            f.seek(start)
            remaining = length
            buf_size = 2 * 1024 * 1024  # 2 MB chunks for fast transfer
            while remaining > 0:
                chunk = f.read(min(buf_size, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except BrokenPipeError:
                    break
                remaining -= len(chunk)

    def log_message(self, format, *args):
        # suppress noisy per-request logs; only show errors
        if args and isinstance(args[0], str) and args[0].startswith("GET /video"):
            return
        super().log_message(format, *args)


def make_handler(video_path, detections_json_ref, det_path, html_content):
    """Factory to create handler instances with bound data."""

    class Handler(RangeHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                video_path=video_path,
                detections_json=None,
                detections_json_ref=detections_json_ref,
                det_path=det_path,
                html_content=html_content,
                **kwargs,
            )

    return Handler


def kill_port(port):
    """Kill any process listening on the given port. Returns True if something was killed."""
    import time
    try:
        if sys.platform == "win32":
            # Windows: use netstat + taskkill
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True, timeout=5)
            time.sleep(0.3)
            return True
        else:
            # macOS/Linux: use lsof
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split()
            if pids:
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except (ProcessLookupError, ValueError):
                        pass
                time.sleep(0.3)
                return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description="Interactive shot review player")
    parser.add_argument("video", help="Path to the video file (mp4), or video name when using --video-url")
    parser.add_argument("--detections", "-d", help="Path to detection JSON (auto-discovered if omitted)")
    parser.add_argument("--video-url", help="Remote URL for video (e.g. http://windows:9000/IMG_0994.mp4)")
    parser.add_argument("--port", "-p", type=int, default=8765, help="HTTP port (default 8765)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    video_url = args.video_url  # None means serve locally
    video_path = os.path.abspath(args.video)

    if not video_url and not os.path.isfile(video_path):
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    det_path = args.detections
    if det_path:
        det_path = os.path.abspath(det_path)
    else:
        det_path = find_detections(video_path)
    if not det_path or not os.path.isfile(det_path):
        print(f"Error: detection JSON not found. Use --detections to specify.", file=sys.stderr)
        sys.exit(1)

    with open(det_path) as f:
        det_data = json.load(f)

    detections_json_ref = [json.dumps(det_data)]
    duration = det_data.get("duration", 0)
    fps = det_data.get("fps", 0)
    num_shots = len(det_data.get("detections", []))
    video_filename = os.path.basename(args.video)
    html = build_html(video_filename, duration, video_url=video_url or "/video",
                      video_path=video_path, det_path=det_path, fps=fps,
                      num_shots=num_shots)

    handler_class = make_handler(video_path, detections_json_ref, det_path, html)

    # Kill any stale process on the port before starting
    if kill_port(args.port):
        print(f"Killed stale process on port {args.port}")

    # Bind to 0.0.0.0 on Windows (accessible from network), 127.0.0.1 on Mac (local only)
    bind_addr = "0.0.0.0" if sys.platform == "win32" else "127.0.0.1"
    class ThreadedServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    with ThreadedServer((bind_addr, args.port), handler_class) as httpd:
        url = f"http://localhost:{args.port}"
        print(f"Shot Review: {video_filename}")
        print(f"Detections:  {os.path.basename(det_path)} ({len(det_data.get('detections', []))} shots)")
        print(f"Serving at:  {url}")
        print(f"Press Ctrl+C to stop.\n")

        if not args.no_open:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
