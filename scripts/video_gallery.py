"""Video gallery for tennis pipeline.

Serves a modern UI for browsing and playing videos from R2 storage.
"""

import os
from datetime import datetime, timezone
from typing import Optional

import boto3
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

load_dotenv('/opt/tennis/.env')

CF_ACCOUNT_ID = os.environ.get('CF_ACCOUNT_ID')
CF_R2_ACCESS_KEY_ID = os.environ.get('CF_R2_ACCESS_KEY_ID')
CF_R2_SECRET_ACCESS_KEY = os.environ.get('CF_R2_SECRET_ACCESS_KEY')
R2_BUCKET = "tennis-videos"

router = APIRouter()


def get_r2_client():
    return boto3.client('s3',
        endpoint_url=f"https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=CF_R2_ACCESS_KEY_ID,
        aws_secret_access_key=CF_R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}),
        region_name='us-east-1')


@router.get("/videos")
async def list_videos(prefix: str = "raw/"):
    client = get_r2_client()
    try:
        response = client.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
        videos = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith(('.mp4', '.mov', '.MP4', '.MOV', '.json', '.jpg', '.jpeg', '.png')):
                videos.append({
                    "key": key,
                    "name": key.split('/')[-1],
                    "size_mb": round(obj['Size'] / 1024 / 1024, 1),
                    "modified": obj['LastModified'].isoformat(),
                })
        return {"count": len(videos), "videos": videos}
    except Exception as e:
        raise HTTPException(500, f"Error listing videos: {e}")


@router.get("/videos/url/{key:path}")
async def get_video_url(key: str, expires: int = 3600):
    client = get_r2_client()
    try:
        url = client.generate_presigned_url(
            'get_object',
            Params={'Bucket': R2_BUCKET, 'Key': key},
            ExpiresIn=expires
        )
        return {"url": url, "expires_in": expires}
    except Exception as e:
        raise HTTPException(500, f"Error generating URL: {e}")


@router.get("/thumb/{key:path}")
async def get_thumbnail(key: str):
    """Proxy thumbnail images from R2."""
    from fastapi.responses import Response
    client = get_r2_client()
    try:
        response = client.get_object(Bucket=R2_BUCKET, Key=f"thumbs/{key}")
        return Response(
            content=response['Body'].read(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"}
        )
    except Exception as e:
        raise HTTPException(404, f"Thumbnail not found: {e}")


@router.get("/watch/{key:path}", response_class=HTMLResponse)
async def watch_video(key: str):
    return await gallery_page()


@router.get("/", response_class=HTMLResponse)
async def gallery_page():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tennis Analysis</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --bg-hover: #22222e;
            --accent: #6366f1;
            --accent-hover: #818cf8;
            --text-primary: #f1f1f1;
            --text-secondary: #888;
            --text-muted: #555;
            --border: #2a2a35;
            --success: #22c55e;
            --warning: #f59e0b;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }

        /* Header */
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent), #a855f7);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        .logo-text {
            font-size: 20px;
            font-weight: 700;
        }

        .logo-text span {
            color: var(--accent);
        }

        /* Stats Bar */
        .stats-bar {
            display: flex;
            gap: 30px;
        }

        .stat {
            text-align: center;
        }

        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--accent);
        }

        .stat-label {
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Main Content */
        .main {
            max-width: 1600px;
            margin: 0 auto;
            padding: 30px 40px;
        }

        /* Tabs */
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 30px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 15px;
        }

        .tab {
            padding: 10px 20px;
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-secondary);
            cursor: pointer;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
            font-family: inherit;
        }

        .tab:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .tab.active {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }

        /* Grid */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 24px;
        }

        /* Video Card */
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid var(--border);
        }

        .card:hover {
            transform: translateY(-4px);
            border-color: var(--accent);
            box-shadow: 0 20px 40px rgba(99, 102, 241, 0.1);
        }

        .card-thumb {
            width: 100%;
            aspect-ratio: 16/9;
            background: var(--bg-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }

        .card-thumb img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .card-thumb .play-icon {
            position: absolute;
            width: 60px;
            height: 60px;
            background: rgba(99, 102, 241, 0.9);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            opacity: 0;
            transition: opacity 0.2s;
        }

        .card:hover .play-icon {
            opacity: 1;
        }

        .card-body {
            padding: 16px;
        }

        .card-title {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 8px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .card-meta {
            display: flex;
            gap: 16px;
            font-size: 12px;
            color: var(--text-secondary);
        }

        .card-meta span {
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .badge {
            display: inline-block;
            padding: 2px 8px;
            background: var(--success);
            color: white;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge.pending {
            background: var(--warning);
        }

        /* Modal */
        .modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.95);
            z-index: 1000;
            padding: 20px;
        }

        .modal.active {
            display: flex;
            flex-direction: column;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0 20px;
        }

        .modal-title {
            font-size: 18px;
            font-weight: 600;
        }

        .close-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            width: 40px;
            height: 40px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 20px;
            transition: all 0.2s;
        }

        .close-btn:hover {
            background: var(--accent);
            border-color: var(--accent);
        }

        .share-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
            font-family: inherit;
        }

        .share-btn:hover {
            background: var(--accent);
            border-color: var(--accent);
        }

        .share-btn.copied {
            background: var(--success);
            border-color: var(--success);
        }

        .video-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 0;
        }

        .modal video {
            max-width: 100%;
            max-height: calc(100vh - 200px);
            background: #000;
            border-radius: 8px;
        }

        /* Controls */
        .controls {
            display: flex;
            gap: 10px;
            padding: 20px 0 10px;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
        }

        .control-group {
            display: flex;
            gap: 4px;
            background: var(--bg-card);
            padding: 4px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .controls button {
            padding: 8px 14px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
            font-family: inherit;
        }

        .controls button:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .controls button.active {
            background: var(--accent);
            color: white;
        }

        .controls-label {
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-right: 8px;
        }

        .keyboard-hint {
            text-align: center;
            color: var(--text-muted);
            font-size: 12px;
            padding-top: 10px;
        }

        .keyboard-hint kbd {
            background: var(--bg-card);
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid var(--border);
            font-family: inherit;
        }

        /* Loading */
        .loading {
            text-align: center;
            padding: 60px;
            color: var(--text-secondary);
        }

        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 16px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Empty State */
        .empty {
            text-align: center;
            padding: 80px 20px;
            color: var(--text-secondary);
        }

        .empty-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">
            <div class="logo-icon">🎾</div>
            <div class="logo-text">Tennis<span>Analysis</span></div>
        </div>
        <div class="stats-bar">
            <div class="stat">
                <div class="stat-value" id="stat-videos">-</div>
                <div class="stat-label">Videos</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-poses">-</div>
                <div class="stat-label">Poses</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-size">-</div>
                <div class="stat-label">Total Size</div>
            </div>
        </div>
    </header>

    <main class="main">
        <div class="tabs">
            <button class="tab active" data-prefix="raw/">Videos</button>
            <button class="tab" data-prefix="poses/">Pose Data</button>
            <button class="tab" data-prefix="thumbs/">Thumbnails</button>
        </div>

        <div class="grid" id="grid">
            <div class="loading">
                <div class="loading-spinner"></div>
                Loading videos...
            </div>
        </div>
    </main>

    <div class="modal" id="modal">
        <div class="modal-header">
            <span class="modal-title" id="modal-title"></span>
            <div style="display:flex;gap:10px;align-items:center;">
                <button class="share-btn" onclick="copyVideoUrl()" id="share-btn" title="Copy shareable link">📋 Copy Link</button>
                <button class="close-btn" onclick="closeModal()">×</button>
            </div>
        </div>
        <div class="video-container">
            <video id="player" controls></video>
        </div>
        <div class="controls">
            <span class="controls-label">Speed</span>
            <div class="control-group">
                <button onclick="setSpeed(0.25)">0.25×</button>
                <button onclick="setSpeed(0.5)">0.5×</button>
                <button onclick="setSpeed(1)" class="active" id="speed-1">1×</button>
                <button onclick="setSpeed(1.5)">1.5×</button>
                <button onclick="setSpeed(2)">2×</button>
            </div>
            <span class="controls-label" style="margin-left: 20px;">Frames</span>
            <div class="control-group">
                <button onclick="skipFrames(-10)">−10</button>
                <button onclick="skipFrames(-1)">−1</button>
                <button onclick="togglePlay()" id="play-btn">⏸</button>
                <button onclick="skipFrames(1)">+1</button>
                <button onclick="skipFrames(10)">+10</button>
            </div>
        </div>
        <div class="keyboard-hint">
            <kbd>Space</kbd> Play/Pause &nbsp;
            <kbd>↑</kbd><kbd>↓</kbd> Speed &nbsp;
            <kbd>←</kbd><kbd>→</kbd> ±1 frame &nbsp;
            <kbd>Shift</kbd>+<kbd>←</kbd><kbd>→</kbd> ±10 frames &nbsp;
            <kbd>Esc</kbd> Close
        </div>
    </div>

    <script>
        let currentSpeed = 1;
        const speeds = [0.25, 0.5, 1, 1.5, 2];
        let currentPrefix = 'raw/';
        let currentVideoUrl = '';

        async function loadStats() {
            try {
                const [rawRes, posesRes] = await Promise.all([
                    fetch('/videos?prefix=raw/'),
                    fetch('/videos?prefix=poses/')
                ]);
                const raw = await rawRes.json();
                const poses = await posesRes.json();

                document.getElementById('stat-videos').textContent = raw.count;
                document.getElementById('stat-poses').textContent = poses.count;

                const totalMB = raw.videos.reduce((sum, v) => sum + v.size_mb, 0);
                document.getElementById('stat-size').textContent =
                    totalMB > 1000 ? (totalMB/1000).toFixed(1) + ' GB' : totalMB.toFixed(0) + ' MB';
            } catch (e) {}
        }

        async function loadVideos(prefix) {
            currentPrefix = prefix;
            const grid = document.getElementById('grid');
            grid.innerHTML = '<div class="loading"><div class="loading-spinner"></div>Loading...</div>';

            try {
                const res = await fetch(`/videos?prefix=${prefix}`);
                const data = await res.json();

                if (data.count === 0) {
                    grid.innerHTML = '<div class="empty"><div class="empty-icon">📁</div>No files found</div>';
                    return;
                }

                grid.innerHTML = data.videos.map(v => {
                    const isVideo = !v.key.endsWith('.json') && !v.key.endsWith('.jpg');
                    const baseName = v.name.replace(/\.(mp4|mov|MP4|MOV)$/i, '');
                    const thumbUrl = '/thumb/' + encodeURIComponent(baseName + '.jpg');
                    const date = new Date(v.modified).toLocaleDateString();

                    return `
                    <div class="card" onclick="openFile('${v.key}', '${v.name}')">
                        <div class="card-thumb">
                            ${v.key.endsWith('.json') ? '<span style="font-size:32px;color:var(--text-muted)">{}</span>' :
                              v.key.endsWith('.jpg') ? '' :
                              '<img src="' + thumbUrl + '" onerror="this.outerHTML=\\'<span style=font-size:32px>🎬</span>\\'">'}
                            ${isVideo ? '<div class="play-icon">▶</div>' : ''}
                        </div>
                        <div class="card-body">
                            <div class="card-title" title="${v.name}">${v.name}</div>
                            <div class="card-meta">
                                <span>${v.size_mb} MB</span>
                                <span>${date}</span>
                            </div>
                        </div>
                    </div>`;
                }).join('');

                
                            } catch (e) {
                grid.innerHTML = `<div class="empty"><div class="empty-icon">⚠️</div>Error: ${e.message}</div>`;
            }
        }

        async function openFile(key, name) {
            if (key.endsWith('.json')) {
                const res = await fetch(`/videos/url/${key}`);
                const data = await res.json();
                window.open(data.url, '_blank');
                return;
            }

            if (key.endsWith('.jpg')) {
                const res = await fetch(`/videos/url/${key}`);
                const data = await res.json();
                window.open(data.url, '_blank');
                return;
            }

            const modal = document.getElementById('modal');
            const player = document.getElementById('player');
            const title = document.getElementById('modal-title');

            title.textContent = name;
            modal.classList.add('active');
            currentSpeed = 1;
            updateSpeedButtons();

            // Update URL bar
            history.pushState({video: key}, name, '/watch/' + encodeURIComponent(key));

            const res = await fetch('/videos/url/' + key + '?expires=86400');
            const data = await res.json();
            currentVideoUrl = data.url;
            player.src = data.url;
            player.play();
        }

        async function copyVideoUrl() {
            if (!currentVideoUrl) return;
            try {
                await navigator.clipboard.writeText(currentVideoUrl);
                const btn = document.getElementById('share-btn');
                btn.textContent = '✓ Copied!';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = '📋 Copy Link';
                    btn.classList.remove('copied');
                }, 2000);
            } catch (e) {
                // Fallback for older browsers
                prompt('Copy this link:', currentVideoUrl);
            }
        }

        function closeModal() {
            const modal = document.getElementById('modal');
            const player = document.getElementById('player');
            modal.classList.remove('active');
            player.pause();
            player.src = '';
            currentVideoUrl = '';
            history.pushState({}, 'Tennis Analysis', '/');
        }

        function setSpeed(speed) {
            const player = document.getElementById('player');
            player.playbackRate = speed;
            currentSpeed = speed;
            updateSpeedButtons();
        }

        function updateSpeedButtons() {
            document.querySelectorAll('.control-group button').forEach(b => {
                if (b.textContent.includes('×')) {
                    const s = parseFloat(b.textContent);
                    b.classList.toggle('active', s === currentSpeed);
                }
            });
        }

        function togglePlay() {
            const player = document.getElementById('player');
            if (player.paused) player.play();
            else player.pause();
        }

        function skipFrames(n) {
            const player = document.getElementById('player');
            player.pause();
            player.currentTime = Math.max(0, player.currentTime + n / 60);
        }

        // Update play button state
        document.getElementById('player').addEventListener('play', () => {
            document.getElementById('play-btn').textContent = '⏸';
        });
        document.getElementById('player').addEventListener('pause', () => {
            document.getElementById('play-btn').textContent = '▶';
        });

        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                loadVideos(tab.dataset.prefix);
            });
        });

        // Keyboard controls
        document.addEventListener('keydown', e => {
            const modal = document.getElementById('modal');
            if (!modal.classList.contains('active')) return;

            if (e.key === 'Escape') closeModal();
            else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
            else if (e.key === 'ArrowUp') {
                e.preventDefault();
                const idx = speeds.indexOf(currentSpeed);
                if (idx < speeds.length - 1) setSpeed(speeds[idx + 1]);
            }
            else if (e.key === 'ArrowDown') {
                e.preventDefault();
                const idx = speeds.indexOf(currentSpeed);
                if (idx > 0) setSpeed(speeds[idx - 1]);
            }
            else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                skipFrames(e.shiftKey ? -10 : -1);
            }
            else if (e.key === 'ArrowRight') {
                e.preventDefault();
                skipFrames(e.shiftKey ? 10 : 1);
            }
        });

        // Handle back/forward buttons
        window.onpopstate = (e) => {
            if (e.state && e.state.video) {
                const key = e.state.video;
                const name = key.split('/').pop();
                openFile(key, name);
            } else {
                closeModal();
            }
        };

        // Check for direct video URL on load
        async function checkDirectUrl() {
            const path = window.location.pathname;
            if (path.startsWith('/watch/')) {
                const key = decodeURIComponent(path.replace('/watch/', ''));
                const name = key.split('/').pop();
                await loadVideos('raw/');
                openFile(key, name);
                return true;
            }
            return false;
        }

        // Initial load
        loadStats();
        checkDirectUrl().then(handled => {
            if (!handled) loadVideos('raw/');
        });
    </script>
</body>
</html>'''
