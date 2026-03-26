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
    pp = os.path.join(PROJECT_ROOT, 'preprocessed', f'{vid}.mp4')

    if os.path.exists(thumb):
        return True
    if not os.path.exists(pp):
        return False

    subprocess.run(
        ['ffmpeg', '-y', '-ss', '5', '-i', pp, '-vframes', '1',
         '-q:v', '8', '-vf', 'scale=480:-1', thumb],
        capture_output=True, timeout=30
    )
    return os.path.exists(thumb)


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


UPLOAD_SECTION = '''
<div class="upload-section">
  <button class="upload-toggle" onclick="this.nextElementSibling.classList.toggle('open')">Upload Video</button>
  <div class="upload-panel">
    <div class="mode-tabs">
      <button class="mode-tab active" onclick="setMode('file',this)">File Upload</button>
      <button class="mode-tab" onclick="setMode('link',this)">iCloud Drive</button>
    </div>
    <div id="fileMode">
      <input type="file" id="videoFile" accept=".mov,.mp4">
    </div>
    <div id="linkMode" style="display:none">
      <div class="link-hint" style="padding:12px 0">
        On iPhone: open video in Photos &gt; Share &gt; Save to Files &gt;<br>
        iCloud Drive &gt; <b>Tennis Highlights</b> folder.<br><br>
        Video will sync automatically and get processed.
        No upload needed!
      </div>
    </div>
    <input type="password" id="uploadPwd" placeholder="Password">
    <button id="uploadBtn" onclick="startUpload()">Upload</button>
    <div class="progress-container" id="progressContainer">
      <div class="progress-bar" id="progressBar"></div>
      <span id="progressText"></span>
    </div>
    <div id="uploadStatus"></div>
  </div>
</div>
<script>
const CHUNK_SIZE = 5 * 1024 * 1024;
let uploadMode = 'file';
let wakeLock = null;

async function acquireWakeLock() {
  try { if (navigator.wakeLock) wakeLock = await navigator.wakeLock.request('screen'); } catch {}
}
function releaseWakeLock() {
  if (wakeLock) { wakeLock.release(); wakeLock = null; }
}

function setMode(mode, el) {
  uploadMode = mode;
  document.getElementById('fileMode').style.display = mode === 'file' ? '' : 'none';
  document.getElementById('linkMode').style.display = mode === 'link' ? '' : 'none';
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
}

async function uploadChunk(url, chunk, retries) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const r = await fetch(url, { method: 'PUT', body: chunk });
      if (r.ok) return await r.json();
      if (attempt === retries) throw new Error('HTTP ' + r.status);
    } catch (err) {
      if (attempt === retries) throw err;
      await new Promise(ok => setTimeout(ok, 2000 * (attempt + 1)));
    }
  }
}

async function startUpload() {
  const password = document.getElementById('uploadPwd').value;
  if (!password) { alert('Enter password'); return; }

  const btn = document.getElementById('uploadBtn');
  const prog = document.getElementById('progressContainer');
  const bar = document.getElementById('progressBar');
  const txt = document.getElementById('progressText');
  const st = document.getElementById('uploadStatus');

  btn.disabled = true;
  bar.style.width = '0%';
  bar.style.background = '#FF8C00';

  if (uploadMode === 'link') { return; }

  const file = document.getElementById('videoFile').files[0];
  if (!file) { alert('Select a file'); btn.disabled = false; return; }

  prog.style.display = 'block';
  st.textContent = 'Initializing...';
  await acquireWakeLock();

  try {
    const initRes = await fetch('/api/upload/init', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password, filename: file.name }),
    });
    if (!initRes.ok) { const e = await initRes.json(); throw new Error(e.error); }
    const { id } = await initRes.json();

    st.textContent = 'Uploading...';
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const parts = [];

    for (let i = 0; i < totalChunks; i++) {
      const start = i * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const chunk = file.slice(start, end);

      const partData = await uploadChunk('/api/upload/' + id + '/' + (i + 1), chunk, 3);
      parts.push(partData);

      const pct = Math.round((end / file.size) * 100);
      bar.style.width = pct + '%';
      txt.textContent = pct + '% (' + (end / 1048576).toFixed(0) + ' / ' + (file.size / 1048576).toFixed(0) + ' MB)';
    }

    st.textContent = 'Finalizing...';
    const cr = await fetch('/api/upload/' + id + '/complete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parts }),
    });
    if (!cr.ok) throw new Error('Finalize failed');

    bar.style.width = '100%';
    bar.style.background = '#27AE60';
    st.innerHTML = 'Upload complete! Processing will begin shortly.<br><small>Video will appear on this page when ready.</small>';
    pollStatus(id);
  } catch (err) {
    st.textContent = 'Error: ' + err.message;
    bar.style.background = '#E74C3C';
  } finally {
    btn.disabled = false;
    releaseWakeLock();
  }
}

function pollStatus(id) {
  const st = document.getElementById('uploadStatus');
  const iv = setInterval(async () => {
    try {
      const r = await fetch('/api/status/' + id);
      const d = await r.json();
      if (d.status === 'not_found') {
        st.innerHTML = 'Video processed! <a href="javascript:location.reload()">Refresh page</a>';
        clearInterval(iv);
      } else if (d.status === 'failed') {
        st.textContent = 'Processing failed.';
        clearInterval(iv);
      } else if (d.status === 'pending') {
        st.textContent = 'Queued for processing...';
      } else if (d.status === 'processing') {
        st.textContent = 'Processing video...';
      }
    } catch {}
  }, 30000);
}
</script>
'''


def build_index_html(videos_meta):
    """Build the HTML index page."""
    sorted_vids = sorted(videos_meta.keys(),
                         key=lambda v: videos_meta[v].get('created', ''),
                         reverse=True)

    label_map = {
        'timeline': ('Timeline', '#FF8C00'),
        'rally': ('Rally', '#27AE60'),
        'rally_slowmo': ('Rally Slow-Mo', '#9B59B6'),
        'grouped': ('By Type', '#5DADE2'),
        'highlights': ('Highlights', '#2ECC71'),
        'highlights_slowmo': ('Highlights Slow-Mo', '#8E44AD'),
        'grouped_slowmo': ('Grouped Slow-Mo', '#3498DB'),
    }

    cards = ''
    for vid in sorted_vids:
        m = videos_meta[vid]
        dur = m.get('duration', 0)
        dur_str = f'{int(dur)//60}:{int(dur)%60:02d}' if dur else '?'
        shots = m.get('shots', '?')
        date_str, time_str = format_date_time(m.get('created', ''))

        bd = m.get('breakdown', {})
        abbrev = {'serve': 'S', 'forehand': 'FH', 'backhand': 'BH', 'unknown_shot': '?'}
        bd_parts = []
        for st in ['serve', 'forehand', 'backhand']:
            if st in bd:
                bd_parts.append(f'{bd[st]} {abbrev[st]}')
        bd_str = ', '.join(bd_parts) if bd_parts else ''

        if m.get('has_thumb'):
            thumb_url = f'https://media.playfullife.com/highlights/thumbs/{vid}.jpg'
            thumb_html = f'<img src="{thumb_url}" alt="{vid}" loading="lazy">'
        else:
            thumb_html = f'<div class="no-thumb">{vid}</div>'

        links = []
        for f in m.get('files', []):
            url = f'https://media.playfullife.com/highlights/{vid}/{f}'
            key = f.replace(vid + '_', '').replace('.mp4', '')
            label, color = label_map.get(key, (key, '#5DADE2'))
            links.append(f'<a href="{url}" style="background:{color}">{label}</a>')

        time_html = f'<span class="time">{time_str}</span>' if time_str else ''
        bd_html = f'<div class="breakdown">{bd_str}</div>' if bd_str else ''

        cards += f'''
    <div class="card">
      <div class="thumb">{thumb_html}</div>
      <div class="info">
        <div class="title">{vid}</div>
        <div class="meta">
          <span class="date">{date_str}</span>
          {time_html}
          <span class="dur">{dur_str}</span>
          <span class="shots">{shots} shots</span>
        </div>
        {bd_html}
        <div class="links">{" ".join(links)}</div>
      </div>
    </div>'''

    return f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tennis Highlights</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; background: #0a0a0a; color: #eee; padding: 20px; }}
  h1 {{ color: #FF8C00; text-align: center; margin: 20px 0 30px; font-size: 1.8em; }}
  .grid {{ max-width: 900px; margin: 0 auto; }}
  .card {{
    display: flex; gap: 16px; padding: 16px;
    background: #1a1a1a; border-radius: 12px;
    margin-bottom: 12px; border: 1px solid #2a2a2a;
    transition: border-color 0.2s;
  }}
  .card:hover {{ border-color: #FF8C00; }}
  .thumb {{ flex-shrink: 0; width: 200px; }}
  .thumb img {{ width: 200px; border-radius: 8px; display: block; }}
  .no-thumb {{
    width: 200px; height: 112px; background: #222; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: #555; font-size: 0.9em;
  }}
  .info {{ flex: 1; display: flex; flex-direction: column; gap: 6px; }}
  .title {{ font-size: 1.15em; font-weight: 700; color: #fff; }}
  .meta {{ display: flex; gap: 12px; flex-wrap: wrap; font-size: 0.85em; color: #999; }}
  .meta span {{ display: inline-flex; align-items: center; gap: 4px; }}
  .date::before {{ content: "\\1F4C5 "; }}
  .time::before {{ content: "\\1F552 "; }}
  .dur::before {{ content: "\\23F1 "; }}
  .shots::before {{ content: "\\1F3BE "; }}
  .breakdown {{ font-size: 0.85em; color: #bbb; }}
  .links {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }}
  .links a {{
    color: #fff; text-decoration: none; font-size: 0.8em; font-weight: 600;
    padding: 5px 12px; border-radius: 6px; display: inline-block;
    opacity: 0.9; transition: opacity 0.15s;
  }}
  .links a:hover {{ opacity: 1; }}
  .upload-section {{ max-width: 900px; margin: 0 auto 24px; text-align: center; }}
  .upload-toggle {{
    padding: 10px 24px; background: #FF8C00; color: #fff; border: none;
    border-radius: 8px; font-size: 1em; font-weight: 600; cursor: pointer;
  }}
  .upload-panel {{
    max-height: 0; overflow: hidden; transition: max-height 0.3s, padding 0.3s;
    text-align: left; margin-top: 12px; background: #1a1a1a; border-radius: 12px;
  }}
  .upload-panel.open {{ max-height: 500px; padding: 20px; border: 1px solid #2a2a2a; }}
  .upload-panel input[type="file"], .upload-panel input[type="password"] {{
    display: block; width: 100%; margin-bottom: 10px; padding: 10px;
    background: #222; border: 1px solid #333; border-radius: 6px; color: #eee; font-size: 0.95em;
  }}
  .upload-panel button {{
    padding: 8px 20px; background: #FF8C00; color: #fff; border: none;
    border-radius: 6px; font-weight: 600; cursor: pointer;
  }}
  .progress-container {{
    display: none; margin-top: 12px; background: #222; border-radius: 6px;
    overflow: hidden; position: relative; height: 24px;
  }}
  .progress-bar {{ height: 100%; background: #FF8C00; width: 0; transition: width 0.3s; }}
  #progressText {{
    position: absolute; top: 3px; left: 0; right: 0; text-align: center;
    font-size: 0.8em; color: #fff;
  }}
  #uploadStatus {{ margin-top: 10px; font-size: 0.9em; color: #bbb; }}
  .mode-tabs {{ display: flex; gap: 0; margin-bottom: 12px; }}
  .mode-tab {{
    flex: 1; padding: 8px; background: #222; border: 1px solid #333; color: #888;
    cursor: pointer; font-size: 0.9em; font-weight: 600;
  }}
  .mode-tab:first-child {{ border-radius: 6px 0 0 6px; }}
  .mode-tab:last-child {{ border-radius: 0 6px 6px 0; }}
  .mode-tab.active {{ background: #FF8C00; color: #fff; border-color: #FF8C00; }}
  .upload-panel input[type="text"] {{
    display: block; width: 100%; margin-bottom: 6px; padding: 10px;
    background: #222; border: 1px solid #333; border-radius: 6px; color: #eee; font-size: 0.95em;
  }}
  .link-hint {{ font-size: 0.8em; color: #666; margin-bottom: 10px; }}
  @media (max-width: 600px) {{
    .card {{ flex-direction: column; }}
    .thumb {{ width: 100%; }}
    .thumb img {{ width: 100%; }}
    .no-thumb {{ width: 100%; }}
  }}
</style>
</head><body>
<h1>Tennis Highlights</h1>
{UPLOAD_SECTION}
<div class="grid">{cards}
</div>
</body></html>
'''


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
    c.upload(tmp.name, 'highlights/index.html', content_type='text/html')
    os.unlink(tmp.name)
    print(f'Updated index: {len(all_meta)} videos')
    print('https://media.playfullife.com/highlights/index.html')


if __name__ == '__main__':
    update_index()
