## Tennis Analysis Pipeline — System Overview

### Recording & Source
- **iPhone 16 Pro Max** — 240fps @ 1080p HEVC
- **iCloud Photos** — automatic sync of original-quality .MOV files
- **pyicloud** — Python library for authenticated iCloud download with 2FA support

### Video Processing
- **FFmpeg** — VFR→CFR conversion (240fps variable → 60fps constant)
- **FFmpeg + NVENC** (Windows PC) — GPU-accelerated H.264 encoding via RTX 5080
- **Encoding settings** — CQ 32, preset p4, yuv420p, 240fps native or 60fps 4x slow-mo
- **Slow motion** — `setpts=4*PTS` filter to bake 4x slow-mo for YouTube at 60fps

### Machine Learning Pipeline
- **MediaPipe** — 33-keypoint pose estimation per frame (model_complexity=2)
- **TensorFlow/Keras** — GRU neural network for temporal shot classification
- **Architecture** — 2-layer bidirectional GRU (128 units), dropout 0.3, Dense(64) → Softmax
- **Training data** — sliding-window clips (30 frames = 0.5s at 60fps), normalized 99 features (33 keypoints × 3 coords)
- **Current model** — 2-class (serve/neutral), trained on 378 clips (253 neutral, 125 serve)
- **Target** — 4-class (forehand, backhand, serve, neutral), >85% validation accuracy
- **Inference** — sliding window with stride=5, segment merging, confidence filtering

### Machines & Networking
All machines connected via **Tailscale** mesh VPN — accessible from anywhere (home, cellular, remote).

| Machine | Tailscale IP | Role | Specs |
|---------|-------------|------|-------|
| MacBook Air | `100.115.41.118` | Development, orchestration, uploads | Apple Silicon, Python 3.9 |
| Windows PC (windows) | `100.81.64.103` | Primary GPU processing, NVENC encoding | RTX 5080 16GB, CUDA 12.6, Python 3.11, Windows 11 |
| Windows PC (tmassena) | `100.98.226.93` | Secondary GPU processing, NVENC encoding | RTX 4080 16GB, CUDA 12.6, Python 3.11, Windows 11 |
| iPhone 16 Pro Max | `100.73.29.124` | Recording, remote SSH access | iOS, Tailscale app |

**SSH access:**
- Mac: `ssh andrewhome@100.115.41.118`
- PC #1: `ssh amass@100.81.64.103` (aliased as `ssh windows` from Mac)
- PC #2: `ssh amass@100.98.226.93` (aliased as `ssh tmassena` from Mac)

### Cloud Services
| Service | Purpose | Auth |
|---------|---------|------|
| **iCloud** | Video download from iPhone, highlight archival | pyicloud + 2FA, credentials in `.env` |
| **YouTube Data API v3** | Unlisted video uploads | OAuth 2.0, `config/client_secrets.json` |
| **Google Cloud Console** | YouTube API project ("Tennis Highlights") | OAuth consent screen, test user added |
| **GitHub** | Source control | SSH key (ed25519) on MacBook Air |

### Repository (github.com/amassena/tennis_analysis)
```
scripts/
  icloud_download.py   — Download videos from iCloud Photos
  extract_poses.py     — MediaPipe pose extraction → JSON
  visual_label.py      — OpenCV GUI for labeling shot segments
  label_clips.py       — Generate training clips from labels
  train_model.py       — Train GRU model on labeled clips
  detect_shots.py      — Run inference on full video poses
  render_detections.py — Overlay shot labels on video
  upload.py            — Upload to YouTube (unlisted) + iCloud Drive

config/
  settings.py          — Central config (paths, hyperparams, video settings)
  client_secrets.json  — YouTube OAuth (gitignored)
  youtube_credentials.json — Saved OAuth tokens (gitignored)
  icloud_session/      — Persisted iCloud 2FA session (gitignored)
```

### Pipeline Flow
```
iPhone 240fps .MOV
  → iCloud sync
  → icloud_download.py → raw/
  → FFmpeg VFR→CFR (60fps) → preprocessed/
  → extract_poses.py (MediaPipe) → poses/*.json
  → visual_label.py → *_labels.csv
  → label_clips.py → training_data/{type}/
  → train_model.py → models/shot_classifier.h5
  → detect_shots.py → shots_detected.json
  → FFmpeg + NVENC clip extraction → clips/
  → FFmpeg + NVENC highlight compilation (4x slow-mo) → highlights/
  → upload.py → YouTube (unlisted) + iCloud Drive
```

### What's Working End-to-End
- Full pipeline verified on **IMG_0870** (Mac) and **IMG_0864** (Mac + Windows PC)
- YouTube upload confirmed: https://youtu.be/0Y2w9579i9s
- iCloud Drive upload confirmed: `Tennis/` folder

### What's Working End-to-End (continued)
- **Multi-machine parallel dispatch** — auto_pipeline.py distributes across windows + tmassena when 2+ videos queued

### What's Remaining
- **Forehand/backhand labeling** — 0 clips for each (need 50+)
- **4-class model retraining** — currently serve/neutral only
- **TensorFlow GPU on tmassena** — CUDA 13.1 too new for TF 2.19; inference runs on CPU (NVENC encoding still uses GPU)
- **Batch automation** — no script yet to run full pipeline on all 7 videos unattended
