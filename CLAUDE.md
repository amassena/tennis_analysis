# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tennis video analysis pipeline that processes 240fps iPhone footage to auto-detect shot types (forehand, backhand, serve, neutral), extract clips, and compile highlight reels. Uses MediaPipe pose estimation feeding into a GRU neural network for temporal shot classification.

## Three-Machine Setup

- **Mac (M4)**: iCloud downloads, manual labeling (visual_label.py GUI), lightweight tasks, pipeline orchestration
- **Windows PC "windows" (RTX 5080, CUDA 12.6, Python 3.11)**: Primary GPU worker via `ssh windows`
- **Windows PC "tmassena" (RTX 4080, CUDA 12.6, Python 3.11)**: Secondary GPU worker via `ssh tmassena`
- Both GPU machines run:
  - Video preprocessing (NVENC via `preprocess_nvenc.py`)
  - Pose extraction (MediaPipe)
  - Model training (TensorFlow)
  - Shot detection (inference)
  - Clip extraction & highlight compilation (FFmpeg h264_nvenc)
- Connected via **Tailscale SSH**, file transfer with `scp`
- All GPU machines mirror the project at `C:\Users\amass\tennis_analysis`
- When multiple videos are queued, `auto_pipeline.py` dispatches them in parallel across both GPU machines

**Rule: Always run matching, editing, and encoding on the GPUs (windows/tmassena). It is significantly faster.**

## Technology Stack

- **Python 3.9.6** (Mac) / **Python 3.11** (Windows PCs) with virtual environments
- **MediaPipe 0.10.18**: Pose estimation (33 keypoints per frame). Note: must use 0.10.18 for `solutions` API compatibility
- **TensorFlow/Keras**: GRU neural network for shot classification (2.16 Mac, 2.20 Windows)
- **FFmpeg**: Video preprocessing (VFR->CFR), clip extraction, highlight compilation. Uses **NVENC (h264_nvenc)** on Windows for GPU-accelerated encoding
- **iCloud**: Video download from iPhone (240fps @ 1080p or 4K @ 120fps)
- **GPUs**: NVIDIA RTX 5080 (16GB, "windows") + RTX 4080 (16GB, "tmassena") for training, inference, and video encoding

## Pipeline Architecture

```
iPhone (240fps VFR .mov)
  -> iCloud Download (Mac) -> raw/
  -> scp raw files to Windows
  -> VFR->CFR conversion (NVENC, 60fps) -> preprocessed/      [GPU]
  -> MediaPipe pose extraction -> poses/ (.json, 33 kp/frame)  [GPU]
  -> scp poses back to Mac
  -> Manual labeling (Mac GUI) -> {video}_labels.csv
  -> Auto-fill neutral gaps -> updated CSV
  -> label_clips.py -> training_data/{forehand,backhand,serve,neutral}/
  -> GRU model training -> models/shot_classifier.h5           [GPU]
  -> Shot detection (inference) -> shots_detected.json          [GPU]
  -> Clip extraction (NVENC) -> clips/{type}/*.mp4              [GPU]
  -> Highlight compilation (NVENC) -> highlights/*_highlights.mp4 [GPU]
  -> scp highlights back to Mac
  -> Upload to YouTube (unlisted) + iCloud Drive
```

## Project Structure

```
tennis_analysis/
├── raw/                  # Original .mov files from iCloud
├── preprocessed/         # CFR-converted .mp4 at 60fps
├── poses/                # Per-frame pose JSON (33 keypoints)
├── training_data/        # Labeled sliding-window clips for training
│   ├── forehand/
│   ├── backhand/
│   ├── serve/
│   └── neutral/
├── models/               # Trained model files (.h5 + _meta.json)
├── clips/                # Extracted per-shot clips by type
│   ├── forehand/
│   ├── backhand/
│   └── serve/
├── highlights/           # Compiled highlight reels by shot type
├── scripts/
│   ├── icloud_download.py
│   ├── preprocess_videos.py    # Mac (libx264)
│   ├── extract_poses.py        # --visualize for skeleton overlay
│   ├── visual_label.py         # GUI labeler (Mac only, needs display)
│   ├── label_clips.py          # CSV + poses -> training clips
│   ├── train_model.py          # GRU training pipeline
│   ├── detect_shots.py         # Run model on full video poses
│   ├── extract_clips.py        # Clip extraction + highlights (NVENC-aware)
│   └── auto_pipeline.py        # Automated iCloud -> GPU -> YouTube daemon
├── preprocess_nvenc.py         # Windows NVENC preprocessing
├── pipeline_state.json         # Tracks processed videos and YouTube URLs
├── {video}_labels.csv          # Per-video manual + auto labels
└── shots_detected*.json        # Detection output per video
```

## Key Design Decisions

- **60fps CFR** for ML processing (downsampled from 240fps; balances speed and quality)
- **MediaPipe** over alternatives (fast, accurate, free, 33-keypoint model)
- **GRU** over simple classifiers (captures temporal motion patterns in pose sequences)
- **NVENC** for all video encoding (h264_nvenc on RTX 5080, falls back to libx264 on Mac)
- **iCloud integration** for native iPhone camera workflow
- **Neutral auto-labeling**: gaps between labeled shots auto-filled as neutral (min 30 frames / 0.5s)
- Custom solution (not SwingVision) for complete control over pipeline

## Current Model Performance

- **Val accuracy: 93.6%** (exceeds 85% target)
- Trained on 1 video (IMG_6665): 2045 clips (164 forehand, 69 backhand, 372 serve, 1818 neutral)
- Per-class F1: forehand 0.78, backhand 0.72, serve 0.95, neutral 0.96
- Weakest: backhand precision (0.59), forehand precision (0.71)
- **To improve**: label more videos and retrain

## Automated Pipeline

Fully automated mode: add videos to an iCloud album on iPhone, and the pipeline downloads, processes, and uploads highlights to YouTube with no manual intervention. Two albums control clip ordering:
- **"Tennis Videos"**: chronological order (see flow of play)
- **"Tennis Videos Group By Shot Type"**: grouped by serve, forehand, backhand

```
iPhone -> add video to album -> iCloud sync
  -> Mac polls iCloud album (5 min interval)
  -> Downloads new videos to raw/
  -> Assigns each video to a GPU machine (round-robin across windows/tmassena)
  -> SCP raw .mov to assigned GPU machine
  -> SSH GPU: preprocess (NVENC 60fps) -> extract poses -> detect shots -> extract clips
  -> SSH GPU: compile combined video (normal speed + 0.25x slow-mo from raw 240fps)
  -> SCP combined video to Mac
  -> Upload to YouTube: "Training session (YYYY-MM-DD) video N" (unlisted)
  -> When 2+ videos found: processes in parallel across both GPU machines
```

**Running:**
- Daemon mode: `python scripts/auto_pipeline.py` (polls every 5 minutes)
- Single pass: `python scripts/auto_pipeline.py --once`
- Debug (dumps iCloud field names): `python scripts/auto_pipeline.py --debug`

**Launchd service** (auto-starts on login, restarts on crash):
- Start: `launchctl load ~/Library/LaunchAgents/com.tennis-analysis.auto-pipeline.plist`
- Stop: `launchctl unload ~/Library/LaunchAgents/com.tennis-analysis.auto-pipeline.plist`
- Status: `launchctl list | grep tennis`
- Logs: `pipeline.log` and `pipeline_error.log` in project root

**State tracking:** `pipeline_state.json` records processed asset IDs, YouTube URLs, and daily video counts.

**Combined video format:** The output highlight has two sections:
1. Normal-speed highlights (all shots concatenated from 60fps preprocessed video)
2. 0.25x slow-motion highlights (same shots extracted from original 240fps, using `setpts=4.0*PTS`)

## Iterative Workflow

1. **Record** practice on iPhone (240fps)
2. **Download** via iCloud to Mac (`scripts/icloud_download.py`)
3. **Transfer** raw files to Windows (`scp`)
4. **Preprocess** on GPU (`preprocess_nvenc.py` -- NVENC, 60fps CFR)
5. **Extract poses** on GPU (`scripts/extract_poses.py`)
6. **Transfer** poses back to Mac (`scp`)
7. **Label** on Mac (`scripts/visual_label.py` with skeleton video)
8. **Auto-fill neutrals** from label gaps
9. **Generate training clips** (`scripts/label_clips.py --csv`)
10. **Train model** on GPU (`scripts/train_model.py`)
11. **Detect shots** on GPU (`scripts/detect_shots.py`)
12. **Extract clips + highlights** on GPU (`scripts/extract_clips.py --highlights`)
13. **Transfer** highlights to Mac (`scp`)
14. **Review** and iterate (label more videos, retrain if needed)

## Target Metrics

- Pose detection rate: >90% of frames
- Shot classification accuracy: >85% on validation set
- Processing time: <1 hour per 15min practice video

## Known Issues

- **mediapipe version**: Must use 0.10.18 (0.10.32+ dropped `solutions` API)
- **protobuf**: Windows needs protobuf==5.28.3 despite mediapipe wanting <5
- **opencv**: Use opencv-contrib-python only (conflicts with opencv-python)
- **Windows encoding**: Avoid Unicode arrows/special chars in print statements (cp1252)
- **visual_label.py**: Must run from interactive terminal with display, not subprocess
