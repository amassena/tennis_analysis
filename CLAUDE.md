# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tennis video analysis pipeline that processes 240fps iPhone footage to auto-detect shot types (forehand, backhand, serve, neutral), extract clips, and compile highlight reels. Uses MediaPipe pose estimation feeding into a GRU neural network for temporal shot classification.

## Technology Stack

- **Python 3.9.6** with virtual environment
- **MediaPipe**: Pose estimation (33 keypoints per frame)
- **TensorFlow/Keras**: GRU/LSTM neural network for shot classification
- **FFmpeg**: Video preprocessing (VFR→CFR conversion, clip extraction, highlight compilation)
- **iCloud**: Video download from iPhone (240fps @ 1080p or 4K @ 120fps)
- **GPU**: NVIDIA Founders Edition for training/inference

## Pipeline Architecture

```
iPhone (240fps VFR .mov)
  → iCloud Download → raw/
  → VFR→CFR conversion (60fps) → preprocessed/
  → MediaPipe pose extraction → poses/ (.json, 33 keypoints/frame)
  → Manual labeling → training_data/{forehand,backhand,serve,neutral}/
  → GRU model training → models/shot_classifier.h5
  → Shot detection → shots_detected.json
  → FFmpeg clip extraction → clips/{forehand,backhand,serve}_*.mp4
  → Highlight compilation → highlights/{type}_highlights.mp4
  → Upload to YouTube (unlisted) + iCloud Drive
```

## Project Structure

```
tennis_analysis/
├── raw/                  # Original .mov files from iCloud
├── preprocessed/         # CFR-converted .mp4 at 60fps
├── poses/                # Per-frame pose JSON (33 keypoints)
├── training_data/        # Labeled clips for model training
│   ├── forehand/
│   ├── backhand/
│   ├── serve/
│   └── neutral/
├── models/               # Trained model files (.h5)
├── clips/                # Extracted per-shot clips
├── highlights/           # Compiled highlight reels by shot type
└── shots_detected.json   # Detection output (timestamps + types)
```

## Key Design Decisions

- **60fps CFR** for ML processing (downsampled from 240fps; balances speed and quality)
- **MediaPipe** over alternatives (fast, accurate, free, 33-keypoint model)
- **GRU** over simple classifiers (captures temporal motion patterns in pose sequences)
- **iCloud integration** for native iPhone camera workflow
- Custom solution (not SwingVision) for complete control over pipeline

## Target Metrics

- Pose detection rate: >90% of frames
- Shot classification accuracy: >85% on validation set
- Processing time: <1 hour per 15min practice video

## Development Phases

1. **Environment Setup**: Project structure, venv, dependencies, GPU/FFmpeg verification
2. **iCloud Integration**: 2FA auth, video listing, original-quality download
3. **Video Preprocessing**: VFR detection, CFR conversion at 60fps
4. **Pose Estimation**: MediaPipe keypoint extraction, >90% detection verification, skeleton overlay visualization
5. **Shot Classification**: Label 50+ clips per category, build/train GRU, achieve >85% accuracy
6. **Clip Extraction & Compilation**: Run model on full videos, extract 1-2s clips, compile highlights
7. **Upload & Distribution**: YouTube (unlisted) + iCloud Drive archival
