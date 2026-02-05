# Tennis Analysis Pipeline — Roadmap

## Vision
An end-to-end tennis training analysis system that automatically captures, analyzes, and provides feedback on tennis practice sessions. Starting with automated highlight reels, evolving toward real-time on-device coaching.

## Current State (v1.0) ✓

### What's Built
- **Automated pipeline**: iPhone → iCloud → GPU processing → YouTube upload
- **Shot classification**: GRU neural network (93.6% accuracy) detecting forehand/backhand/serve
- **Pose estimation**: MediaPipe 33-keypoint extraction at 60fps
- **Highlight compilation**: Normal speed + 0.25x slow-mo from 240fps source
- **Multi-GPU processing**: Parallel dispatch across 2 Windows machines (RTX 5080 + 4080)
- **Album-based workflow**: Two iCloud albums control clip ordering (chronological vs grouped by type)
- **Push notifications**: ntfy.sh integration for processing status
- **Launchd service**: Auto-start daemon on Mac login

### Current Limitations
- Single camera angle (back-court only)
- Shot type only (no spin, quality, or technique analysis)
- Manual labeling required for training data
- Large file uploads (no pre-trimming)

---

## Roadmap

### Phase 1: Expand Detection Capabilities (High Impact, Low Effort)

#### 1.1 Multi-Angle Detection
- **Goal**: Train model to recognize shots from any camera position
- **Camera views**: back-court (current), left-side, right-side, front, overhead
- **Approach**:
  - Add `view_angle` label to training pipeline
  - Label existing + new videos with view metadata
  - Retrain model with view-aware architecture (or separate models per view)
- **Files to modify**: `visual_label.py`, `config/settings.py`, `train_model.py`

#### 1.2 Audio-Based Shot Detection
- **Goal**: Use ball-on-racket sound as shot timing signal
- **Approach**:
  - Extract audio from video, detect amplitude spikes
  - Optional: Train classifier on spectrogram for ball-hit vs other sounds
  - Use audio timestamps to assist/validate pose-based detection
- **Benefit**: Works regardless of camera angle or occlusion

#### 1.3 Good vs Bad Shot Classification
- **Goal**: Classify shot quality, not just shot type
- **Labels**: good / needs-work (or 1-5 scale)
- **Training**: Requires labeled examples of good vs bad technique
- **Use case**: Filter highlights to show only good shots, or create "areas to improve" reels

#### 1.4 Smart Trimming
- **Goal**: Reduce file size before upload (40-50% reduction)
- **Approach**: Detect periods of inactivity (no player movement) and trim
- **Implementation**: ~100 lines, run before preprocessing
- **Benefit**: Faster uploads, less storage

---

### Phase 2: Enhanced Analysis (Medium Impact, Variable Effort)

#### 2.1 Spin Type Classification
- **Labels**: topspin, slice, flat (per shot type)
- **Approach**: May require higher-fidelity pose data or ball tracking
- **Dependency**: Multi-angle detection helps here

#### 2.2 Side-by-Side Comparison
- **Goal**: Compare your shot to a reference (pro or previous best)
- **Approach**: Pose alignment + overlay visualization
- **Use case**: "Here's your forehand vs Federer's forehand"

#### 2.3 Player Level Classification
- **Goal**: Tag videos/shots by skill level (beginner/intermediate/advanced)
- **Use case**: Track improvement over time, personalize feedback

#### 2.4 Metadata Tagging
- **Goal**: Auto-tag clips with searchable metadata
- **Tags**: shot type, quality, spin, court position, rally length
- **Use case**: "Show me all my backhand slices from the baseline"

---

### Phase 3: iOS App (High Impact, High Effort)

#### 3.1 MVP iOS App
- **Features**:
  - Pre-trim videos on-device before upload
  - Quick review of clips before syncing
  - Manual shot labeling UI (faster than desktop)

#### 3.2 On-Device Processing
- **Features**:
  - CoreML model for shot detection
  - Real-time pose overlay during recording
  - Instant highlight compilation (no cloud round-trip)
- **Benefit**: Works offline, immediate feedback

#### 3.3 Real-Time Coaching
- **Features**:
  - Live audio/haptic feedback during practice
  - "Good shot!" / "Watch your elbow" style cues
  - Session summary with key moments

#### 3.4 Coach Feedback Automation
- **Goal**: Generate natural language feedback from shot analysis
- **Example**: "Your backhand slice is landing short. Try more follow-through."
- **Dependency**: Good/bad classification + technique analysis

---

### Phase 4: Scale (If Needed)

#### 4.1 Cloud Migration
- **When**: If processing demand exceeds home GPU capacity
- **Options**: AWS/GCP with GPU instances, or dedicated ML inference service
- **Consideration**: Cost vs convenience tradeoff

#### 4.2 Multi-User Support
- **When**: If sharing with coach or training partners
- **Features**: User accounts, shared video library, permission controls

---

## Priority Matrix

| Feature | Impact | Effort | Status |
|---------|--------|--------|--------|
| Multi-Angle Detection | High | Low | Next up |
| Audio Shot Detection | High | Low | Planned |
| Good/Bad Classification | High | Low | Planned |
| Smart Trimming | High | Low | Planned |
| iOS App MVP | High | High | Future |
| Real-Time Feedback | High | High | Future |
| Side-by-Side Comparison | Medium | Low | Backlog |
| Player Level Classification | Medium | Low | Backlog |
| Spin Type Classification | Medium | Medium | Backlog |
| Metadata Tagging | Low | Low | Backlog |
| Cloud Migration | Medium | High | If needed |

---

## Next Steps
1. Label left-side-view video, add view_angle to training pipeline
2. Implement audio peak detection for shot timing
3. Retrain model with multi-view data
