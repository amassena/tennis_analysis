# Clip Review & Verification Session

Use this document to start a parallel Claude Code session for reviewing and verifying training clips while the pipeline is being built/run.

## Current State (as of 2026-02-25)

**202 total clips** in `training/clips/`:

| Shot Type | Total | Verified | Unverified | Priority |
|-----------|-------|----------|------------|----------|
| serve     | 114   | 57       | 57         | Medium   |
| forehand  | 76    | 39       | 37         | Medium   |
| backhand  | 12    | 4        | 8          | **HIGH** |

**Backhand is critically underrepresented** - only 12 clips total (4 verified). This is the #1 priority for labeling new videos.

**Camera angles**: All 4 source videos (IMG_0864, IMG_0865, IMG_0920, IMG_6665) have `unknown` camera angles. These need to be tagged.

**Pose data**: 100/202 clips have pose data extracted. The remaining 102 need pose extraction before they can be used for training.

## How to Run Verification

```bash
cd ~/tennis_analysis

# Verify all unverified clips (opens each in QuickTime)
python scripts/verify_clips.py

# Verify only backhand clips (highest priority)
python scripts/verify_clips.py --type backhand

# Verify only serve clips
python scripts/verify_clips.py --type serve

# Verify clips from a specific source video
python scripts/verify_clips.py --source IMG_0920
```

### Verification commands (during interactive review):
- `Enter` = label is correct (verify)
- `s` = relabel as serve
- `f` = relabel as forehand
- `b` = relabel as backhand
- `d` = delete clip (bad/unusable)
- `x` = skip (come back later)
- `q` = quit and save

Progress auto-saves every 10 clips.

## Priority Tasks

### 1. Verify backhand clips (HIGHEST PRIORITY)
```bash
python scripts/verify_clips.py --type backhand
```
Only 4/12 backhands are verified. We need at least 50 verified backhand clips for decent model training. When reviewing other shot types, watch for mislabeled backhands.

### 2. Tag camera angles on source videos
Open `training/metadata.json` and update the `camera_angles` section:
```json
"camera_angles": {
    "IMG_0864": "back",        // or "side_left", "side_right", "front"
    "IMG_0865": "back",
    "IMG_0920": "side_left",
    "IMG_6665": "back"
}
```

To determine the angle, watch the first few seconds of each preprocessed video:
```bash
open preprocessed/IMG_0864.mp4
open preprocessed/IMG_0865.mp4
open preprocessed/IMG_0920.mp4
open preprocessed/IMG_6665.mp4
```

Angle definitions:
- **back**: Camera behind the player, facing the court
- **side_left**: Camera on the left side of the court
- **side_right**: Camera on the right side of the court
- **front**: Camera facing the player from the other side of the net

### 3. Verify remaining serve and forehand clips
```bash
python scripts/verify_clips.py --type serve
python scripts/verify_clips.py --type forehand
```

### 4. Extract poses for unverified clips
After verifying clips, clips that are confirmed correct but missing pose data need pose extraction. This happens automatically when you run `process_labels.py` on new label files, but for existing clips without poses, you can run the pipeline on each source video.

## How metadata.json Works

Location: `training/metadata.json`

Each clip entry looks like:
```json
{
    "filename": "IMG_0864_serve_10.mp4",
    "shot_type": "serve",
    "source_video": "IMG_0864",
    "verified": false,
    "has_pose": false,
    "camera_angle": "unknown"
}
```

Fields updated during verification:
- `verified` -> `true` when you confirm the label
- `shot_type` -> changes if you relabel (clip file is also moved)
- Clips are deleted from the list and filesystem if you press `d`

The `camera_angles` top-level dict maps source video names to their angle. When you update this, all clips from that video inherit the angle.

## Training Targets

To train a good GRU model, we need approximately:

| Shot Type       | Target | Current (verified) | Gap    |
|-----------------|--------|--------------------|--------|
| serve           | 100    | 57                 | 43     |
| forehand        | 100    | 39                 | 61     |
| backhand        | 100    | 4                  | **96** |
| forehand_volley | 50     | 0                  | 50     |
| backhand_volley | 50     | 0                  | 50     |
| overhead        | 50     | 0                  | 50     |
| neutral         | 80     | 0                  | 80     |

## Adding New Labels

When you watch a video and identify shots, create a label file:

```bash
# Create label file
cat > labels/IMG_XXXX_labels.txt << 'EOF'
# Video: IMG_XXXX.mp4
# Date labeled: 2026-02-25
# Notes: Practice session, back view

0:05 serve
0:12 forehand
0:15 backhand
0:23 forehand
0:31 serve
1:02 backhand
1:15 forehand
EOF

# Process it (extracts clips + poses, updates metadata)
python scripts/process_labels.py labels/IMG_XXXX_labels.txt --camera-angle back
```

## Quick Diagnostic Commands

```bash
# Check overall training data state
python -c "
import json
with open('training/metadata.json') as f:
    data = json.load(f)
clips = data['clips']
total = len(clips)
verified = sum(1 for c in clips if c['verified'])
has_pose = sum(1 for c in clips if c['has_pose'])
print(f'Clips: {total} total, {verified} verified, {has_pose} with poses')
by_type = {}
for c in clips:
    st = c['shot_type']
    by_type.setdefault(st, {'total': 0, 'verified': 0})
    by_type[st]['total'] += 1
    if c['verified']: by_type[st]['verified'] += 1
for st, counts in sorted(by_type.items()):
    print(f'  {st}: {counts[\"verified\"]}/{counts[\"total\"]} verified')
"

# Count pose files per type
ls training/poses/*/  | head -40

# Count clip files per type
for d in training/clips/*/; do echo "$(ls "$d" | wc -l) $(basename $d)"; done
```
