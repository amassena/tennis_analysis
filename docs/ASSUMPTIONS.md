# Tennis Shot Detection — Assumptions & Constraints

*Last updated: 2025-02-25*

## Current Constraints (Accepted)

### Camera Angle
- **PRIMARY:** Back/baseline view (most footage)
- **STATUS:** Camera angles not yet tagged - need to review source videos
- **ACTION NEEDED:** Tag each source video with its angle in `training/metadata.json`

| Source Video | Camera Angle | Notes |
|--------------|--------------|-------|
| IMG_0864 | unknown | |
| IMG_0865 | unknown | |
| IMG_0920 | unknown | |
| IMG_6665 | unknown | |

Once you identify the angles, update `training/metadata.json` → `camera_angles` section.

### Shot Types We're Training
| Shot Type | Min Samples Needed | Current Verified | Status |
|-----------|-------------------|------------------|--------|
| serve | 50 | 38 | ⚠️ Need ~12 more |
| forehand | 30 | 12 | ⚠️ Need ~18 more |
| backhand | 30 | 8 | ⚠️ Need ~22 more |
| forehand_volley | 20 | 0 | ✗ Need to collect |
| backhand_volley | 20 | 0 | ✗ Need to collect |
| overhead | 20 | 0 | ✗ Need to collect |
| neutral | 30 | 116 | ✓ Sufficient (over-represented) |

### Data Format
- **Frame rate:** 240fps source, downsampled to 60fps for pose extraction
- **Pose model:** MediaPipe (33 keypoints)
- **Sequence length:** 30 frames per sample (~0.5 sec at 60fps)
- **Coordinate system:** 2D screen coordinates + confidence scores

## Labeling Format

When manually labeling a video, use this format:
```
# Video: [filename]
# Date labeled: [date]
# Notes: [any context]

MM:SS shot_type
MM:SS shot_type
...
```

Example:
```
# Video: IMG_1234.mp4
# Date labeled: 2025-02-25
# Notes: Practice session, back view from baseline

0:05 serve
0:12 forehand
0:15 backhand
0:23 forehand_volley
0:31 overhead
0:45 serve
```

## Shot Type Definitions

| Shot Type | Description | Key Visual Cues |
|-----------|-------------|-----------------|
| serve | Service motion | Ball toss, overhead contact, baseline position |
| forehand | Groundstroke, dominant side | Racket on right (for righty), full swing |
| backhand | Groundstroke, non-dominant | Racket crosses body, one or two-handed |
| forehand_volley | Net shot, dominant side | Short punch, near net, no backswing |
| backhand_volley | Net shot, non-dominant | Short punch, near net, crosses body |
| overhead | Smash/overhead | Above head contact, not a serve |
| neutral | Ready position, walking, etc. | No active shot |

## Future Considerations (Parked)

- [ ] Side-view camera angle support
- [ ] Front-view camera angle support
- [ ] Slice vs topspin classification
- [ ] First serve vs second serve
- [ ] Return of serve detection
- [ ] Rally length tracking
- [ ] Player identification (multi-player court)

## Revision History

| Date | Change |
|------|--------|
| 2025-02-25 | Initial assumptions documented. Back-view only constraint accepted. |
