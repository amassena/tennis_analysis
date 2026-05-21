---
from: filmstrip
to: any
created: 2026-05-21T04:15:00+00:00
status: done
priority: medium
topic: Filmstrip contact-precision — what shipped, what's still wrong, where the bottleneck moved
---

# Filmstrip contact-precision — status

Two commits landed on main today on top of `4db0f75` (the 11-panel symmetric window already on main from 2026-05-19):

- **0bf0566** — non-uniform dense sampling (3 wide backswing panels + 5 dense ~37ms-step panels around contact + 3 wide follow-through)
- **0bf0566 + later** — audio-peak contact-frame snap (±300ms window, 1.5× threshold, cached to `audio_hits/{vid}.json`)

Both are now active on every freshly-generated filmstrip from `scripts/swing_composite.py`. Existing R2 thumbnails won't reflect this until the GPU pipeline reprocesses each video.

## What we validated (IMG_1120, 240fps, 37 shots inspected via local server)

Rough split:
- **~40% clear wins** — orange contact panel shows ball-at-strings or peak extension (most serves, close groundstrokes)
- **~25% close but not exact** — within 1-2 frames of true contact, still useful
- **~35% still broken** — orange shows wrong moment (trophy position for serves, prep phase for groundstrokes, or empty court)

## Where the bottleneck moved

Two distinct failure modes for the remaining ~35%, both **outside `swing_composite.py` scope**:

### 1. Detector frame error > 300ms

Cases like IMG_1120 shots 3, 4, 26: the 1D-CNN's `raw_contact` is so far from true contact that even the audio peak (which usually pinpoints the strike accurately) lies outside our ±300ms search window. Widening the audio window introduces false positives (snaps to opponent's hit or ball bounce on court).

**Owner:** `feature/detection/improve-shot-classification`. That branch landed per-class peak finding (F1 0.929 → 0.934) but appears to optimize classification, not contact-frame timing precision. Worth asking that stream whether contact-frame precision is on the roadmap, and if not, raising it.

### 2. Two-player rally framing

Cases like IMG_1120 shots 21, 26: MediaPipe pose tracks whichever player it considers "primary" per frame, and jumps between our player and the opponent across the 11 panels. The per-frame torso anchor then drifts wildly, producing crops that show the wrong half of the court at the contact moment.

**Owner:** ambiguous. Could be fixed in `swing_composite.py` (pick "our" player via largest bbox or consistent tracking), OR in the pose-extraction pipeline (multi-person tracking with identity persistence). Probably both.

## Comparison: old vs new videos

Spot-checked iphone_9ca0a615 (processed 2026-05-18 with current production model approved 2026-05-02) vs IMG_1120 (processed 2026-04-17 with the older model).

**Surprising finding:** detector contact-frame error magnitudes are roughly the same (±150-280ms typical shifts) on both. The current detector update didn't dramatically improve contact-frame precision; it improved classification confidence (forehands now consistently 0.93-0.98 conf vs. earlier 0.5-0.99 range).

What's actually better on newer iphone_* videos is **filming style** — solo practice with one player in frame, so the two-player framing bug doesn't fire and the filmstrip looks much cleaner. Rally footage with opponent visible will keep having issues until pose disambiguation lands.

## Files / artifacts

- `scripts/swing_composite.py` — owns the filmstrip generation, audio-snap, non-uniform sampling
- `audio_hits/{vid}.json` — new per-video cache, ~5-10s extraction cost first time
- `~/tennis_worktrees/filmstrip/` — branch worktree, still active

## Suggested next steps (in priority order)

1. **Surface to `improve-shot-classification`**: ask whether contact-frame timing precision is in scope, or whether the eval metric is purely classification. If the latter, propose adding contact-frame MAE as a secondary metric.
2. **Pro side-by-side filmstrip** (`feature/comparison/pro-library`): blocked on having prepared 4-5s mp4 clips at `pros/{name}/{shot}_NNN.mp4`. Once one Djokovic serve clip exists with pose data, the side-by-side script is ~50 lines (call `generate_composite` twice + `cv2.vconcat`). The pro-clip `contact_frame` is hand-labeled in `pros/index.json` so the pro side will be frame-precise.
3. **Pose disambiguation** in `swing_composite.py`: pick the player with largest pose bbox or most consistent per-frame tracking. Would fix the two-player rally case without needing new pose extraction.

## Pointers for whoever picks this up

- `/tmp/filmstrip_validate/` has 37 + 4 strips from IMG_1120 and iphone_9ca0a615 ready for inspection
- Local server on port 8090 was up at end of session; restart with `cd /tmp/filmstrip_validate && python3 -m http.server 8090`
- To regenerate filmstrips after code changes: `scp scripts/swing_composite.py tmassena:'C:/Users/amass/tennis_analysis/scripts/'` then run on tmassena, scp results back
