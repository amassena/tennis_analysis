# Precise Contact Detection — Core Feature Spec

Status: **proposed** (2026-06) · Owner: contact worktree · Epic: (GitHub issue TBD)

## Why this is core

Every contact signal we ship today is a **proxy**, never the ball itself:
- **Audio strike** (`extract_audio_peaks`/`snap_to_audio_peak`) — dies on pro
  highlights (music/crowd), only works on the user's own clean-audio clips.
- **Pose kinematics** (`detect_pro_contact.py`) — peak wrist speed lags into the
  follow-through; ±many frames; fails on some clips (e.g. Wawrinka → backswing).
- **Detector grid frame** (`detect_shots_sequence.py`) — quantized to the 0.1s
  inference step.

Result: filmstrips off by 1–2 frames on our own video, and **way** off on
downloaded clips; pro-comparison alignment was up to ~350ms wrong. Contact is
load-bearing for filmstrips, pro comparison, and every contact-relative biomech
metric — so imprecision contaminates the whole product. We fix it at the source:
**track the ball and find the actual strike.**

## Goal

Given a tennis clip of **any angle** (behind-player, broadcast, side) and **any
stroke** (forehand, backhand, serve, slice), output the **precise contact
instant** (sub-frame), with an explicit **uncertainty window** when the ball is
occluded/blurred at impact. Never silently guess — when the ball is invisible
across the strike, say so and bound it (e.g. "occluded frames 121–124 / 12ms;
contact estimated 122.6 ±6ms").

## Strategy: best content first, then degrade deliberately

Prove the method on the **easiest** input, then walk *down* the quality stack
measuring where it breaks:

1. **4K / 120fps (our own, recently recorded)** — ball is large & sharp, blur
   minimal, 8ms/frame, short occlusion windows. Establish ground truth + the
   precision ceiling here.
2. **Downsample our 4K → 60fps, then 30fps** — isolates the fps effect on
   precision with the *same* swings as truth.
3. **1080p/60 phone footage** (current user uploads).
4. **Pro broadcast / highlight clips** — hardest (small ball, varied angle,
   compression). 

This yields a principled "accurate to ±X ms down to Y fps / Z angle" envelope
instead of fighting all degradations at once.

## Approach

**Contact = intersection of incoming & outgoing ball trajectories.** You never
need the ball *at* the contact frame — fit a parabola to the visible incoming
arc and another to the visible outgoing arc, intersect them → sub-frame contact.
Occlusion at impact becomes a reported uncertainty (the gap), not a failure.

Pipeline:
1. **Ball detection per frame** — WASB-SBDT (MIT, tennis weights) heatmap →
   `(x, y, confidence, present)`. Multi-frame, handles blur/tiny ball. Start with
   TennisProject/TrackNet for turnkey speed; move detector core to WASB for
   accuracy. Fine-tune on our angles (behind-player is out-of-distribution).
2. **Track build** — confidence-gate, interpolate short gaps; do NOT smooth
   across the candidate contact (that's the discontinuity we want).
3. **Candidate localization** — velocity sign-flip + speed minimum near the
   swing window (window located via MediaPipe wrist + the detector's shot time).
4. **Sub-frame contact** — piecewise parabola fit of incoming vs outgoing
   segments (excluding the candidate frame), solve for intersection time.
5. **Hit vs bounce** — gate by pose proximity (ball near wrist/racket = hit; near
   court homography ground plane = bounce). Reuse TennisProject court + bounce
   model features.
6. **Cross-checks & fusion** — optical-flow reversal near the racket
   (appearance-independent), audio peak (when clean), pose kinematics. Fuse into
   one confidence-scored estimate; agreement → high confidence, disagreement →
   flag + widen the uncertainty window.

Run detection on the **native-fps** copy, not the 60fps CFR export — high fps is
the precision asset.

## Eval-first (the scoreboard)

No detector ships without a ms-error score against hand-labeled truth.
- **Labeling UI** (web): load any clip, frame-step (←/→), mark exact contact
  frame; mark "ball occluded here" + the last-visible-before / first-visible-after
  frames. Stores GT.
- **GT set**: spanning fps × angle × stroke × source (start with the 4K/120fps
  clips). Target ~60–100 labeled shots.
- **Metric**: |predicted − truth| in **ms**, reported median / p90, **sliced by
  fps, angle, stroke, source**, plus occlusion-rate and coverage. Same harness
  reviews predictions vs truth.

## Phases & parallel streams

Largely independent until convergence — can run concurrently:
- **A — Ball tracking** (GPU/tmassena): WASB/TennisProject running, per-frame
  ball track on a 4K/120fps clip.
- **B — Contact math** (pure logic, unit-testable on synthetic arcs first):
  trajectory fit + intersection + flow-reversal + hit/bounce + fusion +
  occlusion model.
- **C — Eval/label harness + GT** (web/Mac): labeling UI + scoring, no GPU.

Then **P-converge**: fuse, score on GT, iterate; **productionize**: wire the
unified contact record into the 5 integration points; backfill pro library.

## Unified contact record (replaces the 3 proxies)

```jsonc
{
  "timestamp": 12.453,        // seconds, final estimate (sub-frame)
  "frame": 747,               // grid-aligned for back-comat
  "confidence": 0.94,         // 0–1
  "occluded": { "from_frame": 745, "to_frame": 748, "ms": 25 } | null,
  "uncertainty_ms": 6.0,
  "sources": {                // per-signal, for audit/calibration
    "ball_trajectory": { "timestamp": 12.452, "confidence": 0.9, "active": true },
    "flow_reversal":   { "timestamp": 12.455, "confidence": 0.7, "active": true },
    "audio":           { "timestamp": 12.453, "rel_amp": 1.8,    "active": true },
    "pose":            { "timestamp": 12.455, "method": "reach", "active": true }
  },
  "method": "ball_intersection" | "ensemble" | "audio" | "pose"
}
```

### Integration points (where it plugs in)
1. `detect_shots_sequence.py` — replace grid `frame = int(ts*fps)` refinement.
2. `swing_composite.py` `generate_composite` — replace audio-only
   `snap_to_audio_peak`; add `contact_source`/`confidence` to the strip info +
   `strips.json`.
3. `compare_filmstrip.py` — user side now precise; pro side already per-clip
   (`pros/index.json.contact_frame`). Both align on true contact.
4. `contact_accuracy.py` — measure unified estimate vs truth; per-signal error.
5. `biomechanical_analysis.py` — anchor metrics on the precise contact +
   propagate `confidence` (filter low-confidence shots).

## Detector options (condensed; full research in epic issue)

| System | License | Tennis weights | Note |
|---|---|---|---|
| **WASB-SBDT** | MIT | yes | accuracy leader (F1 95.6); tracker emits present/absent; research-grade UX (write ~100-line infer loop) |
| TennisProject + TrackNet | unstated* | yes (broadcast) | turnkey run-on-mp4 + court homography + bounce model — best scaffold to start |
| TrackNetV3 | MIT | badminton (retrain) | best occlusion recovery (InpaintNet) |
| YOLO single-frame | AGPL* | community | wrong tool — fails at the blurred contact frame |

\* verify license before any redistribution (we're single-user research).

## Open decisions
- Detector: start TennisProject → swap core to WASB (recommended).
- Sub-frame needed below 120fps? (yes — parabola intersection gives it; matters
  most at 30/60fps.)
- GT size / who labels (web UI; user labels the 4K set).
