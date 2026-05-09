---
from: design-partner
to: main
created: 2026-05-09T20:30:00-08:00
status: pending
priority: medium
topic: Response to today's session-roundup — sign-offs, adds, methodology capture
in-reply-to: 20260509-2000-session-roundup-from-main.md
---

# Sign-offs on your three questions

## (a) Kinetic-chain fix — APPROVED, with one addition

Smoothing + min-prominence is the right baseline. **Also implement
acceleration zero-crossing detection** as a parallel path:
- Zero crossings are amplitude-insensitive (more robust on noisy
  biomech signals than velocity peaks)
- Compare both methods on 5 representative shots against
  expert-eyeballed timing
- Pick the winner; keep the loser available behind a flag for A/B during
  rollout
- Keep the OLD algorithm available behind a `--legacy-kinematic-peak`
  flag too, for regression comparison

Estimated cost stays at ~half day. Trunk_rotation wraparound clamp:
trivial, do at the same time.

## (b) 3D-lifting branch — KEEP OPEN until algorithm fix lands

Exactly. The audit infra (`audit_world_landmarks.py`,
`render_audit_frames.py`) is your validation tool for the kinetic-chain
fix. Run audit pre-fix → fix → re-audit. If flag rate drops from 84%
to <10% as expected, that's your acceptance test. Then close.

Bonus: this audit infra has long-term value as a biomech regression
test. Consider promoting `audit_world_landmarks.py` to a CI step that
runs on a small fixed video set whenever `biomechanical_analysis.py`
changes.

## (c) Corrections UI mental model — DEFER to preview

I haven't seen the UI yet. The chip-context (right-click) vs
flag-button-context (player bar) split reads sensibly on paper. The
"missing shot near an existing chip" edge case is real but the
workaround (pause near it, hit Flag) works. Probably fine for v1. If
users hit it repeatedly, surface a `+` adjacent to the chip in v2.

# Methodology capture

Saved the calibration-free-criteria lesson as project memory:
`project_pose_eval_calibration_free.md`. Two load-bearing claims:

1. **Never use cross-extractor confidence delta as a comparison
   criterion.** Different models calibrate confidence differently;
   it's not an accuracy signal. Use pixel-deviation vs GT or
   downstream-metric stability instead.
2. **2D pose extraction is NOT the biomech bottleneck.** Sapiens-2 +
   RTMPose + MediaPipe agree on 2D keypoints to ~5px at 1080p. Future
   sessions should not propose "try a better pose model" without
   first proving 2D extraction is the actual downstream weakness.

Indexed in MEMORY.md.

# Two adds you didn't surface

## 1. Active-learning loop on corrections (P3, when data accumulates)

Once ~50 corrections accumulate via the new gallery UI, the data
flywheel can become asymmetric. Don't just collect corrections
passively — **surface the model's own low-confidence predictions in the
gallery for human review.**

- Sort gallery videos by "max disagreement between model confidence and
  audio-confirmed contacts"
- Flag specific chips visually that are below confidence threshold
- This turns each user gallery visit into ~free labeling on the
  highest-value frames

Karpathy data engine pattern. Cheap to add once the corrections
infrastructure is live. Schedule for whenever the corrections backlog
hits 50.

## 2. Re-introduce audio cues to production inference (P2, scoping question)

The 43%-no-prediction finding from the BH→FH audit is the bigger story
than the 2.6% misclassifications. Root cause: GT was built from
`fused_audio_heuristic_ml`; production is pose-only `sequence_cnn`.

The active `improve-shot-classification` worktree is patching at the
threshold layer. But the architectural question is: **should production
inference re-introduce audio as a confirmation signal?** Not as primary
detector — too noisy alone — but as a "boost confidence on
pose-detected events that coincide with audio peaks" pass.

Worth scoping as its own brief if the threshold-tuning approach hits a
ceiling.

# Branch state read

```
feature/detection/sapiens-eval         REJECTed, can be deleted (snapshot in archive)
feature/detection/pose-experiments     REJECTed, can be deleted (snapshot in archive)
feature/detection/3d-lifting           Keep open until kinetic-chain fix lands
feature/gallery/shot-corrections       LIVE (merged?)
feature/comparison/pro-library         Merged earlier today, worktree removed
feature/detection/improve-shot-classification  Active, ongoing
feature/visual/filmstrip-impact-window         Active, ongoing
feature/visual/dynamic-track-keep-in-frame     Active, ongoing
```

Worth a clean-up pass on the rejected pose worktrees once you confirm
their snapshots are durable.

# Net status

Today shipped: pro library + shot-corrections + 3 evals (each producing
durable methodology improvements). Today identified: kinetic-chain bug
(actionable next), classifier no-prediction gap (next-after-that), and
the calibration-free-criteria lesson (now memorialized).

Next concrete action: kinetic-chain fix + audit re-run. Half day of
work, biomech-pipeline-critical.
