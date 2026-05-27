---
from: detection
to: design-partner
created: 2026-05-20T21:15:00-07:00
status: pending
priority: medium
topic: Event-level eval + regression-head experiments — what shipped, what didn't, where we are
---

# Detection iteration session — report

## What was asked

User wanted (a) event-level evaluation (TenniSet ±6f convention) added next to
the legacy window-level F1, then (b) "keep improving" the detector once we
knew the real number.

## Headline result

**On the frozen 6-video holdout, the production model gets event tight F1
= 0.298 (with subpixel refinement; 0.268 without). The best clean model we
trained this session lands at 0.296 — matching prod under honest evaluation
on data the model never saw.**

This is the apples-to-apples comparison: prod's headline 0.894 legacy F1 is
inflated because it trained with the holdout 6 videos in the corpus (saw
them during training). On the same 32-video clean training set, a
regression-head model gets within 1 point of prod. The +0.086 lift over
the no-regression control (0.210 → 0.296) is the real measure of the
regression head's value.

We've identified the ceiling of pose-only at ~0.30 tight F1. To go higher
needs structural change — audio-impact supervision is the cheapest big
lever (4-6 hr, plan below).

## What shipped (4 commits on `feature/detection/improve-shot-classification`)

| Commit | What |
|---|---|
| `02a6f22` | `--per-class-thresh` and `--demote-below` flags on detect_shots_sequence.py (opt-in calibration tools). F1 0.929 → 0.934 on LOOCV. |
| `48f7887` | Event-level F1 (±6f), per-class PR curves, per-video and handedness stratification in validate_pipeline.py. `--pr-curves` mode in detect_shots_sequence.py. Wild-eval scaffold under `eval/in_the_wild/`. **This is the metric machinery `eval_holdout.py` later reused.** |
| `bd8adb0` | Multi-task contact-time regression head on ShotClassifierCNN. New `--lambda-regr` flag. New `--jitter-positives`/`--jitter-range` flags on prepare_sequence_data.py. NPZ schema adds a `delta` field. Inference auto-applies `delta/fps` correction. |
| `621bd2a` | Parabolic subpixel peak refinement (default on, +0.012 to +0.030 tight F1 free). Holdout-aware NPZ via `--holdout-manifest` (required after f4dcd21 leak guard). Class-conditional regression and probability-curve smoothing opt-in flags. |

## What we tested this session (full ablation on holdout)

All numbers are event tight F1 at ±6 frames (~100ms @60fps), class-strict,
on the frozen 6-video holdout. All "clean" models trained on the same
12,615-sample NPZ that EXCLUDES holdout videos.

| Configuration | tight F1 | Training data | Notes |
|---|---|---|---|
| prod (sequence_detector.pt) | 0.298 | 14,448 incl. holdout | Contaminated baseline |
| λ=0.0 control | 0.210 | 12,615 clean | Pure no-regression baseline |
| λ=0.2 narrow jitter | 0.286 | 12,615 clean | First regression-head winner |
| λ=0.2 wide jitter (±25f) | 0.269 | 16,749 clean wide | Wider didn't help at low λ |
| λ=0.2 class-conditional | 0.247 | 12,615 clean | Per-class head HURT |
| λ=0.5 narrow jitter | 0.244 | 12,615 clean | Too much regression weight on narrow data |
| **λ=0.5 wide jitter** | **0.296** | 16,749 clean wide | **Best clean model** |
| λ=0.5 class-conditional | 0.257 | 12,615 clean | Class-cond hurt λ=0.5 too |

Models on disk under `models/sequence_detector_regr_*.pt` with sidecars.

## What didn't work, with disconfirming evidence

1. **Class-conditional regression head.** Hypothesis: backhand's -175ms
   systematic offset came from forehand dominating the shared head's
   gradient (FH has 4106 samples vs BH 2615). Per-class head should let
   each class learn its own bias. Trained two models (λ=0.2 and λ=0.5).
   Both *underperformed* class-agnostic by 0.04 / +0.013. The head
   over-specialises to training-time bias and that bias doesn't transfer
   to the holdout video distribution. Code retained behind
   `--class-conditional-regression` flag.

2. **Post-hoc per-class median bias correction.** Computed per-class
   median signed offset on 32 calibration videos for four models, applied
   to holdout. Net effect: -0.034 to +0.022. Per-video bias varies more
   than the inter-class median, so a constant shift can't fix it.
   Disconfirms the "regression head ≈ just a per-class shift" hypothesis
   — it's doing more than that, but it's also why a simpler approach
   doesn't suffice.

3. **Savitzky-Golay smoothing of p_shot before peak finding.** Hypothesis:
   MediaPipe frame jitter propagates to noisy peaks; smoothing should
   yield better localised maxima. Tested windows {0, 3, 5, 7, 9, 11} on
   three models. window=3 with order-2 polynomial is a no-op (identity);
   window≥5 *collapsed* prod from 0.291 to 0.129. The model already
   averages over a 1.5s window internally — extra smoothing destroys the
   sharpness peak-finding relies on. Code retained behind
   `smooth_window=N` parameter on `find_shots` / `detect_video`, default 0.

## What we now know that we didn't before

- Window F1 ≠ event F1. The 0.965 number quoted everywhere was at ±1.5s
  tolerance. At the ±100ms tolerance that biomech actually consumes,
  prod was at 0.268 going in. The gap was hidden by the metric.
- The peak-vs-contact offset is dominated by **variance**, not bias.
  Median offset is small (~50ms) but std is 360-500ms. A constant shift
  fixes nothing; the regression head fixes some of it; pose-only can't
  fix more.
- Per-class behaviour is asymmetric: forehand peak ≈ contact, backhand
  peak ~100ms early, serve peak ~150-230ms late (because the model
  picks the racket-over-head frame, not the racket-ball contact frame).
- **Manual GT timestamps have ~16ms median bias vs acoustic contact
  but ~262ms std** — the user's M-key reaction noise is the dominant
  source of label noise. This is what makes audio supervision attractive.

## Proposed next move — audio supervision (4-6 hr)

Strongest expected lever. The diagnostic earlier showed audio peaks
match GT to ±16ms median across 395 shots — far tighter than the model's
pose-only ±400ms variance.

Plan:

1. Run `detect_audio_hits.py` over all 40 GT videos.
2. For each labeled positive, find the audio peak within ±300ms of the
   GT timestamp. Use the audio peak (not the GT timestamp) as the
   contact reference when constructing jittered training windows.
3. Drop shots where no audio peak exists in the ±300ms window
   (opponent shots, soft contact, audio gaps — probably 5-15%).
4. Train λ=0.5 wide with audio-aligned NPZ. Head-to-head against
   human-aligned λ=0.5 wide on the holdout.
5. If audio-aligned beats human-aligned at tight F1, the cap was label
   noise — ship it. If equal, label noise wasn't the bottleneck and
   pose-only is the real ceiling.

Risks: opponent audio confusion (handle via amplitude gating), missing
audio (filtered out), slo-mo audio artifacts (verify per video).

Optional phase 2 (~1 day): pseudo-label scraped tennis videos with audio
peaks for regression-head supervision. 10-100x more training data
without manual labels (regression only, not classification).

Asking design-partner whether this is the right next bet, or whether
there's a higher-priority item elsewhere (verification system components
C/D? data-engine cleanup? something else from the BACKLOG.md
data-engine section?). Detection branch has the headroom to start
audio work tomorrow if greenlit.

## Where the artifacts live

- Modified scripts: `scripts/{detect_shots_sequence,sequence_model,prepare_sequence_data,train_sequence_model,validate_pipeline}.py`
- Trained candidate models: `models/sequence_detector_regr_*.pt` (7 variants + sidecars)
- Wild-eval scaffold: `eval/in_the_wild/{README.md,manifest.json,.gitignore}`
- PR curves output: `training/pr_curves.{json,png}` (from `--pr-curves`)
- Per-model holdout eval results: `eval_results/*.json` (from `eval_holdout.py`)

Branch tip: `621bd2a`. Diverged from `main` at the merge point in `b7aed33`.
