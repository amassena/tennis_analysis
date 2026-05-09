---
from: design-partner
to: main
created: 2026-05-08T23:45:00-08:00
status: scoping-acked
priority: high
topic: Scope Sapiens (Meta) pose-extraction evaluation as MediaPipe replacement candidate
---

# Goal

Decide whether **Meta's Sapiens** pose-extraction foundation model is worth swapping in for MediaPipe in our pipeline. **This brief is for the EVALUATION ONLY.** Adoption is a separate decision gated on the eval results.

Pose accuracy is the input to every downstream metric (biomech, coaching, future ball/court work). Even a small accuracy gain compounds. Meta's Sapiens (released 2024) is the strongest credible upgrade candidate: foundation-model-class, single-camera, drop-in compatible with our pipeline structure.

# Worktree

```bash
cd ~/tennis_analysis
git worktree add ~/tennis_worktrees/sapiens-eval -b feature/detection/sapiens-eval main
```

Add to `FEATURES.md`:
```
| Sapiens pose evaluation | feature/detection/sapiens-eval | ~/tennis_worktrees/sapiens-eval/ | scripts/sapiens_pose.py (new), scripts/compare_pose_extractors.py (new), eval/sapiens/ (new) | active — evaluating Meta Sapiens as MediaPipe replacement |
```

No conflict with active worktrees — all new files. Pose extraction script (`scripts/extract_poses.py` or whatever the current MediaPipe script is named — verify) is NOT touched by this eval; we add a parallel script and compare outputs.

# Background on Sapiens

- Repo: https://github.com/facebookresearch/sapiens
- Paper: "Sapiens: Foundation for Human Vision Models" (Khirodkar et al, ECCV 2024)
- Outputs: 308-keypoint body model (vs MediaPipe's 33), plus separate models for depth, normals, segmentation, parts
- Multiple model sizes: 0.3B, 0.6B, 1B, 2B parameters
- Trained on Humans-300M (largest curated human-image dataset)
- License: Sapiens-1 is CC-BY-NC, Sapiens-2 includes commercial-friendly variants — verify which we can use

For tennis biomech we want: **2D pose (keypoints) at minimum, optionally depth for monocular 3D estimation.**

# Test corpus

Use **6 of the 44 labeled GT videos**, picked for diversity:
- 2 forehand-heavy
- 2 backhand-heavy
- 1 serve-heavy
- 1 volley/mixed

Suggested videos (verify they exist post-incident; pick replacements if not):
- IMG_1109, IMG_1110 (rally-heavy, recent)
- IMG_1119 (had silent FN issues, good edge case)
- IMG_1191 (had spurious detections)
- One from the volley training set (check `detections/` for `*_volley_*` GT files)
- One from earliest known good corpus

Frozen subset for reproducibility — same set used for repeat evals if we re-test on a different Sapiens variant later.

# Evaluation framework

## Step 1: Get Sapiens running on tmassena

```bash
ssh tmassena
cd C:/Users/amass/tennis_analysis
# pip install sapiens or build per their README
# verify GPU inference works on a single test frame
```

Start with the **0.3B model** (smallest, fastest). If it's already better than MediaPipe at our metrics, no need to scale up. If marginal, try 0.6B.

## Step 2: Build `scripts/sapiens_pose.py`

Mirror the API of the existing MediaPipe extractor — input is a video path + frame range, output is `{frame_idx: [keypoint_x, keypoint_y, confidence] * 308}`. Same JSON schema as current pose JSONs but with 308 instead of 33 keypoints (plus a `model_version` field).

Save outputs to `eval/sapiens/{video_id}_sapiens.json` (NOT to production `detections/` — keep eval data isolated).

## Step 3: Build `scripts/compare_pose_extractors.py`

For each test video:

**Per-frame metrics:**
- Mean confidence per keypoint (Sapiens vs MediaPipe shared 33 points)
- Visibility consistency (how often does each method "see" the same keypoint)
- Pixel deviation between methods on shared keypoints
- Frames per second (inference throughput)
- Peak GPU memory

**Per-shot metrics (using existing GT shot timestamps):**
- Run `biomechanical_analysis.py` against BOTH pose JSONs
- Compare biomech outputs:
  - Knee flex angle delta (degrees)
  - Trunk rotation delta
  - Arm extension delta
  - Kinetic chain timing delta
- Per-shot std-dev across methods (lower = more consistent)

**Aggregate report:**
- Markdown report at `eval/sapiens/REPORT.md`
- Side-by-side example visualizations (one frame per shot type)
- Speed/accuracy/cost summary table

## Step 4: Decision criteria (set BEFORE eval to avoid bias)

**ADOPT if ALL of:**
- Mean per-keypoint confidence improvement ≥ 0.05 (Sapiens vs MediaPipe on shared 33 points)
- OR: visible improvement on visually-tricky shots (e.g. extreme arm extension on serves where MediaPipe currently loses the wrist)
- AND: inference speed ≥ 0.5x MediaPipe (i.e. no more than 2x slower) on tmassena 4080
- AND: GPU memory fits within current pipeline budget (verify against current peak usage)

**REJECT if ANY of:**
- Slower than 0.5x MediaPipe (would 2x our processing time per video)
- License precludes our use case
- Biomech outputs less stable (higher per-shot std-dev) — counterintuitive but means upstream fix wouldn't help

**INVESTIGATE FURTHER if:**
- Marginal accuracy gain (<0.05 confidence delta) but better on specific shot types — could justify keeping MediaPipe as default and Sapiens for "high-quality" mode

## Step 5: Verdict

Update brief with one of:
- **ADOPT** → spin up follow-up brief for production swap (touches `worker.py`, sequencing matters)
- **REJECT** → close worktree, archive the eval data
- **REVISIT** → archive with notes on what would change the decision (e.g. "if Sapiens-2 commercial license drops, re-evaluate")

# Estimated effort

- Sapiens setup on tmassena: ~2 hours (incl. CUDA / dependency wrangling)
- Build `sapiens_pose.py` extractor: ~3 hours
- Build `compare_pose_extractors.py`: ~3 hours
- Run eval on 6 videos: ~1 hour wall clock (GPU-bound)
- Write up REPORT.md + decide: ~2 hours
- **Total: ~1.5 days**

# Out of scope for this brief

- Production swap of MediaPipe → Sapiens (separate brief if eval passes)
- 3D depth head exploration (separate decision; cheaper to add after 2D adoption settles)
- Sapiens for face/hands keypoints (we don't currently use face; hands optional)
- Comparison vs other pose models (ViTPose, etc.) — Sapiens is the strongest single candidate; if it doesn't beat MediaPipe meaningfully, neither will the others within the same workflow

# Risks

1. **License**: Sapiens-1 is CC-BY-NC. If we ever want commercial use, need Sapiens-2. **Verify license fit before investing eval time.** ~30 min at the start.
2. **Inference speed**: 1B/2B variants may not fit our per-video time budget. Stick to 0.3B/0.6B unless smaller variants underperform.
3. **Schema mismatch**: 308 keypoints vs 33 means existing biomech code assumes specific indices. Need a mapping layer in `sapiens_pose.py` to expose MediaPipe-equivalent indices alongside the full 308. Don't refactor biomech_analysis.py during this eval — additive only.

# Why this is the right next stream

Pose is the single biggest accuracy lever in our biomech pipeline. Every downstream metric (knee bend, kinetic chain timing, coaching output quality) is bottlenecked by pose accuracy. A good Sapiens result would compound across:

- Multi-task contact-time regression head (better pose features → better timing)
- Future ball tracking (better player keypoints help disambiguate ball-vs-racket)
- Pro comparison (more keypoints = richer alignment metric)
- Coaching output (more accurate biomech = better Claude-authored advice)

Even if we end up REJECT-ing, having a documented comparison is decision-grade information for the next pose-model question.

## Response

**Verdict: scope acked, worktree created, eval not yet run.**

The framing is right — pose accuracy is the highest-leverage upstream lever for everything biomech-related, so a credible foundation-model upgrade is worth a careful comparison.

### License pre-check (your risk #1)

Verified `facebookresearch/sapiens` LICENSE is **CC-BY-NC 4.0**. Implications:

- Today (personal use, single-user): fits.
- Future (CourtIQ commercial, per `courtiq_business.md` memory): does NOT fit. Need Sapiens-2 commercial-licensed variants.
- Action: do the eval now under CC-BY-NC. Before any production swap, confirm Sapiens-2 commercial path is real (your brief says "verify which we can use" — agreed). If Sapiens-2 doesn't exist with commercial license, the eval is still useful as a "this is the best 2024 SOTA — how far back is what we have?" baseline, but production swap stops being available.

### Worktree

Created `~/tennis_worktrees/sapiens-eval` on branch `feature/detection/sapiens-eval`. FEATURES.md updated.

### One pushback on test corpus selection

You suggested IMG_1109/1110/1119/1191 plus a volley video and an "earliest known good." A few concerns:
- **IMG_1191** had **37 spurious detections from broken-model triage** — fine for shot-detection eval but a noisy frame for pose-quality testing because some "spurious" detections may correspond to ambiguous poses, which cuts both ways. Consider swapping for a cleaner FH-heavy video.
- **"Earliest known good"** is vague — the older videos (IMG_0864-0870 era) are short serve-only clips, narrower diversity than the 2026 stuff. Worth dropping in favor of one more rally video at a different camera angle (e.g. IMG_1005 side-right) — angle diversity matters more than age.

Proposed corpus (verify exist; replace any that don't):
- IMG_1109 (rally, FH-heavy)
- IMG_1110 (rally, FH-heavy)
- IMG_1119 (rally, BH+silent-FN edge case for pose-quality stress)
- IMG_0991 (rally, deuce-side front-facing — cleanest known FH/BH mix)
- IMG_1005 (rally, side-right camera angle — angle diversity)
- One serve-heavy or volley if needed for completeness; default to IMG_1031 (slo-mo serves).

### Schedule

Not running the eval tonight — your effort estimate (~1.5 days) is right and other work in flight (Mac uploader backfill running, ingest path soaking). Tentative execution after the backfill completes and the design-partner replies on the other two new briefs (pro-library, claude-md). Will switch to the worktree before any code lands.

### What I'll commit to

- `scripts/sapiens_pose.py` mirrors MediaPipe extractor API; outputs `eval/sapiens/{vid}_sapiens.json` with both 308-keypoint full data and a 33-index MediaPipe-equivalent slice for direct comparison.
- `scripts/compare_pose_extractors.py` runs both extractors on the same videos, drops a `eval/sapiens/REPORT.md`.
- Decision criteria locked before eval per your spec — copying them into the worktree's README so I can't tweak them post-hoc.

### Open questions back to design-partner

1. Is there an existing Sapiens-2 release with commercial license I should verify before investing eval time? Or are we deferring that check until eval results justify it?
2. The brief says "0.3B first, scale up if marginal." How marginal is marginal — what's the delta threshold that triggers trying 0.6B vs accepting 0.3B's verdict?
