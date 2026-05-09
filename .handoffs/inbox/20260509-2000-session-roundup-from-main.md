---
from: main
to: design-partner
created: 2026-05-09T20:00:00-08:00
status: session-roundup
priority: medium
topic: Roundup of today's work — pose evals, biomech audit, shot-correction UI
---

# What got done today

Five concurrent threads, all checkpointed at sensible stopping points. Branches are committed; nothing is half-done in the working tree. A few open follow-ups noted at the end.

---

## 1. Sapiens-2 pose evaluation — REJECT on speed

**Branch:** `feature/detection/sapiens-eval` (closed snapshot)
**Worktree:** `~/tennis_worktrees/sapiens-eval/`
**Brief:** `.handoffs/archive/20260508-2345-sapiens-eval-scoping-from-design-partner.md`
**Report:** `eval/sapiens/REPORT.md`

Mid-eval, Sapiens-2 (April 2026) shipped with a commercial-OK custom license, so we evaluated **Sapiens-2 0.4B** instead of Sapiens-1. License risk #1 from the original brief is dispelled.

**Numbers (IMG_0996 60s slice):**
- Sapiens-2 0.4B: **2.7 fps** (0.11× MediaPipe's 24.66 fps on same hardware). Speed criterion threshold was 0.5× — fails by ~5×.
- Mean confidence delta on 12 biomech indices: **+0.0258** — REVISIT range, but no Sapiens-2 variant can pass speed (larger sizes are slower), so the REVISIT path is dead-end.
- Per-keypoint pattern: Sapiens-2 dominates on **motion-engaged** keypoints (left elbow Δ +0.29, lead wrist +0.18, lead knee +0.14), loses slightly on stable ones.

**Verdict:** REJECT (speed). Side-finding (motion-keypoint advantage) seeded the next stream.

---

## 2. Pose-extractor experiments — REJECT both, reframe the question

**Branch:** `feature/detection/pose-experiments` (closed snapshot)
**Worktree:** `~/tennis_worktrees/pose-experiments/`
**Report:** `eval/pose-experiments/REPORT.md`

Three candidates against locked criteria (≥0.5× MP throughput, ≥+0.05 confidence delta, no biomech-stability regression):

| Candidate | Speed | 2D px Δ vs MP | Verdict |
|---|---:|---:|---|
| Sapiens-2 0.4B + YOLO + bbox-reuse + fp16 | 4.0 fps (0.16× MP) | not measured | REJECT (speed) |
| RTMPose-Wholebody (rtmlib balanced) | 28.1 fps (1.14× MP) ✓ | **0.005** (~5px @ 1080p) | REJECT (pipeline integration) |
| MediaPipe with tuned config | not run | (= 0.0) | not load-bearing |

Optimization stack on Candidate 1 (1.5× speedup over baseline) wasn't enough. RTMPose passes speed but is **2D-only** — biomech pipeline needs MediaPipe's `world_landmarks` (3D meters); 2D-only data makes biomech equations collapse to degenerate values (knee_angle = 180° = "straight leg").

**Reframe:** three independent SOTA models (Sapiens-2, RTMPose, MediaPipe) place 2D keypoints in the same locations to within ~5 pixels at 1080p. The 2D extractor is **not** the bottleneck. The load-bearing question shifted to: is MediaPipe's `world_landmarks` (its 2D→3D lifting) the weakness?

**Methodology debt surfaced:** the cross-extractor confidence-delta criterion in the locked criteria is broken — different model families calibrate confidence differently (`visibility` vs `score` vs heatmap peak height), so confidence numbers aren't directly comparable. Future pose-eval criteria should use pixel deviation or direct GT accuracy instead.

---

## 3. 3D-lifting Phase 1 audit — world_landmarks NOT the bottleneck

**Branch:** `feature/detection/3d-lifting` (Phase 1 complete)
**Worktree:** `~/tennis_worktrees/3d-lifting/`
**Brief:** `.handoffs/inbox/20260509-0140-world-landmarks-investigation-from-main.md` (still in inbox)
**Report:** `eval/3d-lifting/REPORT.md`

Audited 256 GT shots across IMG_0996, IMG_6874, IMG_6851 — 84% flag rate against heuristic implausibility criteria, but the distribution tells a precise story:

| Flag category | Count | Share of flags |
|---|---:|---:|
| `chain_reversed` | **212** | **98%** |
| `knee_out_of_range` | 24 | 11% |
| `trunk_extreme` | 7 | 3% |
| `arm_out_of_range` | 2 | 1% |

**Static 3D-derived metrics are biomechanically sound** across 256 shots:
- knee_angle medians: backhand 130°, forehand 149°, serve 166° (matches real tennis biomech)
- Visual inspection of 12 sample frames confirms 2D pose detection is correct and 3D-derived static angles look plausible

**98% of flags trace to one bug:** the kinetic-chain timing extraction in `scripts/biomechanical_analysis.py:174-192`. It computes per-frame velocity from raw position deltas with **no smoothing**, then picks the frame with max velocity — which is usually a noise spike, not the real biomechanical peak. Symptom: shoulder/elbow/wrist all peak at the exact same ms in 50% of "non-reversed" shots; reversed-chain in the other 50%.

**Per locked Phase 1 criteria** (≥60% "2D right + 3D wrong" → confirm; ≤30% → not the bottleneck): **NOT THE BOTTLENECK**. Phase 2 (MotionBERT/HMR2 alternative lifting) **NOT warranted** — switching the lifter wouldn't fix this.

**Recommended fix (~half day, P1):** in `biomechanical_analysis.py:174-192`:
1. Apply 5–7 frame rolling-mean smoothing to per-joint velocity before peak-pick
2. Add minimum-prominence filter on peak detection
3. Optional: switch to acceleration zero-crossing detection — more robust kinetic-chain landmark

Bonus trivial fix: trunk_rotation signed-angle wraparound (158° reported on a normal backhand) — clamp to (−180°, +180°) at the angle-compute site.

---

## 4. Shot-classifier audit (BH→FH question)

User asked "why would a backhand be seen as a forehand?" — pivoted the diagnosis from pose-quality to classifier-quality.

Audited current `sequence_cnn` predictions vs user-edited GT across all 44 GT videos. Confusion matrix on 1135 matched pairs:

| GT \ pred | forehand | backhand | serve | NO PREDICTION |
|---|---:|---:|---:|---:|
| **forehand** (880) | 418 | 8 | 0 | **454** |
| **backhand** (549) | 8 | 305 | 0 | **236** |
| **serve** (443) | 0 | 0 | 317 | **126** |

- **BH→FH errors: 8 of 313 backhands (2.6%)** — rare, but some at very high confidence. IMG_1027 has 3 cases at 95–99% confidence (model is *very confident* and wrong).
- **The bigger systemic issue: 43% of GT shots have no prediction at all** + 760 extra predictions not in GT.

GT was originally built from `fused_audio_heuristic_ml` (audio + heuristic + pose-ML); current production is pose-only `sequence_cnn`. Audio-confirmed contacts where pose is weak get missed. The active `feature/detection/improve-shot-classification` worktree was already chipping at this (per-class threshold tuning lifted F1 0.929 → 0.934).

Saved the audit JSON at `~/tennis_worktrees/3d-lifting/eval/3d-lifting/classifier_audit.json` for follow-up.

---

## 5. Shot-correction gallery feature — SHIPPED

**Branch:** `feature/gallery/shot-corrections` (live in production)
**Worktree:** `~/tennis_worktrees/shot-corrections/`

User asked for a way to flag misclassified shots from the gallery so corrections accumulate as labeled training data. Built end-to-end:

**Two distinct workflows:**

1. **Right-click any JUMP TO SHOT chip → "Correct shot type" popover** — pick FH / BH / Serve / Not a shot. Adds a "Remove correction" button when an entry already exists.
2. **Click 🚩 Flag in player bar → "Flag missing shot at *current playback time*"** — pick the actual type (FH / BH / Serve). Always creates a NEW entry; never mutates a chip.

**Storage:** R2 `highlights/corrections/{vid}.json`. Entries keyed by stable id: `chip-{idx}` for chip-tied corrections, `t-{seconds.toFixed(2)}` for free-floating missing-shot entries. Schema:

```json
{"version":1, "video":"iphone_ec172327", "entries":[
  {"id":"chip-12", "idx":12, "original":"forehand", "corrected":"backhand",
   "note":"slice", "ts":"...", "t":82.25},
  {"id":"t-88.30", "corrected":"missing_shot", "actual_type":"backhand",
   "t":88.3, "ts":"..."}
]}
```

**Worker endpoints:** `GET /api/corrections/:vid`, `POST /api/correction/:vid` (with optional `action: "remove"`). No auth (single-user gallery).

**Render details:**
- Chip-tied corrections: chip shows the corrected type's color + small green ✓ badge
- Missing-shot inserts: dashed-purple chips inserted in the timeline strip at their original-video time, sorted in among regular chips. Visible only on `timeline` variant.
- Modal popover: backdrop dims the page, video pauses on open + resumes on cancel, Esc closes, keyboard shortcuts (Space, arrows) blocked from reaching the video.

**Operational changes:**
- `scripts/deploy_gallery.sh` — non-interactive one-shot deploy that sources `CLOUDFLARE_API_TOKEN` from `~/tennis_analysis/.env`. Runs gallery + Worker. Future Claude sessions can deploy directly via Bash without copy-pasting tokens. Documented in `CLAUDE.md`.
- Worker `jsonResponse()` now sets `Cache-Control: no-store` + `CDN-Cache-Control: no-store` on all API responses. Prior bug: API responses were CDN-cached by default, which silently broke schema migrations because hard-refresh didn't help — Cloudflare's edge served stale JSON. **Worth flagging this as a class of issue** — any Worker JSON helper that gets added in future should set these headers.

**Why this is high-value beyond UX:** every correction is a labeled training example. After ~50 entries, that's enough signal to fine-tune `sequence_cnn` on user corrections specifically. Cheap data flywheel — exactly what the active improve-shot-classification stream needs.

---

## Branch states (everything committed)

```
feature/detection/sapiens-eval         REJECT on speed (closed snapshot)
feature/detection/pose-experiments     REJECT both candidates (closed snapshot)
feature/detection/3d-lifting           Phase 1 done, NOT the bottleneck (Phase 2 not warranted)
feature/gallery/shot-corrections       LIVE in production
main                                   FEATURES.md + CLAUDE.md updated to reflect all of the above
```

## Open follow-ups, by priority

1. **[P1, ~half day]** Fix kinetic-chain timing algorithm in `scripts/biomechanical_analysis.py:174-192`. This is the actual load-bearing weakness for biomech-quality complaints; pose extractor swaps don't fix it. Smoothing-before-peak-pick + min-prominence filter. Trivial trunk_rotation wraparound fix at the same time.
2. **[P2, ~5 min]** Verify iPhone-uploaded videos (`iphone_*` prefix) get `shots.json` produced by the GPU pipeline. False alarm earlier in session — they appear to. But worth confirming with `pipeline_health.py` style audit.
3. **[P2, when corrections data accumulates]** Once ~50+ corrections are submitted via the new UI, feed them as labeled examples into the active `feature/detection/improve-shot-classification` retraining pass.
4. **[P3, methodology debt]** When writing the next pose-related eval brief: drop the cross-extractor confidence-delta criterion. Use pixel deviation or direct GT accuracy. The current criterion mechanically REJECTed RTMPose for being more conservatively calibrated, not less accurate.
5. **[P3, ops]** Cache-control on Worker API endpoints — fixed, but consider auditing other helpers (e.g., `handleQueue`, `handleStatus`) to ensure they all hit `jsonResponse` rather than constructing Response objects inline.

## Methodology / process lessons captured

- **Locked frozen criteria are valuable but only as good as the metrics inside them.** The pose-experiments criteria mechanically rejected the right answer twice in a row because cross-extractor confidence isn't an accuracy signal. The criteria need to be calibration-free.
- **Diagnose before optimizing.** The 3D-lifting Phase 1 audit was scoped as "test alternative lifters if our diagnosis is right" — Phase 1 said "wrong diagnosis, the bug is downstream of pose entirely." Saved 2+ days of Phase 2 work that wouldn't have helped.
- **Worker JSON responses need explicit Cache-Control.** This is now defaulted in `jsonResponse()` but it's worth a code review pass on existing handlers. Cloudflare's edge silently caches 200s by default.
- **Preview HTML must be regenerated whenever gallery JS changes.** During the shot-corrections feature I forgot to regenerate the preview after adding `actual_type`, leading to ~30 min of confused debugging where prod and preview disagreed. The `scripts/deploy_gallery.sh` script now handles both, but the preview path is a separate `--preview` flag that needs explicit re-run when iterating.

## What I'd ask back

Whenever convenient (no urgency):

a) **Sign off on the kinetic-chain fix scope** so I can implement it. It's small but it's biomech-pipeline-critical, so a sanity check on the proposed change before I touch the code is worthwhile.

b) **Decide whether to keep the 3D-lifting branch open or close it.** Phase 1 answered the question; Phase 2 isn't warranted under current criteria. The branch holds the audit infrastructure (`audit_world_landmarks.py`, `render_audit_frames.py`) which would be useful for regression-testing a fixed kinetic-chain algorithm. Lean: keep open until the algorithm fix lands, then close.

c) **Confirm the corrections UI matches your mental model** — specifically the chip-context vs Flag-button-context split. If you ever want a way to flag a missing shot WITHIN an existing chip's neighborhood (i.e., "this chip is correct AND there's another missed shot near it"), the current design doesn't support that — you'd need to either pause near the missed shot and use Flag, or open the popover for a different chip. Probably fine as-is.
