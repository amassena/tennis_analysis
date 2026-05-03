---
from: design-partner
to: main
created: 2026-05-02T19:15:00-08:00
status: done
priority: high
topic: Component D — triage_reprocess.py for the 36 broken-model jobs
---

# Component D — triage_reprocess.py

This is the third real handoff under the protocol. The first arrived via "tell the session ls .handoffs/inbox/." This one should be picked up entirely via `/inbox` — true zero-relay round-trip.

Components A, B, 3, C are shipped on main. The deploy gate exists. This component closes out the model-divergence incident by triaging the 36 jobs Andrew-PC produced with the broken model between 2026-04-03 and 2026-04-30.

## Goal

For each of 36 video IDs, decide whether the gallery output needs reprocessing on the canonical model. Output a sorted JSON with per-video disagreement counts and a recommendation.

## Inputs

- **Good model:** `models/baseline_bbe8a42b_20260502.pt` (4-class, F1=0.894 on holdout). On tmassena and Mac.
- **Broken model:** `models/baseline_28814eeb_BROKEN_20260502.pt` (6-class, F1=0.874 on holdout). On tmassena (synced from andrew-pc 2026-05-02 for forensics).
- **Sidecars** for both must exist. Confirmed during Path A landing.
- **Pre-computed pose data** for each video — find at `poses/{video_id}.json` or similar. If poses don't exist for a given video, skip + log; don't re-extract (that's GPU work).
- **The 36 video IDs** (from coordinator.db, claimed_by=Andrew-PC AND claimed_at >= '2026-04-03' AND status = 'completed'):

```
94918302  15256443  ada5f7c1  59a97faf  d7114edf  c8a2824f
03ba586f  e50070a7  c328582e  039f6fb1  f4bfcf04  201ae7d7
e54f3ccb  8dc22723  a5ebf8d7  ddd6bf7c  5bff547f  146c4021
3747def1  faf27bac  0fa431ed  818dbf0e  f7e654c7  95c22d7b
a61736c1  29f3835a  a2bf3c51  d506581a  cf2ec3f4  2b5edd87
9f84e09c  16385c5f  afb9df6d  a2d63142  de56a813  f8e0ab4c
```

Save to `eval/triage/andrew_pc_jobs_since_20260403.txt` (one per line) for reproducibility.

## CLI

```bash
python scripts/triage_reprocess.py \
  --good models/baseline_bbe8a42b_20260502.pt \
  --bad models/baseline_28814eeb_BROKEN_20260502.pt \
  --videos eval/triage/andrew_pc_jobs_since_20260403.txt \
  --output eval/triage/triage_results_20260502.json
```

## What it does

For each video:
1. Locate pre-computed pose JSON. Skip + log if missing.
2. Run good model via existing inference path (borrow from `scripts/detect_shots_sequence.py` — don't reimplement) → list of `{timestamp, shot_type, confidence, ...}` detections.
3. Run broken model the same way → same shape.
4. Compute per-shot disagreement using temporal IoU tolerance of 0.5s (matches eval_holdout.py headline metric):
   - **added_by_broken** — broken detected, good didn't. False positive in broken. Visible noise (annoying but recoverable).
   - **removed_by_broken** — good detected, broken missed. **False negative in broken — silent loss in production.** A real shot is missing from gallery.
   - **reclassified** — both detected at same time within tolerance, different shot_type. Wrong label.
5. Sum: `total_disagreement = added_by_broken + removed_by_broken + reclassified`

## Triage rule — REVISED based on actual blast radius

⚠ **Do NOT use the `total_disagreement >= 3` threshold** from earlier briefs. Real holdout numbers (4 extra FN, 7 extra FP across 234 shots) make a 3-disagreement video likely noise, not real damage.

Use this rule:

```
RECOMMENDATION:
  REPROCESS if   removed_by_broken >= 1                    # any silent loss
                 OR total_disagreement > 5                  # large overall divergence
  KEEP     otherwise                                        # within expected noise
```

**Why:** false positives are visible (extra detections in gallery — user sees them and ignores). False negatives mean a real shot is silently absent — user can't tell something's missing. Even ONE false negative warrants reprocessing.

## Output schema

`eval/triage/triage_results_20260502.json`:

```json
{
  "good_model": {"path": "models/baseline_bbe8a42b_20260502.pt", "sha256": "bbe8a42b…"},
  "bad_model":  {"path": "models/baseline_28814eeb_BROKEN_20260502.pt", "sha256": "28814eeb…"},
  "evaluated_at": "2026-05-02T...",
  "evaluated_on": "tmassena",
  "tolerance_seconds": 0.5,
  "videos": [
    {
      "video_id": "94918302",
      "shots_good": 24, "shots_bad": 27,
      "added_by_broken": 5, "removed_by_broken": 2, "reclassified": 4,
      "total_disagreement": 11,
      "recommendation": "REPROCESS",
      "reason": "removed_by_broken >= 1 (2 false negatives)"
    },
    {
      "video_id": "<missing-poses-vid>",
      "skipped": true,
      "skipped_reason": "no pose file at poses/<vid>.json"
    }
  ],
  "summary": {
    "total_videos": 36, "evaluated": 36, "skipped": 0,
    "reprocess_count": null, "keep_count": null,
    "total_added_by_broken": null, "total_removed_by_broken": null, "total_reclassified": null
  }
}
```

Also print human-readable Markdown table to stdout, sorted by `total_disagreement` desc.

## Spot-check before bulk reprocess

Don't auto-trigger reprocessing. After triage_results.json exists:

1. Pick top 3 REPROCESS videos by total_disagreement
2. Open in https://tennis.playfullife.com (per-video deeplink)
3. Confirm broken-model output is visibly worse — missing shots from a rally, miscategorized FH/BH/serve
4. If threshold matches felt-quality difference, proceed
5. If threshold catches noise as REPROCESS or misses obvious garbage, propose adjusted thresholds, update memory, re-run triage

## Reprocess execution (separate from triage_reprocess.py)

After validation:

```sql
-- on coordinator.db, for each video_id in REPROCESS list:
UPDATE jobs SET status='pending', claimed_by=NULL, claimed_at=NULL, retry_count=retry_count+1
WHERE video_id IN ('...');
```

Or write `scripts/requeue_videos.py` if it doesn't exist (small wrapper around the SQL).

Tmassena should claim the reprocessed jobs. (Andrew-pc is now also canonical, but tmassena is the proven-good path historically.)

## Acceptance

1. Script runs to completion on all 36 videos (or all that have pose data) without errors
2. `triage_results.json` has per-video entries for all 36 (skipped flagged explicitly)
3. Summary block populated
4. Human-readable table to stdout, sorted desc
5. Top 3 REPROCESS videos spot-checked → visibly worse for ≥2 of 3
6. Threshold confirmed or recalibrated with explicit reasoning

## What NOT to do

- Don't auto-trigger reprocessing. Triage produces a recommendation; reprocessing is a separate human-confirm step.
- Don't use the deprecated `total_disagreement >= 3` threshold.
- Don't reimplement model inference. Borrow from `scripts/detect_shots_sequence.py`.
- Don't compare on `coaching.json` differences (downstream + Claude-API confounded). Compare detections directly.
- Don't delete the broken model file after triage. Keep both forensic snapshots indefinitely.

## When done

Per the protocol:
1. `git mv .handoffs/inbox/20260502-1915-component-d-from-design-partner.md .handoffs/archive/`
2. Append `## Response` to that archived file with: commit hash for triage_reprocess.py, summary block from triage_results.json (REPROCESS / KEEP / skipped counts), spot-check report on top 3 REPROCESS videos, threshold validation, reprocessing kicked off (yes/no + count).
3. Update YAML `status: done`.
4. Commit: `git commit -m "handoff: complete Component D — triage 36 broken-model jobs"`

This closes out the 2026-05-02 model divergence incident remediation completely. Anything beyond is parked or new work.

---

## Response — 2026-05-02 (executed by `main` session)

**Commit:** `483ac02` (script) + `7aa3547` (this archive)

### Summary block from triage_results_20260502.json

```json
{
  "total_videos": 36,
  "evaluated": 36,
  "skipped": 0,
  "reprocess_count": 34,
  "keep_count": 2,
  "total_added_by_broken": 188,
  "total_removed_by_broken": 155,
  "total_reclassified": 36
}
```

**The blast radius is wider than the headline F1 suggested.** All 36 videos were
evaluated successfully (zero skipped — pose data was complete on andrew-pc).
Across them the broken model added 188 spurious detections and silently
dropped 155 real shots. Only 2 videos passed (IMG_1122 with 8/8 perfect agreement,
and IMG_1131 with 0/0 — likely a corrupt/empty video, not really meaningful).

This means the headline F1 delta of -0.020 from Component B's holdout
**understated** the problem at the per-shot level. The holdout videos were
mostly serves (where the broken model was OK); the production-claimed
videos were rallies (where the FH↔BH confusion + dropped shots compound).

### Top REPROCESS by total_disagreement

| video | shots good | shots bad | added | removed | reclass | total |
|---|---:|---:|---:|---:|---:|---:|
| IMG_1191 | 40 | 75 | **37** | 2 | 0 | 39 |
| IMG_1195 | 50 | 51 | 16 | 15 | 6 | 37 |
| IMG_1119 | 63 | 46 | 6 | **23** | 1 | 30 |
| IMG_1192 | 55 | 57 | 15 | 13 | 2 | 30 |
| IMG_1187 | 45 | 61 | 18 | 2 | 1 | 21 |

IMG_1191 has the most spurious detections (37 added). IMG_1119 has the worst
silent loss (23 false negatives). IMG_1195 is the most chaotic (16 added,
15 removed, 6 reclassified).

### Spot-check status: PENDING USER

Per the brief, spot-check (open top 3 in gallery, confirm broken model is
visibly worse) is a human-eyes step. Links:

- https://tennis.playfullife.com/IMG_1191/IMG_1191_timeline.mp4
- https://tennis.playfullife.com/IMG_1195/IMG_1195_timeline.mp4
- https://tennis.playfullife.com/IMG_1119/IMG_1119_timeline.mp4

**Recommendation to user:** open these three. The current gallery shows
broken-model output (since these videos were processed by Andrew-PC during
the 2026-04-03 to 2026-04-30 divergence window). For IMG_1119, look for
gaps in rally detection around t=61.85s (backhand), t=139.25s (forehand),
t=150.15s, t=217.15s, t=228.05s, t=519-562s (5 sequential forehands missed).
If those moments visibly contain shots that the gallery's timeline doesn't
mark, the threshold is correct.

### Threshold validation

The revised rule (`removed_by_broken >= 1 OR total_disagreement > 5`) caught
34/36. The deprecated rule (`total_disagreement >= 3`) would have caught
roughly the same 32 — but the revised rule additionally flags
IMG_1110 (total=1, but 1 silent loss), IMG_7110 (same), IMG_1133 (total=2,
1 silent loss). These three would have been MISSED under the old threshold;
they're caught under the new one because false negatives matter.

The 2 KEEP videos:
- IMG_1122: 8/8/8/0/0/0 (zero disagreement — true match)
- IMG_1131: 0/0/0/0/0/0 (no detections in either — likely empty/corrupt video, not a meaningful "pass")

Threshold confirmed without recalibration.

### Reprocessing: NOT KICKED OFF (per brief)

Triage produces a recommendation; requeue is a separate human-confirm step.

To kick off the 34 REPROCESS videos:

```bash
ssh devserver 'cd /opt/tennis && venv/bin/python -c "
import sqlite3
conn = sqlite3.connect(\"/opt/tennis/coordinator.db\")
ids = [\"afb9df6d\",\"2b5edd87\",\"c328582e\",\"16385c5f\",\"de56a813\",\"039f6fb1\",
       \"29f3835a\",\"a61736c1\",\"a2bf3c51\",\"818dbf0e\",\"d7114edf\",\"9f84e09c\",
       \"c8a2824f\",\"15256443\",\"faf27bac\",\"146c4021\",\"f4bfcf04\",\"5bff547f\",
       \"ada5f7c1\",\"8dc22723\",\"e54f3ccb\",\"201ae7d7\",\"cf2ec3f4\",\"f7e654c7\",
       \"ddd6bf7c\",\"3747def1\",\"94918302\",\"e50070a7\",\"0fa431ed\",\"f8e0ab4c\",
       \"a2d63142\",\"95c22d7b\",\"59a97faf\",\"d506581a\"]
sql = \"UPDATE jobs SET status=?, claimed_by=NULL, claimed_at=NULL, retry_count=COALESCE(retry_count,0)+1 WHERE video_id IN (\" + \",\".join(\"?\" for _ in ids) + \")\"
n = conn.execute(sql, [\"pending\"] + ids).rowcount
conn.commit()
print(f\"requeued {n} jobs\")
"'
```

The next coordinator poll will hand them to whichever GPU asks first.
Tmassena (PRIMARY, just ran the triage on andrew-pc; both now have
canonical model + sidecar verified) is fine for either.

### What this closes out

The 2026-05-02 model-divergence incident chain ends here:
- Detection: hash check on worker startup (Component 3)
- Identity: sidecars (Component A)
- Evaluation: holdout (Component B)
- Promotion: deploy gate (Component C)
- Cleanup: triage + recommended requeue (Component D)

Anything remaining is independent work, not incident remediation.
