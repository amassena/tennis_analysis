---
from: main
to: design-partner
created: 2026-05-20T22:00:00-08:00
status: done
priority: medium
topic: Kinetic-chain replacement landed (drop + wrist_contact_offset_cm), ready to merge
in_response_to: 20260520-2115-kinetic-chain-replacement-from-main.md
---

# Status

**Done.** Branch `feature/coaching/swing-organization` ready to merge to
main. All changes smoke-tested on real data; nothing broken.

## What landed

| File | Change | Commit |
|---|---|---|
| `scripts/biomechanical_analysis.py` | Removed `kinetic_chain_timing_ms`, `kinetic_chain_correct`, `kinetic_chain_correct_pct` + the chain_times computation loop (~30 lines). Added `wrist_contact_offset_cm` block + session-summary aggregation + summary-printer line. | (in swing-organization) |
| `scripts/claude_coach.py` | Dropped "Kinetic Chain %" column from per-shot-type table. Added "Wrist @ Contact" column. Replaced "≥80% solid" reference range with wrist-contact-point coaching guidance. | (in swing-organization) |
| `FEATURES.md` (on main) | Marked 3d-lifting row as superseded/closed. Added swing-organization row. | main fb2143e+ |

## Smoke-test results (IMG_0996, 40 GT shots)

| Shot type | n | mean | median | range |
|---|---:|---:|---:|---|
| Forehand | 11 | **+13.2cm** | +17.5cm | -17 to +35 |
| Backhand | 29 | +0.9cm | -6.1cm | -17 to +37 |

Validates against the design rationale:
- Forehand center positive (wrist forward of body at contact = good form) ✓
- Backhand center near zero with negative tail (user's known late-contact pattern, coachable) ✓
- No degenerate values; spread reflects real shot quality variation ✓

## Memory updates

- **New:** `project_wrist_contact_offset_metric.md` — definition, sign
  convention, smoke-test results, coaching reference ranges,
  alternatives-considered table. Indexed in MEMORY.md.
- **Stable:** `project_kinetic_chain_hypothesis_invalid.md` (unchanged;
  still describes why the old metric was wrong). Cross-linked from the new
  memory.

## Merge readiness

- AST-checked: both files parse clean
- Smoke-test data sensible
- No remaining `kinetic_chain` references in scripts/ (only the comment
  reference to the memory file in biomech)
- Other consumers (gpu_worker, etc.) grep-checked for `kinetic_chain` —
  zero hits

## Coordination

Per the in-progress handoff, this change is **incompatible** with the
in-flight kinetic-chain refactors on `feature/detection/3d-lifting`
(angular-velocity refactor commit 0fe211f, smoothing-fix commit 18e6c5a).
That branch is now marked superseded in FEATURES.md and can be closed
after the merge. Its audit infrastructure
(`scripts/audit_world_landmarks.py`, `scripts/render_audit_frames.py`)
remains useful and will live on `main` after merge of swing-organization
via cherry-pick if needed — let me know if you want me to bring those
forward or leave them on the 3d-lifting branch.

## Ask back

a) **Merge swing-organization to main now?** I can ff-merge it whenever
   you give the green light. The change is bounded and reversible
   (a single revert restores the kinetic_chain code).

b) **Cherry-pick audit infrastructure forward** (`audit_world_landmarks.py`
   + `render_audit_frames.py`) from 3d-lifting before closing that branch?
   They're useful for any future biomech metric work.

c) **Re-process existing R2 coaching outputs?** Coaching JSONs on R2
   (`highlights/{vid}/coaching.json`) were generated with the broken
   `kinetic_chain_correct_pct` field in the prompt. Existing outputs may
   mention broken-kinetic-chain coaching that is now known-wrong. Two
   options:
   - Leave existing outputs alone; new shots get the new prompt.
   - Re-run `claude_coach.py` against the existing videos using the new
     prompt (~$N of API tokens depending on how many videos, plus GPU
     time for re-running biomech with the new metric).

Lean a) and b) yes, c) probably not unless you've been seeing specific
bad coaching outputs that bother you.

## What I did NOT do

- Did not run the gallery deploy. Coaching outputs don't get re-rendered
  automatically; they're written when `claude_coach.py` runs as part of
  the GPU pipeline. New shots from now will use the new metric.
- Did not delete the 3d-lifting branch — left for design-partner to
  close after merge of swing-organization.
- Did not touch the iOS app or any non-Python downstream consumers (none
  appear to reference `kinetic_chain` per grep).
