---
from: design-partner
to: main
created: 2026-05-20T23:30:00-08:00
status: pending
priority: high
topic: Finish-line plan — code is merged but features aren't live to the user yet
---

# Reality check

User just course-corrected me. I was treating "code merged to main" as
"done." That's wrong. The actual product-finish bar is:

1. Side-by-side comparison **surfaces in the gallery** — currently the
   `scripts/compare_filmstrip.py` exists but nothing in the gallery
   calls it. User can't click "compare this shot to a pro" today.
2. Filmstrip impact accuracy is **validated by the user** end-to-end —
   last measurement was 40% perfect / 25% close / 35% broken on
   IMG_1120 (filmstrip stream's report). User hasn't signed off.
3. **All 23 pros work** in comparison — original 4 (alcaraz/djokovic/
   federer/nadal) are R2-only with no local clips + no per-clip pose,
   so they fall through to other pros when matched. Comparison library
   is effectively 19 pros until this is fixed.

Everything else (detection improvements, dashboard PR-A/B/C,
dyntrack 4K/120) is either continuous-improvement or
physically-blocked.

# The four concrete items, in execution order

## 1. Original-4 filmstrip parity (~1 hour)

For each of `alcaraz`, `djokovic`, `federer`, `nadal`:

a. R2-download their existing clips to `pros/<slug>/*.mp4` locally
   (per pro-footage convention)
b. Run pose extraction on tmassena: `python scripts/extract_pro_clip_poses.py --pro <slug>` (or equivalent based on actual script signature)
c. Verify `compare_filmstrip.py --user IMG_xxxx --shot N --pro federer` produces output

Why first: small, bounded, immediately unblocks comparison against the
"famous" pros user actually wants to compare against. After this, all
23 pros work.

## 2. Gallery integration of side-by-side comparison (~half day)

Wire `compare_filmstrip.py` into the gallery so users see it. Specifics:

- New Worker endpoint `POST /api/compare/{vid}/{shot_n}?pro=<slug>` —
  invokes a job that runs compare_filmstrip, uploads output PNG to R2
  at `compare/{vid}_{shot_n}_{pro}.png`, returns URL.
- Gallery UI: per-shot chip → "Compare to pro" button → opens modal
  showing the side-by-side filmstrip. Defaults to auto-pick pro
  (Murray-preferred per `feedback_preferred_comparison_pros.md`).
- Optional v1: precompute compares for every shot at processing time
  so the modal opens instantly instead of waiting on render.

Why second: unblocks user-validation. Without gallery surface, the
user can't actually USE the feature to validate it.

## 3. Impact-accuracy validation (depends on user eyes)

After (1) and (2) ship, user clicks through ~20 recent videos in
gallery, validates side-by-side filmstrip accuracy.

Three possible outcomes:
- **A**: visibly good → ship as-is, this is the finish line
- **B**: visibly broken in the same ways filmstrip stream flagged
  (35% wrong-frame, two-player rally pose jumps) → next item
- **C**: visibly broken in new ways → file targeted bug, fix, re-validate

This is a physical-constraint item (needs user eyes) — can't be done
autonomously.

## 4. (Conditional on outcome B) Pose disambiguation for two-player rally (~half day)

If impact accuracy is failing on rally footage with opponent visible:
fix in `scripts/swing_composite.py` to pick the player with largest
pose bbox or most-consistent track. Already in BACKLOG.md from earlier
this session.

# Parallel streams that DON'T block finish

These can run in main sessions whenever main has cycles, but they
don't gate the user calling this "done":

- **Detection audio supervision** (4-6 hr training) — improves the
  upstream 35% broken cases over time. Greenlit. Detection branch can
  start whenever.
- **Dashboard PR-A** (`/dashboard` route + per-stage progress) — real
  dashboard. Independent of comparison feature.
- **Dyntrack 4K/120 test** — physically blocked on iPhone 17 Pro
  capture. Tomorrow per dyntrack's brief.
- **Side-angle Murray harvest** — improves Murray-coverage. In BACKLOG.

# Net actions for main session

1. **Right now**: execute item 1 (original-4 filmstrip parity). Probably
   ~1 hour of script-running + small plumbing. Likely paths:
   - Use existing `scripts/extract_pro_clip_poses.py` or
     `scripts/process_pro_raw.py` (now merged from pro-footage)
   - R2 download via existing `boto3` / wrangler tooling
   - Verify with one `compare_filmstrip.py` invocation per pro
2. **Next**: execute item 2 (gallery integration). Half-day. New Worker
   endpoint + gallery modal + R2 storage convention. Validate one
   end-to-end click-through.
3. **Then**: ping user for item 3 (visual validation). Hand off briefing
   the user that they can now click "compare to pro" in the gallery.
4. **Greenlight item-4 conditional on user feedback.**

# What I'm doing here

Just the brief + sequence. Per design-partner expanded scope (memory:
`feedback_design_partner_handoff_write_exception.md`), production code
changes (gallery JS, Worker endpoints, pose extraction runs) are
outside my scope — main session executes.

But the goal is "finish" — so this brief is the LAST scoping piece.
After this lands, the work is execution, not design.

# Memory updates

After finish: a memory file capturing the actual user-facing
definition of "done" for this app:

```
project_done_means_user_validated.md:
- Done = user can click through the feature on the live site AND
  has visually validated it works on real data.
- Not done = code merged to main.
- This was the gap that surfaced 2026-05-20.
```

I'll write that after the four items above are confirmed shipped.
