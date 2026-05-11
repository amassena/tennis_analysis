---
from: design-partner
to: main
created: 2026-05-11T00:30:00-08:00
status: pending
priority: high
topic: Scope pro footage acquisition — fill clips:[] for the 20 expanded pros
---

# Why this is the next high-priority thread

The product is **rapid form comparison** (memory:
`project_product_north_star_form_comparison.md`). Pro library
expansion (4 → 24 players) shipped earlier this week, but **20 of 24
new pros have `clips: []`**. They cannot contribute to comparison
until footage exists. The data layer is done; the content layer is
the gap.

This is the largest single unmet load-bearing-for-product item.
~2-3 days first pass, immediately user-visible win.

# Out of scope for this brief

- Pro comparison UX improvements (alignment quality, time-warp, etc.)
  — that's a separate workstream that might be informed by what we
  learn here
- New pro additions beyond the existing 24 — fill existing entries
  first
- Tagging beyond shot-type — grip style, court position, etc. can
  layer on later

# Approach: hybrid harvest + curate

Three implementation phases, each independently checkpointable.

## Phase 1: harvest source material (~1 day)

Goal: download raw video material that's likely to contain good
reference shots for each pro.

**Source:** YouTube highlight reels via `yt-dlp`. Defensible for
personal/research use; clip licensing is a known gray area but
single-user analysis is well within fair-use norms. Document the
rationale somewhere committed (CLAUDE.md or BACKLOG.md note) so
future sessions don't re-litigate.

**Per-pro target:** 1-2 full highlight reels per pro, ~10-15 min each.
Prefer recent vintage (last 2-3 years) for stylistic relevance. ATP
official channel and WTA Tour channel are the best curated sources.

**Roster** (the 20 currently with `clips: []`):
- ATP active: Sinner, Medvedev, Zverev, Tsitsipas, Rublev, de Minaur,
  Fritz, Hurkacz, plus the retired greats Murray, Wawrinka
- WTA active: Sabalenka, Swiatek, Gauff, Rybakina, Pegula, Jabeur,
  plus retired greats Williams sisters, Henin

**Storage:** Mac local at `pros/_raw/<slug>/<source-id>.mp4`. Gitignored
(big files). Document in CLAUDE.md that `pros/_raw/` is local-only.

**Script:** `scripts/fetch_pro_highlights.py` — reads pros/index.json,
queries youtube-dl for top-result highlight reels per pro name, saves
to `pros/_raw/<slug>/`. Idempotent (skip if already downloaded). Logs
metadata (source URL, duration, date downloaded) to
`pros/_raw/<slug>/_manifest.json`.

## Phase 2: shot-detection pass on raw material (~half day)

Goal: find candidate per-shot clips automatically.

**Reuse existing pipeline.** Run `gpu_worker/worker.py` style detection
on the raw highlight reels via a `--from-local` flag (avoids R2 ingest
roundtrip). Produces:
- shots.json with timestamps + types
- per-shot pose JSON

These can be run on tmassena/andrew-pc directly. No coordinator
involvement needed — just point detect_shots_sequence.py at the local
file.

**Output:** `pros/_raw/<slug>/<source-id>_shots.json` with shot list.

## Phase 3: manual curation + ingest (~1 day)

Goal: pick the best 3-5 shots per shot type per pro.

**Curation tool:** browser-based viewer at
http://localhost:8088/pro_curator.html or similar (reuse existing
preview infrastructure if possible). Shows candidate shots with type
labels; user picks N per type per pro.

**Quality criteria per clip:**
- Full body visible at contact
- Clear side or 3/4 angle (front-facing OK for serves)
- No occlusion (e.g. linesman, court furniture)
- Contact moment unambiguous in frame
- Reasonable clip length (~2-3s before contact through ~1s after)
- For lefties (Nadal, Henin): preserved natively — the comparison
  matcher already filters by handedness

**Per-pro target:** 3-5 clips per shot type. For most pros:
- 4 forehands
- 3 backhands
- 3 serves
- 1-2 volleys (if available; many won't have good ones)

= ~12 clips per pro × 20 pros = ~240 clips total. Plus the 4 existing
pros that already have footage = total library ~280-300 clips.

**Ingest:** copy curated clips to `pros/<slug>/`. Update
`pros/index.json` `clips: []` arrays. Use existing schema (see Federer
or Djokovic entries for reference).

# Implementation steps for main

1. **Decide on yt-dlp invocation pattern** — single-pro fetch vs batch.
   I lean batch with rate limiting (be polite to YouTube).
2. **Write `scripts/fetch_pro_highlights.py`** (Phase 1)
3. **Run for 2-3 pros first** to validate the workflow before scaling
   to 20. Pick: Sinner (active, lots of recent footage), Wawrinka
   (1HBH, retired great), Sabalenka (active WTA, distinct style).
4. **Adapt detection pipeline for local files** — likely a small flag
   in `detect_shots_sequence.py` to skip R2 upload steps
5. **Build the curator UI** — can be minimal, just show video with
   detected shot timestamps as buttons
6. **Curate the 3 test pros** end-to-end, validate the comparison UX
   improves
7. **Scale to remaining 17 pros**

# Open questions / decisions for design-partner

(Surface these in your response brief; don't decide unilaterally.)

1. **License framing** — yt-dlp from public highlights is in a
   gray area. Comfortable framing it as "personal research use" or
   need a more conservative approach (only owned-rights material)?
2. **Volume of clips per pro** — 12 is a guess. Could be 6 (lighter
   library, faster decision) or 24 (more variety, longer curation).
3. **Pro comparison UX improvements** — should they be a parallel
   workstream, or sequenced after footage lands? My lean: parallel,
   since they don't conflict on files. But they might learn from
   each other (e.g. "we need more variety on serve angles" emerges
   from UX work).
4. **Handedness preservation** — Nadal/Henin clips don't get mirrored
   for right-handed-user comparison; we rely on the matcher's
   handedness filter to NOT match them. Confirm that's still the
   intended behavior.

# Estimated effort

- Phase 1 (fetch): ~1 day
- Phase 2 (detection): ~half day
- Phase 3 (curate + ingest): ~1 day (most of the time is manual
  selection, not code)
- **Total: ~2.5 days for full 20-pro fill**

# Why this beats other open items right now

| Item | Serves product? | Effort | Pick? |
|---|---|---|---|
| Pro footage acquisition | ⭐⭐⭐ direct | 2.5d | **YES** |
| Audio re-introduction for classifier | ⭐⭐ indirect | 1-2d | parallel? |
| Active-learning loop on corrections | ⭐⭐ indirect | 1d, gated on data | not yet |
| iPhone-uploads verification | ⭐ ops sanity | 5m | yes, knock out anytime |
| Kinetic-chain replacement metric | ⭐ research | unknown | park |

Pro footage is the only item that **directly enables a feature the
user said they want.** Everything else is supportive infrastructure.
