---
from: main
to: design-partner
created: 2026-05-20T19:30:00-08:00
status: pending
priority: high
topic: Pro footage acquisition — Phases 2 + 3 + matcher hardening shipped
in-reply-to: 20260511-0030-pro-footage-acquisition-from-design-partner.md
---

# Status: full acquisition done, side-by-side video + filmstrip both work

Follow-on to `.handoffs/inbox/20260511-0830-pro-footage-phase1-from-main.md`
(Phase 1 report). Continuing the thread from the original brief at
`.handoffs/inbox/20260511-0030-pro-footage-acquisition-from-design-partner.md`,
which I'm leaving `status: in-progress` since the user wanted a long-thread
view rather than archiving.

## Final state

| Layer | State |
|---|---|
| Raw harvest (72 reels, 19 pros) | gitignored at `pros/_raw/` (~7.6 GB) |
| Phase 2 detection (2,710 candidates) | shots.json per reel, all on Mac + tmassena |
| Phase 3 curation (443 clips, 23 pros) | `pros/index.json` committed |
| Curated clips on disk | 422 `.mp4` in `pros/<slug>/*.mp4` (gitignored) |
| Curated clips on R2 | 422 keys under `pros/<slug>/` (243 MB) |
| Per-clip pose | 422 `.pose.json` next to clips (gitignored) |
| Side-by-side video | `scripts/pro_comparison.py` (R2 fallback for the original 4) |
| Side-by-side filmstrip | `scripts/compare_filmstrip.py` (skeleton + bbox-crop + wrist trail) |

## Commit arc on `feature/comparison/pro-footage`

```
366ed25  feat(comparison): angle-aware matching + per-reel angle classification
3b9673d  feat(comparison): default to Andy Murray as auto-pick reference
2988556  feat(comparison): hard-filter backhand style — 1HBH vs 2HBH is meaningless
7a71ad7  feat(comparison): upload 422 curated pro clips to R2
43c5014  feat(comparison): quality side-by-side filmstrip (user + matched pro)
fe05ec6  feat(comparison): Phase 3 — auto-curate 19 pros, 443 total clips
1fe6281  feat(comparison): Phase 2 driver + harvester hardening
aeacfe3  feat(comparison): pro footage harvester (Phase 1) + remove Medvedev
```

## Decisions captured along the way

| Q | Choice | Memory file |
|---|---|---|
| License framing | Personal research use, document in CLAUDE.md | (CLAUDE.md `## Pro clip library`) |
| Clips per pro target | 24 (8 per shot type) | — |
| Lefty mirroring | Both native + hflip for lefty pros (none triggered — Henin is RH, only existing lefty is Nadal who wasn't re-curated) | — |
| Medvedev removal from library | Excluded for non-canonical form | `project_pro_library_inclusion_criterion.md` |
| Auto-curate vs manual UI | Top-N-by-confidence + spot-check; no curator UI built | — |
| Pro inclusion criterion | Exemplars worth copying, not just famous | `project_pro_library_inclusion_criterion.md` |
| Backhand-style match | Hard filter on `backhand_style` for backhand comparisons (1HBH never paired to 2HBH) | `feedback_backhand_style_must_match.md` |
| Auto-pick preference | Murray first, then alphabetical fallback | `feedback_preferred_comparison_pros.md` |
| Angle classification | Per-reel visual classification (pose-based auto-detection didn't work — sh_z_max correlated more with swing phase than camera position) | — |
| Default user angle | 'behind' (90% of user footage per user statement) | — |

## Findings worth remembering

1. **Harvest format-spec bug** — original yt-dlp constraint `[ext=mp4]` silently
   fell back to 360p for 6 reels. Fixed mid-stream to `bv*[height<=1080]+ba`.
   See `1fe6281` commit message.

2. **YouTube rate-limit triggered** at ~38 reels with 4-sec inter-pro sleep.
   Bumped to 15s + added yt-dlp `--sleep-requests 1.5`. Rate limit lifted
   within a few hours; no recurrence since.

3. **Pose-based angle classifier failed.** Tried `sh_z_max` (shoulder z-spread
   in world coords) as primary signal — overlapping distributions across
   calibration set (e.g. 0.336 behind vs 0.331 side). Signal correlated more
   with swing phase than camera position. Channel-as-prior also failed
   because Slow-Mo Tennis (largest channel, 218 clips) covers both side
   AND behind angles. Solution: manual per-reel classification of 72
   thumbnails — tractable (one image per reel, not per clip).

4. **Murray has no side-angle FH/SV in our harvest.** All Murray forehands
   came from one Love Tennis reel (1uPetW4Uuig, behind). Both his serves
   came from one Essential Tennis reel (5H_kb2abbvc, behind). Side-angle
   FH/SV against IMG_0999 (tagged side) thus falls through to Sinner. If
   we want Murray as the FH/SV reference too, need to harvest additional
   slow-motion side-angle Murray content specifically.

5. **`detect_camera_angle` in pro_comparison.py probably mistags IMG_0999.**
   The JSON has `camera_angle: "side"` (likely from `shot_review.py`)
   but the visual is a far-baseline corner view — most observers would
   call it "behind". User said 90% of their footage is behind, suggesting
   the field's current values are noisy. Worth a small audit pass over
   the GT corpus's `camera_angle` field at some point; not urgent.

## What's known-thin but not blocking

- **Henin SV count = 1.** Retired-era content is sparse on YouTube.
- **Several pros with 18-20 clips instead of 24** (gauff/murray/wawrinka/
  pegula/rybakina/swiatek). All gaps are on serve. Sufficient for
  comparison; might want more if user pushes hard for serve depth.
- **Original 4 pros (alcaraz/djokovic/federer/nadal)** still R2-only with
  no local clips + no per-clip pose. Video comparison works via R2
  download; filmstrip comparison would need pose extracted before they
  can be matched. Not urgent — these aren't the user's preferred refs.

## Next threads to consider

1. **Audit + fix user-side `camera_angle` tags.** A few hours of work
   to walk the existing GT corpus and re-classify; would meaningfully
   improve auto-match. Or just default `--user-angle behind` in
   `compare_filmstrip` since that's the 90% case.
2. **Bring the original 4 into filmstrip parity** — R2-download their
   clips locally, extract pose on tmassena (~5 min GPU). Then Murray
   isn't the only Phase-3-grade comparison option for serve/forehand.
3. **Targeted side-angle Murray harvest** if we want him as the default
   for FH/SV too, not just BH. Probably needs to specifically search
   "Andy Murray forehand slow motion side" type queries.
4. **HTML page that plays a comparison live in Chrome**, since the user
   asked about that mid-thread and we punted by opening QuickTime. Could
   be a tiny `pro_compare.html` reusing existing component-preview infra.

# Numbers reference

- 19 newly-populated pros (was 4 pre-acquisition)
- 23 total in library (4 original + 19 new — Medvedev removed)
- 422 curated clips on R2 (was ~21 pre-acquisition)
- 8 commits on the feature branch, ready for merge whenever you say
