---
from: pro-library
to: design-partner
created: 2026-05-09T07:10:00-08:00
status: done
priority: medium
topic: Pro library expansion + handedness refactor — implemented
in_reply_to: 20260508-2350-pro-library-expansion-from-design-partner.md
---

# What landed

All 7 brief steps executed in worktree `~/tennis_worktrees/pro-library/`
on branch `feature/comparison/pro-library`. Ready to merge.

## Files changed
- `pros/index.json` — bumped to v3. 4 → 24 players. Per-player schema:
  `name, qid, handedness, backhand_style, gender, birth_year, clips[]`.
  Existing 4 players' clip arrays preserved verbatim. 20 new players have
  `clips: []` (footage deferred per Step 5a).
- `pros/wikidata_cache.json` — NEW. Raw Wikidata fetch (P741/P21/P569).
  Separate from index.json so a refresh can't clobber human curation.
- `scripts/fetch_pros_from_wikidata.py` — NEW. Resolves a roster of names
  → QIDs via wbsearchentities, bulk SPARQL for attributes, writes cache.
  Run with no args to refresh. Stdlib only (urllib).
- `scripts/pro_comparison.py` — `PRO_HANDEDNESS` dict deleted. Matcher
  now reads `library["players"][slug]["handedness"]`. Added gender filter
  (default same-gender) with `--cross-gender` opt-in flag.
- `FEATURES.md` — pro-library row added to Active table; ownership of
  `pros/*` and the two scripts claimed in file-conflict map.

## Decisions captured

- **Sinner QID**: Q54812588 (verified — born 2001-08-16). Q83232664 looked
  plausible but resolves to a non-entity.
- **Wikidata coverage gap**: handedness tagged for ~all players, but
  `backhand_style` (P741 grip-style values) is missing for many big names
  (Wawrinka, Tsitsipas — both famously 1HBH; Sinner, Alcaraz, Medvedev,
  Zverev — 2HBH). Filled manually in `index.json`. The cache file shows
  raw_p741 so future refreshes can detect when Wikidata catches up.
- **Default-right-handed for missing P741 handedness QID**: Wikidata
  oddly omits "right-handedness" Q3039938 for some players (Federer is
  the canonical example — his P741 only carries grip-style values). The
  fetcher defaults `handedness="right"` when no `Q789447 left-handedness`
  is present. Documented in code.
- **No `country` field**: per your reframe in the response section.
- **Roster of 24** (vs ~25 target): ATP active 11, ATP retired greats 4,
  WTA active 6, WTA retired greats 3. Cross-gender comparison possible
  via `--cross-gender` flag (e.g. study Henin's 1HBH as a male user).

## Memory housekeeping
`project_handedness_constant.md` got a `RESOLVED 2026-05-09` note saying
pro_comparison now reads handedness declaratively. The user-side
`dominant_hand` default-to-R shortcut is unchanged.

## Verification
Ran `match_pro_clip` against `detections/IMG_1027_fused.json` (132
eligible shots, default user_hand=R since GT files don't carry the field):
- 0 Nadal matches (correctly filtered as lefty)
- All matches went to right-handed pros (Alcaraz/Djokovic/Federer)
- New empty-clip entries correctly contribute zero candidates
- `cross_gender=False` blocks female pros from male user matches; flag
  flips correctly with `--cross-gender`.

Did NOT run end-to-end ffmpeg comparison generation — Mac doesn't run
the pipeline (golden rule #1) and the matching layer is what carried the
old bug.

## Out of scope (kept as-is)

- Footage acquisition for the 20 new entries — `clips: []`.
- Style-similarity matching ("find pros with similar grip to user").
- Multi-handedness UI (left-handed user toggle).
- Refreshing the cache on a schedule.
