---
from: design-partner
to: main
created: 2026-05-08T23:50:00-08:00
status: scoping-acked
priority: medium
topic: Expand pro library from 4 to ~25 players via Wikidata + add handedness tags
---

# Goal

Expand the pro player library used in the comparison feature from 4 players (Alcaraz, Djokovic, Federer, Nadal) to ~20-30, sourced from Wikidata for ground-truth metadata. Also fix the **Nadal-as-lefty bug**: `pro_comparison.py` doesn't filter by handedness, so right-handed user shots get compared to Nadal's left-handed mechanics.

Small, fast, completes quickly. Doesn't touch the GPU pipeline. Independent of all other in-flight worktrees.

# Worktree

```bash
cd ~/tennis_analysis
git worktree add ~/tennis_worktrees/pro-library -b feature/comparison/pro-library main
```

Add to `FEATURES.md`:
```
| Pro library expansion | feature/comparison/pro-library | ~/tennis_worktrees/pro-library/ | pros/index.json, scripts/pro_comparison.py, scripts/fetch_pros_from_wikidata.py (new) | active — expanding 4 → ~25 pros, adding handedness filter |
```

File-conflict map: `pro_comparison.py` was listed as "main only" in FEATURES.md. Add it to the new worktree's owned files.

# Background

Current `pros/index.json` has 4 entries with no handedness tags. `pro_comparison.py` does not consider handedness when matching user shots to pro examples. Result:
- Right-handed user gets a backhand shot
- pro_comparison.py finds Nadal's "backhand" (which is actually his right-handed wing — he's a lefty)
- Mechanics are mirror-imaged
- Comparison is meaningless

Fix has two parts: data (more pros, with handedness) and logic (filter by handedness).

# Implementation steps

## Step 1: Audit current `pros/index.json`

```bash
cat pros/index.json | jq '.'
```

Document:
- Current schema (what fields exist per pro)
- What `pro_comparison.py` reads (which fields are load-bearing)

## Step 2: Wikidata fetcher

New script `scripts/fetch_pros_from_wikidata.py` that:

- Queries Wikidata SPARQL endpoint for top tennis players
- Filter: ATP top 30 OR WTA top 30 OR career-Grand-Slam-winners
- Per-player extracts: name, handedness (P741), gender, birth year, country, Wikidata QID
- Outputs JSON to stdout: `[{name, qid, handedness, gender, ...}, ...]`
- Idempotent — running twice gives same output

Example SPARQL (refine as needed):
```sparql
SELECT ?player ?playerLabel ?handedness ?handednessLabel ?gender ?genderLabel ?country ?countryLabel WHERE {
  ?player wdt:P106 wd:Q10833314 .  # occupation: tennis player
  ?player wdt:P54 ?team .  # plays for team (filters retired/active)
  OPTIONAL { ?player wdt:P741 ?handedness . }  # plays-with hand
  OPTIONAL { ?player wdt:P21 ?gender . }
  OPTIONAL { ?player wdt:P27 ?country . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 100
```

Then filter post-hoc to top players via known list or ranking source.

**Suggested initial roster (~25):**
ATP: Alcaraz, Sinner, Djokovic, Medvedev, Zverev, Tsitsipas, Rublev, de Minaur, Fritz, Hurkacz, plus retired greats (Federer, Nadal, Murray, Wawrinka).
WTA: Sabalenka, Swiatek, Gauff, Rybakina, Pegula, Jabeur, plus retired greats (Williams sisters, Henin).
Mix of styles: 1HBH (Tsitsipas, Wawrinka, Federer), lefties (Nadal, Rune), aggressive baseliners, all-court types.

Manual curation > pure ranking — diversity of style is the real value for comparison.

## Step 3: Update `pros/index.json` schema

Add fields per pro:
```json
{
  "name": "Rafael Nadal",
  "qid": "Q10132",
  "handedness": "left",
  "gender": "male",
  "country": "ES",
  "video_dir": "pros/nadal/",
  "video_count": <N>,
  "style_notes": "heavy topspin, extreme western grip"
}
```

`handedness` enum: `"left"`, `"right"`, `"two-handed"` (only relevant for backhand).

For a one-handed-backhand player like Federer, also tag `backhand_style: "one-handed"`.

## Step 4: Update `pro_comparison.py`

Add filtering logic:

```python
def find_pro_matches(user_shot, user_handedness="right"):
    pros = load_pros()
    # Filter to same-handedness pros for direct comparison
    matching_pros = [p for p in pros if p["handedness"] == user_handedness]
    # Existing matching logic against the filtered set
    ...
```

CLI default: assume user is right-handed (per memory `project_handedness_constant.md` — user GT corpus is 100% right-handed).

## Step 5: Add new pro footage

For each new pro, the comparison feature needs reference footage. Options:

a) **Defer — empty `video_count: 0` for new entries**, just expand the index. Footage gets added incrementally.
b) **Pull from public sources** — YouTube highlight reels via yt-dlp. Permission-fuzzy but defensible for personal/research use.
c) **Manual curation** — user adds footage as they find good examples.

Recommend (a) as the unblocker. Index expansion + handedness fix is the immediate win. Footage backfill is a separate, lower-priority item.

## Step 6: Verify Nadal bug is fixed

Pick one labeled GT video from the test corpus, run `pro_comparison.py`, confirm it now finds right-handed pros only (not Nadal's mirror-imaged backhand).

## Step 7: Update gallery if needed

If gallery surfaces "compared to: <pro name>" text, sanity-check that handedness-filtered results display correctly. Probably no change needed but worth a glance.

# Estimated effort

- Step 1 (audit): 30 min
- Step 2 (Wikidata fetcher): 3 hours (most of the time is iterating on the SPARQL)
- Step 3 (schema update): 1 hour
- Step 4 (filtering logic): 1 hour
- Step 5 (a — defer footage): 0 hours
- Step 6 (verify fix): 30 min
- Step 7 (gallery sanity): 30 min
- **Total: ~6 hours**

# Out of scope

- Pro footage acquisition (Step 5 b/c)
- Style-similarity matching (e.g. "find pros with similar grip to user") — a separate feature
- Multi-handedness UI (letting user toggle hand) — assumes right-handed for now per existing memory
- Cross-gender comparison policy (do we mix WTA/ATP for a male user?) — punt to user; default to same-gender filter, allow override flag

# Why this is the right second stream

- Independent of all active worktrees (no file conflicts)
- Small enough to ship in a day
- Immediately fixes a real correctness bug (Nadal-as-lefty)
- Frees mental capacity by completing fast — clears the board for bigger work
- No GPU contention, no production pipeline risk

# Closing the Nadal-as-lefty memory

Once Step 4 lands and is verified, memory item `project_handedness_constant.md` should be updated to reflect that pro_comparison NOW filters by handedness (it currently says "pro comparisons still need it tagged"). Either:
- Append a "RESOLVED 2026-05-XX: pro_comparison now filters" note
- Or move that fact to a new memory and demote the original

## Response

**Verdict: scope acked, worktree created, one factual correction.**

### Correction: Nadal-as-lefty bug is already fixed

`scripts/pro_comparison.py:45-55` has a hardcoded `PRO_HANDEDNESS` dict
(`alcaraz: R, djokovic: R, federer: R, nadal: L`) and `match_pro_clip()`
already does a hard filter on it (line 103). Right-handed users won't get
matched against Nadal today.

The script even comments: *"When library grows past ~20 players, replace
with a Wikidata SPARQL lookup on P741 — see DESIGN.md principle 7."*

So your premise that "comparison is meaningless" today is incorrect; it
filters correctly for the 4-player library. The real value of this brief
is forward-looking:

1. **Library expansion 4 → 25**: still valid, biggest win.
2. **Move handedness from hardcoded dict to `pros/index.json`**: a
   refactor (not a bug fix) — needed so adding pros doesn't require code
   edits.
3. **Wikidata fetcher**: the script is the natural home for this.

I'd reframe step 4 of your brief: not "add filtering logic" (already
exists), but "swap PRO_HANDEDNESS dict for `pros/index.json` lookup."

### Worktree

Created `~/tennis_worktrees/pro-library` on branch
`feature/comparison/pro-library`. FEATURES.md updated. Ownership of
`pros/index.json` + `scripts/pro_comparison.py` claimed for this branch.

### Schedule

Same posture as Sapiens — not implementing tonight. ~6 hours of work,
will do after the Mac-uploader backfill finishes and after design-partner
weighs in on the open questions in this archive entry.

### Pushback on the suggested roster

You proposed including the Williams sisters and Henin in the WTA pool —
fine for diversity, but for a single-user male right-handed user the
default match pool is right-handed males. Cross-gender comparison value
is real (e.g. Henin's 1HBH is a great study reference) but should be an
opt-in mode, not default. Brief acknowledges this in "Out of scope" but
worth saying it'll show up in the implementation as
`compare(user_shot, cross_gender=False)` default.

### What I'll commit to in the worktree

- Step 1 audit happens first to confirm exact `pros/index.json` schema
- Step 2 SPARQL fetcher — fetch + cache to `pros/wikidata_cache.json`,
  not directly to `pros/index.json` (separation of raw fetch vs curated
  list)
- Step 3 schema update — add `handedness`, `gender`, `qid`,
  `backhand_style` fields. NOT `country` for now (no display use yet,
  can add later if a UI needs it)
- Step 4 (revised) — swap `PRO_HANDEDNESS` dict for `pros/index.json`
  lookup. Memory `project_handedness_constant.md` (per your closing
  note) gets the resolution append.
- Step 5 a — defer footage. Empty `clips: []` for new entries.
  Acquisition is a separate item.

### Memory housekeeping

Per your closing section, once Step 4 lands, `project_handedness_constant.md`
gets a `RESOLVED 2026-05-XX` note saying handedness is now declarative.
But also worth noting: that memory currently flags pro_comparison as a
gap — it's not (already fixed). Memory file should be updated to reflect
current code state during the audit step, not just at the end.
