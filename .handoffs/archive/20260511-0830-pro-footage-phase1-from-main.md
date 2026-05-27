---
from: main
to: design-partner
created: 2026-05-11T08:30:00-08:00
status: pending
priority: high
topic: Pro footage acquisition — Phase 1 (harvest) done + decisions captured + Phase 2/3 plan
---

# Status: Phase 1 shipped, 3-pro validation green, ready for Phase 2 decision

Response to `.handoffs/inbox/20260511-0030-pro-footage-acquisition-from-design-partner.md`.

## Decisions captured (from user, in-session)

| Q | Choice |
|---|---|
| License framing | **Personal research use** — proceed with yt-dlp from public highlights. Documented in CLAUDE.md (`## Pro clip library`). |
| Volume per pro | **24 clips per pro (~480 total)** — bumped up from your guess of 12, prefer more variety per shot type |
| UX work timing | **Sequenced after footage** — finish acquisition first, then UX work informed by what real diverse footage looks like |
| Handedness | **Store BOTH native + mirrored for lefty pros** — user wants the matcher to have either option. Schema change deferred to Phase 3 ingest. |

The handedness decision changes the brief's Phase 3 spec slightly: each lefty
clip (Nadal, Henin) will produce two index.json entries — one native, one
mirrored — so the comparison matcher can pick either based on user
handedness. Implementation detail: store `forehand_001.mp4` (native) and
`forehand_001_mirrored.mp4` (`hflip` via ffmpeg) under `pros/<slug>/`, and
tag each clip with a `mirrored: true/false` field in index.json. Defer
exact schema until Phase 3.

## Phase 1 outcome

**Built:** `scripts/fetch_pro_highlights.py` (~200 LOC, single file, no new deps — uses local `yt-dlp` 2026.02.21).

Mechanism:
- Reads `pros/index.json`, filters to pros with empty `clips:[]`
- Per pro: `ytsearch12:<name> tennis highlights`, returns JSON metadata
- Scores candidates: tier 3 = preferred channel + ≥2022 + 5-30 min duration; tier 2 = recent + duration window; tier 1 = duration only
- Downloads top 2 per pro (bestvideo+bestaudio ≤1080p, MP4 muxed)
- Idempotent (skips if `<id>.*` already present in slug dir)
- Logs to `pros/_raw/<slug>/_manifest.json` (id, title, channel, upload_date, duration, view_count, url, downloaded_at)
- 4s sleep between pros to be polite

**Preferred channel list** (matched as lowercased substring against `channel`/`uploader`):
ATP Tour, WTA, Tennis TV, Tennistv, US Open Tennis, Roland-Garros, Wimbledon, Australian Open, SlowMoTennis, EssentialTennis, Top Tennis Training.

**Validation run** (Sinner, Wawrinka, Sabalenka, your suggested test set):

| Pro | Pick 1 | Pick 2 | Total size |
|---|---|---|---|
| sinner | Tennis TV, Madrid 2026 Final, 8.1 min, 1.21M views | Tennis TV, Madrid 2026 SF, 6.4 min, 756K views | 242 MB |
| wawrinka | US Open Channel, vs Djokovic 2016 Final, 20.4 min, 505K views | Australian Open, vs Nadal 2014 Final, 11.1 min, 354K views | 152 MB |
| sabalenka | Roland-Garros, vs Gauff 2025 Final, 12 min, 3.0M views | Wimbledon, vs Raducanu 13-min game, 13.8 min, 1.3M views | 249 MB |

All 6 reels: tier 3 picks, official channels, recent (2024-2026), good duration. **643 MB total for 3 pros**, so projecting ~5 GB for the full 20-pro fill.

## Finding worth flagging: highlight reels mix two players

Every reel above is a *match* (Sinner vs Zverev, Wawrinka vs Djokovic, etc.).
That's expected — official channels post match highlights, not single-player
reels. So Phase 3 curation will need a **player-disambiguation step**: the
user has to label which shots are Sinner's vs Zverev's, etc.

Two options to address this:

1. **Add disambiguation to curator UI** (Phase 3). For each candidate shot,
   show the clip and let user tag both shot-type AND which-player-is-hitting.
   Slower per-shot curation, but works with the material we have.
2. **Bias search toward single-player content**. Append "slow motion" or
   "practice" or "training" to the search query — e.g.
   `ytsearch:Jannik Sinner forehand slow motion`. Less material per pro
   (fewer reels exist), but cleaner labeling.

**My lean: do both.** Run a second pass with a slow-motion/practice query
modifier first, see if that alone yields enough material per pro; fall
back to match-highlight reels (with disambiguation step) for pros where
single-player content is sparse (likely retired pros — Williams sisters,
Henin, Wawrinka may have less recent practice footage).

Decision needed before Phase 2/3 work continues. Quick to add `--query-style
slow-motion|highlights|both` flag to `fetch_pro_highlights.py`.

## Implementation plan for remaining phases

### Phase 2 — shot detection on raw material (~half day, not yet started)

Adapt `scripts/detect_shots_sequence.py` to accept a local file path
(skip R2 ingest entirely). Run on GPU (tmassena primary) per the golden
rules — Mac is not for pose extraction.

Output: `pros/_raw/<slug>/<vid_id>_shots.json` with shot list (timestamp,
type, confidence) for each downloaded reel.

Run pattern:
```
# SCP raw reels to tmassena, run detection, scp results back
scp -r pros/_raw/sinner/ tmassena:'C:/Users/amass/tennis_analysis/pros/_raw/'
ssh tmassena 'cd C:/Users/amass/tennis_analysis && venv/Scripts/python scripts/detect_shots_sequence.py --from-local pros/_raw/sinner/<id>.mp4'
scp tmassena:'C:/Users/amass/tennis_analysis/pros/_raw/sinner/*_shots.json' pros/_raw/sinner/
```

(Could automate via a one-shot harness script; not critical for a 20-pro one-time backfill.)

### Phase 3 — curator UI + ingest (~1 day, not yet started)

Browser-based viewer at `http://localhost:8088/pro_curator.html`:
- Shows shot candidates from `_shots.json` as a vertical list
- Each row: thumbnail (frame at contact), video player snippet (2.5s before → 1s after), shot-type dropdown, player tag dropdown (if disambiguation enabled), keep/skip
- Bulk-export: writes selected shots to `pros/<slug>/<type>_NNN.mp4` (trimmed + re-encoded at 60fps 640×480 per `clip_spec`)
- Optional: lefty handler — for lefty pros, ALSO writes `<type>_NNN_mirrored.mp4` (ffmpeg hflip) for the both-options handedness behavior

Ingest finalization:
- Update `pros/index.json` `clips: []` arrays via a small append helper
- Upload `pros/<slug>/*.mp4` to R2 under `pros/<slug>/<file>` (matches `pro_comparison.py:164` resolution)
- Commit the index.json change (clips themselves stay in R2)

## What's not done yet

- [ ] Phase 2 detection adapter (`--from-local` flag in detect_shots_sequence.py)
- [ ] Decision on search query style (slow-motion vs highlights vs both) — needs your call
- [ ] Phase 3 curator UI
- [ ] Lefty mirror handler (Phase 3 detail)
- [ ] Scale to remaining 17 pros (after validating Phase 2+3 on the 3 already-fetched)

## What's done

- [x] Decisions on the 4 open questions
- [x] `scripts/fetch_pro_highlights.py` (committed)
- [x] `.gitignore` entry for `pros/_raw/`
- [x] CLAUDE.md note on pro clip library + license framing
- [x] Validation run on Sinner, Wawrinka, Sabalenka (643 MB raw material staged)

## Recommended next thread

Decide on search-query strategy (sec above), then I'll:
1. Patch `fetch_pro_highlights.py` to take `--query-style` if needed
2. Run for remaining 17 pros (~2-3 hr download time, mostly idle)
3. Move to Phase 2 on GPU
