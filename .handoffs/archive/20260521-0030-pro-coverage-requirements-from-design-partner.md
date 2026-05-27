---
from: design-partner
to: main
created: 2026-05-21T00:30:00-08:00
status: pending
priority: high
topic: Pro comparison library coverage requirements grid — what we need vs what we have
---

# The goal

For ANY user shot in the gallery, the "vs pro" comparison must show
a pro clip filmed at the **same camera angle as the user**, doing the
**same shot type**, with the **same handedness** (and same backhand
style for backhands).

Today's library doesn't meet this because we sourced mostly
broadcast-angle YouTube highlights. The fix is **sourcing more
footage, targeted at specific angle gaps**, not image processing.

# Required coverage grid

The matching axes that must all align between user shot and pro clip:

| Axis | Values | Notes |
|---|---|---|
| **Shot type** | forehand, backhand, serve, volley_FH, volley_BH, overhead | Volley + overhead are nice-to-have; FH/BH/Serve are required. |
| **Handedness** | right, left | User is right; library mostly right; Nadal is the only sourced left. |
| **Backhand style** | one-handed, two-handed | Required ONLY for backhands. User is 2H. |
| **Camera angle** | behind-player, side-deuce, side-ad, front-broadcast | The current "side" vs "behind" tags are too coarse — see below. |

## Camera-angle taxonomy (re-defined)

Current `angle` tag has two values (`side`, `behind`) and is
inconsistently applied. Replacing with four:

| Tag | Camera position | What the user sees on screen |
|---|---|---|
| `behind-player` | Behind the player's baseline, same side as player | Player's back, player facing AWAY from camera |
| `side-deuce` | To the player's right side (deuce sideline) | Player's right profile |
| `side-ad` | To the player's left side (ad sideline) | Player's left profile |
| `front-broadcast` | Opposite baseline, opposite side from player | Player facing camera (broadcast feed) |

For a **right-handed forehand**:
- `behind-player`: racket swings on screen-right (away from camera)
- `side-deuce`: racket swings TOWARD camera (forearm extends into frame)
- `side-ad`: racket swings AWAY from camera (forearm extends out of frame)
- `front-broadcast`: racket swings on screen-left (player's right side facing us)

Side-deuce and side-ad ARE NOT mirrors of each other (different
relationship to swing direction). Both need separate coverage.

# Minimum coverage target

For a right-handed user with 2HBH:

| Shot | Handedness | Backhand style | Angle | Min clips per pro | Min pros |
|---|---|---|---|---|---|
| forehand | right | n/a | behind-player | 3 | 3 |
| forehand | right | n/a | side-deuce | 3 | 3 |
| forehand | right | n/a | side-ad | 3 | 3 |
| forehand | right | n/a | front-broadcast | 3 | 3 |
| backhand | right | two-handed | behind-player | 3 | 3 |
| backhand | right | two-handed | side-deuce | 3 | 3 |
| backhand | right | two-handed | side-ad | 3 | 3 |
| backhand | right | two-handed | front-broadcast | 3 | 3 |
| serve | right | n/a | behind-player | 3 | 3 |
| serve | right | n/a | side-deuce | 3 | 3 |
| serve | right | n/a | side-ad | 3 | 3 |
| serve | right | n/a | front-broadcast | 3 | 3 |

**Total: 36 cells × 3 clips = 108 clips minimum for right-handed user.**
Add equivalent for lefties = ~216 total. Currently we have 399 clips
but heavily skewed to a few angles.

## Current coverage audit (399 clips, 19 pros)

Approximate, based on the existing `angle` tags which collapsed
`behind-player` + `front-broadcast` into `behind`, and split into
`side`:

| Approx. angle | Forehand | Backhand | Serve |
|---|---|---|---|
| "behind" (mix of behind-player + broadcast) | ~25 | ~30 | ~58 |
| "side" (mix of side-deuce + side-ad) | ~110 | ~115 | ~105 |

Two problems:
1. Most "behind"-tagged clips are actually broadcast-from-opposite,
   not behind-the-player. **True `behind-player` coverage is near zero.**
2. "side" doesn't distinguish deuce-side from ad-side, which matters
   because the swing direction differs.

# Sourcing plan

## Phase A: re-tag existing clips (~1 hour)

Audit all 399 clips visually, re-classify by the 4-tag taxonomy.
The curation UI we already built (`/tmp/curate`) can be extended to
add an "angle" picker per clip — click ✓/✗ and also pick angle.

Result: accurate per-clip angle metadata. Reveals exactly which
angle/shot cells are under-covered.

## Phase B: targeted sourcing for under-covered cells (~4-8 hours)

YouTube channels that film at specific angles:

| Angle | Likely sources |
|---|---|
| `behind-player` | First-person hitting partner POV, ball-machine practice videos, drone-from-behind training drills |
| `side-deuce` / `side-ad` | Tennis Warehouse "stroke breakdown" series, Essential Tennis "form analysis", IntuiTennis side-angle slo-mo, Modern Tennis Coaches |
| `front-broadcast` | Already abundant (ATP/WTA highlights — what we have today) |

For each under-covered cell, write targeted yt-dlp queries:
- `"forehand slow motion side view"`
- `"behind the baseline slow motion forehand"`
- `"tennis stroke analysis [pro name]"`
- `"on court ball machine [stroke] slo mo"`

## Phase C: re-curate + re-deploy (~2 hours)

1. Run yt-dlp on new sources → `pros/_raw/<slug>/<id>.mp4`
2. Phase 2 detection (`process_pro_raw.py`)
3. Curation pass via the existing UI, also tag angle per clip
4. Update `pros/index.json` with new `angle` taxonomy
5. Update `pro_comparison.py` matcher to use the 4-value angle
6. Upload new pro clips to R2
7. Re-run compare generation on user videos

## Phase D: detect user-shot camera angle (~2 hours)

Currently user shots have `camera_angle: null` in most cases. We need
to either:
- Auto-detect angle per shot using the pose-derived facing-direction
  signals (the work I just did on `/tmp/classify_facing.py` is the
  start — needs more training samples)
- Or have user tag the angle once per video upload

Once detected/tagged, the matcher can pick a same-angle pro clip.

# What I'd build next

In order:

1. **Extend curation UI** to add per-clip angle picker (4 values).
   User runs through 399 clips re-tagging angle. ~1 hour user time.
2. **Audit coverage** programmatically: per-cell clip counts post-tag.
   Identify under-covered cells.
3. **Targeted yt-dlp sourcing** for missing cells. Multi-hour batch.
4. **Re-extend `pros/index.json` and matcher** with 4-angle taxonomy.
5. **Pose-based auto-detection of user camera angle** so the matcher
   has something to filter on. Or fallback: user tags per video.
6. **Pre-render comparisons across all videos** with the new matching.

Total realistic time: 1-2 days for the data work, plus ongoing
curation as new pros/clips are added.

# What I'm NOT doing

- Image-based mirror/rotation to "fix" angle (impossible — confirmed
  with user 2026-05-21 — only legitimate use is handedness conversion)
- Generative model rendering of pros at user's angle (would require
  per-pro NeRF or similar — out of scope)

# Asking back

a) Does the 4-angle taxonomy match what you want? Should `behind-player`
   include "high-behind" (above-and-behind, common in coaching POV)
   as a separate tag, or rolled in?

b) Approve the sourcing strategy (yt-dlp from coaching channels) or
   should we look at non-YouTube sources too?

c) Are we OK doing the curation as a 1-hour-of-user-time pass
   (extended UI), or do you want me to attempt auto-detection of angle
   on existing clips first?

d) Multi-user future: when other people use the app, they may film
   from completely different angles than you. Should the library be
   tagged with absolute angles (deuce-side, ad-side as I've defined)
   that are user-agnostic, or relative (matches-my-typical-filming)?
   I lean absolute — the matcher then maps each user's filming to
   the same angle.
