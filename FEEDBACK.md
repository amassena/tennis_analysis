# Feedback loop cheat sheet

Quick reference for testing without disturbing production. Type `/howto` at any Claude session to see this.

---

## See what's deployed (read-only, fast)

| Need | Command / URL |
|---|---|
| Browse all component previews | http://localhost:8088/components/ |
| One filmstrip | http://localhost:8088/components/filmstrip.html?vid=IMG_1195&shot=5 |
| Sequences modal content | http://localhost:8088/components/sequences.html?vid=IMG_1195 |
| Coach card content | http://localhost:8088/components/coach.html?vid=IMG_1195 |
| Alignment video | http://localhost:8088/components/align.html?vid=IMG_6665&shot=5&pro=djokovic |
| Card on mobile/desktop side-by-side | http://localhost:8088/components/card.html?vid=IMG_1195 |
| Live gallery | https://tennis.playfullife.com |

## Test UI changes locally (no deploy)

```bash
.venv/bin/python scripts/update_r2_index.py --preview
# → http://localhost:8088/preview-<branch>/
```

10-second cycle: edit code → re-run → reload browser. Reads live R2 data; only the HTML is local.

## Per-branch shareable URL (does not affect production)

```bash
.venv/bin/python scripts/update_r2_index.py --staging
# → tennis.playfullife.com/staging/<branch>/highlights/index.html
```

Use for: iOS WebView testing, sharing with someone for review.

## Generate one artifact for review (no R2 upload)

```bash
.venv/bin/python scripts/swing_composite.py --video IMG_1195 --shot 5 --no-skeleton
# writes exports/IMG_1195/sequences/shot_005_*.jpg

.venv/bin/python scripts/align_strokes.py --video IMG_1195 --shot 5 --pro djokovic
# writes exports/IMG_1195/comparisons/align_005_*.mp4
```

Skip `--upload` to keep local. Open with `open exports/.../foo.jpg`.

## Claude self-verifies (no screenshot from you)

When Claude needs to see a UI, it runs:

```bash
~/bin/screenshot "http://localhost:8088/components/filmstrip.html?vid=IMG_1195&shot=5" \
  -o /tmp/x.png --wait 2
```

Then `Read /tmp/x.png` — no need to paste a screenshot.

## Production deploy (only from main)

```bash
.venv/bin/python scripts/update_r2_index.py
# → tennis.playfullife.com (live, public)
```

Don't run from a worktree branch unless you're certain.

## Multi-stream context

| File | Purpose |
|---|---|
| `BACKLOG.md` | Drop bugs/ideas here as they hit. One-liners. |
| `FEATURES.md` | Active branches + file ownership |
| `DESIGN.md` | Vision + cross-platform rules |

`/sync` re-reads all three and reports state.
