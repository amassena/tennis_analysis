---
from: main
to: any
created: 2026-05-28T09:30
status: ready
priority: normal
topic: overnight playlist refactor + UX polish (Phases 1-4, builds 13-17)
---

# Overnight summary — playlist refactor complete

You went to bed mid-Phase-4b. Phases 1, 2, 3, 4a, 4b all shipped.
Phase 4c (rally.mp4 back-catalog cleanup) is queued with a safer
script but not executed — that's the only thing waiting on you.

## What's live

### Pipeline (new videos)
- `gpu_worker/worker.py` now runs `--types timeline` only.
- No more bytype (forehands/backhands/serves/volleys), no `--slow-motion`,
  no rally. Saves ~8 ffmpeg passes + ~450 MB R2 per session vs the
  pre-refactor pipeline.
- Synced + restarted on both tmassena and andrew-pc.

### Web gallery
- Cards collapsed to a single `▶ Watch <N>` button. The chip explosion
  of 9 buttons per card is gone.
- In-player chip filter row above the shot strip:
  `[All N] [Rally N] [Serve n] [FH n] [BH n] [Volley n] [OH n] · [🐢 Slo]`
- Auto-seek skips gaps between segments of the selected type.
- Rally chip uses point-grouping (shots within 8s = one segment),
  matching the old rally.mp4 cut.
- Filter & Sort row now expanded by default (toggle still works).
- Sequences modal: filmstrip fits to width on mobile (no more orphaned
  labels).
- Coach modal: tighter padding + font sizing on small screens.
- Player title now shows recorded time, e.g. `9:21 PM — iphone_9ca0a615`.
- Bug: renamed in-player `currentFilter` → `playerFilter` to avoid a
  `var` collision with the gallery's older session-filter code.

### iOS app — TestFlight build 17 (`1.2 (17)`)
- `FilterablePlayerView` (SwiftUI) — full chip filter row + 🐢 Slo
  toggle + segment auto-seek + Rally chip (parity with web).
- Locked to portrait so device rotation doesn't trigger Apple's
  landscape fullscreen takeover (which had been hiding the chip row).
- AVPlayer container now fills available vertical space (was sized to
  intrinsic 16:9, leaving big black bars in portrait).
- WKUIDelegate bridge for `confirm()` so the gallery delete button
  works on phone.
- Bumped through builds 13 → 14 → 15 → 16 → 17. Earlier builds also
  attached to "Internal" beta group manually to clear TestFlight
  propagation lag.

### R2 cleanup
- Phase 4a executed: 525 files / 65.5 GB freed (per-type +
  *_slowmo + grouped + highlights files for back catalog).
- Cleanup script (`scripts/cleanup_legacy_exports.py`) is conservative:
  dry-run by default, requires `--execute` to delete, defensive
  parsing that treats parent dir as the canonical `<vid>` (caught a
  bug that would otherwise have deleted timeline files — see commit
  aa09d2b).

## The one thing left

Once you've installed build 17 and confirmed the Rally chip works
end-to-end:

```
python scripts/cleanup_legacy_exports.py --all-users --execute
```

That drops ~99 rally.mp4 files (15 GB). The script preserves
`IMG_1044` because it's the one video without a timeline.mp4 export
— rally is its only fallback.

## What I considered but didn't ship

- **Upload cancel/retry deeper than UI buttons** — UploadManager
  already exposes `retry()` and `discard()` (wired to Retry button +
  trash icon). Real URLSession task cancellation would need task
  tracking added — risk vs reward didn't favor an overnight change.
- **Pro Comparison view in iOS** — web has it via the per-shot "vs
  pro" pill; iOS doesn't. ~half-day port, deferred for explicit
  user direction.
- **Local notifications (PR-I)** — explicitly avoided overnight.
- **Jump back to last-viewed video** — minor; would surprise users.

## Builds 13–17 changelog (TestFlight)

- 13: WKUIDelegate bridge for JS alert/confirm/prompt (delete button)
- 14: FilterablePlayerView debut — chip filter + slo toggle
- 15: always render chip row + diagnostic logging
- 16: portrait fill fix + PortraitHostingController to block AVPlayer's
       landscape takeover
- 17: Rally chip (web + iOS parity)

## Commits on `feature/gallery/per-user` since you went to bed

```
b1a719c gallery: human-friendly player title (time + video id)
976990e gallery: mobile UX polish — sequences + coach modal
f43f3bf ops: cleanup script learns rally-deletion safety check
bcba843 Phase 4b: Rally chip replaces rally.mp4 + UX polish (build 17)
aa09d2b ops: Phase 4a back-catalog cleanup script + executed
1501b17 ios: fix player layout in portrait, lock to portrait (build 16)
a54dfd4 ios: always render chip row + diagnostic logging (build 15)
9603e51 ios: native chip filter + slo-mo toggle in player (Phase 3, build 14)
eef2e61 fix(gallery): bridge JS alert/confirm/prompt to native UIAlerts (build 13)
b5cc36a pipeline: stop emitting bytype + *_slowmo files (Phase 2)
322be7b gallery: collapse card to single Watch button (Phase 1 cleanup)
dc6d366 gallery: in-player shot-type filter + slo-mo toggle (Phase 1)
800c4ba gallery: hide videos with 0 shots detected (probably not tennis)
```
