# In-flight features

Active development streams. Each line is a feature with branch, files touched, and status.

When starting a feature: add a row. When merging back to main: remove the row, move accomplishments to changelog.

Use `git worktree add ../tennis_<short-name> -b feature/<category>/<name>` to start a new one.

---

## Active

| Feature | Branch | Worktree | Files | Status |
|---|---|---|---|---|
| Mac PhotoKit uploader (replaces pyicloud watcher + iOS Shortcut path) | (main) | (none) | scripts/upload_tennis_album.py, worker/upload-worker.js, coordinator/iphone_upload_poller.py, gpu_worker/worker.py, ~/Library/LaunchAgents/com.tennis.uploader.plist | **Live.** Mac launchd agent `com.tennis.uploader` runs `upload_tennis_album.py --watch 300`. Enumerates slo-mo videos in local Photos via PhotoKit (since 2026-04-01 default), dedups against Worker `/check`, uploads single-shot or chunked (50 MB parts via `/api/upload/iphone/{init,part,complete}`). Worker writes to R2 `source/{vid}.{ext}` + marker; Hetzner poller registers coordinator job; GPU R2-first source resolution pulls and runs pipeline. Desktop `Start/Stop Tennis Uploader.command` shortcuts. iOS Shortcut path abandoned (iOS automation timeout on multi-GB uploads). |
| Sapiens pose evaluation | `feature/detection/sapiens-eval` | `~/tennis_worktrees/sapiens-eval/` | scripts/sapiens_pose.py (new), scripts/compare_pose_extractors.py (new), eval/sapiens/ (new) | scoping — worktree created, eval not yet run. License: Sapiens-1 = CC-BY-NC (personal use OK, commercial needs Sapiens-2). |
| Pro library expansion | `feature/comparison/pro-library` | `~/tennis_worktrees/pro-library/` | pros/index.json, scripts/pro_comparison.py, scripts/fetch_pros_from_wikidata.py (new) | scoping — worktree created. Goal: library 4→~25 + handedness as data-layer field instead of hardcoded dict (Nadal-bug itself is already fixed in code). |
| Filmstrip impact window | `feature/visual/filmstrip-impact-window` | `~/tennis_worktrees/filmstrip/` | scripts/swing_composite.py | active |
| Dynamic track keep-in-frame | `feature/visual/dynamic-track-keep-in-frame` | `~/tennis_worktrees/dyntrack/` | scripts/dynamic_track.py | active |
| Shot detection accuracy | `feature/detection/improve-shot-classification` | `~/tennis_worktrees/detection/` | scripts/detect_shots_sequence.py, scripts/sequence_model.py, training/ | active — per-class peak finding + class-conf demote-to-unknown_shot opt-in flags landed (F1 on GT 0.929 → 0.934 with `--per-class-thresh "serve=0.8,forehand=0.8,backhand=0.8"`). Production default unchanged. |
| Pro library expansion | `feature/comparison/pro-library` | `~/tennis_worktrees/pro-library/` | pros/index.json, pros/wikidata_cache.json, scripts/pro_comparison.py, scripts/fetch_pros_from_wikidata.py | ready-to-merge — index.json bumped to v3 (4 → 24 players, +qid/handedness/gender/birth_year/backhand_style); PRO_HANDEDNESS dict removed from pro_comparison.py and replaced with index.json lookup; new `--cross-gender` flag (default same-gender). 20 new entries have empty `clips:[]` — footage backfill is a separate item. |

## Categories (for naming new features)

- **visual** — filmstrips, composites, dynamic tracking, render outputs
- **comparison** — pro library, alignment, side-by-side
- **detection** — shot classifier, pose extraction, ball tracking, court detection
- **coaching** — Claude prompts, biomech analysis, drill prescription
- **gallery-ux** — web gallery UI, sequences modal, coach modal
- **ios-live** — SwingLab app, on-device LLM, live AR coaching
- **measurement** — court homography, ball speed, line calls
- **ops** — pipeline reliability, notifications, error handling

## File-conflict map (avoid parallel work on same file)

| File | Owners |
|---|---|
| `scripts/swing_composite.py` | filmstrip-impact-window |
| `scripts/dynamic_track.py` | dynamic-track-keep-in-frame |
| `scripts/detect_shots_sequence.py` | improve-shot-classification |
| `scripts/pro_comparison.py` | pro-library |
| `scripts/fetch_pros_from_wikidata.py` | pro-library |
| `pros/index.json` | pro-library |
| `pros/wikidata_cache.json` | pro-library |
| `scripts/update_r2_index.py` | (none — main only) |
| `scripts/claude_coach.py` | (none — main only) |
| `ios/CourtIQ/**` | (none — main only) |
