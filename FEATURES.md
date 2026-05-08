# In-flight features

Active development streams. Each line is a feature with branch, files touched, and status.

When starting a feature: add a row. When merging back to main: remove the row, move accomplishments to changelog.

Use `git worktree add ../tennis_<short-name> -b feature/<category>/<name>` to start a new one.

---

## Active

| Feature | Branch | Worktree | Files | Status |
|---|---|---|---|---|
| iPhone Shortcut ingest | (main) | (none) | worker/upload-worker.js, gpu_worker/worker.py, coordinator/api.py | Phase 0 done — Worker `/api/upload/iphone` deployed, smoke-test green via curl. Phase 1: GPU worker R2-first source resolution + coordinator job creation hook. |
| Filmstrip impact window | `feature/visual/filmstrip-impact-window` | `~/tennis_worktrees/filmstrip/` | scripts/swing_composite.py | active |
| Dynamic track keep-in-frame | `feature/visual/dynamic-track-keep-in-frame` | `~/tennis_worktrees/dyntrack/` | scripts/dynamic_track.py | active |
| Shot detection accuracy | `feature/detection/improve-shot-classification` | `~/tennis_worktrees/detection/` | scripts/detect_shots_sequence.py, scripts/sequence_model.py, training/ | active — per-class peak finding + class-conf demote-to-unknown_shot opt-in flags landed (F1 on GT 0.929 → 0.934 with `--per-class-thresh "serve=0.8,forehand=0.8,backhand=0.8"`). Production default unchanged. |

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
| `scripts/update_r2_index.py` | (none — main only) |
| `scripts/claude_coach.py` | (none — main only) |
| `ios/CourtIQ/**` | (none — main only) |
