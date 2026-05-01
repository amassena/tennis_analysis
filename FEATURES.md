# In-flight features

Active development streams. Each line is a feature with branch, files touched, and status.

When starting a feature: add a row. When merging back to main: remove the row, move accomplishments to changelog.

Use `git worktree add ../tennis_<short-name> -b feature/<category>/<name>` to start a new one.

---

## Active

| Feature | Branch | Worktree | Files | Status |
|---|---|---|---|---|
| Filmstrip impact window | `feature/visual/filmstrip-impact-window` | `~/tennis_filmstrip/` | scripts/swing_composite.py | not started |
| Dynamic track keep-in-frame | `feature/visual/dynamic-track-keep-in-frame` | `~/tennis_dyntrack/` | scripts/dynamic_track.py | not started |
| Shot detection accuracy | `feature/detection/improve-shot-classification` | `~/tennis_detection/` | scripts/detect_shots_sequence.py, scripts/sequence_model.py, training/ | not started |

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
