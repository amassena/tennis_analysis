# Tennis Analysis — Claude Code Context

Auto-loaded by Claude Code when working in this repo. Keep this file current; it's the single source of truth for project conventions.

For deeper pipeline history and archived decisions, see `docs/CLAUDE.md`.

---

## Golden Rules

1. **Never run the video pipeline on Mac.** Pose extraction, rendering, encoding, training, and `claude_coach.py` all run on the GPU machines. Mac is only for iCloud downloads, manual labeling (`shot_review.py`), and orchestration.
2. **GPU priority**: **tmassena is PRIMARY** (`ssh tmassena`, RTX 4080). andrew-pc (`ssh windows`, RTX 5080) is fallback. If tmassena is unreachable, ASK the user before falling back.
3. **Never downsample source video.** 240fps / 1080p is the product. Optimize by GPU/parallelism, not quality loss.
4. **Always use `.venv/bin/python`** on Mac — system Python lacks sklearn and the pipeline silently degrades.
5. **Never overwrite production models or data without archiving first.** `models/` is gitignored. Train new models under new names; only rename to `shot_classifier.pkl` after validation.
6. **Don't re-enable the Mac launchd watcher** (`com.tennis.watcher`). It was disabled 2026-04-15 because it duplicated the GPU pipeline locally. Parked at `~/Library/LaunchAgents/.disabled/`.

## Architecture (as of April 2026)

```
iPhone (240fps slo-mo) → iCloud
      │
      ▼
Hetzner VM (5.78.96.237)                       Local GPU machines
┌────────────────────────────┐                  ┌──────────────────────────┐
│ tennis-watcher.service     │                  │ tmassena (RTX 4080)      │
│   (iCloud poller,          │                  │ andrew-pc (RTX 5080)     │
│    verify_session_health)  │                  │                          │
│ tennis-coordinator.service │─── HTTP poll ───▶│ gpu_worker/worker.py     │
│   (FastAPI job queue,      │                  │   pipeline steps 1-7     │
│    SQLite state)           │                  │                          │
└────────────────────────────┘                  └──────────────────────────┘
      │                                                      │
      │                                                      ▼
      │                                         ┌──────────────────────────┐
      │                                         │ Cloudflare R2            │
      │                                         │ bucket: tennis-videos    │
      │                                         │   highlights/{vid}/…mp4  │
      │                                         │   highlights/{vid}/meta.json │
      │                                         │   highlights/{vid}/coaching.json │
      │                                         │   highlights/thumbs/*.jpg │
      └──────── Cloudflare Worker ──────────────│   highlights/index.html  │
               tennis-media                     └──────────────────────────┘
               (tennis.playfullife.com/*)
```

**Gallery URL: https://tennis.playfullife.com** (old `media.playfullife.com` redirects here.)

## Pipeline Steps (`gpu_worker/worker.py run_pipeline_with_stages`)

1. **Preprocess** — `preprocess_nvenc.py`, VFR→60fps CFR, NVENC
2. **Thumbnail** — ffmpeg `thumbnail=100` filter for a non-dark frame, upload to R2 `thumbs/{vid}.jpg`
3. **Prescan / dead-section skip**
4. **Pose extraction** — MediaPipe (33 keypoints) on GPU, `--skip-dead`
5. **Shot detection** — `detect_shots_sequence.py` (1D-CNN, F1=96.5%, threshold=0.90)
   - 5b. Upload `meta.json` to R2 (duration, shots, breakdown, iCloud date fallback)
   - 5c. `biomechanical_analysis.py` — per-shot kinetic chain, knee bend, trunk rotation, arm extension, fatigue
   - 5d. `claude_coach.py` — sends metrics + per-shot timeline to Claude Sonnet 4.5, saves `coaching.json` to R2
6. **Export** — `export_videos.py --types timeline rally bytype --slow-motion --upload`
   - `bytype` = per-shot-type compilations (forehands, backhands, serves, volleys), each with `_slowmo` variant, auto-zoomed to player bounding box
   - Legacy `grouped` kept for older videos
7. **Gallery index regen** — `update_r2_index.py` (runs JS syntax check before upload)

## Model deploy gate

No model file may be promoted to `deploy_status: approved` (or copied to `models/sequence_detector.pt` on a worker) without first passing `compare_models.py` against the current production baseline.

**Workflow to promote a candidate:**

1. Train: `train_sequence_model.py --output models/<candidate>.pt` (writes `.sidecar.json` with `deploy_status: candidate` automatically; refuses to train if any holdout video appears in training set)
2. Evaluate: `eval_holdout.py models/<candidate>.pt` (writes `eval_results/<sha8>_<date>.json`, populates sidecar)
3. Gate: `compare_models.py models/baseline_<sha8>_<date>.pt models/<candidate>.pt`
   - Exit 0 → PASS, candidate may be promoted
   - Exit 1 → BLOCK, see printed reasons
4. (If PASS) human flips sidecar's `deploy_status` from `candidate` to `approved`
5. SCP candidate + sidecar to GPU machines as `models/sequence_detector.pt` + `.sidecar.json`
6. Worker on each machine FATALs unless sha matches sidecar AND status is `approved`

The deploy gate rules are committed at `~/.claude/projects/-Users-andrewhome/memory/project_model_deploy_gate_rules.md`. Do not change without explicit user approval. Acceptance regression test: broken model `baseline_28814eeb_BROKEN_20260502.pt` MUST exit 1 against canonical baseline.

## Machine Setup

Every GPU machine needs:
- Project at `C:\Users\amass\tennis_analysis`, venv at `venv/`
- `.env` with ICLOUD_*, CF_ACCOUNT_ID, CF_R2_*, GMAIL_*, ANTHROPIC_API_KEY (SCP the full Mac `.env` — don't append)
- `pip install anthropic` (0.95.0+)
- iCloud session cookies at `config/icloud_session/` (copy from a working machine to skip 2FA)
- `TennisGPUWorker` scheduled task for auto-start (survives SSH disconnect)

Sync code when you change scripts:
```
scp scripts/*.py tmassena:'C:/Users/amass/tennis_analysis/scripts/'
scp gpu_worker/worker.py tmassena:'C:/Users/amass/tennis_analysis/gpu_worker/'
# Restart — running process doesn't pick up file changes:
ssh tmassena 'powershell -c "Get-CimInstance Win32_Process | Where-Object {\$_.CommandLine -like \"*worker.py*\"} | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"'
ssh tmassena 'schtasks /run /tn TennisGPUWorker'
```

## Key Scripts

| Script | Runs on | Purpose |
|---|---|---|
| `cloud_icloud_watcher.py` | Hetzner | Polls iCloud Slo-mo album, creates jobs in coordinator |
| `coordinator/api.py` | Hetzner | FastAPI job queue (SQLite) |
| `gpu_worker/worker.py` | tmassena / andrew-pc | Claims jobs, runs full pipeline |
| `scripts/detect_shots_sequence.py` | GPU | Sequence CNN shot detection (F1=96.5%) |
| `scripts/biomechanical_analysis.py` | GPU | Per-shot-type biomech summaries |
| `scripts/claude_coach.py` | GPU | Claude-authored coaching (reads local detections + biomech, uploads coaching.json) |
| `scripts/export_videos.py` | GPU | Exports timeline/rally/bytype + slow-mo, auto-zoom crop, R2 upload |
| `scripts/update_r2_index.py` | GPU or Mac | Rebuilds gallery HTML, JS-validated, pushes to R2 |
| `scripts/pipeline_health.py` | Mac | End-to-end status check (watcher, coordinator, GPU workers, queue, gallery) |
| `scripts/shot_review.py` | Mac | Manual labeling GUI (local HTTP + browser) |

## Cloudflare Worker (`worker/upload-worker.js`)

Deploy: `cd worker && CLOUDFLARE_API_TOKEN=... npx wrangler deploy`

Routes: `tennis.playfullife.com/*`, `media.playfullife.com/*` (301→tennis), `playfullife.com/*`, `www.*`

Endpoints:
- Asset serving with path-fallback (tries `key` then `highlights/key`)
- `?dl=1` → `Content-Disposition: attachment`, skips Range path so response is 200 (Chrome rejects 206 downloads)
- `GET /api/queue`, `POST /api/status/:id/update`, `POST /api/upload/…` (chunked upload)
- `GET /api/tags`, `POST /api/tags` (session tagging, password-protected)
- `POST /api/video/:vid/delete` (password `deletevideo`, logs to `deleted.json`)

**Critical Worker gotcha:** R2's `BUCKET.get()` sometimes sets `obj.range` even when the client sent no Range header. Returning 206 to a plain GET breaks `<img>` tags (blank thumbnails) and Chrome downloads. The 206 branch must check `obj.range && rangeHeader && !isDownload`. Do NOT relax that guard.

**HTML must never be CDN-cached.** Worker sets `Cache-Control: no-store, no-cache, must-revalidate, max-age=0` + `cdn-cache-control: no-store` for HTML responses. Reverting to `max-age=300` will cause "my update isn't showing" bug reports.

## Gallery Features

- Session grouping by date, copyable `#YYYY-MM-DD` anchors
- Filter chips: All / Type (Serves/FH/BH) / Month (auto-generated) / People (from tags)
- Sort: Date Recorded, Most/Fewest Shots, Longest/Shortest
- Search: matches video ID, date, tagged person
- Per-card: thumbnail click plays, expanded view shows Coach summary + per-type play buttons with shot counts, delete 🗑 at bottom
- Coach summary card → click opens modal with clickable timestamp buttons that jump to example shots in the timeline video
- Download: ⇩ arrow per video link, Download button in player overlay
- Session tagging: `+ person` with autocomplete

## Known Labeled GT Videos (training corpus — 44 videos, 1252 shots)

Files at `detections/{vid}_fused.json` (user-edited ground truth) — distinct from `{vid}_fused_detections.json` (auto output). Must back up before model retraining; `models/` is gitignored so there's no safety net.

## Common Pitfalls

- **Windows cp1252 encoding**: Avoid Unicode arrows / box-drawing chars in `print` statements. For Windows Python: `PYTHONIOENCODING=utf-8 && chcp 65001`.
- **pyicloud stale session**: Auth succeeds but returns empty/old library data. Watcher has `verify_session_health()` that catches this every 30 polls and sends SMS+email alerts. Re-auth: `ssh -t devserver 'cd /opt/tennis && venv/bin/python cloud_icloud_watcher.py --auth'` (needs `-t` for TTY or 2FA prompt gets EOFError).
- **Quote escaping in gallery JS**: Never write `onclick="foo('bar')"` from a Python f-string. Use `data-action` attributes + event delegation. `update_r2_index.py` runs `node --check` before R2 upload.
- **Processed file mtime ≠ recording date**: For gallery dates, always prefer iCloud asset `created`, then raw MOV ffprobe, then detection JSON `created`. Preprocessed mtime is the day of processing, not recording.
- **Running process doesn't pick up synced files.** Kill + restart after `scp`.

## Workflow rules — multi-stream development

This repo runs multiple parallel work streams via `git worktree`. Three coordination files live at the repo root:

- **`BACKLOG.md`** — captured bugs, deferred ideas, UX papercuts (one-liners)
- **`FEATURES.md`** — what's actively in flight: branch, worktree, files touched
- **`DESIGN.md`** — vision, product principles, cross-platform parity rules, anti-patterns

**Always-on rules:**

1. **Append-on-capture.** When the user mentions a bug, deferred idea, or "we should also..." — APPEND it to `BACKLOG.md` immediately, in the right section. Acknowledge in one sentence and continue current work.
2. **Read on context switch.** When the user shifts topics ("let's work on X", "now do Y"), READ `FEATURES.md` and the relevant section of `BACKLOG.md` before diving in.
3. **Scan before suggesting.** Before proposing new work, check `BACKLOG.md` to avoid duplicating an existing entry.
4. **Respect file ownership.** `FEATURES.md` lists which files belong to which active branch. If a file is owned by another worktree's branch, do not edit it on `main` without coordinating first.
5. **Honor DESIGN.md.** When making cross-cutting decisions, check `DESIGN.md` principles. If something violates them, flag it before doing it.

**`/sync` command.** When the user types `/sync`, re-read `CLAUDE.md`, `FEATURES.md`, `BACKLOG.md`, `DESIGN.md` and report:
- What's currently in flight (from FEATURES.md)
- Top 5 backlog items by section
- Any apparent drift from DESIGN.md principles

## Component preview endpoints (self-verification)

When the user describes a UI issue, **don't ask for a screenshot**. Verify it yourself:

1. **Component preview pages** at http://localhost:8088/components/ — single-purpose HTML pages that render ONE component reading live R2 data:
   - `filmstrip.html?vid=ID&shot=N` — one shot's filmstrip
   - `sequences.html?vid=ID` — all filmstrips for a video
   - `coach.html?vid=ID` — coaching card content
   - `card.html?vid=ID` — single video card (mobile + desktop iframes)
   - `align.html?vid=ID&shot=N&pro=NAME` — alignment video player

2. **Screenshot helper**: `~/bin/screenshot <url> -o /tmp/x.png --wait 2` runs headless Chrome and produces a PNG you can `Read`. Use this for visual verification.

3. **Workflow when user reports a UI issue**:
   - WebFetch the component preview page to inspect HTML/data
   - Run `~/bin/screenshot` to capture the visual state
   - `Read` the resulting PNG to see what's rendered
   - Then propose the fix — don't ask the user to screenshot it for you

The component pages read from PRODUCTION R2 (read-only), so they're always up to date with whatever's deployed. Use `update_r2_index.py --preview` for testing UI CHANGES locally before deploy.

**Feedback-loop reminders:**
- `FEEDBACK.md` at repo root is the cheat sheet for all testing/preview commands.
- The user has a `/howto` slash command that prints it on demand.
- When the user asks "how do I test/verify/preview X" or seems uncertain about the testing flow, point them to `/howto` (or print FEEDBACK.md yourself).
- When you complete a UI-affecting task and the user might want to verify it, mention the relevant component preview URL — don't make them go hunt for it.

## Related memory files (`~/.claude/projects/-Users-andrewhome/memory/`)

The `MEMORY.md` index there is my personal auto-memory (user preferences, past incidents). Project-technical content belongs in THIS file (repo-committed). If adding new learnings, prefer this file unless it's a user-preference pattern.
