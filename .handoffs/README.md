# Cross-session handoff protocol

This directory is the message bus between Claude Code sessions working on
this repo. Replaces clipboard-based relay.

## Layout

- `inbox/` — pending briefs waiting to be picked up
- `archive/` — completed briefs with embedded responses (paper trail)

## File naming

`YYYYMMDD-HHMM-<short-topic>-from-<sender>.md`

Examples:
- `20260502-1430-path-a-merge-from-design-partner.md`
- `20260503-0915-nadal-handedness-from-design-partner.md`

Use lowercase, hyphens between words, no spaces.

## File format

Every brief starts with YAML frontmatter:

```yaml
---
from: design-partner          # session role that wrote this
to: main                      # specific role, OR "any" for first-available
created: 2026-05-02T14:30:00-08:00
status: pending               # pending | in-progress | done | blocked
priority: high                # low | medium | high | urgent
topic: Path A - land verification infrastructure to main
---
```

Body is the brief itself in Markdown. When the receiving session completes
the work, it appends a `## Response` section at the bottom and moves the
file from `inbox/` to `archive/` (no rename — same filename in archive).

If blocked, append a `## Blocked` section with the reason and leave in
`inbox/` with `status: blocked`.

## Protocol — for receiving sessions

At the start of every session AND after completing each task:

```bash
ls .handoffs/inbox/
```

For each file, check the YAML `to:` field:
- If `to: <your-role>` or `to: any` → it's for you
- Otherwise → ignore

When you start work on a brief: edit `status: pending` → `status: in-progress`.

When done:
1. Append `## Response` at the end of the file with your report
2. Update YAML `status` to `done` (or `blocked`)
3. `git mv .handoffs/inbox/<file> .handoffs/archive/<file>`
4. Commit: `git commit -m "handoff: complete <topic>"`

## Protocol — for sending sessions

Write your brief directly to `.handoffs/inbox/<filename>.md` with the
YAML frontmatter at the top.

For design-partner session (read-only on project code): a narrow
write exception applies to `.handoffs/inbox/*.md` only. This is
coordination metadata, not project mutation.

To check for responses to your briefs:

```bash
ls .handoffs/archive/ | grep <your-role>
```

Or just glance at recently-modified files:

```bash
ls -lt .handoffs/archive/ | head
```

## Session roles (current)

- `design-partner` — read-only, runs in `~/tennis_worktrees/design-partner` (detached HEAD). Drafts plans, runs forensics, writes briefs. Does NOT execute.
- `main` — runs in `~/tennis_analysis` on `main` branch. Executes
  infrastructure, incident response, gallery work.
- `detection` — runs in `~/tennis_worktrees/detection` on
  `feature/detection/improve-shot-classification`. Owns shot detection
  and classification work.
- `filmstrip` — runs in `~/tennis_worktrees/filmstrip`. Owns
  `scripts/swing_composite.py`.
- `dyntrack` — runs in `~/tennis_worktrees/dyntrack`. Owns
  `scripts/dynamic_track.py`.

New roles? Add to this list when you add a worktree.

## Why this exists

Earlier flow: design-partner generates a brief → user pbcopies → switches
tabs → pastes into executing session. Failure modes: clipboard gets
overwritten by a terminal selection, user forgets which session is which,
no paper trail of what was handed off.

New flow: design-partner writes a file → user tells executing session
"check inbox" → executing session reads, works, writes back. No clipboard.
History preserved in git.
