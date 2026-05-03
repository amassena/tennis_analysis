---
from: design-partner
to: main
created: 2026-05-02T19:25:00-08:00
status: done
priority: medium
topic: /archive command + design-partner SessionStart for completed responses
---

# Handoff tooling follow-up: /archive + design-partner inbound nudge

Tiny follow-up to the `/inbox` work. After this, the design-partner session can self-check for completed responses without the user typing "check archive."

This is small (~20 min). Bundle with whatever else you're doing or do as a one-off.

## Two pieces

### Piece 1 — `/archive` slash command

Create `.claude/commands/archive.md` (project-level):

```markdown
---
description: List recent completions in .handoffs/archive/, filtered by session role
---

1. List `.handoffs/archive/*.md` sorted by mtime desc (Bash: `ls -lt .handoffs/archive/*.md 2>/dev/null | head -20`)
2. For each, Read the YAML frontmatter and parse `from`, `to`, `topic`, `status`, `created`
3. Filter to entries where `from:` matches THIS session's role (responses to briefs I sent)
4. Print compact list: filename, topic, status, who handled it (the `to:` field — that session is who archived it), age (created → now), age since archived (file mtime → now)
5. If exactly ONE matching, offer to read its `## Response` section
6. If MULTIPLE, ask which to pick or default to most recent
7. If NONE, say "no archived responses to your briefs"

Session role detection: same as `/inbox` — check working directory.
```

### Piece 2 — Extend SessionStart hook for design-partner

Currently the SessionStart hook surfaces inbox count. For design-partner, inbox is usually empty (most briefs go `to: main` or `to: detection`). The relevant nudge for design-partner is "anything new in archive responding to MY briefs."

Update `.claude/settings.json` SessionStart hook to also check archive:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "if [ -d \"$CLAUDE_PROJECT_DIR/.handoffs/inbox\" ]; then in_count=$(ls \"$CLAUDE_PROJECT_DIR/.handoffs/inbox/\"*.md 2>/dev/null | wc -l | tr -d ' '); if [ \"$in_count\" -gt 0 ]; then echo \"📬 $in_count brief(s) in .handoffs/inbox/ — run /inbox\"; fi; fi; if [ -d \"$CLAUDE_PROJECT_DIR/.handoffs/archive\" ]; then recent=$(find \"$CLAUDE_PROJECT_DIR/.handoffs/archive/\" -name \"*.md\" -mtime -1 2>/dev/null | wc -l | tr -d ' '); if [ \"$recent\" -gt 0 ]; then echo \"📨 $recent recent archive completion(s) (last 24h) — run /archive to view\"; fi; fi"
      }]
    }]
  }
}
```

The archive check is generic (24h window, not role-specific) because the hook runs as shell, not as the session — it can't easily know which session role it's in. The `/archive` command does the role filtering. The SessionStart hook just nudges that something happened recently.

## Acceptance tests

1. **`/archive` filters correctly:** with the existing archive containing entries where `from: design-partner` (the handoff_tooling brief, after Component D completes etc.), run `/archive` from a `~/tennis_worktrees/design-partner` working directory → lists those. Run from `~/tennis_analysis` (role=main) → lists entries where `from: main`.
2. **SessionStart shows recent archive:** open a fresh session in `~/tennis_analysis` after a recent archive entry exists → first thing in conversation is "📨 N recent archive completion(s)" line.
3. **SessionStart silent when nothing recent:** if all archive entries are older than 24h, the archive line doesn't appear.
4. **Both nudges can fire:** if inbox has briefs AND archive has recent completions, both lines show.

## File paths

- `.claude/commands/archive.md` — new
- `.claude/settings.json` — edit existing SessionStart hook (don't add a second one, modify the command)

## What NOT to do

- Don't make the SessionStart hook role-aware. Keep it stupid (just file counts). The slash command does role filtering.
- Don't add state files to track "what the user has already seen." Stateless 24h window is fine.
- Don't extend `/archive` to also show responses from OTHER roles unless asked. Default to "responses to MY briefs" because that's what the user usually cares about.

## When done

Per protocol:
1. `git mv .handoffs/inbox/20260502-1925-archive-tooling-from-design-partner.md .handoffs/archive/`
2. Append `## Response` with: commit hash, output of acceptance test 1 (filtering works), output of acceptance test 2 (SessionStart shows recent), confirmation `/archive` lives at `.claude/commands/archive.md`.
3. Update YAML `status: done`.
4. Commit: `git commit -m "handoff: complete /archive + design-partner SessionStart nudge"`

After this lands, the design-partner ↔ main loop is fully zero-relay. User just sits at sessions, types `/inbox` or `/archive` as needed, both auto-surface on session start.

---

## Response — 2026-05-02 (executed by `main` session)

**Commit:** `0b93d79`

### Acceptance test 1 — `/archive` filtering by from

Frontmatter parse + filter on `from:` matches role:

- From `main` cwd → 1 entry: `20260502-1900-protocol-works-from-main.md`
  ("protocol confirmation"; main sent → design-partner handled).
- From `design-partner` cwd → 2 entries:
  - `20260502-1915-component-d-from-design-partner.md` (Component D triage)
  - `20260502-1830-handoff-tooling-from-design-partner.md` (/inbox + SessionStart)

The `from: main` view shows what main sent and the other role responded to;
the `from: design-partner` view shows what design-partner sent and main
responded to. Asymmetric and correct.

Pass.

### Acceptance test 2 — SessionStart shows recent archive

With current state (1 brief in inbox, 4 archive entries, all <24h):

```
📬 1 brief(s) in .handoffs/inbox/ — run /inbox
📨 3 recent archive completion(s) (last 24h) — run /archive to view
```

(3 not 4 because this brief itself was just moved to archive and would
make it 4 on the next run.)

Pass.

### Acceptance test 3 — silent when nothing recent

Temporarily set all archive mtimes to 25h ago and emptied inbox: hook
produced no output. Restored.

Pass.

### Acceptance test 4 — both nudges fire together

Verified by Test 2 above (both lines printed).

Pass.

### Files

- `.claude/commands/archive.md` — new
- `.claude/settings.json` — SessionStart hook command updated in place
  (single hook entry, not duplicated)

The design-partner ↔ main loop is now zero-relay. Either side opens a session
and immediately sees what's pending (inbox briefs) and what was completed
recently (archive nudge). Type `/inbox` or `/archive` for filtered detail.
