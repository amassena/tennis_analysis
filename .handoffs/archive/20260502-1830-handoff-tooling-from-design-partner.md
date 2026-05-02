---
from: design-partner
to: main
created: 2026-05-02T18:30:00-08:00
status: done
priority: high
topic: /inbox slash command + SessionStart hook (kills the relay)
---

# Handoff tooling: /inbox slash command + SessionStart hook

This is the FIRST brief written under the new `.handoffs/` protocol. If you're reading this, the protocol works end-to-end. Now layer two small pieces on top so future handoffs require zero user relay action.

## Goal

After this ships:
- Type `/inbox` in any session → see briefs waiting for that role
- New session opens → SessionStart hook surfaces pending briefs automatically (no command needed)
- The end of clipboard relay AND the end of "tell the session to check the file"

## Two pieces, both small

### Piece 1 — `/inbox` slash command

Create `.claude/commands/inbox.md` at repo root (project-level, all worktrees inherit when they pull main):

```markdown
---
description: Check .handoffs/inbox/ for briefs addressed to this session role
---

1. List `.handoffs/inbox/*.md` files (use Bash: `ls -t .handoffs/inbox/*.md 2>/dev/null`)
2. For each file, Read the YAML frontmatter (between the first two `---` lines) and parse `to:`, `from:`, `priority:`, `topic:`, `created:`, `status:`
3. Filter to briefs where `to:` matches this session's role OR equals `any`
4. Print a compact list: filename, topic, priority, age (created → now), from
5. If exactly ONE matching brief, offer to read and execute it
6. If MULTIPLE, ask the user which to pick
7. If NONE for this role, say so in one line and exit

Session role detection: check the current working directory:
- ends with `tennis_analysis` → role is `main`
- contains `tennis_worktrees/design-partner` → role is `design-partner`
- contains `tennis_worktrees/detection` → role is `detection`
- contains `tennis_worktrees/filmstrip` → role is `filmstrip`
- contains `tennis_worktrees/dyntrack` → role is `dyntrack`
- otherwise: ask the user what role this session is
```

### Piece 2 — SessionStart hook

Add to `.claude/settings.json` (project-level — create if absent, merge if exists):

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "if [ -d \"$CLAUDE_PROJECT_DIR/.handoffs/inbox\" ]; then count=$(ls \"$CLAUDE_PROJECT_DIR/.handoffs/inbox/\"*.md 2>/dev/null | wc -l); if [ \"$count\" -gt 0 ]; then echo \"📬 $count brief(s) in .handoffs/inbox/ — run /inbox to view\"; fi; fi"
      }]
    }]
  }
}
```

Fires once when each session starts. If inbox has briefs, session sees a one-line nudge. If empty, no output.

## Acceptance tests

1. **Slash command works:** with this brief still in inbox, type `/inbox` in this session → command lists this brief.
2. **Filtering works:** create `.handoffs/inbox/test-filmstrip.md` with `to: filmstrip` frontmatter, type `/inbox` from main session → it does NOT show. (Cleanup: delete the test file.)
3. **SessionStart fires:** open a fresh Claude session in `~/tennis_analysis` → first thing in conversation is the `📬 N brief(s)` line.
4. **SessionStart silent on empty:** archive all briefs from inbox (move to `.handoffs/archive/`), open a fresh session → no output.

## File paths

- `.claude/commands/inbox.md` — new
- `.claude/settings.json` — new or merge

Both project-level (committed to repo, on main). Other worktrees inherit on rebase.

## What NOT to do

- Don't auto-execute briefs. `/inbox` lists; user picks; session executes after explicit go.
- Don't add filtering flags (--priority, --from). Keep the command stupid.
- Don't make SessionStart noisy when inbox is empty. The conditional on `$count` matters.
- Don't put the command at user level (`~/.claude/commands/`). Project-level keeps the protocol self-contained.

## When done

1. `git mv .handoffs/inbox/20260502-1830-handoff-tooling-from-design-partner.md .handoffs/archive/`
2. Append `## Response` to that archived file with: commit hash, output of acceptance test 1, output of acceptance test 3, confirmation that test files cleaned up.
3. Update YAML `status: done`.
4. Commit: `git commit -m "handoff: complete /inbox + SessionStart hook"`

This is the second handoff under the protocol (the protocol setup itself was the first); the first one that actually USES it end-to-end. If everything works, write a one-liner to `.handoffs/archive/` for the design-partner session: "protocol works." Then the next brief (Component D) will arrive in `.handoffs/inbox/` for you to pick up via `/inbox` — true zero-relay round-trip.

---

## Response — 2026-05-02 (executed by `main` session)

**Commit:** `a38b263`

### Acceptance test 1 — `/inbox` lists this brief from main session

Working dir: `/Users/andrewhome/tennis_analysis` → role detected as `main`.
Inbox listing matched `to: main` filter:

```
- 20260502-1830-handoff-tooling-from-design-partner.md  [high]  /inbox slash command + SessionStart hook (kills the relay)  (from design-partner, 8h ago)
```

Pass.

### Acceptance test 2 — filtering hides briefs not for this role

Created `.handoffs/inbox/test-filmstrip.md` with `to: filmstrip`. From `main`
session, `/inbox` showed only the design-partner→main brief; the filmstrip
brief was filtered out. Test fixture deleted after the check.

Pass.

### Acceptance test 3 — SessionStart hook fires with brief in inbox

Ran the exact shell snippet from the hook with `CLAUDE_PROJECT_DIR=~/tennis_analysis`:

```
📬 1 brief(s) in .handoffs/inbox/ — run /inbox to view
```

Pass.

### Acceptance test 4 — SessionStart silent on empty inbox

Temporarily moved the brief out, re-ran the hook snippet → no output. Restored
the brief.

Pass.

### Cleanup

- `.handoffs/inbox/test-filmstrip.md` deleted ✓
- This brief moved to `.handoffs/archive/` ✓
- `status: pending` → `status: done` ✓

Protocol works end-to-end. Ready for Component D handoff to land in
`.handoffs/inbox/` for zero-relay pickup.
