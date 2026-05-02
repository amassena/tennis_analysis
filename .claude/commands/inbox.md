---
description: Check .handoffs/inbox/ for briefs addressed to this session role
---

Check the handoffs inbox for briefs addressed to this session.

## Steps

1. **Detect this session's role from the current working directory** (use `pwd`):
   - ends with `tennis_analysis` → role is `main`
   - contains `tennis_worktrees/design-partner` → role is `design-partner`
   - contains `tennis_worktrees/detection` → role is `detection`
   - contains `tennis_worktrees/filmstrip` → role is `filmstrip`
   - contains `tennis_worktrees/dyntrack` → role is `dyntrack`
   - otherwise: ask the user what role this session is

2. **List briefs**: `ls -t .handoffs/inbox/*.md 2>/dev/null` (newest first).
   If none exist, print "📭 inbox empty" in one line and stop.

3. **Parse each brief's YAML frontmatter** (between the first two `---` lines).
   Read each file; extract: `to`, `from`, `priority`, `topic`, `created`, `status`.

4. **Filter to briefs addressed to this role**: keep briefs where `to:` matches
   this session's role exactly, or `to:` equals `any`.

5. **If zero matches for this role**: print one line like
   "📭 no briefs for `<role>` (N total in inbox for other roles)" and stop.

6. **Otherwise print a compact list**, newest first. Format per line:
   `- <filename>  [<priority>]  <topic>  (from <from>, <age>)`
   where `<age>` is a human "5m ago", "2h ago", "yesterday" derived from `created`.

7. **If exactly ONE matching brief**, offer to read and execute it:
   ask "Execute now?" — if yes, Read the full file and proceed.

8. **If MULTIPLE**, ask the user which to pick by filename or topic keyword.

## Behavior rules

- **List, don't auto-execute.** The user always picks.
- **Don't filter by anything other than `to:`.** No `--priority`, no `--from`.
- **Be quiet on empty.** No "no briefs found" preamble; one line max.
- The slash command is read-only with respect to inbox state. Archiving / status
  updates are the executing session's responsibility (per protocol).
