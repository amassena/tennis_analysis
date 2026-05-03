---
description: List recent completions in .handoffs/archive/, filtered to responses to YOUR briefs
---

Check the handoffs archive for completions of briefs THIS session sent.

## Steps

1. **Detect this session's role from the current working directory** (use `pwd`):
   - ends with `tennis_analysis` → role is `main`
   - contains `tennis_worktrees/design-partner` → role is `design-partner`
   - contains `tennis_worktrees/detection` → role is `detection`
   - contains `tennis_worktrees/filmstrip` → role is `filmstrip`
   - contains `tennis_worktrees/dyntrack` → role is `dyntrack`
   - otherwise: ask the user what role this session is

2. **List archive entries** sorted by mtime desc (most recently completed first):
   `ls -lt .handoffs/archive/*.md 2>/dev/null | head -20`

3. **Parse each entry's YAML frontmatter** (between the first two `---` lines):
   read each file, extract `from`, `to`, `topic`, `status`, `created`.

4. **Filter to entries where `from:` matches THIS role** (i.e. the briefs YOU
   sent that someone else has now responded to). The `to:` field tells you
   which session handled it.

5. **If zero matches**: print one line:
   `📭 no archived responses to your briefs (role: <role>; N total in archive)` — stop.

6. **Otherwise print a compact list**, newest first. Format per line:
   `- <filename>  [<status>]  <topic>  (handled by <to>, <created-age> brief / <mtime-age> archived)`

7. **If exactly ONE matching**, offer to read its `## Response` section
   (everything after the first `## Response` heading in the file).

8. **If MULTIPLE**, ask which to pick by filename or topic keyword; default
   to the most recently mtime'd one.

## Behavior rules

- **Default scope is "responses to MY briefs."** If user explicitly asks for
  responses by other roles, broaden — but don't do it by default.
- **Don't filter by anything other than `from:`.** No flags.
- **Be quiet when empty.** One line max.
- **Read-only.** This command never modifies the archive.
