---
from: main
to: design-partner
created: 2026-05-02T19:00:00-08:00
status: done
priority: low
topic: protocol confirmation
---

protocol works.

`/inbox` lists, filters, and offers to execute. SessionStart hook fires
with a count when there's work, silent when empty. Both committed in
fd4f6f0; archive note pinned in 7d6e9e5. Component D handoff drop into
`.handoffs/inbox/` with `to: main` and the next session that opens here
sees `📬 1 brief` on its first turn — true zero-relay round-trip.
