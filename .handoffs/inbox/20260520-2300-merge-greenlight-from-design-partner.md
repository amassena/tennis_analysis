---
from: design-partner
to: main
created: 2026-05-20T23:00:00-08:00
status: pending
priority: high
topic: Merge green-light — pro-footage + swing-organization. No duplication found. Follow-ups noted.
in-reply-to: 20260520-2200-kinetic-chain-replacement-done-from-main.md
---

# Audit summary

After reading the pro-footage report-in brief (which lives on the
feature branch at `~/tennis_worktrees/pro-footage/.handoffs/inbox/`,
not yet on main) and inspecting `scripts/compare_filmstrip.py` vs
`scripts/swing_composite.py`:

**No actual code duplication.** `compare_filmstrip.py:30` imports
`generate_composite` from `swing_composite`. The two are complementary
features — `swing_composite` is the single-shot renderer; `compare_filmstrip`
calls it twice (user + pro) and stacks with `cv2.vconcat`. This is
exactly what the filmstrip stream sketched in their report-in
("~50 lines wrapping `generate_composite` twice").

Initial duplication concern: dismissed. Pro-footage did the right thing.

# Decisions on the three swing-organization questions

| Q | Decision | Reason |
|---|---|---|
| (a) Merge swing-organization to main now | **YES** | Validated on real data (FH +13.2cm forward = textbook good form; BH negative tail = the user's known late-contact tendency). Single revert restores old code. |
| (b) Cherry-pick `audit_world_landmarks.py` + `render_audit_frames.py` from 3d-lifting before closing it | **YES** | Audit infra is metric-agnostic and useful for any future biomech work. Costs nothing to bring forward. |
| (c) Re-process existing R2 coaching outputs with new prompt | **NO** | New shots get new metric. Historical artifacts can stay. Bulk reprocess = real API + GPU cost for unclear user benefit. If user notices specific bad historical coaching, address case-by-case. |

# Pro-footage: merge green-light

Branch `feature/comparison/pro-footage` is **approved for merge to
main**. 8 commits, ready to go.

What this lands:
- 19 newly-populated pros (4 → 23 total in library; Medvedev excluded)
- 422 curated clips on R2 (~243 MB)
- `scripts/compare_filmstrip.py` — side-by-side user-vs-pro filmstrip
  using `swing_composite.generate_composite` (no duplication)
- `scripts/pro_comparison.py` updates — angle-aware matching, hard
  backhand-style filter, Murray-default auto-pick
- Several new ancillary scripts: `fetch_pro_highlights.py`,
  `process_pro_raw.py`, `curate_pro_clips.py`,
  `extract_pro_clip_poses.py`, `upload_pro_clips_to_r2.py`,
  `apply_reel_angles.py`
- `pros/index.json` schema/contents updates

What this does NOT land (deliberately out of scope):
- Original-4 pros (alcaraz/djokovic/federer/nadal) at filmstrip parity
- camera_angle field correctness on user GT corpus
- Additional side-angle harvest for any specific pro

# Sequencing

Order matters slightly. Recommend:

1. **Merge `feature/coaching/swing-organization` first.** Small scope,
   bounded change to two files. Smoke-tested. Lower risk.
2. **Cherry-pick audit infra** from 3d-lifting to main (no rush;
   anytime in the next week).
3. **Close `feature/detection/3d-lifting`** — mark superseded in
   FEATURES.md, remove worktree once main has the audit cherry-picks.
4. **Merge `feature/comparison/pro-footage`.** Larger scope, but no
   blockers identified.
5. **Push pro-footage's report-in brief to main's inbox** so future
   `/inbox` sessions can see what landed (currently invisible because
   it lives only on the feature branch).

# Follow-up items for BACKLOG.md

Three items the merge brings into focus but doesn't address. Add to
BACKLOG.md after merges land:

```markdown
- **Original-4 pros at filmstrip-comparison parity** — alcaraz, djokovic, federer, nadal are R2-only with no local clips + no per-clip pose, so they work for video comparison but not for side-by-side filmstrip. Need: R2-download clips locally, extract pose on tmassena (~5 min GPU per pro). After this lands, all 23 pros work end-to-end with `compare_filmstrip.py`. Surfaced 2026-05-20 by pro-footage acquisition report.
- **Audit user-side `camera_angle` field on GT corpus** — pro-footage flagged that IMG_0999 is tagged `camera_angle: "side"` but visually looks "behind". User stated 90% of footage is behind. Worth a one-pass audit + re-classification of the GT corpus's camera_angle values, OR change the default to `--user-angle behind` in compare_filmstrip and pro_comparison. Surfaced 2026-05-20.
- **Targeted side-angle Murray harvest** — Murray is the auto-pick default, but his current clips are all behind-angle (one Love Tennis FH reel, one Essential Tennis SV reel). For side-angle user shots, comparison falls through to Sinner. If Murray-default for FH/SV is important, need targeted yt-dlp search for slow-motion side-angle Murray content. Surfaced 2026-05-20.
```

# What I'm doing in this brief

Just the green-light + sequencing. I will NOT:
- Merge for you (main does merges)
- Touch any code (out of design-partner scope)
- Reprocess R2 coaching outputs (decision = no)
- Speak for the other 4 streams beyond what their report-ins covered

# Coordination snapshot (for record)

All sessions have report-ins filed:
- main: 4 briefs (dashboard honest, inbox status, windows outage, kinetic-chain pivot + done)
- detection: 1 brief (event-level eval + regression head)
- filmstrip: 1 brief (contact-precision shipped)
- dyntrack: 1 brief (overhaul done, 4K/120 hypothesis)
- pro-footage: 1 brief (acquisition Phases 2+3+matcher) — **but only on branch, not on main inbox yet**
- swing-organization: subsumed under main's kinetic-chain pivot + done briefs

# Net actions for whoever picks this up

1. Merge swing-organization → main
2. Merge pro-footage → main
3. Cherry-pick audit infra forward; close 3d-lifting branch
4. Push pro-footage's report-in brief to main's `.handoffs/inbox/` so `/inbox` shows the full picture
5. Add 3 BACKLOG entries above
