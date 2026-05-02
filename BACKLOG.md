# Backlog

Drop-anywhere capture for bugs, ideas, and UX papercuts. Add a one-liner the moment a thought hits — don't lose it.

## Bugs

- [ ] **Filmstrip misses ball impact** — the orange contact frame often shows the player just before or just after, not at racket-ball contact. Wrist-speed peak refinement is close but not always right.
- [ ] **Filmstrip too short** — current window (0.5s before, 0.3s after) doesn't capture the full backswing-to-finish sequence. Should extend.
- [ ] **Coach modal slo-mo vs full-speed inconsistent** — clicking a coaching example sometimes opens the slow-mo variant, sometimes the full-speed timeline. Should be deterministic (slo-mo for instructional context).
- [ ] **Missed second serve detection** — the 1D-CNN catches the first serve but misses the second serve in many sessions. Likely confidence threshold or nms gap issue.
- [ ] **Shot misclassification (FH↔serve, FH↔BH)** — confidence often >0.95 even when wrong. Calibration issue, not just threshold.
- [ ] **Camera angle misidentification** — comparison alignment picks wrong-angle pro clips because user's angle isn't reliably detected.
- [ ] **Coordinator routing imbalance** — Andrew-PC has done 4× more jobs than tmassena despite CLAUDE.md saying tmassena is PRIMARY. Decide doc-update vs routing-fix; track here. Discovered 2026-05-02 during model-divergence triage.

## Research loop / data engine

- [ ] **Verification system build** — sidecars, eval_holdout.py, compare_models.py gate, triage_reprocess.py. Spec in design-partner session. Components reuse most of validate_pipeline.py from feature/detection/improve-shot-classification branch. ~1 day total.
- [ ] **Triage of 36 broken-model jobs** from Andrew-PC since 2026-04-03. Job IDs in design-partner session memory. Run after triage_reprocess.py ships. Threshold: total_disagreement >= 3 → REPROCESS.
- [ ] **Backfill camera_angle on 44 GT files** via shot_review.py. Only ~1 of 6 sampled has it populated. Skip dominant_hand backfill (all righties) but enforce required-at-save going forward.
- [ ] **MV-0 backups to R2 + Drive** (andrew@massena.com, 42TB available). GT is currently triple-redundant on Mac+andrew-pc+tmassena so no SPOF, but no off-site copy exists. Drive primary for raw/preprocessed, R2 primary for GT/models. Rclone copy (NOT sync). Restore-test required.
- [ ] **Wikidata-backed pro library** when library grows past ~20 players. P741 SPARQL lookup, cache to pros/wikidata_cache.json. See DESIGN principle 7.

## UX papercuts

- [ ] Sequences modal labels get cut off when image scrolls horizontally on mobile
- [ ] Cards too wide on phone, content overflows
- [ ] Coach summary text gets clipped on small screens
- [ ] No way to jump back to last-viewed video on gallery reload
- [ ] Filter & Sort hidden behind a toggle — not discoverable

## Ideas (deferred)

- [ ] **Dynamic player tracking** — keep player centered in video frame analysis (not just filmstrips)
- [ ] **Cross-platform parity audit** — what works on web that doesn't work on iOS, and vice versa
- [ ] **Per-shot inline coaching** — short tip per shot in sequences modal, not just session-level summary
- [ ] **Session-over-session progression** — graph of knee bend, trunk rotation, etc. across recent sessions
- [ ] **Friend feed / social** — see friends' sessions, comments, reactions
- [ ] **Coach inbox** — async voice memos from coach on specific shots
- [ ] **Drill prescription grounded in actual deficits** — "you average 16° trunk on FH, here's a 5-min drill"
- [ ] **History view per video** — re-process old videos with newer models, see how grades changed

## Cross-platform gaps

- [ ] iOS app has no Pro Comparison view (web has it via gallery)
- [ ] iOS app session review doesn't show Coach summary (only filmstrips)
- [ ] Web has search; iOS doesn't (since iOS is just a WebView wrapper, this is fine — but the in-app camera screen has no way to find prior session)
