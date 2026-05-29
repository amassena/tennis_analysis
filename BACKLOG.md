# Backlog

Drop-anywhere capture for bugs, ideas, and UX papercuts. Add a one-liner the moment a thought hits — don't lose it.

## Bugs

- [ ] **Filmstrip misses ball impact** — the orange contact frame often shows the player just before or just after, not at racket-ball contact. Wrist-speed peak refinement is close but not always right.
- [ ] **Filmstrip too short** — current window (0.5s before, 0.3s after) doesn't capture the full backswing-to-finish sequence. Should extend.
- [ ] **Coach modal slo-mo vs full-speed inconsistent** — clicking a coaching example sometimes opens the slow-mo variant, sometimes the full-speed timeline. Should be deterministic (slo-mo for instructional context).
- [ ] **Missed second serve detection** — diagnostic: only 1/144 GT serves missed at production threshold. Real production cases likely the same model behavior (peak p_shot 0.85-0.90 zone). Mitigation landed: `--per-class-thresh "serve=0.7,forehand=0.8,backhand=0.8"` lifts serve recall; opt-in for now (production default unchanged in worker.py).
- [ ] **Shot misclassification (FH↔serve, FH↔BH)** — diagnostic: 99.4% classification accuracy on GT, only 9 misclasses (5 conf>0.9). Real issue is FH↔BH ambiguity in split-probability cases (margin <0.15 = 0.6% of production detections). Mitigation landed: `--demote-below 0.55` re-tags low-class-conf shots as `unknown_shot`. Hard-negative mining of remaining 35 high-conf FPs is the next lever — see `scripts/mine_hard_negatives.py`.
- [ ] **Camera angle misidentification** — comparison alignment picks wrong-angle pro clips because user's angle isn't reliably detected.
- [ ] **Sequencing (filmstrips) not generated for iphone_* uploads** — root cause (2026-05-28): `swing_composite.py` is NOT in the GPU worker pipeline; the 56 videos that have sequences got them from manual runs on the old GT corpus. Also `swing_composite` uploads to flat `highlights/{vid}/sequences/` not the per-user `highlights/{hash}/{vid}/sequences/`. Pose data DOES exist (`poses_full_videos/{vid}.json` on tmassena) so backfill is feasible without reprocess. Fix: (1) add `--user-hash` to swing_composite, (2) wire it into worker.py after detection, (3) backfill existing iphone_* videos on GPU.
- [ ] **Cut-up / playlist segment edge cases still wrong on some videos** (2026-05-28) — the in-player segment auto-skip occasionally lands wrong on certain videos even after the keyframe-snap fix (build 23). Need a repro list: which video IDs, which chip, what the timestamp does. Likely remaining causes: sparse keyframes (long GOP) making forward-tolerance seeks overshoot, or shots clustered tighter than the merge tolerance.
- [x] ~~**Coordinator routing imbalance** — Andrew-PC has done 4× more jobs than tmassena despite CLAUDE.md saying tmassena is PRIMARY.~~ **Resolved 2026-05-02:** root cause was tmassena being repeatedly powered off (not a routing bug). Doc is correct (tmassena PRIMARY); the imbalance was just availability. Going forward tmassena stays on.

## Research loop / data engine

- [ ] **Verification system build** — sidecars, eval_holdout.py, compare_models.py gate, triage_reprocess.py. Spec in design-partner session. Components reuse most of validate_pipeline.py from feature/detection/improve-shot-classification branch. ~1 day total.
- [ ] **Triage of 36 broken-model jobs** from Andrew-PC since 2026-04-03. Job IDs in design-partner session memory. Run after triage_reprocess.py ships. Threshold: total_disagreement >= 3 → REPROCESS.
- [ ] **Backfill camera_angle on 44 GT files** via shot_review.py. Only ~1 of 6 sampled has it populated. Skip dominant_hand backfill (all righties) but enforce required-at-save going forward.
- [ ] **MV-0 backups to R2 + Drive** (andrew@massena.com, 42TB available). GT is currently triple-redundant on Mac+andrew-pc+tmassena so no SPOF, but no off-site copy exists. Drive primary for raw/preprocessed, R2 primary for GT/models. Rclone copy (NOT sync). Restore-test required.
- [ ] **Wikidata-backed pro library** when library grows past ~20 players. P741 SPARQL lookup, cache to pros/wikidata_cache.json. See DESIGN principle 7.
- [ ] **Active-learning loop on corrections (P3)** — once ~50 user corrections accumulate via the gallery shot-correction UI, surface low-confidence model predictions in the gallery for review (sort by max disagreement between model conf and audio-confirmed contacts; visually flag chips below threshold). Karpathy data-engine pattern. Surfaced 2026-05-09 by design-partner.
- [ ] **Audio-as-confirmation in production inference (P2)** — production inference (`sequence_cnn`) is pose-only; the 43%-no-prediction gap on BH→FH audit suggests reintroducing audio as a confirmation pass (boost confidence on pose events that coincide with audio peaks). Scope as separate brief if `improve-shot-classification` threshold tuning hits a ceiling.
- [ ] **Promote `audit_world_landmarks.py` to CI** — run on a small fixed video set whenever `biomechanical_analysis.py` changes. Long-term value as biomech regression test. Surfaced 2026-05-09 by design-partner.
- [ ] **Original-4 pros at filmstrip-comparison parity** — alcaraz, djokovic, federer, nadal are R2-only with no local clips + no per-clip pose, so they work for video comparison but not for side-by-side filmstrip. Need: R2-download clips locally, extract pose on tmassena (~5 min GPU per pro). After this lands, all 23 pros work end-to-end with `compare_filmstrip.py`. Surfaced 2026-05-20 by pro-footage acquisition report.
- [ ] **Audit user-side `camera_angle` field on GT corpus** — pro-footage flagged that IMG_0999 is tagged `camera_angle: "side"` but visually looks "behind". User stated 90% of footage is behind. Worth a one-pass audit + re-classification of the GT corpus's camera_angle values, OR change the default to `--user-angle behind` in compare_filmstrip and pro_comparison. Surfaced 2026-05-20.
- [ ] **Targeted side-angle Murray harvest** — Murray is the auto-pick default, but his current clips are all behind-angle (one Love Tennis FH reel, one Essential Tennis SV reel). For side-angle user shots, comparison falls through to Sinner. If Murray-default for FH/SV is important, need targeted yt-dlp search for slow-motion side-angle Murray content. Surfaced 2026-05-20.

## UX papercuts

- [x] ~~Sequences modal labels get cut off when image scrolls horizontally on mobile~~ — fixed 2026-05-28: filmstrip now fit-to-width on <600px so the label always sits directly under the visible image
- [ ] Cards too wide on phone, content overflows
- [x] ~~Coach summary text gets clipped on small screens~~ — fixed 2026-05-28: text replaced by pill that opens a modal; modal got mobile padding + font sizing pass
- [ ] No way to jump back to last-viewed video on gallery reload
- [x] ~~Filter & Sort hidden behind a toggle — not discoverable~~ — fixed 2026-05-28: row now expanded by default (toggle still works to collapse)

## Ideas (deferred)

- [ ] **Dynamic player tracking** — keep player centered in video frame analysis (not just filmstrips)
- [ ] **Cross-platform parity audit** — what works on web that doesn't work on iOS, and vice versa
- [ ] **Per-shot inline coaching** — short tip per shot in sequences modal, not just session-level summary
- [ ] **Session-over-session progression** — graph of knee bend, trunk rotation, etc. across recent sessions
- [ ] **Friend feed / social** — see friends' sessions, comments, reactions
- [ ] **Coach inbox** — async voice memos from coach on specific shots
- [ ] **Drill prescription grounded in actual deficits** — "you average 16° trunk on FH, here's a 5-min drill"
- [ ] **History view per video** — re-process old videos with newer models, see how grades changed

## Video architecture & encoding (captured 2026-05-28)

- [x] **Single-source architecture confirmed** — `timeline.mp4` is the one source of truth. Playlists (Rally / per-shot-type) and slow-mo are derived client-side from timeline + shots.json. Pipeline emits timeline only (Phase 2/4b). DONE.
- [ ] **Variable slow-mo speeds (1/4, 1/8, not just 1/2)** — player currently toggles 1×↔0.5× via `playbackRate`. Add more speed options. Trivial on the playbackRate side.
- [ ] **Slow-mo quality is bounded by source fps, not playbackRate** — KEY NUANCE. `playbackRate` sets *speed* independent of fps, but *smoothness* depends on the encoded fps. `timeline.mp4` is 60fps CFR, so 1/2 ≈ effective 30fps (smooth), 1/4 ≈ 15fps (choppy), 1/8 ≈ 7.5fps (slideshow). So "1/2" today is tied to the 60fps timeline, NOT the original capture. BUT `preprocess_nvenc.py` already ALSO emits a `{vid}_240fps.mp4` for 240fps originals. Decision: for sub-1/2 speeds, the player should pull frames from the 240fps version (240fps @ 1/4 = effective 60fps, buttery). Need: (a) keep/upload the 240fps version to R2 per-user, (b) player switches source when speed < 0.5. This is the right way to honor the "240fps IS the product" rule (CLAUDE.md golden rule #3) instead of losing it to the 60fps timeline.
- [ ] **4K/120fps capture vs 1080/240 — slow-mo math** — recording quality changed to 4K·120fps (was 1080·240). At 120fps source: 1/2 ≈ 60fps (smooth), 1/4 ≈ 30fps (still smooth), 1/8 ≈ 15fps. So 120fps gives *less* deep-slow headroom than 240fps but more spatial detail (4K). Confirm what `preprocess_nvenc.py` does with a 120fps 4K source (does it still make a high-fps version? at what res?). Decide the capture/encode tradeoff: 4K120 (detail) vs 1080p240 (deeper slow-mo). Tied to the product's slow-mo-analysis purpose.
- [ ] **AV1 encoder evaluation** — friend at Netflix: big companies moving to AV1 (≈30% better compression than H.264 at equal quality). RTX 4080 (tmassena) and RTX 5080 (andrew-pc) BOTH have hardware AV1 NVENC encoders, so encode cost is fine. DECODE caveat: iOS hardware AV1 decode requires A17 Pro+ / M3+ (iPhone 15 Pro and newer); Safari and older devices may software-decode or fail. Viable for a recent-device friends group (user is on iPhone 16 Pro Max = A18). Eval: encode a sample timeline in AV1 via NVENC, confirm it plays in the native AVPlayer + WebView gallery on target devices, measure size/quality delta. Park behind the slow-mo source work since both touch preprocess/export.

## Cross-platform gaps

- [ ] iOS app has no Pro Comparison view (web has it via gallery) — **in-flight comparison/filmstrip-alignment work; don't lose it (parked 2026-05-28).**
- [ ] iOS app session review doesn't show Coach summary (only filmstrips)
- [ ] Web has search; iOS doesn't (since iOS is just a WebView wrapper, this is fine — but the in-app camera screen has no way to find prior session)

## Pipeline robustness

- [ ] **R2 source-MOV caching** — every successful preprocess uploads source MOV to `source/{vid}.MOV` in R2. Asset resolution becomes local → R2 → iCloud. Eliminates iCloud-auth coupling for reprocess. ~600 GB / ~$9/mo storage cost. Surfaced 2026-05-07 by iCloud lockout incident.
- [ ] **Coordinator worker kill switch** — `worker_pool` table with `enabled bool`, `POST /api/worker/:id/disable` endpoint, worker checks its own enabled state before claiming. Use cases: thermal throttle, suspect model canary, OS patch, auth failure auto-park. Surfaced 2026-05-07 by iCloud lockout incident.
- [ ] **Atomic iCloud session writes** — pyicloud overwrites cookie jar mid-auth, leaving partial state on failure. Wrap auth in `session.candidate/ → atomic rename` pattern so failed auths never touch the active session dir. Surfaced 2026-05-07 by iCloud lockout incident.
- [ ] **Watcher: distinguish `-20209` from session-stale** — on hard account-lock errors, alert + sleep 1h instead of systemd-restart-loop. Less urgent if R2 source caching lands first. Surfaced 2026-05-07 by iCloud lockout incident.

- **iPhone upload_id linkage broken** — `iphone_upload_poller.py:110` doesn't pass `upload_id` to `VideoJob`, so R2 markers stay at `coordinator_registered` forever. Diagnosed 2026-05-19 in `.handoffs/archive/20260519-0010-0-shots-diagnosis-from-main.md`. 3-touchpoint fix described there. Park until dashboard needs per-upload tracking; not user-blocking.
- [ ] **Original-4 pros at filmstrip-comparison parity** — alcaraz, djokovic, federer, nadal are R2-only with no local clips + no per-clip pose, so they work for video comparison but not side-by-side filmstrip. Need: R2-download clips locally, extract pose on tmassena (~5 min GPU per pro). After this, all 23 pros work end-to-end with compare_filmstrip.py. Surfaced 2026-05-20 by pro-footage acquisition.
- [ ] **Targeted side-angle Murray harvest** — Murray is the auto-pick default (per feedback_preferred_comparison_pros.md), but his current clips are all behind-angle. For side-angle user shots, comparison falls through to Sinner. If Murray-default for FH/SV is important across angles, need targeted yt-dlp search for slow-motion side-angle Murray content. Surfaced 2026-05-20.
- [ ] **Pose disambiguation in two-player rally framing** — MediaPipe jumps between user and opponent across the 11 filmstrip panels in rally footage, producing crops that show the wrong half of the court at contact. Fix in `swing_composite.py` (pick player via largest bbox or consistent tracking) OR in pose-extraction pipeline (multi-person tracking with identity persistence). Probably both. Surfaced 2026-05-20 by filmstrip stream.
