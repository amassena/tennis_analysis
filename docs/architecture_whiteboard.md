# SwingLab / CourtIQ — Architecture Whiteboards

## Whiteboard 1: WHAT WE HAVE TODAY (April 2026)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER'S iPHONE 16 PRO                            │
│                                                                          │
│  ┌─────────────────────┐       ┌──────────────────────────────────┐     │
│  │   SwingLab App      │       │  Native iOS Camera (Photos app)  │     │
│  │                     │       │  ─────────────────────────────   │     │
│  │  ┌──────────────┐   │       │  • 240fps slo-mo recording       │     │
│  │  │  WebView →   │◀──┼───────┤  • Auto-uploads to iCloud Photos │     │
│  │  │  tennis.     │   │       │  ✓ Existing iOS feature           │     │
│  │  │  playfullife │   │       └──────────────────────────────────┘     │
│  │  │  .com        │   │                       │                          │
│  │  └──────────────┘   │                       │                          │
│  │                     │                       ▼                          │
│  │  ┌──────────────┐   │                  ┌─────────┐                    │
│  │  │ Live record  │   │                  │ iCloud  │                    │
│  │  │ + on-device  │   │                  │ Photos  │                    │
│  │  │ pose overlay │   │                  └────┬────┘                    │
│  │  │ (Apple       │   │                       │                          │
│  │  │  Vision)     │   │                       │                          │
│  │  │ + shot ID    │   │                       │                          │
│  │  │ (ONNX 17KB)  │   │                       │                          │
│  │  └──────────────┘   │                       │                          │
│  └─────────────────────┘                       │                          │
└─────────────────────────────────────────────────┼──────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       HETZNER VPS (5.78.96.237)                          │
│  ┌─────────────────────────┐    ┌────────────────────────────────┐      │
│  │ tennis-watcher.service  │───▶│ tennis-coordinator (FastAPI)   │      │
│  │ • polls iCloud album    │    │ • SQLite job queue             │      │
│  │ • detects new videos    │    │ • REST API for GPU workers     │      │
│  │ • cookies in R2 bus     │    └─────────────┬──────────────────┘      │
│  └─────────────────────────┘                  │                         │
└────────────────────────────────────────────────┼─────────────────────────┘
                                                 │ HTTP poll
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LOCAL GPU MACHINES (Tailscale)                        │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐       │
│  │ tmassena (RTX 4080)         │  │ andrew-pc (RTX 5080)        │       │
│  │ PRIMARY                     │  │ FALLBACK                    │       │
│  │                             │  │                             │       │
│  │  Pipeline (gpu_worker):     │  │  Same pipeline              │       │
│  │  1. Preprocess (NVENC)      │  │                             │       │
│  │  2. Thumbnail               │  │                             │       │
│  │  3. Pose extraction         │  │                             │       │
│  │     (MediaPipe heavy)       │  │                             │       │
│  │  4. Shot detection          │  │                             │       │
│  │     (1D-CNN, F1=96.5%)      │  │                             │       │
│  │  5. Biomechanical analysis  │  │                             │       │
│  │  6. Claude coaching         │  │                             │       │
│  │     (Opus 4.7 via API)      │  │                             │       │
│  │  7. Video exports           │  │                             │       │
│  │     (timeline/rally/bytype) │  │                             │       │
│  │  8. Composites (filmstrip)  │  │                             │       │
│  │  9. Pro comparison          │  │                             │       │
│  └──────────────┬──────────────┘  └─────────────────────────────┘       │
└─────────────────┼────────────────────────────────────────────────────────┘
                  │ uploads
                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLOUDFLARE EDGE                                  │
│  ┌──────────────────────────┐    ┌─────────────────────────────────┐    │
│  │ R2 Object Storage        │    │ Worker (upload-worker.js)       │    │
│  │ bucket: tennis-videos    │◀──▶│ • Asset serving + Range support │    │
│  │ • highlights/{vid}/*.mp4 │    │ • Path fallback                 │    │
│  │ • coaching.json          │    │ • API: queue, tags, delete      │    │
│  │ • sequences/*.jpg        │    │ • Routes: tennis.playfullife.com│    │
│  │ • meta.json              │    └─────────────────────────────────┘    │
│  └──────────────────────────┘                  │                         │
└─────────────────────────────────────────────────┼─────────────────────────┘
                                                  │
                                                  ▼
                                      ┌──────────────────────┐
                                      │  GALLERY (web)       │
                                      │  Public URL          │
                                      │  • Session grouping  │
                                      │  • Coach modals      │
                                      │  • Filmstrips        │
                                      │  • Pro comparisons   │
                                      │  • Shot timestamps   │
                                      │  • Tags, search      │
                                      └──────────────────────┘

  STRENGTHS                              WEAKNESSES
  ─────────                              ──────────
  ✓ High-fidelity 240fps capture         ✗ Coaching takes ~5-15 min
  ✓ Pro-grade biomech analysis           ✗ No INSTANT feedback during play
  ✓ Camera-invariant 3D angles           ✗ Single-user (no friend feed)
  ✓ Cheap hosting (R2)                   ✗ No history/progression view
  ✓ Web-shareable (links)                ✗ No remote coach workflow
  ✓ Self-healing iCloud auth             ✗ Sharing is "send link" only
  ✓ 96.5% shot detection F1              ✗ No social features (likes,
                                            comments, leaderboards)
                                         ✗ Cloud-dependent for coaching
```

---

## Whiteboard 2: WHERE WE WANT TO GO

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER'S iPHONE 18 PRO                            │
│                  (A20 chip, 12 GB RAM, ~50 TOPS NPU)                     │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     SwingLab App (native)                        │    │
│  │                                                                  │    │
│  │  ┌──────────────────┐   ┌────────────────────┐                  │    │
│  │  │ LIVE PRACTICE    │   │ INSTANT REVIEW     │                  │    │
│  │  │ MODE             │   │ ──────────────     │                  │    │
│  │  │ ──────────────   │   │ Last shot in 2s:   │                  │    │
│  │  │ AR overlay:      │   │ • Filmstrip        │                  │    │
│  │  │ • pose skeleton  │   │ • Grade A-F        │                  │    │
│  │  │ • angle readouts │   │ • Quick fix tip    │                  │    │
│  │  │ • shot count     │   │   (local 7B LLM)   │ ⚡ < 3 sec        │    │
│  │  │ • grade A-F      │   └────────────────────┘                  │    │
│  │  │   in real-time   │                                            │    │
│  │  └──────────────────┘   ┌────────────────────┐                  │    │
│  │                          │ SESSION SUMMARY    │                  │    │
│  │  ┌──────────────────┐   │ ──────────────     │                  │    │
│  │  │ FRIEND FEED      │   │ End of session:    │                  │    │
│  │  │ ──────────────   │   │ • All shots graded │                  │    │
│  │  │ See friends'     │   │ • Trends vs prev   │                  │    │
│  │  │ sessions, comment│   │ • Drill suggestion │                  │    │
│  │  │ react, challenge │   │ • Auto-share opt   │                  │    │
│  │  └──────────────────┘   └────────────────────┘                  │    │
│  │                                                                  │    │
│  │  ┌──────────────────┐   ┌────────────────────┐                  │    │
│  │  │ MY HISTORY       │   │ COACH MODE         │                  │    │
│  │  │ ──────────────   │   │ ──────────────     │                  │    │
│  │  │ • Progression    │   │ Send specific shot │                  │    │
│  │  │   graphs         │   │ to coach for       │                  │    │
│  │  │ • Streaks        │   │ async voice memo   │                  │    │
│  │  │ • Heatmaps       │   │ feedback           │                  │    │
│  │  │ • Personal bests │   │                    │                  │    │
│  │  └──────────────────┘   └────────────────────┘                  │    │
│  │                                                                  │    │
│  │       ┌─────────────────────────────────────┐                   │    │
│  │       │  ON-DEVICE INFERENCE STACK          │                   │    │
│  │       │  • Pose: Apple Vision (existing)    │                   │    │
│  │       │  • Shot detection: ONNX (existing)  │                   │    │
│  │       │  • Coaching: Qwen 2.5 7B Instruct   │                   │    │
│  │       │    (4-bit MLX, ~2 GB on disk)       │                   │    │
│  │       │  • Voice synth: Apple TTS           │                   │    │
│  │       │    "Bend your knees more next time" │                   │    │
│  │       └─────────────────────────────────────┘                   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                  │                                       │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │ background sync
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    UNCHANGED CLOUD INFRA                                 │
│  Hetzner watcher → GPU pipeline → R2 → Gallery (still does the           │
│  heavy 3D biomech + Opus 4.7 deep coaching for the long view)            │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              NEW: USER ACCOUNTS + SOCIAL LAYER                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │ Cloudflare D1 (SQLite at edge) — replaces anonymous gallery    │     │
│  │  • users, sessions, follows, comments, reactions               │     │
│  │  • per-user history with personal bests, streaks               │     │
│  │  • friend feed query: "show me friends' last 10 sessions"      │     │
│  │  • coach-student relationships (1:N async coaching)            │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │ Push notifications (APNs)                                       │     │
│  │  • "Your coach left you a video reply"                         │     │
│  │  • "Sarah just shared a session — beat her PR?"                │     │
│  │  • "You're 2 sessions from a 30-day streak"                    │     │
│  └────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘

  THE SWEET SPOT
  ──────────────
  Local for SPEED:                       Cloud for VALUE:
  • Live AR coaching during play          • Deep biomech analysis
  • Instant per-shot grade                • Pro comparison library
  • Quick fix tips between points         • Permanent shareable URLs
  • Works offline at any court            • Cross-device history
  • No coaching API costs                 • Friend feed + social
                                          • Coach-student feedback loop
                                          • Long-term progression
                                          • Backups + DR

  USER JOURNEY (FUTURE)
  ────────────────────
  1. PRACTICE  : Live AR coach on phone, A-F grade per shot in 2 seconds
  2. END       : Session auto-uploads to cloud while you walk to your car
  3. CAR       : Phone shows summary "32 BH, avg knee 118° (PR!)"
  4. HOME      : 10 min later, deep cloud coaching ready, share to friends
  5. NEXT DAY : Coach replied with voice memo on shot #14
  6. WEEK     : Progression graph shows knee bend +8° over 2 weeks
```

---

## Migration Path (incremental, ship-as-you-go)

### Phase 1 — Local instant feedback (this quarter)
- [ ] Pull Qwen 2.5 7B Instruct, quantize to 4-bit MLX (~2 GB)
- [ ] Bundle with iOS app (or download on first launch)
- [ ] Wire pose+detection → on-device LLM → voice memo
- [ ] Ship "Live Practice Mode" with AR overlay + instant per-shot feedback

### Phase 2 — Accounts + history (next quarter)
- [ ] Sign in with Apple (no passwords)
- [ ] Cloudflare D1 schema for users, sessions, shots
- [ ] Migrate gallery from public-anonymous to per-user
- [ ] My History view: progression, PRs, streaks

### Phase 3 — Social (Q3)
- [ ] Friend follows
- [ ] Friend feed
- [ ] Reactions + comments
- [ ] Auto-share toggles

### Phase 4 — Coach mode (Q4)
- [ ] Coach role + invite flow
- [ ] Coach inbox (students' shots flagged for review)
- [ ] Async voice memo feedback
- [ ] Subscription billing for coach features
