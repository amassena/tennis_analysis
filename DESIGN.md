# Design — bigger picture

What good looks like across the whole app. Read before making any decision that touches more than one file or affects multiple platforms.

---

## Vision

**SwingLab is a hybrid local+cloud AI tennis coach.**

- **Local for SPEED**: live AR coaching during practice, instant per-shot grade in <3s, works offline at any court.
- **Cloud for VALUE**: deep biomechanical analysis (Opus quality), pro comparison library, permanent shareable URLs, social/coach features, history.

The two halves are NOT redundant. Local does the things humans need fast. Cloud does the things only the cloud can.

## Product principles

1. **The web gallery is the source of truth.** iOS embeds it (WebView). All shareable artifacts live there.
2. **Speed is a feature.** Fast feedback (<3s) beats deeper feedback (>5min) for in-session use.
3. **Slow-mo is the default for instructional context.** Coach example links → slo-mo. "Watch full session" → full-speed.
4. **Pro comparisons must be apples-to-apples.** Same handedness, same camera angle, same shot type. Otherwise it teaches nothing.
5. **Filmstrips are billboards, not videos.** Static, glanceable, optimized for the moment of contact. If you need motion, watch the video.
6. **Confidence-gated content.** If our shot detector is <0.85 confident, don't surface that shot as a coaching example.
7. **Don't ask humans for facts that are already structured-public.**
   For entity attributes that exist in Wikipedia/Wikidata/ATP/etc.
   (handedness, height, tournament surface, event year), look them up
   on import — don't build a tagging UI. Reserve human attention for
   the genuinely subjective: camera_angle (your phone's position), failure-
   mode tags on corrections, label edits. The bar: "could a stranger
   answer this from a search?" If yes, automate. If no, ask.

## Cross-platform parity

| Capability | Web | iOS | Notes |
|---|---|---|---|
| Browse sessions | ✅ | ✅ (WebView) | iOS inherits web gallery |
| Live recording | ❌ | ✅ | iOS-only by design |
| Pro comparison | ✅ | ✅ (via web) | |
| On-device coaching | ❌ | 🚧 (planned) | local LLM phase |
| Friend feed | ❌ | ❌ | future, both platforms |

Rule: any feature that lands on web should have an iOS plan within 2 weeks. If iOS plan can't fit, surface it in BACKLOG.md as a parity gap.

## UX guidelines

- **Big tap targets on mobile.** Min 44pt.
- **Color-coded by shot type.** Serve=orange, FH=green, BH=blue, volley=purple.
- **Coach text is short.** 1-2 sentences per point. If longer, link to detail modal.
- **Confidence shown for grades, hidden for shot types.** A grade with confidence is meaningful; a shot type uncertain feels broken.
- **Always show what's clickable.** No mystery affordances.

## Anti-patterns to avoid

- **Don't auto-play videos.** User chooses when.
- **Don't generate text in images.** Use SVG/HTML for any text. Image gen models misspell.
- **Don't ship deep features without instrumentation.** If we can't measure it, we can't improve it.
- **Don't trust ground-truth labels in pro clips library.** Verify with our own shot classifier.
- **Don't build a UI to tag what's already on Wikipedia.** If you find
  yourself adding a dropdown for player handedness, tournament surface,
  or anything similarly public, write a Wikidata lookup instead.
