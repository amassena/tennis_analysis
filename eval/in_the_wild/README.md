# In-the-wild eval set

The 44-video GT corpus is the user's own iPhone footage from a small set of camera setups. Event-level F1 on that corpus does not predict the metric a new user sees on their first session. This directory holds an independent eval set built to expose distribution shift.

## Target composition (20 clips)

| Slice                        | Target | Why it's here                                           |
|------------------------------|--------|---------------------------------------------------------|
| Doubles (4 players in frame) | 3      | Tracker has to pick the right player, often fails       |
| Indoor / artificial light    | 3      | LED flicker, color cast, low motion blur tolerance      |
| Lefty                        | 3      | Mirror-augmentation correctness check                   |
| Junior (≤14yo, smaller body) | 2      | Bone-length normalization sanity                        |
| Different phone (Pixel/etc.) | 3      | Rolling shutter / encoder differences from iPhone       |
| Shot variety: lobs, drops    | 2      | Classes the 4-class model can't represent               |
| 30fps source (not slo-mo)    | 2      | Verify we don't depend on the 240→60 pipeline           |
| Outdoor harsh backlight      | 2      | High-contrast pose dropout                              |

Pick clips that are 30–90s long with at least 8 trackable shots each. Aim for ~200 total shots.

## Sourcing

- User submissions (gallery feedback / coach inbox)
- Public YouTube under permissive licensing (cite source in `clips/<id>.json`)
- Friends/coaches recording on non-iPhone devices

For each clip, download the original at native fps when possible (no re-encoded YouTube proxy if avoidable). Keep the unedited file in `clips/` for reproducibility.

## Directory layout

```
eval/in_the_wild/
├── README.md                  # this file
├── manifest.json              # one entry per clip — required
├── clips/{vid}.mp4            # source videos (gitignored, large)
├── poses/{vid}.json           # MediaPipe output (gitignored, regenerable)
└── labels/{vid}.json          # human-labeled GT — committed
```

## Manifest format (`manifest.json`)

```json
{
  "clips": [
    {
      "id": "wild_001_doubles_indoor",
      "source_url": "https://... or 'user submission 2026-04-22'",
      "phone": "iphone15pro",
      "fps_source": 240,
      "fps_target": 60,
      "stratify": {
        "session_type": "doubles",
        "lighting": "indoor",
        "dominant_hand": "right",
        "age_group": "adult",
        "rare_shots": ["lob", "drop"]
      },
      "duration_sec": 47.2,
      "n_shots_labeled": 12,
      "notes": "two-handed BH, partner switches sides at ~25s"
    }
  ]
}
```

`stratify` keys feed `validate_pipeline.py`'s stratified breakdown — they end up in the per-clip detection JSON's top level so the existing report picks them up.

## Label format

`labels/{vid}.json` follows the same shape as the existing `detections/{vid}_fused.json` — see one of those for reference. Required fields per shot:

- `timestamp` (seconds, contact frame)
- `shot_type` (one of: serve, forehand, backhand, forehand_volley, backhand_volley, overhead, lob, drop, unknown_shot)
- `source: "manual"` so `validate_pipeline.py` recognizes it as ground truth

Top-level metadata (`fps`, `dominant_hand`, `camera_angle`, `session_type`) must be set — `shot_review.py` writes these by default.

## Workflow to add a clip

1. Drop source video at `clips/{id}.mp4`.
2. Add an entry to `manifest.json`.
3. SCP to a GPU box, run pose extraction, copy poses back to `eval/in_the_wild/poses/{id}.json`.
4. Run `.venv/bin/python scripts/shot_review.py eval/in_the_wild/clips/{id}.mp4` — label every shot.
5. Save labels to `labels/{id}.json` (shot_review's default save path or copy manually).
6. Re-run `validate_pipeline.py --include-all-gt --json-output eval/in_the_wild_results.json` (after symlinking labels into `detections/`, see below).

## Hooking into validate_pipeline

`validate_pipeline.py` discovers GT via files matching `detections/*_fused.json`. To include the wild set without polluting the regression corpus, **symlink only when running the wild eval**:

```bash
# Run wild eval (one-time per session)
for f in eval/in_the_wild/labels/*.json; do
  ln -sf "$(realpath "$f")" "detections/$(basename "$f" .json)_fused.json"
done
.venv/bin/python scripts/validate_pipeline.py --include-all-gt --json-output eval/wild_results.json

# Tear down (so the regression corpus is back to canonical 26 videos)
for f in eval/in_the_wild/labels/*.json; do
  rm "detections/$(basename "$f" .json)_fused.json"
done
```

A future improvement is a `--gt-dir` flag on `validate_pipeline.py` so this dance isn't needed; for now this keeps the corpus boundaries explicit.

## Why this exists

Event-level F1 on the canonical 44-video corpus is currently 26%. We don't know whether that's:
- a labeling-convention issue (GT contact frame ≠ model peak frame),
- a quantization issue (100ms inference step → 50ms quantization error baseline),
- a true distribution shift that gets worse on new users' phones.

The wild set is a clean independent measurement. If event F1 on the canonical corpus and the wild set move together, the bias is in the model. If they diverge, the issue is in the labeling pipeline or distribution shift. We can't tell without it.
