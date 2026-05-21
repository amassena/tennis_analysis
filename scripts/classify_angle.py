#!/usr/bin/env python3
"""Classify camera-angle relative to player from pose data.

4 categories:
  behind-player   — camera behind the player (we see their back/back-of-head)
  side-deuce      — camera on player's RIGHT side (right shoulder closer)
  side-ad         — camera on player's LEFT side  (left shoulder closer)
  front-broadcast — camera in front of player (we see their face, e.g. from opposite baseline)

Signal hierarchy (decision tree):
  1. nose_vis < 0.4         → behind-player (back of head, face hidden)
  2. |shoulder_dz| > 0.10   → side (sign decides deuce vs ad)
  3. nose−shoulder_z < −0.05 → front-broadcast (nose much closer than shoulders)
  4. fallback               → ambiguous; uses shoulder_dx_pix as final disambig

MediaPipe convention: z values are in pose-relative meters with the hip-midpoint
near 0; smaller z = closer to the camera. So shoulder_dz = z(left) − z(right);
positive means left is farther (so right shoulder closer → side-deuce for a
player facing the net), negative the opposite.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def load_pose(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "frames" in data: return data["frames"]
        keys = sorted([k for k in data.keys() if k.isdigit()], key=int)
        if keys: return [data[k] for k in keys]
    return data if isinstance(data, list) else []


def get_field(frame, idx, what):
    lms = frame.get("landmarks") if isinstance(frame, dict) else None
    if not lms or idx >= len(lms):
        return None
    lm = lms[idx]
    if isinstance(lm, dict):
        v = lm.get("visibility", 0)
        if v < 0.3 and what != "visibility": return None
        return lm.get(what, 0)
    if len(lm) < 3: return None
    v = lm[3] if len(lm) > 3 else 1.0
    if v < 0.3 and what != "visibility": return None
    return {"x": lm[0], "y": lm[1], "z": lm[2], "visibility": v}[what]


def signals(frames, center, half=20):
    start = max(0, center - half)
    end = min(len(frames), center + half + 1)
    nv, ld, rd = [], [], []
    nz, lz, rz, lx, rx = [], [], [], [], []
    for i in range(start, end):
        f = frames[i] if i < len(frames) else None
        if not f: continue
        v = get_field(f, NOSE, "visibility")
        if v is not None: nv.append(v)
        nzv = get_field(f, NOSE, "z")
        if nzv is not None: nz.append(nzv)
        lzv = get_field(f, LEFT_SHOULDER, "z")
        if lzv is not None: lz.append(lzv)
        rzv = get_field(f, RIGHT_SHOULDER, "z")
        if rzv is not None: rz.append(rzv)
        lxv = get_field(f, LEFT_SHOULDER, "x")
        if lxv is not None: lx.append(lxv)
        rxv = get_field(f, RIGHT_SHOULDER, "x")
        if rxv is not None: rx.append(rxv)
    if not nz or not lz or not rz:
        return None
    m = lambda xs: sum(xs)/len(xs) if xs else None
    nz_m, lz_m, rz_m = m(nz), m(lz), m(rz)
    sz_m = (lz_m + rz_m) / 2
    return {
        "nose_vis": m(nv),
        "nose_minus_shoulder_z": nz_m - sz_m,
        "shoulder_dz": lz_m - rz_m,
        "shoulder_dx_pix": (m(lx) - m(rx)) if lx and rx else None,
        "n_frames": end - start,
    }


def classify(s):
    if s is None: return "unknown"
    nv = s["nose_vis"] or 0
    nzs = s["nose_minus_shoulder_z"]
    sdz = s["shoulder_dz"]

    # Rule 1: face hidden → looking at back of player's head
    if nv < 0.4:
        return "behind-player"
    # Rule 2: nose strongly forward of shoulders → facing camera (strongest signal)
    if nzs is not None and nzs < -0.08:
        return "front-broadcast"
    # Rule 3: clear shoulder_dz asymmetry → side view
    if sdz is not None and abs(sdz) > 0.05:
        # shoulder_dz = left_z - right_z; sdz>0 → left farther → camera on right (deuce)
        return "side-ad" if sdz < 0 else "side-deuce"
    # Rule 4: nose behind shoulders → behind-player
    if nzs is not None and nzs > 0.05:
        return "behind-player"
    return "ambiguous"


def classify_pro_clip(pose_path):
    """Pro clips use CONTACT_FRAME=120 convention; sample around there."""
    frames = load_pose(pose_path)
    if not frames: return None, None
    center = min(120, len(frames) - 1)
    s = signals(frames, center, half=20)
    return classify(s), s


def classify_user_shot(pose_path, shot_frame):
    """User shots have a specific frame index. Sample around it."""
    frames = load_pose(pose_path)
    if not frames: return None, None
    s = signals(frames, shot_frame, half=20)
    return classify(s), s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pros", action="store_true", help="Classify all pro clips")
    ap.add_argument("--single", help="Single pose JSON path")
    ap.add_argument("--frame", type=int, help="Center frame for --single")
    args = ap.parse_args()

    if args.single:
        label, s = classify_pro_clip(args.single) if not args.frame else classify_user_shot(args.single, args.frame)
        print(f"{Path(args.single).name}: {label}")
        if s:
            for k, v in s.items():
                if isinstance(v, float):
                    print(f"  {k:24s}: {v:+.4f}")
                else:
                    print(f"  {k:24s}: {v}")
        return

    if args.pros:
        root = Path("pros")
        with open(root / "index.json") as f:
            idx = json.load(f)
        results = []
        per_pro_per_type = defaultdict(lambda: defaultdict(Counter))
        for slug, p in sorted(idx["players"].items()):
            if not p.get("clips"):
                continue
            for clip in p["clips"]:
                pose_path = root / slug / Path(clip["file"]).with_suffix(".pose.json").name
                if not pose_path.exists():
                    continue
                label, _ = classify_pro_clip(pose_path)
                clip_type = clip.get("type", "?")
                existing_angle = clip.get("angle", "?")
                results.append({
                    "slug": slug,
                    "file": clip["file"],
                    "type": clip_type,
                    "existing_angle": existing_angle,
                    "detected_angle": label,
                })
                per_pro_per_type[slug][clip_type][label] += 1

        # Summary by per-pro per-type
        print("\n=== per-pro per-type detected-angle distribution ===")
        for slug in sorted(per_pro_per_type):
            for typ in sorted(per_pro_per_type[slug]):
                c = per_pro_per_type[slug][typ]
                items = ", ".join(f"{k}={v}" for k, v in c.most_common())
                print(f"  {slug:12} {typ:8}: {items}")
        # Cross-tab existing-vs-detected
        print("\n=== existing-vs-detected cross-tab ===")
        cross = Counter()
        for r in results:
            cross[(r["existing_angle"], r["detected_angle"])] += 1
        for (ex, det), n in cross.most_common():
            print(f"  existing={ex:6} detected={det:18} : {n}")
        # Overall counts of detected
        overall = Counter(r["detected_angle"] for r in results)
        print("\n=== overall detected coverage ===")
        for k, v in overall.most_common():
            print(f"  {k:18}: {v}")
        # Save full per-clip result
        out = Path("/tmp/pro_clip_angle_classification.json")
        with out.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nfull per-clip result -> {out}")


if __name__ == "__main__":
    main()
