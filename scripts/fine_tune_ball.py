#!/usr/bin/env python3
"""
Fine-tune the WASB-SBDT ball tracker (HRNet heatmap CNN) on OUR footage using
ball labels from our labeler.

Phase-2 enabler for contact detection on all shot types
(docs/contact_detection.md + GitHub issue #32).

Runs on a GPU machine (tmassena, RTX 4080). NOT on Mac.

------------------------------------------------------------------------------
What this does
------------------------------------------------------------------------------
WASB-SBDT is a 3-frame -> 3-heatmap HRNet. For each frame f the model is shown
the stack [f-1, f, f+1] (warped to 512x288, ImageNet-normalized) and predicts a
heatmap per input frame. The production training target (see WASB
dataloaders/dataset_loader.py + utils/heatmap.py) is a BINARY DISC of radius
sigma=2.5 px centered on the ball, at OUTPUT resolution 512x288 (stride 1), or an
all-zero map when the ball is not visible. The loss is the repo's WBCE
(focal-equivalent, gamma=2) applied to the sigmoid of the logits.

We mirror that exactly:
  - same model construction + weight loading as wasb_track.py (no hydra)
  - same letterbox affine transform / 288x512 / ImageNet norm input
  - same gen_binary_map target (sigma=2.5) at 512x288, zeros when invisible
  - same WBCE loss (vendored, identical math)
  - low-LR Adam fine-tune from the pretrained tennis weights
  - clip-level train/val split (never split one clip's frames)
  - light aug: h-flip (with x-mirror of label) + brightness/contrast jitter
  - eval: detection rate (peak > thresh) + mean pixel error vs label,
    BEFORE (pretrained) vs AFTER (fine-tuned)

Decode at eval time reuses wasb_track's connected-component blob decode so the
"detection rate" / "pixel error" numbers are comparable to inference.

------------------------------------------------------------------------------
Labels
------------------------------------------------------------------------------
human : R2 ball_labels/<clip>.json
        {fps,width,height, frames:{ "<frame>": {x,y,visible} }}  (x,y 0..1)
seed  : R2 uploads/ball_seed_<clip>.json   (TrackNet weak labels)
        {fps,width,height, track:[{frame,x,y,visible}]}          (x,y 0..1)
both  : union (human overrides seed per frame)

Video : R2 uploads/strike_<clip>.mp4

NOTE: seed-only training is a PIPELINE SMOKE TEST, not a real accuracy gain --
you'd be training on the tracker's own outputs. Use --labels-source human (or
both) once enough human labels exist.

------------------------------------------------------------------------------
CLI
------------------------------------------------------------------------------
  python fine_tune_ball.py --labels-source seed --epochs 1
  python fine_tune_ball.py --labels-source both --epochs 8 --val-clips 6c891e87 --lr 1e-4

Defaults: epochs=8, lr=1e-4, val-clips=auto (hold out 1 clip), labels-source=human.
"""
import os
import sys
import json
import time
import math
import random
import argparse
import datetime
import importlib.util as _ilu

import numpy as np
import cv2
import torch
from torch import nn

# ----------------------------------------------------------------------------
# Paths / WASB vendoring (mirror wasb_track.py)
# ----------------------------------------------------------------------------
BALLTRACK = os.environ.get("BALLTRACK_DIR", "C:/Users/amass/balltrack")
WASB_ROOT = os.path.join(BALLTRACK, "WASB-SBDT")
WASB_SRC = os.path.join(WASB_ROOT, "src")
DEFAULT_PRETRAINED = os.path.join(WASB_ROOT, "pretrained_weights", "wasb_tennis_best.pth.tar")
TENNIS_ENV = os.environ.get("TENNIS_ENV", "C:/Users/amass/tennis_analysis/.env")

sys.path.insert(0, WASB_SRC)

from models.hrnet import HRNet  # noqa: E402

# Import utils/image.py by file path to dodge utils/__init__.py (pulls in pandas).
_spec = _ilu.spec_from_file_location("wasb_image", os.path.join(WASB_SRC, "utils", "image.py"))
_img = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_img)
get_affine_transform = _img.get_affine_transform
affine_transform = _img.affine_transform

# Import utils/heatmap.py the same way to reuse the EXACT target generator.
_spec_hm = _ilu.spec_from_file_location("wasb_heatmap", os.path.join(WASB_SRC, "utils", "heatmap.py"))
_hm = _ilu.module_from_spec(_spec_hm)
_spec_hm.loader.exec_module(_hm)
gen_binary_map = _hm.gen_binary_map  # disc of radius sigma -> production target

# ----------------------------------------------------------------------------
# Static config lifted from configs/model/wasb.yaml + dataloader/default.yaml
# (identical to wasb_track.py so weights load cleanly)
# ----------------------------------------------------------------------------
class AD(dict):
    def __getattr__(self, k):
        v = self[k]
        return AD(v) if isinstance(v, dict) else v
    __setattr__ = dict.__setitem__


def build_cfg():
    return AD({
        'name': 'hrnet',
        'frames_in': 3, 'frames_out': 3,
        'inp_height': 288, 'inp_width': 512,
        'out_height': 288, 'out_width': 512,
        'rgb_diff': False,
        'out_scales': [0],
        'MODEL': {
            'EXTRA': {
                'FINAL_CONV_KERNEL': 1,
                'PRETRAINED_LAYERS': ['*'],
                'STEM': {'INPLANES': 64, 'STRIDES': [1, 1]},
                'STAGE1': {'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK',
                           'NUM_BLOCKS': [1], 'NUM_CHANNELS': [32], 'FUSE_METHOD': 'SUM'},
                'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC',
                           'NUM_BLOCKS': [2, 2], 'NUM_CHANNELS': [16, 32], 'FUSE_METHOD': 'SUM'},
                'STAGE3': {'NUM_MODULES': 1, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC',
                           'NUM_BLOCKS': [2, 2, 2], 'NUM_CHANNELS': [16, 32, 64], 'FUSE_METHOD': 'SUM'},
                'STAGE4': {'NUM_MODULES': 1, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC',
                           'NUM_BLOCKS': [2, 2, 2, 2], 'NUM_CHANNELS': [16, 32, 64, 128], 'FUSE_METHOD': 'SUM'},
                'DECONV': {'NUM_DECONVS': 0, 'KERNEL_SIZE': [], 'NUM_BASIC_BLOCKS': 2},
            },
            'INIT_WEIGHTS': True,
        },
    })


# ImageNet normalization (WASB build_img_transforms)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# WASB dataloader/default.yaml heatmap config + detector/tracknetv2.yaml
HM_SIGMA = 2.5            # disc radius for gen_binary_map
SCORE_THRESHOLD = 0.5    # postprocessor.score_threshold (decode + det-rate)


# ----------------------------------------------------------------------------
# Input transform (identical to wasb_track.preprocess_frame)
# ----------------------------------------------------------------------------
def get_input_transform(h, w, inp_wh):
    c = np.array([w / 2., h / 2.], dtype=np.float32)
    s = max(h, w) * 1.0
    trans = get_affine_transform(c, s, 0, [inp_wh[0], inp_wh[1]], inv=0)
    trans_inv = get_affine_transform(c, s, 0, [inp_wh[0], inp_wh[1]], inv=1)
    return trans, trans_inv


def warp_frame(frame_bgr, trans, inp_wh):
    """BGR original -> warped BGR uint8 at inp_wh (kept as uint8 so we can apply
    per-sample brightness/contrast jitter before normalizing)."""
    return cv2.warpAffine(frame_bgr, trans, inp_wh, flags=cv2.INTER_LINEAR)


def normalize_chw(warped_bgr):
    rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return np.transpose(rgb, (2, 0, 1))  # 3,H,W


# ----------------------------------------------------------------------------
# Target heatmap (mirror dataloader: gen_binary_map disc at output res, or zeros)
# ----------------------------------------------------------------------------
def make_target(out_wh, label_xy_outpx, visible):
    """out_wh=(W,H). label_xy_outpx = (x,y) in OUTPUT pixel space. Returns HxW."""
    if not visible or label_xy_outpx is None:
        return np.zeros((out_wh[1], out_wh[0]), dtype=np.float32)
    ct_int = np.array(label_xy_outpx, dtype=np.float32).astype(np.int32)
    return gen_binary_map((out_wh[0], out_wh[1]), ct_int, HM_SIGMA, np.float32)


# ----------------------------------------------------------------------------
# Vendored WBCE loss (identical math to losses/wbce.py, gamma=2 focal-equivalent)
# WASB applies it on sigmoid(logits). We pass per-scale dicts {0: tensor}.
# ----------------------------------------------------------------------------
class WBCELoss(nn.Module):
    def forward(self, inputs, targets):
        # inputs/targets: dict {scale: tensor[B,frames_out,H,W]} ; inputs are sigmoid'd
        loss_acc = 0.0
        for scale in inputs.keys():
            p = inputs[scale].clamp(1e-7, 1 - 1e-7)
            t = targets[scale]
            loss = ((1 - p) ** 2) * t * torch.log(p) + (p ** 2) * (1 - t) * torch.log(1 - p)
            loss_acc = loss_acc + torch.mean(-loss)
        return loss_acc


# ----------------------------------------------------------------------------
# Heatmap blob decode (identical to wasb_track.decode_heatmap) -> output px
# ----------------------------------------------------------------------------
def decode_heatmap_peak(hm):
    """hm: HxW float in [0,1] (sigmoid'd). Returns (x,y,score) in heatmap px or
    (None,None,peak) if below threshold. score = blob weight sum."""
    peak = float(hm.max())
    if peak <= SCORE_THRESHOLD:
        return None, None, peak
    _, hm_th = cv2.threshold(hm, SCORE_THRESHOLD, 1, cv2.THRESH_BINARY)
    n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
    best = None
    for m in range(1, n_labels):
        ys, xs = np.where(labels == m)
        ws = hm[ys, xs]
        score = float(ws.sum())
        x = float(np.sum(xs * ws) / np.sum(ws))
        y = float(np.sum(ys * ws) / np.sum(ws))
        if best is None or score > best[2]:
            best = (x, y, score)
    if best is None:
        return None, None, peak
    return best


# ----------------------------------------------------------------------------
# R2
# ----------------------------------------------------------------------------
def r2_client():
    try:
        from dotenv import load_dotenv
        load_dotenv(TENNIS_ENV)
    except Exception:
        # fall back to manual .env parse
        if os.path.exists(TENNIS_ENV):
            for line in open(TENNIS_ENV):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    import boto3
    return boto3.client(
        "s3",
        endpoint_url="https://%s.r2.cloudflarestorage.com" % os.environ["CF_ACCOUNT_ID"],
        aws_access_key_id=os.environ["CF_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CF_R2_SECRET_ACCESS_KEY"],
    )


BUCKET = "tennis-videos"


def r2_get_json(s3, key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.load(obj["Body"])


def r2_download(s3, key, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    s3.download_file(BUCKET, key, dst)
    return dst


def r2_list_clips(s3, source):
    """Return set of clip ids available for the requested label source."""
    clips = set()
    if source in ("seed", "both"):
        r = s3.list_objects_v2(Bucket=BUCKET, Prefix="uploads/ball_seed_")
        for o in r.get("Contents", []):
            name = os.path.basename(o["Key"])
            if name.startswith("ball_seed_") and name.endswith(".json"):
                clips.add(name[len("ball_seed_"):-len(".json")])
    if source in ("human", "both"):
        r = s3.list_objects_v2(Bucket=BUCKET, Prefix="ball_labels/")
        for o in r.get("Contents", []):
            name = os.path.basename(o["Key"])
            if name.endswith(".json"):
                clips.add(name[:-len(".json")])
    return clips


# ----------------------------------------------------------------------------
# Label loading -> {frame_idx: (x_norm, y_norm, visible)}
# ----------------------------------------------------------------------------
def load_labels(s3, clip, source):
    labels = {}  # frame -> (x,y,visible)
    if source in ("seed", "both"):
        try:
            d = r2_get_json(s3, "uploads/ball_seed_%s.json" % clip)
            for rec in d.get("track", []):
                f = int(rec["frame"])
                vis = bool(rec.get("visible")) and rec.get("x") is not None
                labels[f] = (rec.get("x"), rec.get("y"), vis)
        except Exception as e:
            print("  [warn] no seed for %s: %s" % (clip, e))
    if source in ("human", "both"):
        try:
            d = r2_get_json(s3, "ball_labels/%s.json" % clip)
            for fk, rec in d.get("frames", {}).items():
                f = int(fk)
                vis = bool(rec.get("visible")) and rec.get("x") is not None
                labels[f] = (rec.get("x"), rec.get("y"), vis)  # human overrides seed
        except Exception as e:
            print("  [warn] no human labels for %s: %s" % (clip, e))
    return labels


def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
    cap.release()
    return frames, fps


# ----------------------------------------------------------------------------
# Build per-clip sample index: a sample is a labeled center frame f with both
# neighbors f-1, f+1 present. We pre-warp every video frame once.
# ----------------------------------------------------------------------------
class ClipData:
    def __init__(self, clip, frames_bgr, labels, inp_wh):
        self.clip = clip
        self.n = len(frames_bgr)
        self.H, self.W = frames_bgr[0].shape[:2]
        self.inp_wh = inp_wh
        self.trans, self.trans_inv = get_input_transform(self.H, self.W, inp_wh)
        # output transform == input transform (stride 1, out res == in res)
        self.warped = [warp_frame(f, self.trans, inp_wh) for f in frames_bgr]  # uint8 BGR
        self.labels = labels
        # valid center frames: labeled and have both neighbors
        self.centers = [f for f in sorted(labels.keys())
                        if 1 <= f < self.n - 1]

    def label_outpx(self, f):
        """Return (x,y) in OUTPUT/heatmap pixel space for frame f, or None."""
        xn, yn, vis = self.labels[f]
        if not vis:
            return None, False
        px = xn * self.W
        py = yn * self.H
        outxy = affine_transform(np.array([px, py], dtype=np.float32), self.trans)
        return (float(outxy[0]), float(outxy[1])), True


def build_sample_tensor(clip_data, f, augment=False):
    """Return (input_chw[9,H,W], target[3,H,W]) for center frame f.
    frames_out=3: channel k is the heatmap for window frame (f-1+k)."""
    W, H = clip_data.inp_wh
    fin = 3
    win_frames = [f - 1, f, f + 1]

    # brightness/contrast jitter (shared across the 3 frames, like a clip)
    do_flip = augment and random.random() < 0.5
    if augment:
        b = random.uniform(0.85, 1.15)   # brightness
        c = random.uniform(0.85, 1.15)   # contrast
    else:
        b = c = 1.0

    chans = []
    for wf in win_frames:
        warped = clip_data.warped[wf].astype(np.float32)
        if augment:
            mean = warped.mean()
            warped = (warped - mean) * c + mean * b
            warped = np.clip(warped, 0, 255)
        warped = warped.astype(np.uint8)
        if do_flip:
            warped = cv2.flip(warped, 1)  # horizontal
        chans.append(normalize_chw(warped))
    inp = np.concatenate(chans, axis=0)  # 9,H,W

    # targets: one per window frame
    tgts = []
    for wf in win_frames:
        if wf in clip_data.labels:
            outxy, vis = clip_data.label_outpx(wf)
        else:
            outxy, vis = None, False
        if vis and do_flip:
            outxy = (W - 1 - outxy[0], outxy[1])
        tgts.append(make_target((W, H), outxy, vis))
    tgt = np.stack(tgts, axis=0)  # 3,H,W
    return inp.astype(np.float32), tgt.astype(np.float32)


# ----------------------------------------------------------------------------
# Model load (mirror wasb_track.py)
# ----------------------------------------------------------------------------
def load_model(weights_path, device):
    cfg = build_cfg()
    model = HRNet(cfg)
    ck = torch.load(weights_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("  [model] missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    return model.to(device), cfg


# ----------------------------------------------------------------------------
# Eval: detection rate + mean pixel error on the CENTER frame of each window.
# We use the trailing-aware approach of wasb_track (channel k -> frame f-1+k);
# for eval we read the channel that corresponds to the center frame (k=1).
# Pixel error is reported in ORIGINAL pixel space (via trans_inv).
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, clip_datas, device):
    model.eval()
    n_vis = 0          # labeled-visible center frames
    n_detected = 0     # of those, model fired (peak > thresh)
    px_errors = []     # pixel error (orig space) on detected+visible frames
    for cd in clip_datas:
        for f in cd.centers:
            xn, yn, vis = cd.labels[f]
            inp, _ = build_sample_tensor(cd, f, augment=False)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            out = model(inp_t)[0]  # scale-0: 1,frames_out,H,W
            hm = torch.sigmoid(out)[0, 1].cpu().numpy()  # center frame channel
            x, y, score = decode_heatmap_peak(hm)
            if not vis:
                continue
            n_vis += 1
            if x is None:
                continue
            n_detected += 1
            # map heatmap-space (x,y) back to original pixels
            orig = affine_transform(np.array([x, y], dtype=np.float32), cd.trans_inv)
            gx, gy = xn * cd.W, yn * cd.H
            err = math.hypot(orig[0] - gx, orig[1] - gy)
            px_errors.append(err)
    det_rate = (n_detected / n_vis) if n_vis else 0.0
    mean_px = float(np.mean(px_errors)) if px_errors else None
    med_px = float(np.median(px_errors)) if px_errors else None
    return {
        "visible_frames": n_vis,
        "detected": n_detected,
        "detection_rate": round(det_rate, 4),
        "mean_pixel_error": round(mean_px, 2) if mean_px is not None else None,
        "median_pixel_error": round(med_px, 2) if med_px is not None else None,
    }


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def make_batches(clip_datas, batch_size, shuffle=True):
    samples = []  # (clip_index, frame)
    for ci, cd in enumerate(clip_datas):
        for f in cd.centers:
            samples.append((ci, f))
    if shuffle:
        random.shuffle(samples)
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]


def train(model, train_clips, val_clips, device, epochs, lr, batch_size,
          patience, out_path):
    criterion = WBCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_metric = None   # lower mean_px is better; fall back to -det_rate
    best_state = None
    epochs_no_improve = 0
    history = []

    n_train_samples = sum(len(cd.centers) for cd in train_clips)
    print("Train samples=%d (clips=%d)  Val clips=%d" % (
        n_train_samples, len(train_clips), len(val_clips)))

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for batch in make_batches(train_clips, batch_size, shuffle=True):
            inps, tgts = [], []
            for ci, f in batch:
                inp, tgt = build_sample_tensor(train_clips[ci], f, augment=True)
                inps.append(inp)
                tgts.append(tgt)
            inp_t = torch.from_numpy(np.stack(inps)).to(device)
            tgt_t = torch.from_numpy(np.stack(tgts)).to(device)
            out = model(inp_t)  # dict {0: B,frames_out,H,W}
            pred = {s: torch.sigmoid(v) for s, v in out.items()}
            target = {s: tgt_t for s in out.keys()}
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item())
            n_batches += 1
        avg_loss = ep_loss / max(1, n_batches)

        val = evaluate(model, val_clips, device)
        # selection metric: prefer lower median pixel error; if none detected,
        # use a large sentinel so any detecting epoch wins.
        sel = val["median_pixel_error"]
        sel = 1e9 if sel is None else sel
        # tie-break by detection rate (higher better) -> subtract small bonus
        sel = sel - val["detection_rate"] * 1e-3
        history.append({"epoch": ep, "train_loss": round(avg_loss, 6), **val})
        print("epoch %d  loss=%.5f  val det=%.3f  med_px=%s  mean_px=%s" % (
            ep, avg_loss, val["detection_rate"],
            val["median_pixel_error"], val["mean_pixel_error"]))

        if best_metric is None or sel < best_metric:
            best_metric = sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stop (no val improvement for %d epochs)" % patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    # save checkpoint in the same container format wasb_track expects to load
    torch.save({"model_state_dict": model.state_dict()}, out_path)
    print("Saved fine-tuned weights ->", out_path)
    return history


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune WASB-SBDT ball tracker on our footage")
    ap.add_argument("--labels-source", choices=["human", "seed", "both"], default="human")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3, help="early-stop patience (epochs)")
    ap.add_argument("--val-clips", default="", help="comma-separated clip ids to hold out; default auto-pick 1")
    ap.add_argument("--clips", default="", help="restrict to these clip ids (comma-separated)")
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    ap.add_argument("--out", default="", help="output weights path; default wasb_ft_<date>.pth")
    ap.add_argument("--cache-dir", default=os.path.join(BALLTRACK, "ft_cache"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device=%s  labels-source=%s" % (device, args.labels_source))

    s3 = r2_client()

    # which clips
    if args.clips.strip():
        clips = [c.strip() for c in args.clips.split(",") if c.strip()]
    else:
        clips = sorted(r2_list_clips(s3, args.labels_source))
    if not clips:
        print("No clips found for source=%s. Nothing to do." % args.labels_source)
        sys.exit(2)
    print("clips (%d): %s" % (len(clips), clips))

    cfg = build_cfg()
    inp_wh = (cfg["inp_width"], cfg["inp_height"])  # (512,288)

    # load all clip data
    clip_datas = []
    for clip in clips:
        print("loading clip", clip)
        labels = load_labels(s3, clip, args.labels_source)
        if not any(v[2] for v in labels.values()):
            print("  [skip] no visible-labeled frames for", clip)
            continue
        vid_path = os.path.join(args.cache_dir, "strike_%s.mp4" % clip)
        try:
            r2_download(s3, "uploads/strike_%s.mp4" % clip, vid_path)
        except Exception as e:
            print("  [skip] no video for %s: %s" % (clip, e))
            continue
        frames, fps = read_video_frames(vid_path)
        if len(frames) < 3:
            print("  [skip] too few frames for", clip)
            continue
        cd = ClipData(clip, frames, labels, inp_wh)
        if not cd.centers:
            print("  [skip] no usable center frames for", clip)
            continue
        print("  frames=%d  labeled-centers=%d  size=%dx%d" % (
            cd.n, len(cd.centers), cd.W, cd.H))
        clip_datas.append(cd)

    if len(clip_datas) < 2:
        print("Need >=2 usable clips (1 train + 1 val). Got %d." % len(clip_datas))
        sys.exit(2)

    # clip-level split
    by_id = {cd.clip: cd for cd in clip_datas}
    if args.val_clips.strip():
        val_ids = [c.strip() for c in args.val_clips.split(",") if c.strip() in by_id]
    else:
        val_ids = [clip_datas[-1].clip]  # hold out last clip
    if not val_ids:
        val_ids = [clip_datas[-1].clip]
    train_datas = [cd for cd in clip_datas if cd.clip not in val_ids]
    val_datas = [by_id[v] for v in val_ids]
    if not train_datas:
        print("All clips landed in val. Pick fewer --val-clips.")
        sys.exit(2)
    print("TRAIN clips:", [cd.clip for cd in train_datas])
    print("VAL   clips:", [cd.clip for cd in val_datas])

    # BEFORE (pretrained baseline)
    model, _ = load_model(args.pretrained, device)
    print("\n=== BEFORE (pretrained) eval on held-out val ===")
    before = evaluate(model, val_datas, device)
    print(json.dumps(before, indent=2))

    # FINE-TUNE
    date = datetime.date.today().strftime("%Y%m%d")
    out_path = args.out or os.path.join(WASB_ROOT, "pretrained_weights", "wasb_ft_%s.pth" % date)
    print("\n=== FINE-TUNE (lr=%.1e, epochs=%d) ===" % (args.lr, args.epochs))
    t0 = time.time()
    history = train(model, train_datas, val_datas, device,
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    patience=args.patience, out_path=out_path)
    train_secs = round(time.time() - t0, 1)

    # AFTER (best checkpoint already loaded into model)
    print("\n=== AFTER (fine-tuned) eval on held-out val ===")
    after = evaluate(model, val_datas, device)
    print(json.dumps(after, indent=2))

    report = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "labels_source": args.labels_source,
        "pretrained": args.pretrained,
        "out_weights": out_path,
        "device": device,
        "epochs_requested": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "sigma": HM_SIGMA,
        "score_threshold": SCORE_THRESHOLD,
        "input_wh": list(inp_wh),
        "train_clips": [cd.clip for cd in train_datas],
        "val_clips": [cd.clip for cd in val_datas],
        "train_samples": sum(len(cd.centers) for cd in train_datas),
        "train_seconds": train_secs,
        "eval_before": before,
        "eval_after": after,
        "history": history,
        "note": ("seed-only training is a PIPELINE smoke test, not a real "
                 "accuracy gain" if args.labels_source == "seed" else ""),
    }
    report_path = os.path.splitext(out_path)[0] + ".report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved report ->", report_path)

    print("\n========== SUMMARY ==========")
    print("BEFORE: det=%.3f  median_px=%s  mean_px=%s" % (
        before["detection_rate"], before["median_pixel_error"], before["mean_pixel_error"]))
    print("AFTER : det=%.3f  median_px=%s  mean_px=%s" % (
        after["detection_rate"], after["median_pixel_error"], after["mean_pixel_error"]))


if __name__ == "__main__":
    main()
