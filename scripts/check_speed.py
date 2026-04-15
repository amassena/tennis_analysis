"""Check which preprocessed videos are sped up vs their raw source."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocess_nvenc import probe_max_fps, probe_duration

raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")
pre_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocessed")

for f in sorted(os.listdir(raw_dir)):
    if not f.lower().endswith(".mov"):
        continue
    stem = os.path.splitext(f)[0]
    raw_path = os.path.join(raw_dir, f)
    pre_path = os.path.join(pre_dir, stem + ".mp4")
    fps = probe_max_fps(raw_path)
    if fps <= 60:
        continue
    if not os.path.exists(pre_path):
        continue
    raw_dur = probe_duration(raw_path)
    pre_dur = probe_duration(pre_path)
    if raw_dur <= 0:
        continue
    ratio = pre_dur / raw_dur
    flag = " *** SPED UP" if ratio < 0.95 else " OK"
    print(f"{stem}: raw={raw_dur:.1f}s pre={pre_dur:.1f}s ratio={ratio:.3f}{flag}")
