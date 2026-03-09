#!/usr/bin/env python3
"""Detect tennis ball hits using audio peaks.

Outputs a CSV compatible with visual_label.py --load for manual classification.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PREPROCESSED_DIR, PROJECT_ROOT


def extract_audio(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Extract mono audio from video using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name

    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-ac', '1', '-ar', str(sample_rate),
        '-f', 'wav', temp_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Read WAV file
    import wave
    with wave.open(temp_path, 'rb') as wav:
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        audio /= 32768.0  # Normalize to [-1, 1]

    os.unlink(temp_path)
    return audio


def detect_peaks(audio: np.ndarray, sample_rate: int,
                 threshold_percentile: float = 95,
                 min_gap_ms: float = 300,
                 return_amplitudes: bool = False) -> list:
    """Find audio peaks that likely represent ball hits.

    Returns list of peak times in seconds.
    If return_amplitudes=True, returns list of (time, rms_amplitude) tuples.
    """
    # Compute envelope using RMS in short windows
    window_ms = 10
    window_samples = int(sample_rate * window_ms / 1000)
    hop_samples = window_samples // 2

    envelope = []
    times = []
    for i in range(0, len(audio) - window_samples, hop_samples):
        chunk = audio[i:i + window_samples]
        rms = np.sqrt(np.mean(chunk ** 2))
        envelope.append(rms)
        times.append(i / sample_rate)

    envelope = np.array(envelope)
    times = np.array(times)

    # Find threshold
    threshold = np.percentile(envelope, threshold_percentile)

    # Find peaks above threshold
    above = envelope > threshold
    peaks = []
    min_gap_samples = int(min_gap_ms / (window_ms / 2))

    i = 0
    while i < len(above):
        if above[i]:
            # Find the peak in this region
            start = i
            while i < len(above) and above[i]:
                i += 1
            end = i
            # Find max in this region
            peak_idx = start + np.argmax(envelope[start:end])
            if return_amplitudes:
                peaks.append((times[peak_idx], float(envelope[peak_idx])))
            else:
                peaks.append(times[peak_idx])
            # Skip min_gap
            i = end + min_gap_samples
        else:
            i += 1

    return peaks


def detect_spectral_onsets(audio: np.ndarray, sample_rate: int,
                           percentile_threshold: float = 92,
                           min_gap_ms: float = 300) -> list:
    """Detect audio onsets using spectral flux (STFT-based).

    Spectral flux = L2 norm of positive differences in STFT magnitude
    between consecutive frames. Catches broadband transients (ball strikes)
    even when RMS amplitude is low.

    Returns list of (onset_time, flux_value) tuples.
    """
    from scipy.signal import stft, find_peaks

    # STFT parameters: 32ms window, 8ms hop
    nperseg = 512
    hop = 128
    noverlap = nperseg - hop

    _, _, Zxx = stft(audio, fs=sample_rate, nperseg=nperseg,
                     noverlap=noverlap, window='hann')
    mag = np.abs(Zxx)  # shape: (freq_bins, time_frames)

    # Half-wave rectified spectral flux (onset-only: positive differences)
    diff = np.diff(mag, axis=1)
    diff_rectified = np.maximum(diff, 0.0)
    flux = np.sqrt(np.sum(diff_rectified ** 2, axis=0))  # L2 norm per frame

    if len(flux) == 0:
        return []

    # Time axis for flux values (offset by 1 STFT frame due to diff)
    hop_sec = hop / sample_rate
    flux_times = np.arange(len(flux)) * hop_sec + (nperseg / sample_rate)

    # Adaptive threshold
    threshold = np.percentile(flux, percentile_threshold)

    # Find peaks
    min_gap_samples = max(1, int(min_gap_ms / 1000.0 / hop_sec))
    peak_indices, _ = find_peaks(flux, height=threshold, distance=min_gap_samples)

    onsets = []
    for idx in peak_indices:
        onsets.append((float(flux_times[idx]), float(flux[idx])))

    return onsets


def get_video_fps(video_path: str) -> float:
    """Get video FPS using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'csv=p=0', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    fps_str = result.stdout.strip().rstrip(',')
    if '/' in fps_str:
        num, den = fps_str.split('/')
        return float(num) / float(den)
    return float(fps_str)


def main():
    parser = argparse.ArgumentParser(
        description="Detect ball hits using audio and output CSV for visual labeling"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=92,
                        help="Percentile threshold for peak detection (default: 92)")
    parser.add_argument("--min-gap", type=float, default=400,
                        help="Minimum gap between hits in ms (default: 400)")
    parser.add_argument("--shot-type", default="forehand",
                        help="Default shot type for marks (default: forehand)")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.getcwd(), video_path)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = args.output or os.path.join(PROJECT_ROOT, f"{video_name}_labels.csv")

    print(f"Extracting audio from {os.path.basename(video_path)}...")
    sample_rate = 16000
    audio = extract_audio(video_path, sample_rate)
    duration = len(audio) / sample_rate
    print(f"  Duration: {duration:.1f}s")

    print(f"Detecting peaks (threshold={args.threshold}%, min_gap={args.min_gap}ms)...")
    peaks = detect_peaks(audio, sample_rate, args.threshold, args.min_gap)
    print(f"  Found {len(peaks)} potential hits")

    # Get FPS for frame conversion
    fps = get_video_fps(video_path)
    print(f"  Video FPS: {fps}")

    with open(output_path, 'w') as f:
        f.write(f"# Audio-detected hits from {os.path.basename(video_path)}\n")
        f.write(f"# {len(peaks)} contact points detected\n")
        f.write("# shot_type, frame\n")

        for peak_time in peaks:
            contact_frame = int(peak_time * fps)
            f.write(f"{args.shot_type},{contact_frame}\n")

    print(f"\nWrote {len(peaks)} marks to {output_path}")
    print(f"\nNext steps:")
    print(f"  python scripts/visual_label.py {video_path} --load {output_path}")


if __name__ == "__main__":
    main()
