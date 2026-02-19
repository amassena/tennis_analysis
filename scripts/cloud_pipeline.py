#!/usr/bin/env python3
"""Cloud pipeline for tennis video processing.

Runs on Hetzner coordinator. Watches for new jobs and dispatches to RunPod.

Flow:
1. Poll coordinator API for pending jobs
2. For each job: upload raw video to R2
3. Spin up RunPod GPU
4. Run processing (pose extraction, shot detection, clip extraction)
5. Upload highlights to R2
6. Mark job complete
7. Terminate RunPod
"""

import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv('/opt/tennis/.env')

RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
CF_ACCOUNT_ID = os.environ.get('CF_ACCOUNT_ID')
CF_R2_ACCESS_KEY_ID = os.environ.get('CF_R2_ACCESS_KEY_ID')
CF_R2_SECRET_ACCESS_KEY = os.environ.get('CF_R2_SECRET_ACCESS_KEY')

COORDINATOR_URL = "http://localhost:8080"  # Local on Hetzner
R2_ENDPOINT = f"https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2_BUCKET = "tennis-videos"

# RunPod processing script (runs on GPU pod)
# Full pipeline: pose extraction -> shot detection -> clip extraction -> highlight compilation
PROCESSING_SCRIPT = '''
#!/bin/bash
set -e

VIDEO_KEY="$1"
VIDEO_NAME="$2"

echo "=== Installing dependencies ==="
# Install FFmpeg for video processing
apt-get update -qq && apt-get install -y -qq ffmpeg

pip install -q boto3 matplotlib opencv-contrib-python

# Install YOLOv8 for GPU-accelerated pose extraction (~200fps vs 8fps CPU)
pip install -q ultralytics

# Install TensorFlow with h5py for model patching
pip install -q h5py
pip uninstall -y keras protobuf 2>/dev/null || true
pip install -q protobuf
pip install -q tensorflow

echo "=== Downloading video and model from R2 ==="
python3 << ENDPY
import boto3
from botocore.config import Config
import os

client = boto3.client('s3',
    endpoint_url='${R2_ENDPOINT}',
    aws_access_key_id='${CF_R2_ACCESS_KEY_ID}',
    aws_secret_access_key='${CF_R2_SECRET_ACCESS_KEY}',
    config=Config(signature_version='s3v4'))

# Download video
client.download_file('${R2_BUCKET}', '${VIDEO_KEY}', '/workspace/input.mp4')
print("Downloaded video")

# Download model
os.makedirs('/workspace/models', exist_ok=True)
client.download_file('${R2_BUCKET}', 'models/shot_classifier.h5', '/workspace/models/shot_classifier.h5')
client.download_file('${R2_BUCKET}', 'models/shot_classifier_meta.json', '/workspace/models/shot_classifier_meta.json')
print("Downloaded model")
ENDPY

echo "=== Extracting thumbnail ==="
python3 << ENDPY
import cv2

cap = cv2.VideoCapture('/workspace/input.mp4')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Seek to 10% into video for a good frame
cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames // 10, 100))
ret, frame = cap.read()

if ret:
    h, w = frame.shape[:2]
    if w > 480:
        scale = 480 / w
        frame = cv2.resize(frame, (480, int(h * scale)))
    cv2.imwrite('/workspace/thumb.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("Thumbnail extracted")
else:
    print("Failed to extract thumbnail")

cap.release()
ENDPY

echo "=== Preprocessing to 60fps CFR ==="
# Convert VFR to 60fps CFR - reduces frames dramatically and speeds up pose extraction
ffmpeg -y -i /workspace/input.mp4 -r 60 -c:v libx264 -preset fast -crf 18 -an /workspace/preprocessed.mp4
echo "Preprocessed to 60fps"

echo "=== Running pose extraction (YOLOv8 GPU) ==="
python3 << ENDPY
import cv2
import json
import time
from ultralytics import YOLO

# Load YOLOv8-Pose model (GPU accelerated, ~200fps)
model = YOLO("yolov8m-pose.pt")

cap = cv2.VideoCapture('/workspace/preprocessed.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {fps:.0f} fps, {total_frames} frames, {width}x{height}")

frames_data = []
frame_idx = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)

    frame_data = {"frame_idx": frame_idx, "detected": False}

    if len(results) > 0 and results[0].keypoints is not None:
        keypoints = results[0].keypoints
        if len(keypoints.data) > 0:
            kp = keypoints.data[0]  # First person, shape (17, 3)
            if kp.shape[0] == 17:
                frame_data["detected"] = True
                # Normalize coordinates and store as world_landmarks
                landmarks = []
                for i in range(17):
                    x = float(kp[i, 0]) / width
                    y = float(kp[i, 1]) / height
                    conf = float(kp[i, 2])
                    landmarks.append([x, y, 0.0, conf])
                frame_data["world_landmarks"] = landmarks
                frame_data["landmarks"] = landmarks

    frames_data.append(frame_data)
    frame_idx += 1
    if frame_idx % 500 == 0:
        elapsed = time.time() - start
        print(f"Processed {frame_idx}/{total_frames} frames ({frame_idx/elapsed:.1f} fps)")

cap.release()
elapsed = time.time() - start
detected = sum(1 for f in frames_data if f.get("detected"))
print(f"Done: {detected}/{frame_idx} frames with pose in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")

output = {
    "video_info": {"fps": fps, "total_frames": total_frames, "pose_model": "yolov8m-pose", "num_keypoints": 17},
    "frames": frames_data
}
with open('/workspace/poses.json', 'w') as f:
    json.dump(output, f)
ENDPY

echo "=== Running shot detection ==="
python3 << ENDPY
import json
import numpy as np
import h5py

# Patch the model file to remove quantization_config
# This is needed because newer Keras doesn't recognize this key
with h5py.File('/workspace/models/shot_classifier.h5', 'r+') as f:
    if 'model_config' in f.attrs:
        config = json.loads(f.attrs['model_config'])
        modified = False
        for layer in config.get('config', {}).get('layers', []):
            if 'config' in layer and 'quantization_config' in layer['config']:
                del layer['config']['quantization_config']
                modified = True
        if modified:
            f.attrs['model_config'] = json.dumps(config).encode('utf8')
            print("Patched model config")

from tensorflow import keras

# Load model and metadata
model = keras.models.load_model('/workspace/models/shot_classifier.h5')
print("Model loaded successfully")
with open('/workspace/models/shot_classifier_meta.json') as f:
    meta = json.load(f)

mean = np.array(meta["normalization"]["mean"], dtype=np.float32)
std = np.array(meta["normalization"]["std"], dtype=np.float32)
std[std < 1e-8] = 1.0
seq_len = meta["sequence_length"]
inverse_label_map = meta["inverse_label_map"]
view_angle_map = meta.get("view_angle_map", {})
n_features = meta.get("num_features", 99)

# Determine pose feature count (without view angle one-hot)
# Models with view angle: 99+5=104 (MediaPipe) or 51+5=56 (YOLO)
# Models without: 99 (MediaPipe) or 51 (YOLO)
view_angle_one_hot = None
if view_angle_map and n_features in (104, 56):
    # Model uses view angle features
    n_pose_features = n_features - 5
    view_angle_one_hot = [0.0] * 5
    view_angle_one_hot[view_angle_map.get("back-court", 0)] = 1.0
else:
    # Model doesn't use view angle
    n_pose_features = n_features

print(f"Model expects {n_features} features ({n_pose_features} pose + {5 if view_angle_one_hot else 0} view angle)")

# Load poses
with open('/workspace/poses.json') as f:
    data = json.load(f)

fps = data["video_info"]["fps"]
total_frames = data["video_info"]["total_frames"]
raw_frames = data["frames"]

# Build frame array
frames = [None] * total_frames
for fr in raw_frames:
    idx = fr["frame_idx"]
    if idx < total_frames and fr.get("detected") and fr.get("world_landmarks"):
        frames[idx] = [lm[:3] for lm in fr["world_landmarks"]]

print(f"Loaded {sum(1 for f in frames if f is not None)}/{total_frames} poses")

# Run inference with sliding window
stride = 5
predictions = []

for start in range(0, total_frames - seq_len + 1, stride):
    window = []
    for i in range(start, start + seq_len):
        if frames[i] is not None:
            flat = []
            for kp in frames[i]:
                flat.extend(kp[:3])
            while len(flat) < n_pose_features:
                flat.append(0.0)
            pose_features = flat[:n_pose_features]
        else:
            pose_features = [0.0] * n_pose_features

        if view_angle_one_hot is not None:
            frame_features = pose_features + view_angle_one_hot
        else:
            frame_features = pose_features

        window.append(frame_features)

    X = np.array([window], dtype=np.float32)
    X = (X - mean) / std

    probs = model.predict(X, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    label = inverse_label_map[str(pred_idx)]

    center = start + seq_len // 2
    predictions.append((center, label, confidence))

print(f"Made {len(predictions)} predictions")

# Merge into segments
segments = []
if predictions:
    current_label = predictions[0][1]
    current_start = predictions[0][0]
    current_end = predictions[0][0]
    confidences = [predictions[0][2]]

    for center, label, conf in predictions[1:]:
        if label == current_label and (center - current_end) <= 15:
            current_end = center
            confidences.append(conf)
        else:
            if (current_end - current_start) >= 15:
                segments.append({
                    "shot_type": current_label,
                    "start_frame": current_start,
                    "end_frame": current_end,
                    "start_time": round(current_start / fps, 2),
                    "end_time": round(current_end / fps, 2),
                    "avg_confidence": round(sum(confidences) / len(confidences), 3),
                })
            current_label = label
            current_start = center
            current_end = center
            confidences = [conf]

    # Final segment
    if (current_end - current_start) >= 15:
        segments.append({
            "shot_type": current_label,
            "start_frame": current_start,
            "end_frame": current_end,
            "start_time": round(current_start / fps, 2),
            "end_time": round(current_end / fps, 2),
            "avg_confidence": round(sum(confidences) / len(confidences), 3),
        })

# Count by type
by_type = {}
for seg in segments:
    st = seg["shot_type"]
    by_type[st] = by_type.get(st, 0) + 1
print(f"Detected segments: {by_type}")

output = {
    "source_video": "${VIDEO_NAME}",
    "fps": fps,
    "total_frames": total_frames,
    "segments": segments,
}
with open('/workspace/shots_detected.json', 'w') as f:
    json.dump(output, f, indent=2)
ENDPY

echo "=== Extracting clips and compiling highlights ==="
python3 << ENDPY
import json
import subprocess
import os

with open('/workspace/shots_detected.json') as f:
    data = json.load(f)

segments = data["segments"]
video_name = "${VIDEO_NAME}"

# Filter: non-neutral, confidence >= 0.7, duration 0.5-10s
filtered = []
for seg in segments:
    if seg["shot_type"] == "neutral":
        continue
    if seg["avg_confidence"] < 0.7:
        continue
    duration = seg["end_time"] - seg["start_time"]
    if duration < 0.5 or duration > 10:
        continue
    filtered.append(seg)

# Merge close segments (< 3s gap)
merged = []
if filtered:
    merged = [filtered[0].copy()]
    for seg in filtered[1:]:
        gap = seg["start_time"] - merged[-1]["end_time"]
        if gap < 3.0:
            merged[-1]["end_time"] = seg["end_time"]
        else:
            merged.append(seg.copy())

print(f"Filtered to {len(merged)} segments for highlights")

if not merged:
    print("No clips to extract")
    # Create empty highlight
    with open('/workspace/highlights.txt', 'w') as f:
        pass
else:
    # Extract clips
    os.makedirs('/workspace/clips', exist_ok=True)
    clip_paths = []

    for i, seg in enumerate(merged):
        start = max(0, seg["start_time"] - 0.5)
        duration = (seg["end_time"] - seg["start_time"]) + 1.0
        clip_path = f'/workspace/clips/clip_{i:04d}.mp4'

        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', '/workspace/preprocessed.mp4',
            '-t', str(duration),
            '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
            '-an', '-pix_fmt', 'yuv420p',
            clip_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and os.path.exists(clip_path):
            clip_paths.append(clip_path)
            print(f"  Extracted clip {i+1}/{len(merged)}: {seg['shot_type']} {seg['start_time']:.1f}s")

    if clip_paths:
        # Write concat file
        with open('/workspace/concat.txt', 'w') as f:
            for path in clip_paths:
                f.write(f"file '{path}'\\n")

        # Compile highlights
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', '/workspace/concat.txt',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            '/workspace/highlights.mp4'
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            size_mb = os.path.getsize('/workspace/highlights.mp4') / (1024*1024)
            print(f"Compiled highlights: {len(clip_paths)} clips, {size_mb:.1f} MB")
        else:
            print(f"Failed to compile highlights: {result.stderr.decode()}")
ENDPY

echo "=== Uploading results to R2 ==="
python3 << ENDPY
import boto3
from botocore.config import Config
import os

client = boto3.client('s3',
    endpoint_url='${R2_ENDPOINT}',
    aws_access_key_id='${CF_R2_ACCESS_KEY_ID}',
    aws_secret_access_key='${CF_R2_SECRET_ACCESS_KEY}',
    config=Config(signature_version='s3v4'))

video_name = "${VIDEO_NAME}"

# Upload poses
poses_key = f"poses/{video_name}.json"
with open('/workspace/poses.json', 'rb') as f:
    client.put_object(Bucket='${R2_BUCKET}', Key=poses_key, Body=f.read())
print(f"Uploaded poses to {poses_key}")

# Upload thumbnail
if os.path.exists('/workspace/thumb.jpg'):
    thumb_key = f"thumbs/{video_name}.jpg"
    with open('/workspace/thumb.jpg', 'rb') as f:
        client.put_object(Bucket='${R2_BUCKET}', Key=thumb_key, Body=f.read(), ContentType='image/jpeg')
    print(f"Uploaded thumbnail to {thumb_key}")

# Upload shots detected
shots_key = f"shots/{video_name}_detected.json"
with open('/workspace/shots_detected.json', 'rb') as f:
    client.put_object(Bucket='${R2_BUCKET}', Key=shots_key, Body=f.read())
print(f"Uploaded detections to {shots_key}")

# Upload highlights and generate thumbnail for it
if os.path.exists('/workspace/highlights.mp4'):
    # Generate thumbnail from highlights
    import cv2
    cap = cv2.VideoCapture('/workspace/highlights.mp4')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(total // 10, 50))
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        if w > 480:
            scale = 480 / w
            frame = cv2.resize(frame, (480, int(h * scale)))
        cv2.imwrite('/workspace/highlights_thumb.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    cap.release()

    highlights_key = f"highlights/{video_name}_highlights.mp4"
    with open('/workspace/highlights.mp4', 'rb') as f:
        client.put_object(Bucket='${R2_BUCKET}', Key=highlights_key, Body=f.read(), ContentType='video/mp4')
    size_mb = os.path.getsize('/workspace/highlights.mp4') / (1024*1024)
    print(f"Uploaded highlights to {highlights_key} ({size_mb:.1f} MB)")

    # Upload highlights thumbnail
    if os.path.exists('/workspace/highlights_thumb.jpg'):
        thumb_key = f"thumbs/{video_name}_highlights.jpg"
        with open('/workspace/highlights_thumb.jpg', 'rb') as f:
            client.put_object(Bucket='${R2_BUCKET}', Key=thumb_key, Body=f.read(), ContentType='image/jpeg')
        print(f"Uploaded highlights thumbnail to {thumb_key}")
else:
    print("No highlights to upload")
ENDPY

echo "=== Processing complete ==="
'''


def log(msg):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def runpod_query(query):
    """Execute RunPod GraphQL query."""
    response = requests.post(
        "https://api.runpod.io/graphql",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"query": query},
        timeout=30
    )
    return response.json()


def create_pod():
    """Create a RunPod GPU instance."""
    query = '''
    mutation {
      podFindAndDeployOnDemand(input: {
        name: "tennis-cloud"
        imageName: "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
        gpuTypeId: "NVIDIA GeForce RTX 4090"
        cloudType: SECURE
        volumeInGb: 20
        containerDiskInGb: 20
        minVcpuCount: 4
        minMemoryInGb: 16
        gpuCount: 1
        volumeMountPath: "/workspace"
        ports: "22/tcp"
        startSsh: true
      }) {
        id
        machine { gpuDisplayName }
      }
    }
    '''
    result = runpod_query(query)
    if "data" in result and result["data"]["podFindAndDeployOnDemand"]:
        pod = result["data"]["podFindAndDeployOnDemand"]
        log(f"Created pod {pod['id']} ({pod['machine']['gpuDisplayName']})")
        return pod["id"]
    else:
        log(f"Failed to create pod: {result}")
        return None


def wait_for_ssh(pod_id, timeout=300):
    """Wait for pod to be ready with SSH."""
    query = '''
    query {
      pod(input: { podId: "%s" }) {
        runtime {
          ports { ip isIpPublic privatePort publicPort }
        }
      }
    }
    ''' % pod_id

    start = time.time()
    while time.time() - start < timeout:
        result = runpod_query(query)
        pod = result.get("data", {}).get("pod", {})
        runtime = pod.get("runtime")

        if runtime and runtime.get("ports"):
            for port in runtime["ports"]:
                if port["privatePort"] == 22 and port["isIpPublic"]:
                    return port["ip"], port["publicPort"]

        time.sleep(10)

    return None, None


def terminate_pod(pod_id):
    """Terminate a RunPod instance."""
    query = f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}'
    runpod_query(query)
    log(f"Terminated pod {pod_id}")


def run_on_pod(ssh_host, ssh_port, script):
    """Run a script on the pod via SSH."""
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(ssh_port), f"root@{ssh_host}",
        "bash", "-s"
    ]
    result = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=3600)
    return result.returncode, result.stdout, result.stderr


def process_job(job):
    """Process a single video job."""
    video_id = job["video_id"]
    filename = job["filename"]

    log(f"Processing job {video_id}: {filename}")

    # Create RunPod
    pod_id = create_pod()
    if not pod_id:
        return False, "Failed to create pod"

    try:
        # Wait for SSH
        log("Waiting for pod SSH...")
        ssh_host, ssh_port = wait_for_ssh(pod_id)
        if not ssh_host:
            return False, "Pod SSH timeout"

        log(f"Pod ready: {ssh_host}:{ssh_port}")

        # Run processing
        video_key = f"raw/{filename}"
        video_name = filename.rsplit('.', 1)[0]  # Remove extension

        script = PROCESSING_SCRIPT.replace('${VIDEO_KEY}', video_key)
        script = script.replace('${VIDEO_NAME}', video_name)
        script = script.replace('${R2_ENDPOINT}', R2_ENDPOINT)
        script = script.replace('${R2_BUCKET}', R2_BUCKET)
        script = script.replace('${CF_R2_ACCESS_KEY_ID}', CF_R2_ACCESS_KEY_ID)
        script = script.replace('${CF_R2_SECRET_ACCESS_KEY}', CF_R2_SECRET_ACCESS_KEY)

        log("Running processing on pod...")
        returncode, stdout, stderr = run_on_pod(ssh_host, ssh_port, script)

        # Print processing output
        for line in stdout.split('\n'):
            if line.strip():
                log(f"  {line}")

        if returncode != 0:
            log(f"Processing failed: {stderr}")
            return False, stderr

        log("Processing complete!")
        highlights_key = f"highlights/{video_name}_highlights.mp4"
        return True, highlights_key

    finally:
        # Always terminate pod
        terminate_pod(pod_id)


def get_pending_jobs():
    """Get pending jobs from coordinator."""
    try:
        response = requests.get(f"{COORDINATOR_URL}/jobs/pending", timeout=10)
        data = response.json()
        return data.get("jobs", [])
    except Exception as e:
        log(f"Error getting jobs: {e}")
        return []


def claim_job(video_id):
    """Claim a job for processing."""
    try:
        response = requests.post(
            f"{COORDINATOR_URL}/jobs/{video_id}/claim",
            params={"worker_id": "cloud-worker"},
            timeout=10
        )
        return response.json().get("success", False)
    except Exception as e:
        log(f"Error claiming job: {e}")
        return False


def complete_job(video_id, success, result=None, error=None):
    """Mark job as complete."""
    try:
        requests.post(
            f"{COORDINATOR_URL}/jobs/{video_id}/complete",
            params={"worker_id": "cloud-worker"},
            json={"success": success, "youtube_url": result, "error_message": error},
            timeout=10
        )
    except Exception as e:
        log(f"Error completing job: {e}")


def main():
    """Main worker loop."""
    log("Cloud pipeline worker starting...")
    log(f"Coordinator: {COORDINATOR_URL}")
    log(f"R2 Bucket: {R2_BUCKET}")

    while True:
        jobs = get_pending_jobs()

        if jobs:
            job = jobs[0]
            video_id = job["video_id"]

            if claim_job(video_id):
                log(f"Claimed job {video_id}")
                success, result = process_job(job)
                complete_job(video_id, success, result if success else None, result if not success else None)
            else:
                log(f"Failed to claim job {video_id}")

        time.sleep(60)  # Poll every minute


if __name__ == "__main__":
    main()
