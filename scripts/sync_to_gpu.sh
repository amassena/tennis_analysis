#!/bin/bash
set -euo pipefail

# ─── GPU Machine Sync ──────────────────────────────────────────────
# Syncs ALL code + config to GPU machines. Run after ANY code change.
# Verifies each file landed correctly via MD5 checksum.
#
# Usage:
#   ./scripts/sync_to_gpu.sh                    # sync both machines
#   ./scripts/sync_to_gpu.sh tmassena           # sync one machine
#   ./scripts/sync_to_gpu.sh --verify           # check drift, don't sync
#   ./scripts/sync_to_gpu.sh --restart          # sync + restart workers
#   ./scripts/sync_to_gpu.sh tmassena --restart # sync one + restart
# ────────────────────────────────────────────────────────────────────

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MACHINES=("tmassena" "windows")
GPU_PATH="C:/Users/amass/tennis_analysis"
VERIFY_ONLY=false
RESTART=false
TARGET_MACHINES=()

# Parse args
for arg in "$@"; do
    case "$arg" in
        --verify) VERIFY_ONLY=true ;;
        --restart) RESTART=true ;;
        tmassena|windows) TARGET_MACHINES+=("$arg") ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done
[ ${#TARGET_MACHINES[@]} -eq 0 ] && TARGET_MACHINES=("${MACHINES[@]}")

# ─── Files to sync ─────────────────────────────────────────────────
# Every file a GPU machine needs. Add new files HERE so they never
# get missed in a manual scp.
SYNC_FILES=(
    # GPU worker
    "gpu_worker/worker.py"

    # Scripts
    "scripts/ball_tracking.py"
    "scripts/ball_tracking_batch.py"
    "scripts/batch_slowmo_grade.py"
    "scripts/biomechanical_analysis.py"
    "scripts/claude_coach.py"
    "scripts/detect_shots_sequence.py"
    "scripts/email_notify.py"
    "scripts/export_videos.py"
    "scripts/extract_poses.py"
    "scripts/fused_detect.py"
    "scripts/preprocess_nvenc.py"
    "scripts/render_detections.py"
    "scripts/pipeline_health.py"
    "scripts/render_graded.py"
    "scripts/sequence_model.py"
    "scripts/update_r2_index.py"

    # Storage + config modules
    "storage/__init__.py"
    "storage/r2_client.py"
    "config/__init__.py"
    "config/settings.py"
    "config/batch_exclude.py"
    "config/exclude_from_batch.json"

    # TrackNet model definition (not weights — too large, managed separately)
    "models/tracknet/model.py"

    # Credentials
    ".env"
)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }

# ─── MD5 helper ────────────────────────────────────────────────────
local_md5() {
    md5 -q "$1" 2>/dev/null || md5sum "$1" 2>/dev/null | awk '{print $1}'
}

remote_md5() {
    local machine="$1" remote_path="$2"
    ssh -o ConnectTimeout=8 "$machine" \
        "certutil -hashfile \"$remote_path\" MD5 2>nul | findstr /v MD5 | findstr /v CertUtil" \
        2>/dev/null | tr -d '[:space:]' || echo "MISSING"
}

# ─── Main ──────────────────────────────────────────────────────────
TOTAL=0
SYNCED=0
FAILED=0
SKIPPED=0
DRIFTED=0

for machine in "${TARGET_MACHINES[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  $machine"
    echo "════════════════════════════════════════════════════"

    # Check machine is reachable
    if ! ssh -o ConnectTimeout=8 "$machine" "echo ok" &>/dev/null; then
        fail "$machine is OFFLINE — skipping"
        continue
    fi

    for file in "${SYNC_FILES[@]}"; do
        TOTAL=$((TOTAL + 1))
        local_path="$PROJECT_ROOT/$file"
        remote_path="$GPU_PATH/$file"

        # Skip files that don't exist locally
        if [ ! -f "$local_path" ]; then
            warn "$file — not found locally, skipping"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        local_hash=$(local_md5 "$local_path")
        remote_hash=$(remote_md5 "$machine" "$remote_path")

        if [ "$local_hash" = "$remote_hash" ]; then
            ok "$file — in sync"
            continue
        fi

        DRIFTED=$((DRIFTED + 1))

        if $VERIFY_ONLY; then
            fail "$file — DRIFTED (local=$local_hash remote=$remote_hash)"
            continue
        fi

        # Ensure remote directory exists
        remote_dir=$(dirname "$remote_path")
        ssh "$machine" "mkdir \"$remote_dir\" 2>nul" 2>/dev/null || true

        # Sync
        if scp -q "$local_path" "$machine:$remote_path" 2>/dev/null; then
            # Verify
            new_hash=$(remote_md5 "$machine" "$remote_path")
            if [ "$local_hash" = "$new_hash" ]; then
                ok "$file — synced ✓"
                SYNCED=$((SYNCED + 1))
            else
                fail "$file — synced but hash mismatch! (expected=$local_hash got=$new_hash)"
                FAILED=$((FAILED + 1))
            fi
        else
            fail "$file — SCP FAILED"
            FAILED=$((FAILED + 1))
        fi
    done

    # Restart GPU worker if requested
    if $RESTART && ! $VERIFY_ONLY; then
        echo ""
        echo "  Restarting GPU worker on $machine..."
        ssh "$machine" 'powershell -c "Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -like \"*worker.py*\"} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"' 2>/dev/null || true
        sleep 2
        ssh "$machine" 'schtasks /run /tn "TennisGPUWorker" 2>nul' 2>/dev/null && ok "Worker restarted" || warn "Worker restart failed — check schtasks"
    fi
done

# ─── Summary ───────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════════════════"
echo "  Files checked:  $TOTAL"
echo "  In sync:        $((TOTAL - DRIFTED - SKIPPED))"
echo "  Synced:         $SYNCED"
echo "  Drifted:        $DRIFTED"
echo "  Failed:         $FAILED"
echo "  Skipped:        $SKIPPED"

if [ $FAILED -gt 0 ]; then
    echo ""
    fail "Some files failed to sync — check output above"
    exit 1
elif [ $DRIFTED -gt 0 ] && $VERIFY_ONLY; then
    echo ""
    warn "Drift detected — run without --verify to fix"
    exit 1
else
    echo ""
    ok "All files in sync"
fi
