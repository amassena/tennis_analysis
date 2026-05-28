#!/usr/bin/env bash
# Local iOS dev loop — build to Simulator, inject auth, launch, screenshot.
# Replaces the TestFlight round-trip for everything except real video-frame
# decode (the Simulator can't render AVPlayer video — a documented Apple bug,
# see developer.apple.com/forums/thread/727288 — but UI, layout, orientation,
# chips, navigation, auth, and playback *logic* via the time readout all work).
#
# Usage:
#   scripts/sim_run.sh build     # build + install + launch + screenshot
#   scripts/sim_run.sh run       # install + launch + screenshot (no rebuild)
#   scripts/sim_run.sh shot      # just screenshot the booted sim
#
# Auth: reads a JWT from /tmp/courtiq_jwt.txt. Refresh it with:
#   scripts/sim_jwt.sh
#
# Requires: a JWT that returns 200 from https://tennis.playfullife.com/u/<hash>/
set -euo pipefail

SIM_NAME="${SIM_NAME:-iPhone 17}"
BUNDLE="com.amassena.courtiq.CourtIQ"
SCHEME="CourtIQ"
PROJ="$(cd "$(dirname "$0")/.." && pwd)/ios/CourtIQ/CourtIQ.xcodeproj"
DERIVED="$HOME/Library/Developer/Xcode/DerivedData"
JWT_FILE="${JWT_FILE:-/tmp/courtiq_jwt.txt}"
CMD="${1:-run}"

app_path() {
  # Find the most recent Debug-iphonesimulator build of the app.
  find "$DERIVED" -path "*Debug-iphonesimulator/SwingLab.app" -maxdepth 6 -type d \
    -print0 2>/dev/null | xargs -0 ls -dt 2>/dev/null | head -1
}

boot_sim() {
  xcrun simctl boot "$SIM_NAME" 2>/dev/null || true
  open -a Simulator 2>/dev/null || true
  sleep 3
}

do_build() {
  echo "▶ building for $SIM_NAME…"
  xcodebuild -project "$PROJ" -scheme "$SCHEME" -sdk iphonesimulator \
    -destination "platform=iOS Simulator,name=$SIM_NAME" -configuration Debug build \
    2>&1 | tail -2
}

do_launch() {
  local app jwt
  app="$(app_path)"
  [ -z "$app" ] && { echo "✗ no built app found — run with 'build' first"; exit 1; }
  jwt="$(tr -d '\n' < "$JWT_FILE" 2>/dev/null || true)"
  [ -z "$jwt" ] && { echo "✗ no JWT at $JWT_FILE — run scripts/sim_jwt.sh"; exit 1; }
  xcrun simctl terminate booted "$BUNDLE" 2>/dev/null || true
  xcrun simctl install booted "$app"
  echo "▶ launching with injected auth…"
  SIMCTL_CHILD_COURTIQ_DEBUG_JWT="$jwt" xcrun simctl launch booted "$BUNDLE"
  sleep 5
}

do_shot() {
  local out="${1:-/tmp/sim_shot.png}"
  xcrun simctl io booted screenshot "$out" >/dev/null 2>&1
  echo "📸 $out"
}

case "$CMD" in
  build) boot_sim; do_build; do_launch; do_shot ;;
  run)   boot_sim; do_launch; do_shot ;;
  shot)  do_shot "${2:-/tmp/sim_shot.png}" ;;
  *) echo "usage: $0 {build|run|shot}"; exit 2 ;;
esac
