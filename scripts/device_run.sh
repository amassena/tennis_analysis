#!/usr/bin/env bash
# Build + install + launch on a physically connected iPhone — no TestFlight.
# Use this when you need to see REAL video playback (the Simulator can't render
# AVPlayer video; see sim_run.sh notes). Bypasses Apple's daily TestFlight
# upload cap entirely — it's a local dev-signed install straight to the device.
#
# Usage:
#   scripts/device_run.sh            # build + install + launch on the connected device
#   scripts/device_run.sh install    # install last build + launch (no rebuild)
#
# Requires: device connected & paired, a paid Apple Developer account, and the
# ASC API key at ~/.app-store-connect/AuthKey_<id>.p8 for automatic signing.
# Dev-signed builds expire (~1yr on a paid account); just re-run to refresh.
set -euo pipefail

BUNDLE="com.amassena.courtiq.CourtIQ"
SCHEME="CourtIQ"
PROJ="$(cd "$(dirname "$0")/.." && pwd)/ios/CourtIQ/CourtIQ.xcodeproj"
DERIVED="$HOME/Library/Developer/Xcode/DerivedData"
ASC_KEY_ID="${ASC_KEY_ID:-JF7L964MB8}"
ASC_ISSUER_ID="${ASC_ISSUER_ID:-61ee2f13-a9e3-43cd-a2bc-cf3e1dd2bdf4}"
ASC_KEY_PATH="${ASC_KEY_PATH:-$HOME/.app-store-connect/AuthKey_${ASC_KEY_ID}.p8}"
CMD="${1:-build}"

# Resolve the first connected physical device's UDID.
device_id() {
  xcrun xctrace list devices 2>&1 \
    | grep -iE "iphone|ipad" | grep -viE "simulator" \
    | grep -oE '[0-9A-F]{8}-[0-9A-F]{16}|[0-9A-F]{40}' | head -1
}

app_path() {
  find "$DERIVED" -path "*Debug-iphoneos/SwingLab.app" -maxdepth 6 -type d \
    -print0 2>/dev/null | xargs -0 ls -dt 2>/dev/null | head -1
}

DEV="$(device_id)"
[ -z "$DEV" ] && { echo "✗ no connected device found (plug in / pair the iPhone)"; exit 1; }
echo "▶ device: $DEV"

PBXPROJ="$(dirname "$PROJ")/CourtIQ.xcodeproj/project.pbxproj"
if [ "$CMD" = "build" ]; then
  # Auto-bump CURRENT_PROJECT_VERSION so the in-app version label climbs on
  # every device install — otherwise the number only moves on TestFlight
  # uploads and you can't tell whether the latest code is actually on the
  # phone. (Device builds aren't uploaded, so the exact number is free.)
  cur="$(grep -m1 -oE 'CURRENT_PROJECT_VERSION = [0-9]+' "$PBXPROJ" | grep -oE '[0-9]+')"
  if [ -n "$cur" ]; then
    nxt=$((cur + 1))
    sed -i '' "s/CURRENT_PROJECT_VERSION = ${cur};/CURRENT_PROJECT_VERSION = ${nxt};/g" "$PBXPROJ"
    echo "▶ build number ${cur} → ${nxt}"
  fi
  echo "▶ building (dev-signed) for device…"
  xcodebuild -project "$PROJ" -scheme "$SCHEME" \
    -destination "platform=iOS,id=$DEV" -configuration Debug \
    -allowProvisioningUpdates \
    -authenticationKeyID "$ASC_KEY_ID" \
    -authenticationKeyIssuerID "$ASC_ISSUER_ID" \
    -authenticationKeyPath "$ASC_KEY_PATH" \
    build 2>&1 | tail -3
fi

APP="$(app_path)"
[ -z "$APP" ] && { echo "✗ no device build found — run without 'install' to build first"; exit 1; }

echo "▶ installing…"
xcrun devicectl device install app --device "$DEV" "$APP" 2>&1 | grep -iE "App installed|error|fail" | head -3
echo "▶ launching (unlock the phone if it refuses)…"
xcrun devicectl device process launch --device "$DEV" "$BUNDLE" 2>&1 | grep -iE "Launched|error|Locked" | head -3
