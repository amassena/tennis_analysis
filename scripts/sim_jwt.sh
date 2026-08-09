#!/usr/bin/env bash
# Refresh the local-dev JWT used by sim_run.sh.
#
# Requests a magic-link email, then waits for you to paste the consume URL
# (or token) from the email. Curls the consume endpoint, extracts the
# tennis_jwt cookie from the 302 Set-Cookie, writes it to /tmp/courtiq_jwt.txt,
# and verifies it returns 200 from the user gallery.
#
# Why not auto-read the email? The Gmail MCP can't fetch trashed messages and
# the account auto-trashes these; pasting the link is faster than fighting it.
#
# Usage:
#   scripts/sim_jwt.sh                      # interactive
#   scripts/sim_jwt.sh <consume-url-or-token>
set -euo pipefail

EMAIL="${COURTIQ_EMAIL:-amassena@gmail.com}"
HASH="${COURTIQ_HASH:-u_ae629639}"
BASE="https://tennis.playfullife.com"
OUT="/tmp/courtiq_jwt.txt"

arg="${1:-}"
if [ -z "$arg" ]; then
  echo "▶ requesting magic link for $EMAIL…"
  curl -s -X POST "$BASE/api/auth/magic/request" \
    -H "Content-Type: application/json" -d "{\"email\":\"$EMAIL\"}" >/dev/null
  echo "  check $EMAIL, then paste the sign-in URL (or just the token):"
  read -r arg
fi

# Accept either a full URL or a bare token.
if [[ "$arg" == http* ]]; then
  url="$arg"
else
  url="$BASE/api/auth/magic/consume?token=$arg"
fi

echo "▶ consuming…"
jwt="$(curl -s -i "$url" | tr -d '\r' | sed -n 's/^set-cookie: tennis_jwt=\([^;]*\).*/\1/p' | head -1)"
[ -z "$jwt" ] && { echo "✗ no tennis_jwt in response (link expired/used?)"; exit 1; }

printf '%s' "$jwt" > "$OUT"
code="$(curl -s -o /dev/null -w '%{http_code}' -H "Cookie: tennis_jwt=$jwt" "$BASE/u/$HASH/")"
if [ "$code" = "200" ]; then
  echo "✓ JWT saved to $OUT and verified (gallery 200)"
else
  echo "✗ JWT saved but gallery returned $code — may be wrong user/expired"
  exit 1
fi
