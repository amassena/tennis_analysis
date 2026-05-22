#!/usr/bin/env python3
"""
App Store Connect API automation for Tennis Uploader.

One script, four subcommands. Reads credentials from environment:
    ASC_KEY_ID      — 10-char API Key ID from App Store Connect
    ASC_ISSUER_ID   — UUID Issuer ID (shared across keys for your team)
    ASC_KEY_PATH    — path to the .p8 private key file you downloaded
    ASC_BUNDLE_ID   — bundle identifier for the app (default: com.amassena.courtiq.CourtIQ)
    ASC_APP_NAME    — app store display name (default: "Tennis Uploader")
    ASC_SKU         — internal SKU (default: tennis-uploader-001)

Subcommands:
    list-apps                  → list apps already on your team
    register-bundle-id         → register the Bundle ID at developer.apple.com (idempotent)
    create-app                 → create the App Store Connect app entry (idempotent)
    archive-and-upload PATH    → xcodebuild archive, export IPA, upload via altool
    add-tester EMAIL [NAME]    → add an internal TestFlight tester
    full                       → register-bundle-id + create-app, archive-and-upload, ready for TestFlight

JWT auth per Apple's spec:
    https://developer.apple.com/documentation/appstoreconnectapi/generating_tokens_for_api_requests
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import jwt
except ImportError:
    print("PyJWT not installed. Run: pip install PyJWT cryptography requests", file=sys.stderr)
    sys.exit(1)

import requests

API_BASE = "https://api.appstoreconnect.apple.com"


def env_or_die(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if not val:
        print(f"ERROR: {name} not set", file=sys.stderr)
        sys.exit(2)
    return val


def make_token() -> str:
    key_id = env_or_die("ASC_KEY_ID")
    issuer_id = env_or_die("ASC_ISSUER_ID")
    key_path = Path(env_or_die("ASC_KEY_PATH")).expanduser()
    if not key_path.exists():
        print(f"ERROR: {key_path} not found", file=sys.stderr)
        sys.exit(2)
    private_key = key_path.read_text()
    now = int(time.time())
    payload = {
        "iss": issuer_id,
        "iat": now,
        "exp": now + 20 * 60,  # max 20 min per Apple's policy
        "aud": "appstoreconnect-v1",
    }
    headers = {"kid": key_id, "typ": "JWT"}
    return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)


def api(method: str, path: str, **kwargs):
    """Send an authenticated request to App Store Connect API."""
    token = make_token()
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    headers.setdefault("Content-Type", "application/json")
    url = path if path.startswith("http") else f"{API_BASE}{path}"
    resp = requests.request(method, url, headers=headers, **kwargs)
    return resp


def cmd_list_apps(args):
    resp = api("GET", "/v1/apps?limit=50")
    if resp.status_code != 200:
        print(f"FAILED: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    body = resp.json()
    apps = body.get("data", [])
    if not apps:
        print("No apps on this team yet.")
        return
    print(f"{len(apps)} app(s) on the team:")
    for app in apps:
        attrs = app["attributes"]
        print(f"  id={app['id']}  bundleId={attrs.get('bundleId')}  name={attrs.get('name')}  sku={attrs.get('sku')}")


def cmd_register_bundle_id(args):
    bundle_id = env_or_die("ASC_BUNDLE_ID", "com.amassena.courtiq.CourtIQ")
    name = "Tennis Uploader"

    # Check if it already exists
    resp = api("GET", f"/v1/bundleIds?filter[identifier]={bundle_id}")
    if resp.status_code == 200:
        existing = resp.json().get("data", [])
        if existing:
            attrs = existing[0]["attributes"]
            print(f"Bundle ID already registered: id={existing[0]['id']} platform={attrs.get('platform')}")
            return existing[0]["id"]

    # Create it
    payload = {
        "data": {
            "type": "bundleIds",
            "attributes": {
                "identifier": bundle_id,
                "name": name,
                "platform": "IOS",
            }
        }
    }
    resp = api("POST", "/v1/bundleIds", data=json.dumps(payload))
    if resp.status_code not in (200, 201):
        print(f"FAILED: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    bid = resp.json()["data"]["id"]
    print(f"Registered bundle ID: {bundle_id} (id={bid})")
    return bid


def cmd_create_app(args):
    bundle_id = env_or_die("ASC_BUNDLE_ID", "com.amassena.courtiq.CourtIQ")
    app_name = env_or_die("ASC_APP_NAME", "Tennis Uploader")
    sku = env_or_die("ASC_SKU", "tennis-uploader-001")
    primary_locale = "en-US"

    # Check if app already exists for this bundle id
    resp = api("GET", f"/v1/apps?filter[bundleId]={bundle_id}")
    if resp.status_code == 200:
        existing = resp.json().get("data", [])
        if existing:
            print(f"App already exists: id={existing[0]['id']} name={existing[0]['attributes'].get('name')}")
            return existing[0]["id"]

    # Need the bundle id's internal record id
    resp = api("GET", f"/v1/bundleIds?filter[identifier]={bundle_id}")
    bundles = resp.json().get("data", [])
    if not bundles:
        print(f"ERROR: bundle id {bundle_id} not registered. Run register-bundle-id first.", file=sys.stderr)
        sys.exit(1)
    bundle_record_id = bundles[0]["id"]

    payload = {
        "data": {
            "type": "apps",
            "attributes": {
                "bundleId": bundle_id,
                "name": app_name,
                "primaryLocale": primary_locale,
                "sku": sku,
            },
            "relationships": {
                "bundleId": {
                    "data": {"type": "bundleIds", "id": bundle_record_id}
                }
            }
        }
    }
    resp = api("POST", "/v1/apps", data=json.dumps(payload))
    if resp.status_code not in (200, 201):
        print(f"FAILED: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    app_id = resp.json()["data"]["id"]
    print(f"Created app: name={app_name} bundleId={bundle_id} sku={sku} id={app_id}")
    return app_id


def cmd_archive_and_upload(args):
    project_path = "ios/CourtIQ/CourtIQ.xcodeproj"
    scheme = "CourtIQ"
    build_dir = Path("build/asc")
    build_dir.mkdir(parents=True, exist_ok=True)
    archive_path = build_dir / "CourtIQ.xcarchive"
    ipa_dir = build_dir / "ipa"
    export_opts_path = build_dir / "ExportOptions.plist"

    print(f"[1/3] xcodebuild archive → {archive_path}")
    result = subprocess.run([
        "xcodebuild",
        "-project", project_path,
        "-scheme", scheme,
        "-sdk", "iphoneos",
        "-configuration", "Release",
        "-destination", "generic/platform=iOS",
        "archive",
        f"-archivePath={archive_path}",
        "CODE_SIGN_STYLE=Automatic",
    ], capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: xcodebuild archive exited {result.returncode}", file=sys.stderr)
        sys.exit(1)

    # Export IPA
    export_opts_path.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>method</key>
    <string>app-store-connect</string>
    <key>signingStyle</key>
    <string>automatic</string>
    <key>uploadSymbols</key>
    <true/>
    <key>destination</key>
    <string>export</string>
</dict>
</plist>
""")
    print(f"[2/3] xcodebuild -exportArchive → {ipa_dir}")
    result = subprocess.run([
        "xcodebuild",
        "-exportArchive",
        f"-archivePath={archive_path}",
        f"-exportPath={ipa_dir}",
        f"-exportOptionsPlist={export_opts_path}",
    ], capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: -exportArchive exited {result.returncode}", file=sys.stderr)
        sys.exit(1)

    ipa_files = list(ipa_dir.glob("*.ipa"))
    if not ipa_files:
        print(f"ERROR: no .ipa produced in {ipa_dir}", file=sys.stderr)
        sys.exit(1)
    ipa_path = ipa_files[0]

    print(f"[3/3] xcrun altool --upload-app → App Store Connect")
    key_id = env_or_die("ASC_KEY_ID")
    issuer_id = env_or_die("ASC_ISSUER_ID")
    # altool reads .p8 files from ~/.appstoreconnect/private_keys/AuthKey_<KEYID>.p8
    # OR specific environment-via-flags; we pre-stage the file.
    altool_key_dir = Path.home() / ".appstoreconnect" / "private_keys"
    altool_key_dir.mkdir(parents=True, exist_ok=True)
    src = Path(env_or_die("ASC_KEY_PATH")).expanduser()
    dst = altool_key_dir / f"AuthKey_{key_id}.p8"
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        dst.write_bytes(src.read_bytes())
        dst.chmod(0o600)

    result = subprocess.run([
        "xcrun", "altool",
        "--upload-app",
        "-f", str(ipa_path),
        "-t", "ios",
        "--apiKey", key_id,
        "--apiIssuer", issuer_id,
    ], capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: altool --upload-app exited {result.returncode}", file=sys.stderr)
        sys.exit(1)
    print("✅ Upload complete. Build will appear in App Store Connect TestFlight in 5-15 min.")


def cmd_add_tester(args):
    if not args.email:
        print("ERROR: pass --email <addr> [--name 'First Last']", file=sys.stderr)
        sys.exit(2)
    # Find the app
    bundle_id = env_or_die("ASC_BUNDLE_ID", "com.amassena.courtiq.CourtIQ")
    resp = api("GET", f"/v1/apps?filter[bundleId]={bundle_id}")
    apps = resp.json().get("data", [])
    if not apps:
        print("App not found — run create-app first", file=sys.stderr)
        sys.exit(1)
    app_id = apps[0]["id"]

    first_name, last_name = ("", "")
    if args.name:
        parts = args.name.strip().split()
        first_name = parts[0]
        last_name = " ".join(parts[1:]) or ""

    payload = {
        "data": {
            "type": "betaTesters",
            "attributes": {
                "email": args.email,
                "firstName": first_name or None,
                "lastName": last_name or None,
            },
            "relationships": {
                "apps": {
                    "data": [{"type": "apps", "id": app_id}]
                }
            }
        }
    }
    resp = api("POST", "/v1/betaTesters", data=json.dumps(payload))
    if resp.status_code in (200, 201):
        print(f"Invited {args.email} to TestFlight for {bundle_id}")
    else:
        print(f"FAILED: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_full(args):
    print("=== Full automation: register bundle id → create app → archive + upload ===")
    cmd_register_bundle_id(args)
    cmd_create_app(args)
    cmd_archive_and_upload(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-apps")
    sub.add_parser("register-bundle-id")
    sub.add_parser("create-app")
    sub.add_parser("archive-and-upload")
    p_add = sub.add_parser("add-tester")
    p_add.add_argument("--email", required=True)
    p_add.add_argument("--name")
    sub.add_parser("full")

    args = parser.parse_args()
    handlers = {
        "list-apps": cmd_list_apps,
        "register-bundle-id": cmd_register_bundle_id,
        "create-app": cmd_create_app,
        "archive-and-upload": cmd_archive_and_upload,
        "add-tester": cmd_add_tester,
        "full": cmd_full,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
