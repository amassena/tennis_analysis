#!/usr/bin/env python3
"""Simple HTTP file server with range request support for video streaming.

Run on Windows to serve preprocessed videos to Mac review player.

Usage:
    python scripts/serve_videos.py [--port 9000] [--dir preprocessed]
"""

import argparse
import os
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler that supports Range requests for video seeking."""

    def do_GET(self):
        # Get the file path
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            super().do_GET()
            return

        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")

        if range_header:
            # Parse range
            range_spec = range_header.replace("bytes=", "")
            parts = range_spec.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            ctype = self.guess_type(path)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            ctype = self.guess_type(path)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with open(path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.end_headers()

    def log_message(self, format, *args):
        # Quieter logging — only show file requests, not range probes
        msg = format % args
        if "206" not in msg and "OPTIONS" not in msg:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Serve video files over HTTP")
    parser.add_argument("--port", "-p", type=int, default=9000)
    parser.add_argument("--dir", "-d", default="preprocessed",
                        help="Directory to serve (default: preprocessed)")
    args = parser.parse_args()

    serve_dir = os.path.abspath(args.dir)
    if not os.path.isdir(serve_dir):
        print(f"Error: directory not found: {serve_dir}")
        sys.exit(1)

    os.chdir(serve_dir)
    handler = RangeRequestHandler
    httpd = HTTPServer(("0.0.0.0", args.port), handler)
    print(f"Serving {serve_dir}")
    print(f"Listening on http://0.0.0.0:{args.port}")
    print(f"Press Ctrl+C to stop.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
