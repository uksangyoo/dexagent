"""Loopback-only review server with read-only live progress from local run logs."""

import argparse
import json
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.review.export_data import DIST, progress


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route not in {"/api/progress", "/runtime.js"}:
            return super().do_GET()
        if route == "/runtime.js":
            data, kind = b"window.REVIEW_LIVE = true;\n", "text/javascript"
        else:
            data, kind = json.dumps(progress()).encode(), "application/json"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_head(self):
        self.range_remaining = None
        path = Path(self.translate_path(self.path)).resolve()
        if path.suffix not in {".mp4", ".webm"} or not path.is_file() or not path.is_relative_to(DIST):
            return super().send_head()
        size = path.stat().st_size
        start, end = 0, size - 1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match or not any(match.groups()):
                self.send_error(416)
                return None
            first, last = match.groups()
            start = int(first) if first else max(0, size - int(last))
            end = min(size - 1, int(last)) if first and last else size - 1
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end-start+1))
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        source = path.open("rb")
        source.seek(start)
        self.range_remaining = end-start+1
        return source

    def copyfile(self, source, outputfile):
        remaining = self.range_remaining
        try:
            if remaining is None:
                return super().copyfile(source, outputfile)
            while remaining:
                chunk = source.read(min(65536, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Changing cameras can cancel an in-flight video request.

    def log_message(self, format, *args):
        pass  # Keep preview requests from filling the long-running console pipe.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"Review: http://127.0.0.1:{args.port} (loopback only)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler).serve_forever()
