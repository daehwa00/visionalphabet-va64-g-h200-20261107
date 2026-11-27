"""Receive bounded, append-only H200 epoch reports on loopback."""
import argparse
import hmac
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY = 8192
RUN_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")


def reject_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def create_server(directory: Path, token: str, port: int = 8765) -> ThreadingHTTPServer:
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token):
        raise ValueError("Expected a URL-safe token of at least 32 characters")
    directory.mkdir(parents=True, exist_ok=True)
    report_path = f"/report/{token}".encode()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        timeout = 5

        def log_message(self, *_args):
            # The default access log would include the report token.
            pass

        def reply(self, status, payload):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"ok": True})
            else:
                self.reply(404, {"ok": False})

        def do_POST(self):
            if not hmac.compare_digest(self.path.encode(), report_path):
                self.reply(404, {"ok": False})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
                self.reply(400, {"ok": False})
                return
            if not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
                self.reply(400, {"ok": False})
                return
            size = int(lengths[0])
            if size > MAX_BODY:
                self.reply(413, {"ok": False})
                return
            try:
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("Incomplete request")
                payload = json.loads(body.decode("utf-8"), parse_constant=reject_constant)
                run_id = payload.get("run_id") if isinstance(payload, dict) else None
                if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
                    raise ValueError("Invalid run_id")
                line = json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
            except (ValueError, RecursionError, TimeoutError):
                self.reply(400, {"ok": False})
                return
            with lock:
                path = directory / f"{run_id}.jsonl"
                try:
                    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                    with os.fdopen(fd, "a", encoding="utf-8") as stream:
                        stream.write(line)
                except OSError:
                    self.reply(500, {"ok": False})
                    return
                stop = (directory / "STOP_H200").is_file() or (
                    directory / f"STOP_{run_id}"
                ).is_file()
            self.reply(200, {"ok": True, "stop": stop})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/h200_live"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with create_server(args.directory, os.environ["H200_TELEMETRY_TOKEN"], args.port) as server:
        server.serve_forever()
