"""Enroll this pinned pod, decrypt the sealed benchmark inputs, and return their path."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.parse import urlparse
import zipfile

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import requests

BASE = "https://va-hopper-r11-input-20261120.gpupulse-monitor.workers.dev/va64g"
AAD = b"VA-HOPPER-R11-INPUT-v1"
MANIFEST = Path(__file__).with_name("inputs_r11.sha256")


def pinned_hashes():
    return dict(line.split()[::-1] for line in MANIFEST.read_text().splitlines())


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def relay_request(method, path, **kwargs):
    response = requests.request(method, BASE + path, timeout=20, allow_redirects=False, **kwargs)
    if not response.ok:
        raise RuntimeError(f"Relay {path} HTTP {response.status_code}")
    return response.json()


def enroll(token):
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    identity = dict(pod=socket.gethostname(), code_sha=code,
                    token_hash=hashlib.sha256(token.encode()).hexdigest())
    for attempt in range(90):
        try:
            response = requests.post(BASE + "/enroll", json=identity, timeout=10, allow_redirects=False)
        except requests.RequestException:
            response = None
        if response is not None:
            if response.ok:
                return {"Authorization": f"Bearer {token}", "X-Session-ID": response.json()["session_id"]}
            if response.status_code not in (403, 404, 429, 502, 503, 504):
                raise RuntimeError(f"Relay enrollment HTTP {response.status_code}")
        if attempt < 89:
            time.sleep(5)
    raise RuntimeError("Timed out waiting for the pinned pod to enroll")


def ciphertext_bytes(spec):
    url = spec["url"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "release-assets.githubusercontent.com":
        raise RuntimeError("Unexpected encrypted asset origin")
    total = spec["bytes"]
    if not isinstance(total, int) or not 0 < total < 1024**3:
        raise RuntimeError("Invalid encrypted asset size")
    block = 8 << 20

    def fetch(start):
        end = min(start + block, total) - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        for attempt in range(5):
            try:
                with requests.get(url, headers=headers, timeout=(15, 90), allow_redirects=False) as reply:
                    if (reply.status_code == 206 and reply.headers.get("Content-Range") == f"bytes {start}-{end}/{total}"
                            and len(reply.content) == end - start + 1):
                        return reply.content
            except requests.RequestException:
                pass
            time.sleep(2**attempt)
        raise RuntimeError("Encrypted fixture range download failed") from None

    with ThreadPoolExecutor(max_workers=4) as pool:
        ciphertext = b"".join(pool.map(fetch, range(0, total, block)))
    if hashlib.sha256(ciphertext).hexdigest() != spec["cipher_sha256"]:
        raise RuntimeError("Encrypted fixture SHA-256 mismatch")
    return ciphertext


def verify_and_unpack(clear, target, expected):
    with zipfile.ZipFile(io.BytesIO(clear)) as archive:
        if sorted(archive.namelist()) != sorted(expected):
            raise RuntimeError("Unexpected sealed fixture members")
        for name, sha in expected.items():
            with archive.open(name) as source, (target / name).open("xb") as output:
                shutil.copyfileobj(source, output)
            if digest(target / name) != sha:
                raise RuntimeError(f"Fixture SHA-256 mismatch for {name}")


def stage(target):
    expected = pinned_hashes()
    headers = enroll(secrets.token_hex(32))
    artifact = "/artifact/handoff"
    meta = relay_request("GET", artifact + "/latest", headers=headers)
    if not meta or not meta["verified"] or meta["chunks"] != 1:
        raise RuntimeError("Missing verified sealed handoff")
    record = relay_request("GET", f"{artifact}/{meta['sha']}/0", headers=headers)
    blob = base64.b64decode(record["base64"], validate=True)
    if hashlib.sha256(blob).hexdigest() != meta["sha"]:
        raise RuntimeError("Sealed handoff SHA-256 mismatch")
    spec = json.loads(blob)
    key, nonce = bytes.fromhex(spec["key"]), bytes.fromhex(spec["nonce"])
    if len(key) != 32 or len(nonce) != 12:
        raise RuntimeError("Invalid AES-256-GCM parameters")
    clear = AESGCM(key).decrypt(nonce, ciphertext_bytes(spec), AAD)
    with TemporaryDirectory(dir=target.parent, prefix="relay-input-") as tmp:
        verified = Path(tmp) / "verified"
        verified.mkdir(mode=0o700)
        verify_and_unpack(clear, verified, expected)
        verified.rename(target)
    if not relay_request("POST", "/received", headers=headers, json=expected)["closed"]:
        raise RuntimeError("Relay did not close after the verified receipt")
    return target.resolve()


if __name__ == "__main__":
    try:
        print(stage(Path(sys.argv[1])))
    except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException) as exc:
        if type(exc) is RuntimeError:
            print(str(exc), file=sys.stderr)
        else:
            print(f"Fixture staging failed ({type(exc).__name__})", file=sys.stderr)
        sys.exit(1)
