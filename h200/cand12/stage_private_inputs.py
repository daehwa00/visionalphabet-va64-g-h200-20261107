"""Download only the three pinned benchmark fixtures from the experiment relay."""
import concurrent.futures
import base64
import threading
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

import requests

BASE = "https://va-cand12-matched-h200-20261118.gpupulse-monitor.workers.dev/va64g"


def main(target):
    target.mkdir(mode=0o700)
    token = secrets.token_hex(32)
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    identity = {"pod":socket.gethostname(), "code_sha":code,
                "token_hash":hashlib.sha256(token.encode()).hexdigest()}
    for _ in range(60):
        response = requests.post(BASE+"/enroll", json=identity, timeout=20)
        if response.ok:
            break
        if response.status_code not in (403,429,502,503):
            response.raise_for_status()
        time.sleep(5)
    response.raise_for_status()
    session = response.json()["session_id"]
    headers = {"Authorization":"Bearer "+token,"X-Session-ID":session, "Accept":"application/json","Accept-Encoding":"identity"}
    import io
    import zipfile
    from urllib.parse import urlparse
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    meta = requests.get(BASE+"/artifact/handoff/latest", headers=headers, timeout=20).json()
    assert meta["verified"] and meta["chunks"] == 1
    record=requests.get(BASE+"/artifact/handoff/"+meta["sha"]+"/0", headers=headers, timeout=20)
    record.raise_for_status()
    blob=base64.b64decode(record.json()["base64"], validate=True)
    assert hashlib.sha256(blob).hexdigest()==meta["sha"]
    spec=json.loads(blob)
    url=spec["url"]
    assert urlparse(url).scheme=="https" and urlparse(url).hostname=="release-assets.githubusercontent.com"
    ciphertext = None
    for attempt in range(4):
        try:
            with requests.get(url, timeout=(15,120), allow_redirects=False) as reply:
                if reply.status_code==200 and len(reply.content)==spec["bytes"]:
                    ciphertext=reply.content
                    break
        except requests.RequestException:
            pass
        time.sleep(2**attempt)
    if ciphertext is None:
        raise RuntimeError("Encrypted fixture download failed")
    assert hashlib.sha256(ciphertext).hexdigest()==spec["cipher_sha256"]
    clear=AESGCM(bytes.fromhex(spec["key"])).decrypt(bytes.fromhex(spec["nonce"]),ciphertext,b"VA-IN100-B128-v1")
    verified={}
    expected=dict(line.split()[::-1] for line in Path("matched-inputs.sha256").read_text().splitlines())
    with zipfile.ZipFile(io.BytesIO(clear)) as z:
        assert sorted(z.namelist()) == sorted(expected)
        for name,h in expected.items():
            data=z.read(name)
            assert hashlib.sha256(data).hexdigest()==h
            (target/name).write_bytes(data)
            verified[name]=h
            print("VERIFIED_PRIVATE_INPUT",name,h,flush=True)
    ack = requests.post(BASE+"/received", headers=headers, json=verified, timeout=30)
    ack.raise_for_status()
    assert ack.json()["closed"]


if __name__=="__main__":
    main(Path(sys.argv[1]))
