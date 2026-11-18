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

BASE = "https://va-effcli-h200-20261118.gpupulse-monitor.workers.dev/va64g"


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
    # Verify each ranged response. Never log signed URLs or release credentials.
    total=spec["bytes"]
    block=8*1024*1024
    def fetch(start):
        end=min(start+block,total)-1
        for attempt in range(5):
            try:
                headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"}
                with requests.get(url,headers=headers,timeout=(15,90),allow_redirects=False) as reply:
                    if (reply.status_code==206 and reply.headers.get("Content-Range")==f"bytes {start}-{end}/{total}"
                            and len(reply.content)==end-start+1):
                        return reply.content
            except requests.RequestException:
                pass
            time.sleep(2**attempt)
        raise RuntimeError("Encrypted fixture range download failed") from None
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        ciphertext=b"".join(pool.map(fetch,range(0,total,block)))
    assert hashlib.sha256(ciphertext).hexdigest()==spec["cipher_sha256"]
    clear=AESGCM(bytes.fromhex(spec["key"])).decrypt(bytes.fromhex(spec["nonce"]),ciphertext,b"VA-IN100-B128-v1")
    verified={}
    expected=dict(line.split()[::-1] for line in Path("effective_cli/inputs.sha256").read_text().splitlines())
    if "fixed_control_sha256" in spec:
        import re
        control_digest = spec["fixed_control_sha256"]
        assert isinstance(control_digest, str) and re.fullmatch("[0-9a-f]{64}", control_digest)
        expected["previous-fixed.json"] = control_digest
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
