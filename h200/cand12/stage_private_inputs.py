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
    files = {"checkpoint.pt":"checkpoint","images.pt":"images","4090.validation.pt":"validation"}
    verified = {}
    pace_lock=threading.Lock()
    next_read=0.0
    for line in Path("matched-inputs.sha256").read_text().splitlines():
        digest,name=line.split()
        job=files[name]
        url=f"{BASE}/artifact/{job}"
        meta=requests.get(url+"/latest",headers=headers,timeout=30)
        meta.raise_for_status()
        meta=meta.json()
        assert meta["sha"]==digest and meta["bytes"]>0 and meta["verified"]
        def read_chunk(part):
            nonlocal next_read
            for attempt in range(5):
                try:
                    with pace_lock:
                        time.sleep(max(0,next_read-time.monotonic()))
                        next_read=time.monotonic()+.125
                    res=requests.get(f"{url}/{digest}/{part}",headers=headers,timeout=40)
                    res.raise_for_status()
                    chunk=base64.b64decode(res.json()["base64"],validate=True)
                    assert len(chunk)==min(meta["chunk_bytes"],meta["bytes"]-part*meta["chunk_bytes"])
                    return chunk
                except requests.RequestException:
                    if attempt==4:raise
                    time.sleep(2**attempt)
        p=target/name
        h=hashlib.sha256()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool,p.open("xb") as f:
            for blob in pool.map(read_chunk,range(meta["chunks"])):
                f.write(blob);h.update(blob)
        assert h.hexdigest()==digest and p.stat().st_size==meta["bytes"]
        verified[name] = digest
        print("VERIFIED_PRIVATE_INPUT",name,digest,flush=True)

    ack = requests.post(BASE+"/received", headers=headers, json=verified, timeout=30)
    ack.raise_for_status()
    assert ack.json()["closed"]


if __name__=="__main__":
    main(Path(sys.argv[1]))
