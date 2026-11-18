"""Publish only measurement logs, keeping the issue comfortably under the reporting cap."""
import base64, json, zlib
from pathlib import Path
import sys

root=Path(sys.argv[1]);logs={}
for name in ('measurements.txt','training-summary.txt'):
    p=root/name
    if p.exists():logs[name]=p.read_text()
for p in (root/'results').glob('actual-*.json'):
    d=json.loads(p.read_text())
    logs[p.stem]=dict(status=d['status'],checks={k:dict(count=len(v),failed=sum(not c['passed'] for c in v.values())) for k,v in d.get('gradient_checks',{}).items()})
print("POST714_DATA_ZLIB",base64.b64encode(zlib.compress(json.dumps(logs).encode(),9)).decode())
