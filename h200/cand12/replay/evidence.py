"""Emit compact diagnostics and verify timing/validation scope before any record is promoted."""
import base64
import json
import os
from pathlib import Path
import sys
import zlib

def main():
    root=Path(os.environ["RUN_ROOT"])
    if sys.argv[1]=="micro":
        d=json.loads((root/"micro/result.json").read_text())
        print("REPLAY_RESULT_ZLIB",base64.b64encode(zlib.compress(json.dumps(d).encode(),9)).decode())
        return
    for p in sorted((root/"results").glob("*.json")):
        if ".trace." in p.name or ".launches." in p.name:continue
        r=json.loads(p.read_text())
        if p.stem.startswith("actual-"):
            g=r.get("gradient_checks",{})
            print("ACTUAL_GATE",p.stem,json.dumps(dict(status=r["status"],finite=r["validation_finite"],missing=r["missing_gradients"],outputs=r.get("outputs"),params={k:dict(n=len(v),fail=sum(not t["passed"] for t in v.values()),absolute=max(t["max_abs"] for t in v.values()),rel=max(t["relative_l2"] for t in v.values())) for k,v in g.items()})))
        elif "-p" in p.stem or p.stem in ("screen-combo","screen-mixed","reference"):
            keys=("status","median_ms","images_s","peak_allocated_mib","event_ms","validation_finite","missing_gradients","final_finite","input_sha256","checkpoint_sha256","first_loss","final_loss")
            print("TRAIN_STEP",p.stem,json.dumps({k:r.get(k) for k in keys}))

if __name__=="__main__":main()
