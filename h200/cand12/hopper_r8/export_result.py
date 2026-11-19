"""Publish bounded numeric JSON even when a setup or runtime step fails."""
import base64
import json
import lzma
import os
from pathlib import Path
import sys

root = Path(os.environ["RUN_ROOT"])
doc = dict(schema=1, commit=os.environ.get("HOPPER_R8_COMMIT"),
           requested_pool=os.environ.get("HOPPER_REQUESTED_POOL"),
           exit_code=int(sys.argv[1]), run_root=str(root))
for name in ("fixture-source", "fixture-manifest", "environment-source", "setup-error"):
    path = root / name
    if path.is_file():
        doc[name.replace("-", "_")] = path.read_text().strip()
if (root / "run.json").is_file():
    doc["run"] = json.loads((root / "run.json").read_text())
doc["screens"] = {path.stem.removeprefix("screen-"): json.loads(path.read_text())
                  for path in sorted(root.glob("screen-*.json"))}
if (root / "topology.json").is_file():
    doc["gpu_topology"] = json.loads((root / "topology.json").read_text())
if doc.get("run", {}).get("gpu_topology"):
    doc["gpu_topology"] = doc["run"]["gpu_topology"]


def encode(data):
    return base64.b64encode(lzma.compress(
        json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
    )).decode()


(root / "export-summary.json").write_text(json.dumps(doc, indent=2))
payload = encode(doc)
if len(payload) > 48000:
    # Repeated code hashes and proof paths are retained in the persistent copy.
    # Keep all kernel configurations, timings, numeric results, and error rows.
    doc["persistent_full_result"] = str(root / "export-summary.json")
    for screen in doc["screens"].values():
        for case in screen["cases"].values():
            all_codes = [case.get("baseline_code", {})]
            all_codes.extend(row.get("code", {}) for row in case.get("trials", ()))
            for code in all_codes:
                for key in ("ptx_sha256", "ttir_sha256", "local_declarations"):
                    code.pop(key, None)
    payload = encode(doc)
if len(payload) > 48000:
    raise RuntimeError(f"Numeric result exceeds report budget: {len(payload)} bytes")
print("H200_HOPPER_R8_XZ", payload, flush=True)
