"""Export an incremental result marker within the issue comment budget."""
import base64
import json
import lzma
import os
from pathlib import Path
import sys


def encode(data):
    return base64.b64encode(lzma.compress(
        json.dumps(data, separators=(",", ":"), allow_nan=False).encode(), preset=9
    )).decode()


def main(root, exit_code):
    doc = dict(schema=1, commit=os.environ.get("HOPPER_R10_COMMIT"),
               requested_pool=os.environ.get("HOPPER_REQUESTED_POOL"),
               exit_code=exit_code, run_root=str(root),
               remote_plaintext_absent=not (root / "private-input").exists())
    for name in ("fixture-source", "fixture-manifest", "environment-source", "setup-error"):
        path = root / name
        if path.is_file():
            doc[name.replace("-", "_")] = path.read_text().strip()
    for name, file in (("run", "result.json"), ("screen", "screen-vertical.json"),
                       ("gpu_topology", "topology.json")):
        path = root / file
        if path.is_file():
            doc[name] = json.loads(path.read_text())
    full = root / "export-summary.json"
    full.write_text(json.dumps(doc, indent=2))
    payload = encode(doc)
    if len(payload) > 46000:
        doc["persistent_full_result"] = str(full)
        for arm in doc.get("run", {}).get("arms", {}).values():
            if "trace" in arm:
                arm["trace"]["top"] = arm["trace"]["top"][:10]
        for row in doc.get("screen", {}).get("cases", {}).values():
            for trial in row.get("trials", ()):
                for key in ("ptx_sha256", "ttir_sha256", "local_declarations"):
                    trial.get("code", {}).pop(key, None)
        payload = encode(doc)
    if len(payload) > 46000:
        for gate in doc.get("run", {}).get("gates", {}).values():
            gate.pop("raw", None)
        payload = encode(doc)
    if len(payload) > 46000:
        doc.pop("screen", None)
        payload = encode(doc)
    if len(payload) > 46000:
        doc = {key: doc[key] for key in ("schema", "commit", "requested_pool", "exit_code",
                                        "run_root", "persistent_full_result") if key in doc}
        doc["export_error"] = "Result marker exceeded the size budget"
        payload = encode(doc)
    print("H200_HOPPER_R10_XZ", payload, flush=True)


if __name__ == "__main__":
    main(Path(os.environ["RUN_ROOT"]), int(sys.argv[1]))
