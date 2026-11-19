"""Export numeric results on success, partial completion, or shell failure."""
import base64
import json
import lzma
import os
from pathlib import Path
import sys

root = Path(os.environ["RUN_ROOT"])
summary = dict(schema=1, commit=os.environ.get("HOPPER_R7_COMMIT"), exit_code=int(sys.argv[1]))
for name in ("fixture-source", "environment-source"):
    path = root / name
    if path.is_file():
        summary[name.replace("-", "_")] = path.read_text().strip()
for name in ("run", "screen"):
    path = root / f"{name}.json"
    if path.is_file():
        summary[name] = json.loads(path.read_text())
summary["run_root"] = str(root)
enc = base64.b64encode(lzma.compress(json.dumps(summary, separators=(",", ":"), allow_nan=False).encode())).decode()
(root / "export-summary.json").write_text(json.dumps(summary, indent=2))
print("H200_HOPPER_R7_XZ", enc, flush=True)
