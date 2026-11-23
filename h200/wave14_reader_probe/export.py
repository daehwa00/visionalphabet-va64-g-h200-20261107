"""Preserve raw H200 files and emit a self-contained result for the GitHub runner."""

from __future__ import annotations

import base64
import json
import lzma
import sys
from pathlib import Path


def encode(result: dict) -> str:
    data = json.dumps(result, separators=(",", ":")).encode()
    return base64.b64encode(lzma.compress(data, preset=9)).decode()


def main(root: Path, status: int) -> None:
    result: dict = {"runner_exit_code": status}
    for path in sorted(root.glob("*.json")):
        result[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    out = root / "raw-result.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    payload = encode(result)
    if len(payload) > 38_000:
        # Retain medians and parity in the issue result. The raw file keeps every sample.
        for row in result.get("isolated", {}).get("shapes", {}).values():
            for ops in row.values():
                for measurements in ops.values():
                    measurements.pop("samples_ms", None)
        result["issue_result_without_samples"] = True
        payload = encode(result)
    sys.stdout.write(f"WAVE14_READER_XZ {payload}\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
