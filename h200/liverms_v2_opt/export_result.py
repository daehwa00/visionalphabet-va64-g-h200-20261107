"""Export benchmark results even when the verifier exits early."""

from __future__ import annotations

import base64
import json
import lzma
import os
import sys
from pathlib import Path

MARKER = "H200_LIVERMS_V2_OPT_XZ"
MAX_PAYLOAD = 44_000


def collect(root: Path, exit_code: int) -> dict:
    result = {"runner_exit_code": exit_code, "arms": {}, "infer": {}}
    paths = [*root.glob("*.json"), *root.glob("arms/*.json"), *root.glob("infer/*.json")]
    for path in sorted(paths):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            row = {"status": "unreadable", "error": str(exc)}
        parent = result if path.parent == root else result[path.parent.name]
        parent[path.stem] = row
    return result


def encode(result: dict) -> str:
    raw = json.dumps(result, separators=(",", ":")).encode()
    return base64.b64encode(lzma.compress(raw, preset=9)).decode("ascii")


def portable_payload(result: dict) -> str:
    payload = encode(result)
    if len(payload) <= MAX_PAYLOAD:
        return payload
    result["payload_trimmed"] = True
    rows = [result.get(name, {}) for name in ("pytest_targeted", "pytest_full", "probe", "dynamo")]
    rows.extend(result["arms"].values())
    rows.extend(result["infer"].values())
    for limit in (24, 8, 2, 0):
        for row in rows:
            if "tail" in row:
                row["tail"] = row["tail"][-limit:] if limit else []
            if "error" in row:
                row["error"] = row["error"][-max(160, limit * 150) :]
            if "profiler" in row:
                row["profiler"]["top_kernels"] = row["profiler"]["top_kernels"][:limit]
        payload = encode(result)
        if len(payload) <= MAX_PAYLOAD:
            return payload
    return encode(
        {
            "runner_exit_code": result["runner_exit_code"],
            "payload_trimmed": True,
            "export_error": "Portable payload exceeds the limit after trimming",
        }
    )


def main() -> None:
    result = collect(Path(os.environ["RUN_ROOT"]), int(sys.argv[1]))
    if persistence := os.environ.get("PERSIST_ROOT"):
        try:
            (Path(persistence) / "h200_result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            result["persistence_error"] = str(exc)
    sys.stdout.write(f"{MARKER} {portable_payload(result)}\n")


if __name__ == "__main__":
    main()
