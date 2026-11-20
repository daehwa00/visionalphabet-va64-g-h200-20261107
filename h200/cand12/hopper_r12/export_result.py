"""Clean the staging tree and export the R12 numerical result to the issue."""
import base64
import json
import lzma
import os
from pathlib import Path
import shutil
import sys

MARKER = "H200_HOPPER_R12_XZ"
PAYLOAD_LIMIT = 56_000


def encode(data):
    return base64.b64encode(lzma.compress(
        json.dumps(data, separators=(",", ":"), allow_nan=False).encode(), preset=6
    )).decode()


def cleanup_plaintext(root):
    folders = [root / label for label in ("private-input", "r4", "current")]
    folders.extend(root.glob("relay-input-*"))
    captures = list((root / "results").glob("*.capture.pt"))
    errors = []
    for path in folders + captures:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            errors.append(path.name)
    return errors


def main(root, exit_code):
    cleanup_errors = cleanup_plaintext(root)
    doc = dict(schema=1, commit=os.environ.get("HOPPER_R12_COMMIT"),
               requested_pool=os.environ.get("HOPPER_REQUESTED_POOL"),
               exit_code=exit_code or int(bool(cleanup_errors)),
               mode=os.environ.get("HOPPER_R12_MODE"), run_root=str(root),
               remote_plaintext_absent=not cleanup_errors, remote_cleanup_errors=cleanup_errors)
    for name in ("fixture-source", "fixture-manifest", "environment-source", "setup-error"):
        path = root / name
        if path.is_file():
            doc[name.replace("-", "_")] = path.read_text().strip()
    for name, file in (("run", "result.json"), ("gpu_topology", "topology.json")):
        path = root / file
        if path.is_file():
            doc[name] = json.loads(path.read_text())
    persist = os.environ.get("PERSIST_ROOT")
    full = (Path(persist) if persist else root) / "export-summary.json"
    try:
        full.write_text(json.dumps(doc, indent=2))
    except OSError:
        full = root / "export-summary.json"
        full.write_text(json.dumps(doc, indent=2))
    payload = encode(doc)
    if len(payload) >= PAYLOAD_LIMIT:
        failure = {key: doc[key] for key in ("schema", "commit", "requested_pool", "run_root",
                                             "remote_plaintext_absent")}
        failure.update(exit_code=1, persistent_full_result=str(full),
                       export_error=f"Full result marker exceeds size budget ({len(payload)} bytes)")
        print(MARKER, encode(failure), flush=True)
        return 1
    print(MARKER, payload, flush=True)
    return int(bool(cleanup_errors))


if __name__ == "__main__":
    sys.exit(main(Path(os.environ["RUN_ROOT"]), int(sys.argv[1])))
