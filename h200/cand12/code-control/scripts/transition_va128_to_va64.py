"""Preserve a complete VA128 epoch and release only its owned GPU for VA64."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import shlex
import signal
import time
from pathlib import Path
from typing import Any


def atomic(path: Path, payload: object) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def identity(pid: int) -> dict[str, Any] | None:
    root = Path(f"/proc/{pid}")
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
            "start_time": fields[19],
            "cmdline": (root / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip(),
            "cwd": str((root / "cwd").resolve(strict=True)),
        }
    except FileNotFoundError:
        return None


def same_process(expected: dict[str, Any]) -> dict[str, Any]:
    current = identity(expected["pid"])
    if not current or any(
        current[k] != expected[k] for k in ("pid", "ppid", "start_time", "cmdline")
    ):
        raise RuntimeError("process identity changed; refusing to signal")
    return current


def transition(old: Path, new: Path, timeout: int) -> None:
    new.mkdir(parents=True, exist_ok=True)
    handle = (new / "transition.lock").open("a+")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = new / "legacy-pause.json"
    if receipt.exists():
        raise RuntimeError("transition receipt already exists; review before rerunning")
    status = json.loads((old / "status.json").read_text())
    if (status.get("state"), status.get("variant"), status.get("seed")) != ("running", "B", 0):
        raise RuntimeError("expected only original VA128 B0 to be running")
    child = identity(status["pid"])
    if (
        not child
        or str(old / "code-003/scripts/run_va_retrain_imagenet1k.py") not in child["cmdline"]
    ):
        raise RuntimeError("unexpected legacy child")
    parent = identity(child["ppid"])
    parent_args = shlex.split(parent["cmdline"]) if parent else []
    absolute_script = str(old / "code-003/scripts/prepare_va_retrain_campaign.py")
    relative_script_verified = bool(
        parent
        and parent.get("cwd") == str(old / "code-003")
        and "scripts/prepare_va_retrain_campaign.py" in parent_args
        and "--root" in parent_args
        and parent_args[parent_args.index("--root") + 1] == str(old)
    )
    if not parent or (absolute_script not in parent["cmdline"] and not relative_script_verified):
        raise RuntimeError("unexpected legacy queue parent")
    for directory in (old / "runs").glob("va_k128_retrain_*/*"):
        if (directory.name != "seed_0" or directory.parent.name != "va_k128_retrain_B") and any(
            directory.iterdir()
        ):
            raise RuntimeError("another VA128 run has already started")
    run = old / "runs/va_k128_retrain_B/seed_0"
    last = json.loads((run / "steps.jsonl").read_text().splitlines()[-1])
    minimum = last["epoch"]
    record = {
        "state": "waiting_for_complete_epoch",
        "child": child,
        "parent": parent,
        "minimum_epoch": minimum,
        "requested_unix": time.time(),
        "reason": "User prioritized matched VA64 seed-0 B/P/A.",
    }
    (new / "legacy-preserved").mkdir(exist_ok=True)
    for name in ("status.json", "campaign.json"):
        (new / "legacy-preserved" / name).write_bytes((old / name).read_bytes())
    same_process(parent)
    os.kill(parent["pid"], signal.SIGSTOP)
    if same_process(parent)["state"] != "T":
        for _ in range(20):
            time.sleep(0.1)
            if same_process(parent)["state"] == "T":
                break
        else:
            raise RuntimeError("legacy queue did not fence")
    atomic(new / "transition.json", record)
    print(json.dumps(record), flush=True)
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        same_process(parent)
        same_process(child)
        meta = run / "latest-epoch.json"
        row = json.loads(meta.read_text()) if meta.exists() else {}
        if row.get("epoch", 0) >= minimum:
            import torch

            torch.set_num_threads(2)
            data = (run / "checkpoint.pt").read_bytes()
            payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            contract_files = list((run / "contracts").glob("*.json"))
            if len(contract_files) != 1:
                raise RuntimeError("ambiguous legacy contract")
            contract = json.loads(contract_files[0].read_text())
            digest = hashlib.sha256(
                json.dumps(
                    contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ).encode()
            ).hexdigest()
            if payload["contract_sha256"] != digest or payload["completed_epochs"] != row["epoch"]:
                raise RuntimeError("legacy checkpoint contract/epoch mismatch")
            if (
                payload["global_step"] != row["epoch"] * 5005
                or len(payload["history"]) != row["epoch"]
            ):
                raise RuntimeError("legacy checkpoint history/step mismatch")
            if not all(k in payload for k in ("model", "optimizer", "rng")):
                raise RuntimeError("legacy checkpoint is incomplete")
            archive = new / "legacy-preserved" / f"va128-B0-epoch{row['epoch']}.pt"
            with archive.open("xb") as f:
                f.write(data)
            sha = hashlib.sha256(data).hexdigest()
            if hashlib.sha256(archive.read_bytes()).hexdigest() != sha:
                raise RuntimeError("legacy backup checksum mismatch")
            record.update(
                state="checkpoint_preserved",
                completed_epoch=row["epoch"],
                global_step=payload["global_step"],
                backup=str(archive),
                checkpoint_sha256=sha,
                contract_sha256=digest,
            )
            atomic(new / "transition.json", record)
            same_process(child)
            same_process(parent)
            os.kill(child["pid"], signal.SIGINT)
            for _ in range(240):
                current = identity(child["pid"])
                if not current or current["state"] == "Z":
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError(
                    "legacy child did not exit after SIGINT; no forced GPU kill attempted"
                )
            # Only the fenced dispatcher remains; it holds no model/GPU state.
            same_process(parent)
            os.kill(parent["pid"], signal.SIGKILL)
            record.update(
                state="paused_for_va64",
                paused_unix=time.time(),
                source_manifest_sha256=hashlib.sha256(
                    (old / "code-003/SOURCE_MANIFEST.json").read_bytes()
                ).hexdigest(),
                old_campaign_sha256=hashlib.sha256(
                    (old / "campaign.json").read_bytes()
                ).hexdigest(),
                resume="Original code-003 runner, original B0 paths/environment, --resume; no old queue restart.",
            )
            atomic(receipt, record)
            atomic(new / "transition.json", record)
            atomic(
                old / "status.json",
                {
                    "state": "paused_for_va64",
                    "variant": "B",
                    "seed": 0,
                    "completed_epochs": row["epoch"],
                    "checkpoint_path": str(run / "checkpoint.pt"),
                    "transition_receipt": str(receipt),
                    "remaining_queue": "held; do not resume dispatcher",
                },
            )
            print(json.dumps(record), flush=True)
            return
        time.sleep(5)
    record["state"] = "needs_attention_epoch_timeout"
    atomic(new / "transition.json", record)
    raise RuntimeError("epoch wait timed out; learning child left untouched, queue parent fenced")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-root", type=Path, required=True)
    p.add_argument("--new-root", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=7200)
    a = p.parse_args()
    transition(a.old_root, a.new_root, a.timeout)
