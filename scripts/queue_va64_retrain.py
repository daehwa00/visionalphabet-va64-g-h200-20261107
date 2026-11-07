"""Owned, stoppable VA64 seed-0 B/P/A queue on the verified 4090 host."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from queue_va_retrain import atomic, gpu_idle, verify_source

ORDER = [["B", 0], ["P", 0], ["A", 0]]
OWNER = "kau-daehwa/va64-20261107"


def update_claim(root: Path, variant: str, state: str, **values: Any) -> None:
    path = root / "ownership.json"
    ledger = json.loads(path.read_text())
    key = variant + "0"
    if ledger["runs"][key]["owner"] != OWNER:
        raise RuntimeError("run owned by a different dispatcher")
    ledger["revision"] += 1
    ledger["runs"][key].update(state=state, **values)
    atomic(path, ledger)
    with (root / "ownership-events.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {
                    "revision": ledger["revision"],
                    "run": key,
                    "time_unix": time.time(),
                    **ledger["runs"][key],
                }
            )
            + "\n"
        )


def aggregate(root: Path) -> dict[str, Any]:
    records = []
    for v, _seed in ORDER:
        folder = root / "runs" / f"va_k64_retrain_{v}" / "seed_0"
        row = {
            "variant": v,
            "seed": 0,
            "status": "pending",
            "top1": None,
            "top5": None,
            "loss": None,
            "stored_parameters": None,
            "trainable_parameters": None,
            "gpu_hours": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
        if (folder / "result.json").is_file():
            r = json.loads((folder / "result.json").read_text())
            if (
                r["status"] != "completed"
                or r["completed_epochs"] != 100
                or r["final_validation"]["examples"] != 50000
            ):
                raise RuntimeError("invalid final endpoint")
            inventory = json.loads((folder / "parameter-inventory.json").read_text())
            cost = json.loads((folder / "cost.json").read_text())
            row.update(
                status="completed",
                top1=100 * r["final_validation"]["accuracy"],
                top5=100 * r["final_validation"]["top5_accuracy"],
                loss=r["final_validation"]["cross_entropy"],
                stored_parameters=inventory["total"]["stored_elements"],
                trainable_parameters=inventory["total"]["trainable_elements"],
                gpu_hours=cost["gpu_hours"],
                peak_allocated_bytes=cost["peak_allocated_bytes"],
                peak_reserved_bytes=cost["peak_reserved_bytes"],
            )
        elif (folder / "checkpoint.pt").is_file():
            row["status"] = "partial_checkpoint"
        records.append(row)
    completed = sum(row["status"] == "completed" for row in records)
    scores = {r["variant"]: r["top1"] for r in records if r["status"] == "completed"}
    summary = {
        "model": "VA64",
        "seed": 0,
        "planned": 3,
        "completed": completed,
        "records": records,
        "B-P_pp": scores["B"] - scores["P"] if all(v in scores for v in "BP") else None,
        "B-A_pp": scores["B"] - scores["A"] if all(v in scores for v in "BA") else None,
        "seed_sample_sd": None,
        "seed_confidence_interval": None,
        "extra_seeds": False,
        "limits": [
            "Single matched seed; no estimate of training-seed uncertainty.",
            "P remains complex-valued.",
            "A reduces head capacity; all scans remain.",
        ],
    }
    atomic(root / "summary.json", summary)
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if completed == 3:
        lines = [
            "# VA64 full ImageNet-1K: matched seed-0 retraining",
            "",
            "All variants trained from scratch for 100 epochs; final-epoch evaluation on 50,000 images.",
            "",
            "| Variant | Top-1 (%) | Parameters | GPU hours |",
            "|---|---:|---:|---:|",
        ]
        for row in records:
            lines.append(
                f"| {row['variant']} | {row['top1']:.3f} | {row['stored_parameters']} | {row['gpu_hours']:.2f} |"
            )
        lines += [
            "",
            f"B-P difference: {summary['B-P_pp']:+.3f} percentage points.",
            f"B-A difference: {summary['B-A_pp']:+.3f} percentage points.",
            "",
            "These are single-seed observations, not evidence of statistical significance or equivalence.",
            "P fixes effective scan phases to zero while retaining complex representations.",
            "A averages normalized directional energies before log1p and changes the head from 320 to 80 inputs.",
            "The B-A comparison includes reduced head capacity and optimization effects.",
            "These VA64 results do not establish a result for VA128 or a matched-compute advantage.",
        ]
        (root / "rebuttal-single-seed.en.md").write_text("\n".join(lines) + "\n")
    return summary


def queue(root: Path, code: Path, python: Path, *, resume: bool = False) -> None:
    lock = (root / "queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    shared = (root.parent / ".va-4090-gpu0.lock").open("a+")
    fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sha = verify_source(code)
    receipt = json.loads((root / "legacy-pause.json").read_text())
    if receipt["state"] != "paused_for_va64":
        raise RuntimeError("legacy dispatcher has not been fenced")
    for v in "BPA":
        passed = json.loads((root / "preflight" / f"{v}-b128.json").read_text())
        if passed.get("status") != "passed" or passed.get("source_manifest_sha256") != sha:
            raise RuntimeError("unverified VA64 preflight")
    unit = json.loads((root / "preflight/unit-tests.json").read_text())
    if unit.get("status") != "passed" or unit.get("source_manifest_sha256") != sha:
        raise RuntimeError("VA64 unit tests are unverified")
    config = {
        "model": "VA64",
        "order": ORDER,
        "seed": 0,
        "epochs": 100,
        "batch_size": 128,
        "accumulation": 2,
        "owner": OWNER,
        "source_manifest_sha256": sha,
        "code": str(code),
        "extra_seeds": False,
    }
    path = root / "campaign.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("VA64 campaign changed")
    atomic(path, config)
    ledger = root / "ownership.json"
    if not ledger.exists():
        atomic(
            ledger,
            {
                "revision": 0,
                "source_manifest_sha256": sha,
                "runs": {v + "0": {"owner": OWNER, "state": "pending"} for v, _seed in ORDER},
            },
        )
    if json.loads(ledger.read_text())["source_manifest_sha256"] != sha:
        raise RuntimeError("ownership/source mismatch")
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("LNET_"):
            env.pop(name)
    env.update(
        PYTHONPATH=f"{code}/src:{code}/scripts",
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TORCHINDUCTOR_COMPILE_THREADS="4",
        PYTHONUNBUFFERED="1",
        LNET_DISABLE_LAUNCH_AUTOTUNE="1",
        TORCHINDUCTOR_CACHE_DIR=str(root / "cache/inductor"),
        TRITON_CACHE_DIR=str(root / "cache/triton"),
    )
    (root / "logs").mkdir(exist_ok=True)
    for v, _seed in ORDER:
        verify_source(code)
        if (root / "STOP").exists():
            atomic(root / "status.json", {"state": "paused", "reason": "STOP before next run"})
            return
        folder = root / "runs" / f"va_k64_retrain_{v}" / "seed_0"
        claim = json.loads(ledger.read_text())["runs"][v + "0"]
        if claim["owner"] != OWNER:
            raise RuntimeError("foreign ownership; refusing launch")
        if (folder / "result.json").exists():
            aggregate(root)
            continue
        if (folder / "failure.json").exists():
            raise RuntimeError("previous VA64 failure needs review")
        if claim["state"] in ("running", "paused", "interrupted") and not resume:
            raise RuntimeError("explicit resume required for a prior claim")
        while not gpu_idle() or shutil.disk_usage(root).free < 15 * 1024**3:
            atomic(root / "status.json", {"state": "waiting_for_gpu_or_disk", "variant": v})
            if (root / "STOP").exists():
                atomic(root / "status.json", {"state": "paused", "reason": "STOP while waiting"})
                return
            time.sleep(15)
        if (root / "STOP").exists():
            atomic(root / "status.json", {"state": "paused", "reason": "STOP before claim"})
            return
        update_claim(root, v, "reserved", source_manifest_sha256=sha)
        folder.mkdir(parents=True, exist_ok=True)
        command = [
            str(python),
            str(code / "scripts/run_va64_retrain_imagenet1k.py"),
            "--variant",
            v,
            "--seed",
            "0",
            "--data-root",
            "/data/ImageNet/2012",
            "--data-manifest",
            str(root / "dataset-manifest.json"),
            "--output-root",
            str(root / "runs"),
            "--batch-size",
            "128",
            "--workers",
            "8",
            "--stop-file",
            str(root / "STOP"),
        ]
        if (folder / "checkpoint.pt").exists():
            command += ["--resume"]
        atomic(
            folder / "command.json",
            {
                "argv": command,
                "environment": {
                    k: env[k]
                    for k in env
                    if k.startswith(("LNET_", "TORCH", "CUDA_", "OMP_", "MKL_", "PYTHONPATH"))
                },
            },
        )
        log = root / "logs" / f"{v}0-{time.time_ns()}.log"
        with log.open("w") as f:
            child = subprocess.Popen(command, cwd=code, env=env, stdout=f, stderr=subprocess.STDOUT)
            update_claim(root, v, "running", pid=child.pid, log=str(log))
            atomic(
                root / "status.json",
                {"state": "running", "variant": v, "seed": 0, "pid": child.pid, "log": str(log)},
            )
            rc = child.wait()
        if (
            rc == 0
            and (folder / "cost.json").exists()
            and json.loads((folder / "cost.json").read_text())["status"] == "paused"
        ):
            update_claim(root, v, "paused")
            aggregate(root)
            atomic(
                root / "status.json",
                {"state": "paused", "variant": v, "checkpoint": str(folder / "checkpoint.pt")},
            )
            return
        if rc or not (folder / "result.json").exists():
            atomic(folder / "failure.json", {"returncode": rc, "log": str(log)})
            update_claim(root, v, "failed")
            aggregate(root)
            atomic(root / "status.json", {"state": "failed", "variant": v, "log": str(log)})
            raise RuntimeError(f"VA64 {v}0 failed")
        update_claim(root, v, "completed")
        aggregate(root)
    atomic(
        root / "status.json",
        {"state": "completed", "completed": 3, "seed": 0, "extra_seeds": False},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    queue(a.root, a.code, Path(sys.executable), resume=a.resume)
