"""Durable, single-GPU B/P/A queue; never launches on top of another GPU job."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ORDER = [("B", 0), ("P", 0), ("A", 0), ("P", 1), ("A", 1), ("B", 1), ("A", 2), ("B", 2), ("P", 2)]


def atomic(path: Path, value: object) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def verify_source(code: Path) -> str:
    manifest = json.loads((code / "SOURCE_MANIFEST.json").read_text())
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((code / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source snapshot changed: {name}")
    return hashlib.sha256((code / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()


def gpu_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    return not result.stdout.strip()


def aggregate(root: Path) -> dict[str, Any]:
    records = []
    scores = {}
    for variant, seed in ORDER:
        directory = root / "runs" / f"va_k128_retrain_{variant}" / f"seed_{seed}"
        result = directory / "result.json"
        record = {
            "variant": variant,
            "seed": seed,
            "status": "pending",
            "top1": None,
            "top5": None,
            "training_seconds": None,
        }
        if result.exists():
            r = json.loads(result.read_text())
            if r.get("status") != "completed" or r.get("completed_epochs") != 100:
                raise ValueError(f"incomplete result pretending to be completed: {result}")
            record.update(
                status="completed",
                top1=100 * r["final_validation"]["accuracy"],
                top5=100 * r["final_validation"]["top5_accuracy"],
                training_seconds=r["training_seconds"],
            )
            scores[(variant, seed)] = record["top1"]
        elif (directory / "failure.json").exists():
            record["status"] = "failed"
        elif (directory / "checkpoint.pt").exists() or (directory / "steps.jsonl").exists():
            record["status"] = "running_or_interrupted"
        records.append(record)
    means = {}
    for v in "BPA":
        values = [scores[(v, s)] for s in range(3) if (v, s) in scores]
        means[v] = {
            "n": len(values),
            "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        }
    pairs = {}
    for v in "PA":
        values = [
            {"seed": s, "difference_pp": scores[("B", s)] - scores[(v, s)]}
            for s in range(3)
            if ("B", s) in scores and (v, s) in scores
        ]
        ds = [x["difference_pp"] for x in values]
        mean = statistics.mean(ds) if ds else None
        sd = statistics.stdev(ds) if len(ds) > 1 else None
        half = 4.302652729 * sd / math.sqrt(3) if len(ds) == 3 else None
        pairs[f"B-{v}"] = {
            "per_seed": values,
            "mean_pp": mean,
            "sample_sd_pp": sd,
            "t95_ci": [mean - half, mean + half] if half is not None else None,
        }
    summary = {
        "completed": len(scores),
        "planned": 9,
        "records": records,
        "groups": means,
        "paired": pairs,
        "practical_threshold_pp": 0.3,
        "additional_seeds": False,
        "limits": [
            "P is not a real-valued model.",
            "A changes head capacity and readout.",
            "Three-seed t intervals are unstable.",
            "Synthetic benchmark excludes data loading.",
        ],
    }
    atomic(root / "summary.json", summary)
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if len(scores) == 9:
        lines = [
            "# Retrained K128 ImageNet-1K results",
            "",
            "All variants: scratch, 100 epochs, seeds 0/1/2, final-epoch validation.",
            "",
        ]
        for v in "BPA":
            lines.append(
                f"- {v}: {means[v]['mean']:.3f} ± {means[v]['sample_sd']:.3f}% Top-1 (sample SD)."
            )
        lines += [
            "",
            "P removes scan phase freedom while preserving complex computation.",
            "B-A includes reduced head capacity and readout/optimization effects; all four scans remain.",
            "These results do not establish superiority at matched compute.",
        ]
        (root / "rebuttal-results.en.md").write_text("\n".join(lines) + "\n")
    return summary


def queue(root: Path, code: Path, python: Path, batch: int, workers: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sha = verify_source(code)
    for v in "BPA":
        receipt = json.loads((root / "preflight" / f"{v}-b{batch}.json").read_text())
        if (
            receipt.get("status") != "passed"
            or receipt.get("batch_size") != batch
            or receipt.get("source_manifest_sha256") != sha
        ):
            raise RuntimeError(f"preflight receipt mismatch: {v}")
    unit = json.loads((root / "preflight/unit-tests.json").read_text())
    if unit.get("status") != "passed" or unit.get("source_manifest_sha256") != sha:
        raise RuntimeError("missing exact-source unit test receipt")
    config = {
        "order": ORDER,
        "batch_size": batch,
        "workers": workers,
        "source_manifest_sha256": sha,
        "python": str(python),
        "code": str(code),
        "dataset": str(root / "dataset-manifest.json"),
        "epochs": 100,
        "practical_threshold_pp": 0.3,
        "extra_seeds": False,
    }
    if (root / "campaign.json").exists() and json.loads(
        (root / "campaign.json").read_text()
    ) != config:
        raise RuntimeError("campaign configuration changed")
    atomic(root / "campaign.json", config)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("LNET_"):
            env.pop(key)
    env.update(
        PYTHONPATH=f"{code}/src:{code}/scripts",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TORCHINDUCTOR_COMPILE_THREADS="4",
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="0",
        LNET_DISABLE_LAUNCH_AUTOTUNE="1",
        TORCHINDUCTOR_CACHE_DIR=str(root / "cache/inductor"),
        TRITON_CACHE_DIR=str(root / "cache/triton"),
    )
    (root / "logs").mkdir(exist_ok=True)
    for variant, seed in ORDER:
        verify_source(code)
        output = root / "runs" / f"va_k128_retrain_{variant}" / f"seed_{seed}"
        result = output / "result.json"
        if result.exists():
            aggregate(root)
            continue
        if (output / "failure.json").exists():
            raise RuntimeError(f"previous failure requires review: {output}")
        while True:
            free = shutil.disk_usage(root).free
            if free < 15 * 1024**3:
                atomic(
                    root / "status.json",
                    {
                        "state": "waiting_for_disk",
                        "free_bytes": free,
                        "variant": variant,
                        "seed": seed,
                    },
                )
            elif not gpu_idle():
                atomic(
                    root / "status.json",
                    {"state": "waiting_for_gpu", "variant": variant, "seed": seed},
                )
            else:
                break
            time.sleep(30)
        output.mkdir(parents=True, exist_ok=True)
        command = [
            str(python),
            str(code / "scripts/run_va_retrain_imagenet1k.py"),
            "--variant",
            variant,
            "--seed",
            str(seed),
            "--data-root",
            "/data/ImageNet/2012",
            "--data-manifest",
            str(root / "dataset-manifest.json"),
            "--output-root",
            str(root / "runs"),
            "--batch-size",
            str(batch),
            "--workers",
            str(workers),
        ]
        if (output / "checkpoint.pt").exists():
            command += ["--resume"]
        atomic(
            output / "command.json",
            {
                "argv": command,
                "environment": {
                    k: env[k]
                    for k in env
                    if k.startswith(("LNET_", "TORCH", "CUDA_", "OMP_", "MKL_", "PYTHONPATH"))
                },
            },
        )
        log = root / "logs" / f"{variant}{seed}-{time.time_ns()}.log"
        with log.open("w") as stream:
            child = subprocess.Popen(
                command, cwd=code, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            atomic(
                root / "status.json",
                {
                    "state": "running",
                    "variant": variant,
                    "seed": seed,
                    "pid": child.pid,
                    "log": str(log),
                    "started_unix": time.time(),
                },
            )
            returncode = child.wait()
        if returncode or not result.exists():
            atomic(
                output / "failure.json",
                {"returncode": returncode, "log": str(log), "time_unix": time.time()},
            )
            aggregate(root)
            atomic(
                root / "status.json",
                {"state": "failed", "variant": variant, "seed": seed, "log": str(log)},
            )
            raise RuntimeError(f"run failed: {variant}{seed}; inspect {log}")
        aggregate(root)
    atomic(
        root / "status.json",
        {"state": "completed", "completed": 9, "summary": str(root / "summary.json")},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--code", type=Path)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--batch-size", type=int, choices=(64, 128), default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args()
    if a.aggregate_only:
        aggregate(a.root)
    else:
        queue(a.root, a.code, a.python, a.batch_size, a.workers)
