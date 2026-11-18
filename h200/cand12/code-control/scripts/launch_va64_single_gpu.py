"""Claim one explicit 4090 UUID and launch one matched VA64 seed-0 condition."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def atomic(path: Path, value: object) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def gpu_info(uuid: str) -> dict:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True
    )
    found = [
        line.strip().split(", ", 2)
        for line in text.splitlines()
        if line.split(",")[1].strip() == uuid
    ]
    if len(found) != 1 or found[0][2] != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("expected the explicitly assigned RTX4090 UUID")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    pids = [
        int(line.split(",")[1].strip())
        for line in apps.splitlines()
        if line.split(",")[0].strip() == uuid
    ]
    return {
        "physical_index": int(found[0][0]),
        "uuid": uuid,
        "name": found[0][2],
        "compute_pids": pids,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--tools", type=Path, required=True)
    p.add_argument("--variant", choices=("B", "P", "A"), required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--owner", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validator", type=Path)
    p.add_argument("--initialization-dir", type=Path)
    p.add_argument("--initialized-entrypoint", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    slot = (root / f"gpu-{a.gpu_uuid}.lock").open("a+")
    fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_lock = (root / f"{a.variant}0.lock").open("a+")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(a.code / "scripts"))
    from queue_va_retrain import verify_source

    sha = verify_source(a.code)
    claim_path = root / f"claim-{a.variant}0.json"
    if claim_path.exists():
        old = json.loads(claim_path.read_text())
        if old["owner"] != a.owner or old["gpu"]["uuid"] != a.gpu_uuid:
            raise RuntimeError("foreign run ownership")
        if old["state"] == "completed":
            return
        if not a.resume:
            raise RuntimeError("existing claim requires explicit resume")
    gpu = gpu_info(a.gpu_uuid)
    if gpu["compute_pids"]:
        raise RuntimeError(f"assigned GPU is occupied: {gpu}")
    if shutil.disk_usage(root).free < 15 * 1024**3:
        raise RuntimeError("less than 15 GiB free on output filesystem")
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("LNET_"):
            env.pop(name)
    env.update(
        CUDA_VISIBLE_DEVICES=a.gpu_uuid,
        PYTHONPATH=f"{a.code}/src:{a.code}/scripts:{a.tools}",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TORCHINDUCTOR_COMPILE_THREADS="4",
        PYTHONUNBUFFERED="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        LNET_DISABLE_LAUNCH_AUTOTUNE="1",
        TORCHINDUCTOR_CACHE_DIR=str(root / "cache/inductor"),
        TRITON_CACHE_DIR=str(root / "cache/triton"),
    )
    claim = {
        "owner": a.owner,
        "variant": a.variant,
        "seed": 0,
        "gpu": gpu,
        "source_manifest_sha256": sha,
        "state": "preflight",
        "started_unix": time.time(),
        "supervisor_pid": os.getpid(),
    }
    atomic(claim_path, claim)
    logroot = root / "logs"
    logroot.mkdir(exist_ok=True)

    def run(command: list[str], stage: str) -> int:
        log = logroot / f"{a.variant}0-{stage}-{time.time_ns()}.log"
        with log.open("w") as stream:
            child = subprocess.Popen(
                command, cwd=a.code, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            claim.update(state=stage, pid=child.pid, log=str(log))
            atomic(claim_path, claim)
            rc = child.wait()
            if rc:
                claim.update(state="failed", failed_stage=stage, returncode=rc)
                atomic(claim_path, claim)
            return rc

    unit = root / f"unit-{a.variant}.json"
    if not unit.exists():
        rc = run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_va64_retrain.py",
                "tests/test_va_retrain_ablation.py::test_mean_scan_gpu_cpu_parity",
            ],
            "unit_tests",
        )
        if rc:
            raise RuntimeError("VA64 unit tests failed")
        atomic(unit, {"status": "passed", "source_manifest_sha256": sha, "gpu_uuid": a.gpu_uuid})
    benchmark = root / f"benchmark-{a.variant}.json"
    if not benchmark.exists():
        rc = run(
            [
                sys.executable,
                str(a.validator or (a.code / "scripts/validate_va64_retrain.py")),
                "--variant",
                a.variant,
                "--batch-size",
                "128",
                "--output",
                str(benchmark),
            ],
            "benchmark",
        )
        if rc:
            raise RuntimeError("VA64 benchmark failed")
        receipt = json.loads(benchmark.read_text())
        receipt.update(source_manifest_sha256=sha, gpu_uuid=a.gpu_uuid)
        atomic(benchmark, receipt)
    if (root / "STOP").exists():
        claim["state"] = "paused"
        atomic(claim_path, claim)
        return
    target = root / "runs" / f"va_k64_retrain_{a.variant}" / "seed_0"
    command = [
        sys.executable,
        str(a.code / "scripts/run_va64_retrain_imagenet1k.py"),
        "--variant",
        a.variant,
        "--seed",
        "0",
        "--data-root",
        str(a.data_root),
        "--data-manifest",
        str(a.manifest),
        "--output-root",
        str(root / "runs"),
        "--batch-size",
        "128",
        "--workers",
        "8",
        "--stop-file",
        str(root / "STOP"),
    ]
    if a.initialization_dir is not None:
        if a.initialized_entrypoint is None:
            raise RuntimeError("initialization entrypoint is required")
        init_manifest = json.loads((a.initialization_dir / "canonical.json").read_text())
        if init_manifest["source_manifest_sha256"] != sha:
            raise RuntimeError("canonical initialization source mismatch")
        command[1] = str(a.initialized_entrypoint)
        command += [
            "--code",
            str(a.code),
            "--initial-state",
            str(a.initialization_dir / f"initial-{a.variant}0.pt"),
            "--initial-state-sha256",
            init_manifest["variants"][a.variant]["file_sha256"],
        ]
    if (target / "checkpoint.pt").exists():
        command += ["--resume"]
    claim["command"] = command
    atomic(claim_path, claim)
    rc = run(command, "training")
    cost = target / "cost.json"
    if rc == 0 and cost.exists() and json.loads(cost.read_text())["status"] == "paused":
        claim["state"] = "paused"
    elif rc == 0 and (target / "result.json").exists():
        claim["state"] = "completed"
    else:
        claim.update(state="failed", returncode=rc)
    atomic(claim_path, claim)
    if claim["state"] == "failed":
        raise RuntimeError("VA64 training failed")


if __name__ == "__main__":
    main()
