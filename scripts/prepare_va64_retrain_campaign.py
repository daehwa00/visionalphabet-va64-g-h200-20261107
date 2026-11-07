"""Prepare VA64 while VA128 finishes an epoch, then validate and launch three runs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from queue_va64_retrain import queue
from queue_va_retrain import atomic, gpu_idle, verify_source


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    code = Path(__file__).resolve().parents[1]
    supervisor = (root / "supervisor.lock").open("a+")
    fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sha = verify_source(code)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("LNET_"):
            env.pop(name)
    env.update(
        PYTHONPATH=f"{code}/src:{code}/scripts:/data/experiments/alphabet/va-retrain-20261106/tools",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TORCHINDUCTOR_COMPILE_THREADS="4",
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="0",
        LNET_DISABLE_LAUNCH_AUTOTUNE="1",
        TORCHINDUCTOR_CACHE_DIR=str(root / "cache/inductor"),
        TRITON_CACHE_DIR=str(root / "cache/triton"),
    )
    preflight = root / "preflight"
    preflight.mkdir(exist_ok=True)

    def run(command: list[str], name: str, environment: dict[str, str]) -> Path:
        log = preflight / f"{name}-{time.time_ns()}.log"
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=code,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            atomic(root / "status.json", {"state": name + "_failed", "log": str(log)})
            raise RuntimeError(f"{name} failed: {log}")
        return log

    # CPU-only correctness checks cannot interfere with the active legacy CUDA job.
    cpu_env = {**env, "CUDA_VISIBLE_DEVICES": ""}
    cpu_log = run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_va64_retrain.py"], "cpu_tests", cpu_env
    )
    manifest = root / "dataset-manifest.json"
    if not manifest.exists():
        run(
            [
                sys.executable,
                str(code / "scripts/validate_va64_retrain.py"),
                "--data-root",
                "/data/ImageNet/2012",
                "--output",
                str(manifest),
            ],
            "dataset",
            cpu_env,
        )
    reference = json.loads(
        (
            Path("/data/experiments/alphabet/va-retrain-20261106") / "dataset-manifest.json"
        ).read_text()
    )
    actual = json.loads(manifest.read_text())
    for key in ("counts", "classes", "membership_size_sha256"):
        if actual[key] != reference[key]:
            raise RuntimeError("ImageNet split changed")
    while not (root / "legacy-pause.json").is_file():
        if (root / "STOP").exists():
            atomic(
                root / "status.json",
                {"state": "paused", "reason": "STOP while waiting for legacy checkpoint"},
            )
            return
        transition = root / "transition.json"
        if transition.exists() and json.loads(transition.read_text())["state"].startswith(
            "needs_attention"
        ):
            raise RuntimeError("legacy transition needs attention")
        transition_lock = root / "transition.lock"
        if transition_lock.exists():
            with transition_lock.open("a+") as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    if not (root / "legacy-pause.json").is_file():
                        atomic(
                            root / "status.json",
                            {
                                "state": "legacy_transition_failed",
                                "log": str(root / "transition-v2.log"),
                            },
                        )
                        raise RuntimeError("legacy transition exited without a pause receipt")
        atomic(
            root / "status.json",
            {"state": "waiting_for_legacy_checkpoint", "source_manifest_sha256": sha},
        )
        time.sleep(15)
    if json.loads((root / "legacy-pause.json").read_text())["state"] != "paused_for_va64":
        raise RuntimeError("invalid legacy transition receipt")
    while not gpu_idle():
        atomic(root / "status.json", {"state": "waiting_for_gpu"})
        time.sleep(15)
    gpu_log = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_va_retrain_ablation.py::test_mean_scan_gpu_cpu_parity",
            "tests/test_pac_product_scan_pipeline.py",
            "tests/test_imagenet1k_runtime.py",
        ],
        "gpu_tests",
        env,
    )
    atomic(
        preflight / "unit-tests.json",
        {
            "status": "passed",
            "source_manifest_sha256": sha,
            "cpu_log": str(cpu_log),
            "gpu_log": str(gpu_log),
            "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
    )
    for variant in "BPA":
        if (root / "STOP").exists():
            atomic(root / "status.json", {"state": "paused", "reason": "STOP during preflight"})
            return
        output = preflight / f"{variant}-b128.json"
        previous = json.loads(output.read_text()) if output.exists() else {}
        if previous.get("status") == "passed" and previous.get("source_manifest_sha256") == sha:
            continue
        while not gpu_idle():
            time.sleep(15)
        atomic(root / "status.json", {"state": "benchmark", "variant": variant, "batch_size": 128})
        log = run(
            [
                sys.executable,
                str(code / "scripts/validate_va64_retrain.py"),
                "--variant",
                variant,
                "--batch-size",
                "128",
                "--output",
                str(output),
            ],
            "benchmark_" + variant,
            env,
        )
        receipt = json.loads(output.read_text())
        receipt.update(source_manifest_sha256=sha, log=str(log))
        atomic(output, receipt)
    queue(root, code, Path(sys.executable), resume=a.resume)


if __name__ == "__main__":
    main()
