"""Validate an immutable remote snapshot and then own the nine-run campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from queue_va_retrain import atomic, gpu_idle, queue, verify_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    code = Path(__file__).resolve().parents[1]
    sha = verify_source(code)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("LNET_"):
            env.pop(name)
    env.update(
        PYTHONPATH=f"{code}/src:{code}/scripts:{root}/tools",
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
    while not gpu_idle():
        atomic(root / "status.json", {"state": "preflight_waiting_for_gpu"})
        time.sleep(30)
    preflight = root / "preflight"
    receipt = preflight / "unit-tests.json"
    existing = json.loads(receipt.read_text()) if receipt.exists() else {}
    if existing.get("source_manifest_sha256") != sha or existing.get("status") != "passed":
        atomic(root / "status.json", {"state": "unit_tests", "source_manifest_sha256": sha})
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_va_retrain_ablation.py",
            "tests/test_lnet_imagenet1k_runner.py",
            "tests/test_imagenet1k_runtime.py",
            "tests/test_pac_product_scan_pipeline.py",
        ]
        log = preflight / f"unit-tests-{time.time_ns()}.log"
        with log.open("w") as stream:
            result = subprocess.run(
                command, cwd=code, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        if result.returncode:
            atomic(root / "status.json", {"state": "unit_tests_failed", "log": str(log)})
            raise RuntimeError(f"unit tests failed: {log}")
        atomic(
            receipt,
            {"status": "passed", "source_manifest_sha256": sha, "log": str(log), "argv": command},
        )
    manifest = root / "dataset-manifest.json"
    if not manifest.exists():
        subprocess.run(
            [
                sys.executable,
                str(code / "scripts/validate_va_retrain.py"),
                "--data-root",
                "/data/ImageNet/2012",
                "--output",
                str(manifest),
            ],
            cwd=code,
            env=env,
            check=True,
        )
    batch = 128
    for selected in (128, 64):
        oom = False
        for variant in "BPA":
            output = preflight / f"{variant}-b{selected}.json"
            previous = json.loads(output.read_text()) if output.exists() else {}
            if previous.get("source_manifest_sha256") == sha and previous.get("status") == "passed":
                continue
            while not gpu_idle():
                atomic(
                    root / "status.json", {"state": "preflight_waiting_for_gpu", "variant": variant}
                )
                time.sleep(30)
            atomic(
                root / "status.json",
                {"state": "benchmark", "variant": variant, "batch_size": selected},
            )
            command = [
                sys.executable,
                str(code / "scripts/validate_va_retrain.py"),
                "--variant",
                variant,
                "--batch-size",
                str(selected),
                "--output",
                str(output),
            ]
            log = preflight / f"{variant}-b{selected}-{time.time_ns()}.log"
            with log.open("w") as stream:
                result = subprocess.run(
                    command, cwd=code, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            if result.returncode:
                text = log.read_text()
                atomic(
                    preflight / f"{variant}-b{selected}-failure.json",
                    {
                        "returncode": result.returncode,
                        "log": str(log),
                        "source_manifest_sha256": sha,
                    },
                )
                if selected == 128 and ("OutOfMemoryError" in text or "CUDA out of memory" in text):
                    oom = True
                    break
                atomic(
                    root / "status.json",
                    {"state": "benchmark_failed", "variant": variant, "log": str(log)},
                )
                raise RuntimeError(f"benchmark failed: {log}")
            payload = json.loads(output.read_text())
            payload["source_manifest_sha256"] = sha
            payload["log"] = str(log)
            atomic(output, payload)
        if not oom:
            batch = selected
            break
    queue(root, code, Path(sys.executable), batch, 8)


if __name__ == "__main__":
    main()
