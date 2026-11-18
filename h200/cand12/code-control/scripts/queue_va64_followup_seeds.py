"""Wait for one owned VA64 seed-0 run, then launch seeds 1 and 2 on its GPU."""

# The outer exception handler must persist every blocked state before raising.
# ruff: noqa: TRY301

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(code: Path) -> str:
    sys.path.insert(0, str(code / "scripts"))
    from queue_va_retrain import verify_source as verify

    return verify(code)


def gpu_info(uuid: str) -> dict[str, Any]:
    # Restrict both inspection and execution to the explicitly assigned 4090.
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True
    )
    found = [line.split(",", 2) for line in rows.splitlines() if line.split(",")[1].strip() == uuid]
    if len(found) != 1 or found[0][2].strip() != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("expected the explicitly assigned RTX 4090 UUID")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    return {
        "uuid": uuid,
        "physical_index": int(found[0][0]),
        "name": found[0][2].strip(),
        "compute_pids": [
            int(line.split(",")[1])
            for line in rows.splitlines()
            if line.split(",")[0].strip() == uuid
        ],
    }


def target(root: Path, variant: str, seed: int) -> Path:
    return root / "runs" / f"va_k64_retrain_{variant}" / f"seed_{seed}"


def completed(root: Path, variant: str, seed: int) -> None:
    directory = target(root, variant, seed)
    if (directory / "failure.json").exists():
        raise RuntimeError(f"failure artifact requires review: {directory}")
    result = json.loads((directory / "result.json").read_text())
    cost = json.loads((directory / "cost.json").read_text())
    validation = result.get("final_validation", {})
    if (
        result.get("status") != "completed"
        or result.get("completed_epochs") != 100
        or result.get("model_key") != f"va_k64_retrain_{variant}"
        or result.get("seed") != seed
        or validation.get("examples") != 50000
        or cost.get("status") != "completed"
        or cost.get("variant") != variant
        or cost.get("seed") != seed
    ):
        raise RuntimeError(f"invalid completed endpoint: {directory}")
    for name in ("accuracy", "top5_accuracy", "cross_entropy"):
        value = validation.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"invalid final validation metric: {directory}/{name}")
    if (
        not 0 <= validation["accuracy"] <= validation["top5_accuracy"] <= 1
        or validation["cross_entropy"] < 0
    ):
        raise RuntimeError(f"invalid final validation range: {directory}")


def check_claim(path: Path, a: argparse.Namespace, source_sha: str) -> dict[str, Any]:
    claim = json.loads(path.read_text())
    if (
        claim.get("owner") != a.owner
        or claim.get("gpu", {}).get("uuid") != a.gpu_uuid
        or claim.get("variant") != a.variant
        or claim.get("source_manifest_sha256") != source_sha
    ):
        raise RuntimeError(f"foreign or changed run ownership: {path}")
    return claim


def initialization(a: argparse.Namespace, seed: int, source_sha: str) -> tuple[Path, str]:
    directory = a.initialization_dir / f"seed_{seed}"
    canonical = json.loads((directory / "canonical.json").read_text())
    if canonical.get("source_manifest_sha256") != source_sha or canonical.get("seed") != seed:
        raise RuntimeError("canonical initialization source or seed mismatch")
    path = directory / f"initial-{a.variant}{seed}.pt"
    sha = canonical["variants"][a.variant]["file_sha256"]
    if digest(path) != sha:
        raise RuntimeError(f"initialization checksum mismatch: {path}")
    return path, sha


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root",
        "code",
        "data-root",
        "manifest",
        "tools",
        "initialization-dir",
        "entrypoint",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--variant", choices=("B", "P", "A"), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    return parser.parse_args()


def queue(a: argparse.Namespace) -> None:
    if a.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    for name in (
        "root",
        "code",
        "data_root",
        "manifest",
        "tools",
        "initialization_dir",
        "entrypoint",
    ):
        setattr(a, name, getattr(a, name).resolve())
    a.root.mkdir(parents=True, exist_ok=True)
    state_path = a.root / f"queue-{a.variant}.json"
    with (a.root / f"queue-{a.variant}.lock").open("a+") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_sha = verify_source(a.code)
        config = {
            "variant": a.variant,
            "owner": a.owner,
            "gpu_uuid": a.gpu_uuid,
            "source_manifest_sha256": source_sha,
            "data_manifest_sha256": digest(a.manifest),
            "entrypoint_sha256": digest(a.entrypoint),
            "queue_sha256": digest(Path(__file__)),
            "code": str(a.code),
            "entrypoint": str(a.entrypoint),
            "data_root": str(a.data_root),
            "manifest": str(a.manifest),
            "tools": str(a.tools),
            "python": sys.executable,
            "initialization_dir": str(a.initialization_dir),
            "seeds": [1, 2],
            "batch_size": 128,
            "gradient_accumulation_steps": 2,
            "workers": 8,
        }
        if state_path.exists() and json.loads(state_path.read_text()).get("config") != config:
            raise RuntimeError("existing queue configuration changed")
        state: dict[str, Any] = {"config": config, "supervisor_pid": os.getpid()}

        def update(status: str, **fields: Any) -> None:
            state.update(state=status, updated_unix=time.time(), **fields)
            atomic(state_path, state)

        def stop() -> bool:
            if (a.root / "STOP").exists():
                update("stopped")
                return True
            return False

        def verify() -> None:
            if (
                verify_source(a.code) != source_sha
                or digest(a.entrypoint) != config["entrypoint_sha256"]
                or digest(a.manifest) != config["data_manifest_sha256"]
            ):
                raise RuntimeError("source, entrypoint or dataset manifest changed")

        try:
            # Validate future inputs before going into a potentially day-long wait.
            for seed in (1, 2):
                initialization(a, seed, source_sha)
            with (a.root / f"gpu-{a.gpu_uuid}.lock").open("a+") as slot:
                while True:
                    if stop():
                        return
                    old = check_claim(a.root / f"claim-{a.variant}0.json", a, source_sha)
                    if old.get("seed") != 0 or old.get("state") in ("failed", "paused"):
                        raise RuntimeError("seed 0 failed or paused; follow-up requires review")
                    try:
                        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        update("waiting_for_seed0")
                        time.sleep(a.poll_seconds)
                        continue
                    # Seed 0 may finish between the first claim read and lock acquisition.
                    old = check_claim(a.root / f"claim-{a.variant}0.json", a, source_sha)
                    if old.get("state") != "completed":
                        raise RuntimeError("seed 0 released GPU without a completed claim")
                    completed(a.root, a.variant, 0)
                    break
                for seed in (1, 2):
                    if stop():
                        return
                    verify()
                    claim_path = a.root / f"claim-{a.variant}{seed}.json"
                    output = target(a.root, a.variant, seed)
                    if claim_path.exists():
                        old = check_claim(claim_path, a, source_sha)
                        if old.get("state") != "completed" or old.get("seed") != seed:
                            raise RuntimeError(
                                f"existing seed {seed} claim requires explicit review"
                            )
                        completed(a.root, a.variant, seed)
                        update("skipped_completed", seed=seed)
                        continue
                    if output.exists() and any(output.iterdir()):
                        raise RuntimeError(
                            f"existing seed {seed} artifacts require explicit review"
                        )
                    with (a.root / f"{a.variant}{seed}.lock").open("a+") as run_lock:
                        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        while True:
                            if stop():
                                return
                            if shutil.disk_usage(a.root).free < 15 * 1024**3:
                                raise RuntimeError("less than 15 GiB free; no new run started")
                            gpu = gpu_info(a.gpu_uuid)
                            if not gpu["compute_pids"]:
                                break
                            update("waiting_for_gpu", seed=seed, compute_pids=gpu["compute_pids"])
                            time.sleep(a.poll_seconds)
                        verify()
                        initial_path, initial_sha = initialization(a, seed, source_sha)
                        if stop():
                            return
                        command = [
                            sys.executable,
                            str(a.entrypoint),
                            "--code",
                            str(a.code),
                            "--initial-state",
                            str(initial_path),
                            "--initial-state-sha256",
                            initial_sha,
                            "--variant",
                            a.variant,
                            "--seed",
                            str(seed),
                            "--data-root",
                            str(a.data_root),
                            "--data-manifest",
                            str(a.manifest),
                            "--output-root",
                            str(a.root / "runs"),
                            "--batch-size",
                            "128",
                            "--workers",
                            "8",
                            "--stop-file",
                            str(a.root / "STOP"),
                        ]
                        env = {
                            key: value
                            for key, value in os.environ.items()
                            if not key.startswith("LNET_")
                        }
                        env.update(
                            CUDA_VISIBLE_DEVICES=a.gpu_uuid,
                            PYTHONPATH=f"{a.code}/src:{a.code}/scripts:{a.tools}",
                            OMP_NUM_THREADS="4",
                            MKL_NUM_THREADS="4",
                            TORCHINDUCTOR_COMPILE_THREADS="4",
                            PYTHONUNBUFFERED="1",
                            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                            LNET_DISABLE_LAUNCH_AUTOTUNE="1",
                            TORCHINDUCTOR_CACHE_DIR=str(a.root / "cache/inductor"),
                            TRITON_CACHE_DIR=str(a.root / "cache/triton"),
                        )
                        logroot = a.root / "logs"
                        logroot.mkdir(exist_ok=True)
                        log = logroot / f"{a.variant}{seed}-training-{time.time_ns()}.log"
                        claim = {
                            "owner": a.owner,
                            "variant": a.variant,
                            "seed": seed,
                            "gpu": gpu,
                            "source_manifest_sha256": source_sha,
                            "state": "starting",
                            "command": command,
                            "command_sha256": hashlib.sha256(
                                json.dumps(command).encode()
                            ).hexdigest(),
                            "initial_state_sha256": initial_sha,
                            "entrypoint_sha256": config["entrypoint_sha256"],
                            "data_manifest_sha256": config["data_manifest_sha256"],
                            "supervisor_pid": os.getpid(),
                            "started_unix": time.time(),
                            "log": str(log),
                        }
                        # Exclusive creation prevents an unexpected second claimant being overwritten.
                        with claim_path.open("x") as stream:
                            json.dump(claim, stream, indent=2)
                        with log.open("x") as stream:
                            child = subprocess.Popen(
                                command,
                                cwd=a.code,
                                env=env,
                                stdout=stream,
                                stderr=subprocess.STDOUT,
                            )
                            claim.update(state="training", pid=child.pid)
                            atomic(claim_path, claim)
                            update(
                                "training", seed=seed, pid=child.pid, log=str(log), command=command
                            )
                            rc = child.wait()
                        claim["returncode"] = rc
                        claim["finished_unix"] = time.time()
                        if rc or (a.root / "STOP").exists():
                            claim["state"] = "paused" if (a.root / "STOP").exists() else "failed"
                            atomic(claim_path, claim)
                            if stop():
                                return
                            raise RuntimeError(f"seed {seed} exited with code {rc}")
                        try:
                            completed(a.root, a.variant, seed)
                        except Exception:
                            claim["state"] = "failed"
                            atomic(claim_path, claim)
                            raise
                        claim["state"] = "completed"
                        atomic(claim_path, claim)
                        update("seed_completed", seed=seed)
                update("completed", completed_seeds=[0, 1, 2])
        except Exception as error:
            update("blocked", error=str(error))
            raise


if __name__ == "__main__":
    queue(arguments())
