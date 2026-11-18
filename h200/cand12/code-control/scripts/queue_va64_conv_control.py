"""Wait for the predecessor, then run the pinned VA64 convolution seeds 0 and 1."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

MIN_FREE_BYTES = 15 * 1024**3
MODEL_KEY = "va_k64_retrain_C"


def atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(code: Path, manifest: Path) -> str:
    payload = json.loads(manifest.read_text())
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source manifest must contain a nonempty files mapping")
    for relative, expected in files.items():
        path = (code / relative).resolve()
        if not path.is_relative_to(code) or not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"source checksum mismatch: {relative}")
    actual = {str(path.relative_to(code)) for folder in ("src", "scripts")
              for path in (code / folder).rglob("*.py")}
    if not actual.issubset(files):
        raise ValueError("source manifest must cover the complete Python snapshot")
    if "scripts/run_va64_conv_control.py" not in files:
        raise ValueError("source manifest must pin the convolution runner")
    return digest(manifest)


def completed(path: Path, *, seed: int | None = None, model_key: str | None = None) -> None:
    result = json.loads(path.read_text())
    validation = result.get("final_validation", {})
    if (result.get("status") != "completed" or result.get("completed_epochs") != 100
            or (seed is not None and result.get("seed") != seed)
            or (model_key is not None and result.get("model_key") != model_key)
            or validation.get("examples") != 50000):
        raise RuntimeError(f"incomplete or foreign final result: {path}")
    for name in ("accuracy", "top5_accuracy", "cross_entropy"):
        metric = validation.get(name)
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric):
            raise RuntimeError(f"invalid final metric {name}: {path}")
    if not 0 <= validation["accuracy"] <= validation["top5_accuracy"] <= 1 or validation["cross_entropy"] < 0:
        raise RuntimeError(f"invalid final metric range: {path}")
    history = result.get("history")
    if (not isinstance(history, list) or len(history) != 100
            or any(not isinstance(row, dict) or row.get("epoch") != i for i, row in enumerate(history, 1))
            or type(result.get("global_step")) is not int or result["global_step"] <= 0
            or history[-1].get("global_step") != result["global_step"]):
        raise RuntimeError(f"incomplete final history: {path}")


def gpu_pids(uuid: str) -> list[int]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    return [int(row.split(",")[1]) for row in rows.splitlines() if row.split(",")[0].strip() == uuid]


def wait_child(child: subprocess.Popen[bytes], stop: Path, *, grace_seconds: float = 120) -> int:
    stopping_at = None
    while True:
        try:
            return child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            if stop.exists():
                if stopping_at is None:
                    stopping_at = time.monotonic()
                if time.monotonic() - stopping_at >= grace_seconds:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        return child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        return child.wait(timeout=20)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "code", "data-root", "data-manifest", "python", "slot-lock"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--wait-result", type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    return parser.parse_args(argv)


def queue(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    for name in ("root", "code", "data_root", "data_manifest", "slot_lock"):
        setattr(args, name, getattr(args, name).resolve())
    # Keep a virtualenv interpreter path: resolving its symlink changes the environment.
    args.python = Path(os.path.abspath(args.python.expanduser()))
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    source_manifest = (args.source_manifest or root / "source-manifest.json").resolve()
    budget = (args.budget or root / "budget.json").resolve()
    runner = args.code / "scripts/run_va64_conv_control.py"
    config = {
        "source_manifest_sha256": verify_sources(args.code, source_manifest),
        "dataset_manifest_sha256": digest(args.data_manifest),
        "budget_sha256": digest(budget),
        "queue_sha256": digest(Path(__file__)),
        "code": str(args.code), "data_root": str(args.data_root),
        "data_manifest": str(args.data_manifest), "python": str(args.python),
        "gpu_uuid": args.gpu_uuid, "slot_lock": str(args.slot_lock),
        "wait_result": str(args.wait_result.resolve()) if args.wait_result else None,
        "seeds": [0, 1], "epochs": 100, "batch_size": 128,
        "gradient_accumulation_steps": 2, "global_batch": 256, "workers": 8,
    }
    state: dict[str, object] = {"config": config, "supervisor_pid": os.getpid()}
    stop = root / "STOP"

    def update(status: str, **extra: object) -> None:
        state.update(state=status, updated_unix=time.time(), **extra)
        atomic(root / "queue-state.json", state)

    def verify() -> None:
        if (verify_sources(args.code, source_manifest) != config["source_manifest_sha256"]
                or digest(args.data_manifest) != config["dataset_manifest_sha256"]
                or digest(budget) != config["budget_sha256"]):
            raise RuntimeError("pinned source, budget, or dataset manifest changed")

    def stopped() -> bool:
        if stop.exists():
            update("stopped")
            return True
        return False

    def handle_stop(*_: object) -> None:
        stop.touch()

    previous_handlers = {sig: signal.signal(sig, handle_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    owns_queue = False
    try:
        with (root / "queue.lock").open("a+") as own:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owns_queue = True
            config_path = root / "queue-config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise RuntimeError("existing queue configuration changed")
            if not config_path.exists():
                atomic(config_path, config)
            update("waiting")
            args.slot_lock.parent.mkdir(parents=True, exist_ok=True)
            with args.slot_lock.open("a+") as slot:
                while True:
                    if stopped():
                        return
                    verify()
                    if args.wait_result:
                        if not args.wait_result.exists():
                            update("waiting_for_predecessor")
                            time.sleep(args.poll_seconds)
                            continue
                        completed(args.wait_result)
                    try:
                        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        update("waiting_for_slot")
                        time.sleep(args.poll_seconds)
                        continue
                    break
                for seed in (0, 1):
                    if stopped():
                        return
                    verify()
                    result = root / "runs" / MODEL_KEY / f"seed_{seed}" / "result.json"
                    if result.exists():
                        completed(result, seed=seed, model_key=MODEL_KEY)
                        update("skipped_completed", seed=seed)
                        continue
                    for stage, output, extra in (
                        ("smoke", root / "diagnostics", ["--smoke-updates", "20"]),
                        ("training", root / "runs", []),
                    ):
                        if stopped():
                            return
                        verify()
                        log_path = root / "logs" / f"seed{seed}-{stage}.log"
                        # Preserve partial/failed attempts: no implicit overwrite or restart.
                        if log_path.exists() or (output / MODEL_KEY / f"seed_{seed}").exists():
                            raise FileExistsError(f"previous attempt requires review: {log_path}")
                        while True:
                            if stopped():
                                return
                            if shutil.disk_usage(root).free < MIN_FREE_BYTES:
                                raise RuntimeError("less than 15 GiB free; no new run started")
                            pids = gpu_pids(args.gpu_uuid)
                            if not pids:
                                break
                            update("waiting_for_gpu", seed=seed, stage=stage, compute_pids=pids)
                            time.sleep(args.poll_seconds)
                        verify()
                        command = [str(args.python), str(runner), "--code", str(args.code), "--seed", str(seed),
                                   "--data-root", str(args.data_root), "--data-manifest", str(args.data_manifest),
                                   "--output-root", str(output), "--source-manifest", str(source_manifest),
                                   "--budget", str(budget), "--batch-size", "128", "--workers", "8",
                                   "--stop-file", str(stop), *extra]
                        env = os.environ.copy()
                        env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, PYTHONUNBUFFERED="1",
                                   PYTHONPATH=f"{args.code}/src:{args.code}/scripts", OMP_NUM_THREADS="4",
                                   MKL_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="4",
                                   LNET_DISABLE_LAUNCH_AUTOTUNE="1", TRITON_CACHE_DIR=str(root / "cache/triton"),
                                   TORCHINDUCTOR_CACHE_DIR=str(root / "cache/inductor"))
                        update(stage, seed=seed, command=command)
                        with log_path.open("x") as log:
                            child = subprocess.Popen(command, cwd=args.code, env=env, stdout=log,
                                                     stderr=subprocess.STDOUT, start_new_session=True)
                            update(stage, seed=seed, child_pid=child.pid)
                            try:
                                returncode = wait_child(child, stop)
                            finally:
                                if child.poll() is None:
                                    os.killpg(child.pid, signal.SIGTERM)
                                    try:
                                        child.wait(timeout=20)
                                    except subprocess.TimeoutExpired:
                                        os.killpg(child.pid, signal.SIGKILL)
                                        child.wait(timeout=20)
                        if stopped():
                            return
                        if returncode:
                            raise RuntimeError(f"seed {seed} {stage} exited {returncode}; inspect {log_path}")
                        if stage == "smoke":
                            smoke = json.loads((output / MODEL_KEY / f"seed_{seed}" / "smoke/result.json").read_text())
                            if smoke.get("status") != "smoke_passed" or smoke.get("updates") != 20:
                                raise RuntimeError("smoke did not pass all 20 updates")
                        else:
                            completed(result, seed=seed, model_key=MODEL_KEY)
                    update("seed_completed", seed=seed)
                update("completed", finished_unix=time.time())
    except BaseException as error:
        if owns_queue:
            update("stopped" if stop.exists() else "failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    queue(arguments())
