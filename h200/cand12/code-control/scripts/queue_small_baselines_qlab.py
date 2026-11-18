"""Run pinned small-backbone jobs for requested seeds after ADE finishes.

One supervisor owns a durable ledger and one or two assigned GPU lanes. Interrupted/failed attempts
are retained for review; only fully validated results may be skipped on restart.
"""
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
from typing import TypedDict, TextIO, Required

MODELS = ("efficientvmamba_t", "mobilenetv4_conv_small", "repvit_m0_9")
SEEDS = (501, 509, 521)
MIN_FREE_BYTES = 15 * 1024**3
RUNNER = "scripts/run_small_baseline_imagenet1k.py"


class ExecutionChoice(TypedDict):
    batch_size: int
    workers: int
    compile: bool
    images_per_second: float
    selection_file: str
    selection_sha256: str


class SmokeReceipt(TypedDict):
    file: str
    sha256: str
    source_sha256: object
    dataset_sha256: object
    selection_sha256: str


class Task(TypedDict, total=False):
    id: Required[str]
    model: Required[str]
    seed: Required[int]
    state: Required[str]
    gpu_uuid: str
    child_pid: int
    command: list[str]
    log: str
    execution: ExecutionChoice
    returncode: int
    finished_unix: float
    smoke_receipt: SmokeReceipt


class Lane(TypedDict, total=False):
    uuid: Required[str]
    lock: Required[TextIO]
    locked: Required[bool]
    task: Task
    child: subprocess.Popen[bytes]
    stage: str
    ready_training: Task
    ready_smoke: Task


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(code: Path, models_root: Path, manifest: Path) -> str:
    payload = json.loads(manifest.read_text())
    for key, base, folders in (("files", code, ("src", "scripts")),
                               ("model_files", models_root, (".",))):
        files = payload.get(key)
        if not isinstance(files, dict) or not files:
            raise ValueError(f"source manifest needs nonempty {key}")
        for relative, expected in files.items():
            path = (base / relative).resolve()
            if not path.is_relative_to(base) or not path.is_file() or digest(path) != expected:
                raise RuntimeError(f"source checksum mismatch: {key}/{relative}")
        if key == "files":
            actual = {str(path.relative_to(base)) for folder in folders
                      for path in (base / folder).rglob("*.py")}
            if not actual.issubset(files):
                raise ValueError(f"source manifest does not cover complete Python tree: {key}")
    if RUNNER not in payload["files"]:
        raise ValueError("source manifest does not pin the runner")
    return digest(manifest)


def finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def completed(path: Path, model: str, seed: int) -> None:
    result = json.loads(path.read_text())
    validation = result.get("final_validation", {})
    if (result.get("status") != "completed" or result.get("completed_epochs") != 100
            or result.get("seed") != seed or result.get("model_key") != model
            or validation.get("examples") != 50000):
        raise RuntimeError(f"incomplete or foreign final result: {path}")
    if not all(finite(validation.get(key)) for key in ("accuracy", "top5_accuracy", "cross_entropy")):
        raise RuntimeError(f"nonfinite final metrics: {path}")
    if not 0 <= validation["accuracy"] <= validation["top5_accuracy"] <= 1 or validation["cross_entropy"] < 0:
        raise RuntimeError(f"invalid final metric range: {path}")
    history = result.get("history")
    if (not isinstance(history, list) or len(history) != 100
            or any(not isinstance(row, dict) or row.get("epoch") != i for i, row in enumerate(history, 1))
            or type(result.get("global_step")) is not int or result["global_step"] <= 0
            or history[-1].get("global_step") != result["global_step"]):
        raise RuntimeError(f"incomplete final history: {path}")


def ade_completed(result_path: Path) -> bool:
    """A missing endpoint waits; an existing malformed endpoint fails closed."""
    if not result_path.exists():
        return False
    result = json.loads(result_path.read_text())
    progress = result.get("progress", {})
    if progress.get("optimizer_updates") != 80000:
        return False
    if result.get("status") != "complete":
        return False
    evaluation = result.get("evaluation", {})
    if not finite(evaluation.get("mIoU")) or not 0 <= evaluation["mIoU"] <= 1:
        raise RuntimeError("ADE final result has no valid evaluation.mIoU")
    return True


def gpu_pids() -> dict[str, list[int]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    result: dict[str, list[int]] = {}
    for row in output.splitlines():
        if row.strip():
            uuid, pid = row.split(",", 1)
            result.setdefault(uuid.strip(), []).append(int(pid.strip()))
    return result


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "code", "models-root", "source-manifest", "execution-config", "data-root",
                 "data-manifest", "python", "ade-result"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(0, *SEEDS), default=list(SEEDS))
    parser.add_argument("--gpu-uuids", nargs="+", required=True)
    parser.add_argument("--slot-locks", nargs="+", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    return parser.parse_args(argv)


def queue(args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in (0, *SEEDS) for seed in seeds):
        raise ValueError("seeds must be distinct values from 0, 501, 509, 521")
    if (args.poll_seconds <= 0 or not 1 <= len(args.gpu_uuids) <= 2
            or len(args.gpu_uuids) != len(args.slot_locks)
            or len(set(args.gpu_uuids)) != len(args.gpu_uuids)):
        raise ValueError("use one or two distinct GPU lanes, matching slot locks, and a positive poll interval")
    for name in ("root", "code", "models_root", "source_manifest", "execution_config", "data_root",
                 "data_manifest", "ade_result"):
        setattr(args, name, getattr(args, name).resolve())
    args.python = Path(os.path.abspath(args.python.expanduser()))  # Preserve venv symlink.
    args.slot_locks = [path.resolve() for path in args.slot_locks]
    if len(set(args.slot_locks)) != len(args.slot_locks):
        raise ValueError("GPU lanes must use distinct slot locks")
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    stop = root / "STOP"
    config: dict[str, object] = {key: str(getattr(args, key)) for key in (
        "code", "models_root", "source_manifest", "execution_config", "data_root", "data_manifest",
        "python", "ade_result")}
    config.update(gpu_uuids=args.gpu_uuids, slot_locks=list(map(str, args.slot_locks)),
                  seeds=list(seeds), models=list(MODELS), global_batch=256,
                  source_sha256=verify_sources(args.code, args.models_root, args.source_manifest),
                  execution_sha256=digest(args.execution_config), dataset_sha256=digest(args.data_manifest),
                  queue_sha256=digest(Path(__file__)))
    execution = json.loads(args.execution_config.read_text())["models"]
    for model in MODELS:
        choice = execution[model]
        if (choice.get("batch_size") != 128 or choice.get("workers") != 8
                or type(choice.get("compile")) is not bool):
            raise ValueError(f"unsupported execution configuration: {model}")
    tasks: list[Task] = [{"id": f"{model}-seed{seed}", "model": model, "seed": seed, "state": "pending"}
             for index, seed in enumerate(seeds)
             for model in (*MODELS[index:], *MODELS[:index])]
    preflights: list[Task] = [{"id": f"preflight-{model}", "model": model, "seed": seeds[0], "state": "pending"}
                  for model in MODELS]
    resolved: dict[str, ExecutionChoice] = {}
    lanes: list[Lane] = []
    owns_queue = False
    state: dict[str, object] = {"state": "waiting_for_ade", "supervisor_pid": os.getpid(), "jobs": tasks, "preflights": preflights, "resolved_execution": resolved}
    def handle_stop(_signum: int, _frame: object) -> None:
        stop.touch()

    old_handlers = {sig: signal.signal(sig, handle_stop) for sig in (signal.SIGTERM, signal.SIGINT)}

    def save(status: str, **extra: object) -> None:
        state.update(state=status, updated_unix=time.time(), **extra)
        atomic(root / "queue-state.json", state)

    def verify() -> None:
        for model, choice in resolved.items():
            if digest(Path(choice["selection_file"])) != choice["selection_sha256"]:
                raise RuntimeError(f"benchmark selection changed: {model}")
        if (verify_sources(args.code, args.models_root, args.source_manifest) != config["source_sha256"]
                or digest(args.execution_config) != config["execution_sha256"]
                or digest(args.data_manifest) != config["dataset_sha256"]):
            raise RuntimeError("pinned source, execution configuration, or data manifest changed")

    def start(lane: Lane, task: Task, stage: str) -> None:
        verify()
        if shutil.disk_usage(root).free < MIN_FREE_BYTES:
            raise RuntimeError("less than 15 GiB free; no new run started")
        model, seed = task["model"], task["seed"]
        is_preflight = task["id"].startswith("preflight-")
        output = root / ("preflight" if is_preflight else "diagnostics" if stage == "smoke" else "runs")
        log_path = root / "logs" / f"{task['id']}-{stage}.log"
        previous_output = (root / "benchmarks" / model if stage == "benchmark"
                           else output / model / f"seed_{seed}")
        if log_path.exists() or previous_output.exists():
            raise FileExistsError(f"previous attempt requires review: {log_path}")
        command = [str(args.python), str(args.code / ("scripts/benchmark_small_baseline.py" if stage == "benchmark" else RUNNER)),
                   "--model", model, "--models-root", str(args.models_root),
                   "--source-manifest", str(args.source_manifest), "--data-root", str(args.data_root),
                   "--data-manifest", str(args.data_manifest), "--workers", "8", "--stop-file", str(stop)]
        if stage == "benchmark":
            command += ["--output-dir", str(root / "benchmarks" / model), "--seed", str(seed)]
            if not execution[model]["compile"]:
                command += ["--native-only"]
        else:
            command += ["--seed", str(seed), "--output-root", str(output), "--batch-size", "128"]
            if stage == "smoke":
                command += ["--smoke-updates", "20"]
            if not resolved[model]["compile"]:
                command += ["--no-compile"]
            task["execution"] = resolved[model]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=lane["uuid"], PYTHONUNBUFFERED="1",
                   PYTHONPATH=f"{args.code}/runtime-deps:{args.code}/src:{args.code}/scripts", OMP_NUM_THREADS="4",
                   MKL_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="4",
                   TRITON_CACHE_DIR=str(root / "cache" / model / "triton"),
                   TORCHINDUCTOR_CACHE_DIR=str(root / "cache" / model / "inductor"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x") as log:
            child = subprocess.Popen(command, cwd=args.code, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        lane.update(task=task, child=child, stage=stage)
        task.update(state=stage, gpu_uuid=lane["uuid"], child_pid=child.pid,
                    command=command, log=str(log_path))
        save("running")

    def validate_smoke(path: Path) -> None:
        smoke = json.loads(path.read_text())
        if (smoke.get("status") != "smoke_passed" or smoke.get("updates") != 20
                or smoke.get("checkpoint_next_update_replay_passed") is not True):
            raise RuntimeError(f"smoke did not pass 20 updates and checkpoint continuation: {path}")
        batches = smoke.get("full_and_remainder_batches", {})
        train = batches.get("train", {})
        validation = batches.get("validation", {})
        if (batches.get("passed") is not True or batches.get("train_full_batch") != 128
                or batches.get("train_tail_batch") != 15 or batches.get("train_updates") != 1
                or train.get("examples") != 143 or not finite(train.get("loss"))
                or train["loss"] < 0 or batches.get("validation_full_batch") != 128
                or batches.get("validation_tail_batch") != 80 or validation.get("examples") != 208
                or not all(finite(validation.get(key)) for key in ("accuracy", "top5_accuracy", "cross_entropy"))):
            raise RuntimeError(f"smoke lacks valid full/remainder train and validation receipt: {path}")
        if not 0 <= validation["accuracy"] <= validation["top5_accuracy"] <= 1 or validation["cross_entropy"] < 0:
            raise RuntimeError(f"smoke validation metric range invalid: {path}")

    def bind_selection(model: str) -> None:
        path = root / "benchmarks" / model / "selection.json"
        choice = json.loads(path.read_text())
        if (choice.get("batch_size") != 128 or choice.get("workers") != 8
                or type(choice.get("compile")) is not bool
                or not finite(choice.get("images_per_second")) or choice["images_per_second"] <= 0):
            raise RuntimeError(f"invalid benchmark selection: {path}")
        if not execution[model]["compile"] and choice["compile"]:
            raise RuntimeError(f"benchmark selected compile despite native-only policy: {model}")
        resolved[model] = {"batch_size": choice["batch_size"], "workers": choice["workers"],
                           "compile": choice["compile"], "images_per_second": choice["images_per_second"],
                           "selection_file": str(path), "selection_sha256": digest(path)}


    try:
        with (root / "queue.lock").open("a+") as own:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owns_queue = True
            config_path = root / "queue-config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise RuntimeError("existing queue configuration changed")
            if not config_path.exists():
                atomic(config_path, config)
            # Review old ledger before overwriting: failed/running attempts never restart blindly.
            for task in tasks:
                endpoint = root / "runs" / task["model"] / f"seed_{task['seed']}" / "result.json"
                if endpoint.exists():
                    completed(endpoint, task["model"], task["seed"])
                    task["state"] = "completed"
                elif any((root / "logs" / f"{task['id']}-{stage}.log").exists() for stage in ("smoke", "training")):
                    raise FileExistsError(f"previous attempt requires review: {task['id']}")
            for task in preflights:
                model = task["model"]
                selection = root / "benchmarks" / model / "selection.json"
                smoke_path = root / "preflight" / model / f"seed_{seeds[0]}" / "smoke/result.json"
                if selection.exists() and smoke_path.exists():
                    bind_selection(model)
                    validate_smoke(smoke_path)
                    task["state"] = "completed"
                elif any((root / "logs" / f"{task['id']}-{stage}.log").exists() for stage in ("benchmark", "smoke")):
                    raise FileExistsError(f"previous preflight requires review: {model}")
            save("waiting_for_ade")
            while not stop.exists():
                verify()
                if ade_completed(args.ade_result):
                    pids = gpu_pids()
                    if not any(pids.get(uuid) for uuid in args.gpu_uuids):
                        break
                    save("waiting_for_assigned_gpus", compute_pids={uuid: pids.get(uuid, []) for uuid in args.gpu_uuids})
                else:
                    save("waiting_for_ade")
                time.sleep(args.poll_seconds)
            if stop.exists():
                save("stopped")
                return
            for uuid, path in zip(args.gpu_uuids, args.slot_locks, strict=True):
                path.parent.mkdir(parents=True, exist_ok=True)
                lanes.append({"uuid": uuid, "lock": path.open("a+"), "locked": False})
            stopping_at = None
            while True:
                if stop.exists():
                    stopping_at = stopping_at or time.monotonic()
                # Collect finished children before considering another task. Any failure aborts new work.
                for lane in lanes:
                    child = lane.get("child")
                    if child is None or child.poll() is None:
                        continue
                    task, stage = lane.get("task"), lane.get("stage")
                    if task is None or stage is None:
                        raise RuntimeError("active GPU lane has no task/stage")
                    code = child.returncode
                    lane.pop("child")
                    if stop.exists():
                        task.update(state="stopped", returncode=code)
                        continue
                    if code:
                        task.update(state="failed", returncode=code)
                        raise RuntimeError(f"{task['id']} {stage} exited {code}; inspect {task.get('log')}")
                    is_preflight = task["id"].startswith("preflight-")
                    if stage == "benchmark":
                        bind_selection(task["model"])
                        lane["ready_smoke"] = task
                        task["state"] = "benchmarked"
                    elif stage == "smoke":
                        output = root / ("preflight" if is_preflight else "diagnostics")
                        validate_smoke(output / task["model"] / f"seed_{task['seed']}" / "smoke/result.json")
                        if is_preflight:
                            task.update(state="completed", finished_unix=time.time())
                        else:
                            lane["ready_training"] = task
                            task["state"] = "smoke_passed"
                    else:
                        completed(root / "runs" / task["model"] / f"seed_{task['seed']}" / "result.json", task["model"], task["seed"])
                        task.update(state="completed", finished_unix=time.time())
                if stop.exists():
                    active = [lane for lane in lanes if lane.get("child")]
                    if not active:
                        save("stopped")
                        return
                    if stopping_at is not None and time.monotonic() - stopping_at >= 120:
                        for lane in active:
                            child = lane.get("child")
                            if child is not None:
                                os.killpg(child.pid,
                                          signal.SIGKILL if time.monotonic() - stopping_at >= 140 else signal.SIGTERM)
                    save("stopping")
                elif all(task["state"] == "completed" for task in tasks):
                    save("completed", finished_unix=time.time())
                    return
                else:
                    verify()
                    pids = gpu_pids()
                    for lane in lanes:
                        if lane.get("child") or pids.get(lane["uuid"]):
                            continue
                        if not lane["locked"]:
                            try:
                                fcntl.flock(lane["lock"], fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                continue
                            lane["locked"] = True
                        if task := lane.pop("ready_smoke", None):
                            start(lane, task, "smoke")
                        elif task := lane.pop("ready_training", None):
                            start(lane, task, "training")
                        elif not all(row["state"] == "completed" for row in preflights):
                            task = next((row for row in preflights if row["state"] == "pending"), None)
                            if task:
                                start(lane, task, "benchmark")
                        else:
                            task = next((row for row in tasks if row["state"] == "pending"), None)
                            if task:
                                if task["seed"] == seeds[0]:
                                    receipt = root / "preflight" / task["model"] / f"seed_{seeds[0]}" / "smoke/result.json"
                                    validate_smoke(receipt)
                                    task["smoke_receipt"] = {"file": str(receipt), "sha256": digest(receipt),
                                                             "source_sha256": config["source_sha256"],
                                                             "dataset_sha256": config["dataset_sha256"],
                                                             "selection_sha256": resolved[task["model"]]["selection_sha256"]}
                                    start(lane, task, "training")
                                else:
                                    start(lane, task, "smoke")
                    save("running")
                time.sleep(args.poll_seconds)
    except BaseException as error:
        if owns_queue:
            # A systemic error stops any concurrently running sibling normally first.
            stop.touch()
            save("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for lane in lanes:
            child = lane.get("child")
            if child is not None and child.poll() is None:
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=20)
            lane["lock"].close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    queue(arguments())
