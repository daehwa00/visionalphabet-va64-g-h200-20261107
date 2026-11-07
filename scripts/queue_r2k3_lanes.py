#!/usr/bin/env python3
"""Restart-safe multi-GPU lane queue for a multi-variant R2K3 campaign.

The campaign is selected with ``--runner``; the runner supplies VARIANTS and the
JOBS_BY_GPU lane map that --gpu is validated against.
"""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
import argparse
import fcntl
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import a2d_r2k3_runtime as runtime
import r2k3_runner_registry as registry
import torch

WANDB_URL_PATTERN = re.compile(r"https://wandb\.ai/[^\s]+/runs/[a-zA-Z0-9]+")
ENTRY_MODULE = Path(__file__).stem
DEFAULT_RUNNER = "run_a2d_r2k3_path_collapse_overnight_imagenet100"
DEFAULT_COMPILE_CACHE_ROOT = Path.home() / ".cache/lnet/compile"


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # --gpu validates against the selected runner's lane map, so the runner has
    # to be resolved before the full parser is built.
    selector = argparse.ArgumentParser(add_help=False)
    registry.add_runner_argument(selector, default=DEFAULT_RUNNER)
    selected, _ = selector.parse_known_args(argv)
    parser = argparse.ArgumentParser(parents=[selector])
    parser.add_argument(
        "--gpu", type=int, choices=tuple(selected.runner.JOBS_BY_GPU), required=True
    )
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--wandb-project", default="alphabet2d-imagenet100")
    parser.add_argument("--wandb-entity", default="daehwa")
    parser.add_argument("--wandb-group", default="R2K3-PathCollapse-Overnight-S501")
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat()


def _parse_cpus(specification: str) -> set[int]:
    cpus: set[int] = set()
    for part in specification.split(","):
        bounds = part.strip().split("-", maxsplit=1)
        if not bounds[0] or (len(bounds) > 1 and not bounds[1]):
            raise ValueError(f"invalid queue CPU affinity: {specification}")
        try:
            start = int(bounds[0])
            stop = int(bounds[-1])
        except ValueError as error:
            raise ValueError(f"invalid queue CPU affinity: {specification}") from error
        if start < 0 or stop < start:
            raise ValueError(f"invalid queue CPU affinity: {specification}")
        cpus.update(range(start, stop + 1))
    if not cpus:
        raise ValueError("queue CPU affinity cannot be empty")
    return cpus


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _gpu_processes(gpu: int) -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(line) for line in completed.stdout.splitlines() if line.strip()]


def _wait_for_gpu(args: argparse.Namespace, event_log: Path) -> None:
    observed: list[int] | None = None
    while True:
        processes = _gpu_processes(args.gpu)
        if not processes:
            return
        if processes != observed:
            _append_event(
                event_log,
                {"event": "gpu_wait", "gpu": args.gpu, "pids": processes, "time": _now()},
            )
            observed = processes
        time.sleep(args.poll_seconds)


def _source_fingerprint(repo: Path, runner_module: str) -> str:
    paths = runtime.source_dependency_paths(
        repo,
        (
            Path(__file__).stem,
            ENTRY_MODULE,
            runner_module,
            "smoke_r2k3_campaign",
        ),
    )
    return runtime.source_fingerprint(repo, paths)


def _cache_component(value: object) -> str:
    component = re.sub(r"[^A-Za-z0-9_.+-]+", "-", str(value)).strip("-")
    return component or "unknown"


def _gpu_cache_identity(gpu: int) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=compute_cap,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line = completed.stdout.strip().splitlines()
    if len(line) != 1:
        raise RuntimeError(f"GPU {gpu} returned an ambiguous cache identity")
    capability, separator, name = line[0].partition(",")
    if not separator:
        raise RuntimeError(f"GPU {gpu} cache identity omitted its product name")
    architecture = f"sm{capability.strip().replace('.', '')}"
    return f"{architecture}-{_cache_component(name.strip())}"


def _compile_stack_namespace() -> Path:
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = "unavailable"
    cudnn_version = torch.backends.cudnn.version()
    return Path(
        f"python-{sys.version_info.major}.{sys.version_info.minor}",
        f"torch-{_cache_component(torch.__version__)}",
        f"cuda-{_cache_component(torch.version.cuda or 'cpu')}",
        f"cudnn-{_cache_component(cudnn_version or 'none')}",
        f"triton-{_cache_component(triton_version)}",
    )


def _compile_cache_paths(
    args: argparse.Namespace,
    environment: dict[str, str] | None = None,
) -> dict[str, Path]:
    active_environment = os.environ if environment is None else environment
    scope = active_environment.get("LNET_COMPILE_CACHE_SCOPE", "shared").strip().lower()
    if scope == "isolated":
        namespace = Path(args.root) / "cache"
    elif scope == "shared":
        base = Path(
            active_environment.get(
                "LNET_COMPILE_CACHE_ROOT",
                str(DEFAULT_COMPILE_CACHE_ROOT),
            )
        ).expanduser()
        namespace = base / _compile_stack_namespace() / _gpu_cache_identity(args.gpu)
    else:
        raise ValueError("LNET_COMPILE_CACHE_SCOPE must be 'shared' or 'isolated'")
    return {
        "namespace": namespace,
        "torchinductor": namespace / "torchinductor",
        "triton": namespace / "triton",
    }


def _environment(args: argparse.Namespace, name: str) -> dict[str, str]:
    environment = os.environ.copy()
    cache = _compile_cache_paths(args, environment)
    cache["torchinductor"].mkdir(parents=True, exist_ok=True)
    cache["triton"].mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": f"{args.repo / 'src'}:{args.repo / 'scripts'}",
            "PYTHONUNBUFFERED": "1",
            "LNET_COMPILE_MODE": "default",
            "LNET_PERSISTENT_WORKERS": "1",
            "LNET_DATALOADER_WORKERS": str(args.workers),
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(cache["torchinductor"]),
            "TRITON_CACHE_DIR": str(cache["triton"]),
            "LNET_COMPILE_CACHE_NAMESPACE": str(cache["namespace"]),
            "MALLOC_ARENA_MAX": "2",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "WANDB_MODE": environment.get("WANDB_MODE", "online"),
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_GROUP": args.wandb_group,
            "WANDB_NAME": f"{name}-S501",
        }
    )
    return environment


def _run_logged(
    command: list[str],
    *,
    args: argparse.Namespace,
    name: str,
    log: Path,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(json.dumps({"event": "launch", "command": command, "time": _now()}) + "\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=args.repo,
            env=_environment(args, name),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(
            json.dumps({"event": "exit", "returncode": completed.returncode, "time": _now()}) + "\n"
        )
    return completed.returncode


def _valid_smoke(args: argparse.Namespace, fingerprint: str) -> bool:
    evidence = args.root / "evidence/cuda/smoke.json"
    marker = args.root / "evidence/cuda/source.sha256"
    if not evidence.exists() or not marker.exists() or marker.read_text().strip() != fingerprint:
        return False
    try:
        payload = json.loads(evidence.read_text())
        variants = payload["variants"]
        full_batch = payload["full_batch"]
        smoke_variants = tuple(getattr(args.runner, "SMOKE_VARIANTS", args.runner.VARIANTS))
        return bool(
            payload.get("status") == "passed"
            and payload.get("device") == "cuda"
            and payload.get("candidate_count") == len(smoke_variants)
            and set(variants) == set(smoke_variants)
            and set(full_batch) == set(smoke_variants)
            and all(
                variants[name].get("status") == "passed"
                and float(variants[name].get("checkpoint_max_abs", -1.0)) == 0.0
                and float(full_batch[name].get("steps", 0.0)) >= 2.0
                and 0.0 < float(full_batch[name].get("peak_bytes", 0.0)) < 24 * 1024**3
                for name in smoke_variants
            )
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _ensure_smoke_once(args: argparse.Namespace, fingerprint: str, event_log: Path) -> None:
    if _valid_smoke(args, fingerprint):
        return
    # Do not hold the cross-GPU smoke lock while this lane is waiting for its
    # GPU.  Otherwise a busy GPU can prevent the other idle GPU from running
    # the one shared preflight.
    _wait_for_gpu(args, event_log)
    lock_path = args.root / "smoke.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _valid_smoke(args, fingerprint):
            return
        _wait_for_gpu(args, event_log)
        shared_lock_path = _compile_cache_paths(args)["namespace"] / "preflight.lock"
        shared_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with shared_lock_path.open("w") as shared_lock:
            fcntl.flock(shared_lock, fcntl.LOCK_EX)
            if _valid_smoke(args, fingerprint):
                return
            _wait_for_gpu(args, event_log)
            evidence_root = args.root / "evidence/cuda"
            smoke_variants = tuple(getattr(args.runner, "SMOKE_VARIANTS", args.runner.VARIANTS))
            returncode = _run_logged(
                [
                    str(args.python),
                    "-u",
                    str(args.repo / "scripts/smoke_r2k3_campaign.py"),
                    "--runner",
                    args.runner.__name__,
                    "--device",
                    "cuda",
                    "--compile",
                    "--full-batch",
                    "--full-batch-size",
                    str(args.batch_size),
                    "--repeat-steps",
                    "2",
                    "--variants",
                    *smoke_variants,
                    "--root",
                    str(evidence_root),
                ],
                args=args,
                name="path-collapse-smoke",
                log=args.root / "logs" / f"gpu{args.gpu}-preflight.log",
            )
            if returncode != 0:
                raise RuntimeError("path-collapse CUDA preflight failed")
            evidence_root.joinpath("source.sha256").write_text(fingerprint + "\n")
            if not _valid_smoke(args, fingerprint):
                raise RuntimeError("path-collapse CUDA preflight evidence is incomplete")


def _result_is_complete(run_root: Path, variant: str, epochs: int) -> bool:
    result = run_root / "results" / f"{variant}__seed501.json"
    checkpoint = run_root / "checkpoints" / f"{variant}__seed501.pt"
    if not result.exists() or not checkpoint.exists():
        return False
    try:
        payload = json.loads(result.read_text())
        history = payload["history"]
        return bool(history and history[-1].get("epoch") == epochs)
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False


def _wandb_url(log: Path) -> str | None:
    if not log.exists():
        return None
    matches = WANDB_URL_PATTERN.findall(log.read_text(errors="replace"))
    return matches[-1] if matches else None


def _campaign(args: argparse.Namespace) -> dict[str, Any]:
    cache = _compile_cache_paths(args)
    return {
        "seed": 501,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "workers": args.workers,
        "persistent_workers": args.workers > 0,
        "precision": "bfloat16",
        "data_root": str(args.data_root.resolve()),
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
        "compile_cache": {
            "scope": os.environ.get("LNET_COMPILE_CACHE_SCOPE", "shared"),
            "namespace": str(cache["namespace"]),
            "torchinductor": str(cache["torchinductor"]),
            "triton": str(cache["triton"]),
        },
    }


def _load_status(args: argparse.Namespace, fingerprint: str) -> dict[str, Any]:
    path = args.root / f"queue-gpu{args.gpu}.json"
    jobs = args.runner.JOBS_BY_GPU[args.gpu]
    if not path.exists():
        return {
            "schema": "lnet.a2d.r2k3.path_collapse_overnight.queue.v1",
            "gpu": args.gpu,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "source_fingerprint": fingerprint,
            "campaign": _campaign(args),
            "created_at": _now(),
            "updated_at": _now(),
            "jobs": {
                variant: {
                    "variant": variant,
                    "status": "QUEUED",
                    "attempts": 0,
                    "run_root": str(args.root / "runs" / variant),
                    "wandb_url": None,
                    "started_at": None,
                    "ended_at": None,
                    "last_error": None,
                }
                for variant in jobs
            },
        }
    payload = json.loads(path.read_text())
    if payload.get("source_fingerprint") != fingerprint:
        raise RuntimeError("queue source changed; refusing a mixed-source resume")
    if payload.get("campaign") != _campaign(args):
        raise RuntimeError("queue recipe changed; refusing a mixed-recipe resume")
    if set(payload.get("jobs", {})) != set(jobs):
        raise RuntimeError("queue membership changed")
    for state in payload["jobs"].values():
        if state["status"] == "RUNNING":
            state["status"] = "QUEUED"
    return payload


def _write_status(args: argparse.Namespace, status: dict[str, Any]) -> None:
    status["updated_at"] = _now()
    _atomic_json(args.root / f"queue-gpu{args.gpu}.json", status)


def _record_completion_if_ready(
    args: argparse.Namespace,
    state: dict[str, Any],
    status: dict[str, Any],
    *,
    run_root: Path,
    variant: str,
    log: Path,
) -> bool:
    if not _result_is_complete(run_root, variant, args.epochs):
        return False
    state.update(status="COMPLETE", ended_at=state["ended_at"] or _now())
    state["wandb_url"] = state["wandb_url"] or _wandb_url(log)
    _write_status(args, status)
    return True


def _run_job(args: argparse.Namespace, variant: str, status: dict[str, Any]) -> None:
    state = status["jobs"][variant]
    run_root = Path(state["run_root"])
    log = args.root / "logs" / f"gpu{args.gpu}-{variant}.log"
    if _record_completion_if_ready(
        args,
        state,
        status,
        run_root=run_root,
        variant=variant,
        log=log,
    ):
        return

    while True:
        _wait_for_gpu(args, args.root / "logs" / f"gpu{args.gpu}-events.jsonl")
        # A restarted supervisor can wait here for the child launched by its
        # predecessor. Reconcile the completed result before consuming another
        # attempt or declaring an already-finished final attempt failed.
        if _record_completion_if_ready(
            args,
            state,
            status,
            run_root=run_root,
            variant=variant,
            log=log,
        ):
            return
        if int(state["attempts"]) >= args.max_attempts:
            break
        state["attempts"] = int(state["attempts"]) + 1
        state.update(
            status="RUNNING",
            started_at=state["started_at"] or _now(),
            last_error=None,
        )
        _write_status(args, status)
        returncode = _run_logged(
            [
                str(args.python),
                "-u",
                str(args.repo / f"scripts/{args.runner.__name__}.py"),
                "--root",
                str(run_root),
                "--data-root",
                str(args.data_root),
                "--variants",
                variant,
                "--run-seeds",
                "501",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--gradient-accumulation-steps",
                "1",
                "--workers",
                str(args.workers),
                "--precision",
                "bfloat16",
            ],
            args=args,
            name=variant,
            log=log,
        )
        state["wandb_url"] = _wandb_url(log)
        if returncode == 0 and _record_completion_if_ready(
            args,
            state,
            status,
            run_root=run_root,
            variant=variant,
            log=log,
        ):
            return
        state.update(
            status="QUEUED",
            last_error=f"returncode={returncode}; complete_result=false",
        )
        _write_status(args, status)
    state.update(status="FAILED", ended_at=_now())
    _write_status(args, status)


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, _parse_cpus(args.cpus))
    lock_path = args.root / f"queue-gpu{args.gpu}.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"GPU {args.gpu} path-collapse queue is already running") from error
        fingerprint = _source_fingerprint(args.repo, args.runner.__name__)
        status = _load_status(args, fingerprint)
        _write_status(args, status)
        _ensure_smoke_once(
            args,
            fingerprint,
            args.root / "logs" / f"gpu{args.gpu}-events.jsonl",
        )
        status["preflight"] = {
            "status": "PASSED",
            "evidence": str(args.root / "evidence/cuda/smoke.json"),
            "source_fingerprint": fingerprint,
            "completed_at": _now(),
        }
        _write_status(args, status)
        for variant in args.runner.JOBS_BY_GPU[args.gpu]:
            _run_job(args, variant, status)
        status["status"] = (
            "COMPLETE"
            if all(state["status"] == "COMPLETE" for state in status["jobs"].values())
            else "COMPLETE_WITH_FAILURES"
        )
        _write_status(args, status)


if __name__ == "__main__":
    main()
