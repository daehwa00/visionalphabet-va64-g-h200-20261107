#!/usr/bin/env python3
"""Restart-safe two-GPU queue for the same-resolution factorial campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_run_artifacts as artifacts
import a2d_r2k3_source_manifest as source_manifest
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as runner
import run_alphabet2d_imagenet100_nano as harness

if TYPE_CHECKING:
    from collections.abc import Iterable


WANDB_URL_PATTERN = re.compile(r"https://wandb\.ai/[^\s]+/runs/[a-zA-Z0-9]+")
MAXIMAL_VARIANTS = tuple(runner.variant_name(width, runner.RESOLUTIONS) for width in runner.WIDTHS)
COMPILE_VARIANTS = frozenset(
    runner.variant_name(width, placement)
    for width in runner.WIDTHS
    for placement in ((), (56,), (28,), (14,), (7,), runner.RESOLUTIONS)
)
PREDECESSOR_TERMINAL_STATES = frozenset({"COMPLETE", "COMPLETE_WITH_FAILURES"})


def _variant_priority(variant: str) -> tuple[int, int, int, tuple[int, ...]]:
    spec = runner.SPECS[variant]
    placement = spec.resolutions
    explicit = {
        runner.variant_name(128, runner.RESOLUTIONS): 0,
        runner.variant_name(96, ()): 1,
        runner.variant_name(96, runner.RESOLUTIONS): 2,
        runner.variant_name(64, ()): 3,
        runner.variant_name(64, runner.RESOLUTIONS): 4,
    }
    if variant in explicit:
        return explicit[variant], 0, 0, ()
    if not placement:
        category = 5
    elif len(placement) == 1:
        category = 6
    elif len(placement) == 2:
        category = 7
    elif len(placement) == 3:
        category = 8
    else:
        category = 9
    width_order = {128: 0, 96: 1, 64: 2}[spec.width]
    resolution_order = tuple(runner.RESOLUTIONS.index(value) for value in placement)
    return category, width_order, len(placement), resolution_order


def _job_partition() -> dict[int, tuple[str, ...]]:
    """Give GPU 1 the five decisive controls before balancing the remaining cells."""
    priority = sorted(runner.TRAIN_VARIANTS, key=_variant_priority)
    if len(priority) != 44:
        message = f"expected 44 new factorial cells, found {len(priority)}"
        raise RuntimeError(message)
    gpu1_controls = (
        runner.variant_name(128, runner.RESOLUTIONS),
        runner.variant_name(96, ()),
        runner.variant_name(96, runner.RESOLUTIONS),
        runner.variant_name(64, ()),
        runner.variant_name(64, runner.RESOLUTIONS),
    )
    remaining = [variant for variant in priority if variant not in gpu1_controls]
    jobs = {
        1: (*gpu1_controls, *remaining[1::2]),
        0: tuple(remaining[::2]),
    }
    if set(jobs[0]).intersection(jobs[1]):
        message = "GPU queues overlap"
        raise RuntimeError(message)
    if set(jobs[0]).union(jobs[1]) != set(runner.TRAIN_VARIANTS):
        message = "GPU queues do not cover every new factorial cell"
        raise RuntimeError(message)
    return jobs


JOBS_BY_GPU = _job_partition()


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=tuple(JOBS_BY_GPU), required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--predecessor-status", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--wandb-project", default="alphabet2d-imagenet100")
    parser.add_argument("--wandb-entity", default="daehwa")
    parser.add_argument(
        "--wandb-group",
        default="R2K3-SameResolution-Factorial-PF15K-S501",
    )
    return parser.parse_args()


def _parse_cpus(specification: str) -> set[int]:
    cpus: set[int] = set()
    for part in specification.split(","):
        bounds = part.split("-", maxsplit=1)
        if len(bounds) == 1:
            cpus.add(int(bounds[0]))
        else:
            cpus.update(range(int(bounds[0]), int(bounds[1]) + 1))
    if not cpus:
        message = "queue CPU affinity cannot be empty"
        raise ValueError(message)
    return cpus


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _source_paths(repo: Path) -> Iterable[Path]:
    yield from source_manifest.dependency_paths(
        repo,
        (
            "queue_a2d_r2k3_same_resolution_factorial",
            "run_a2d_r2k3_same_resolution_factorial_imagenet100",
            "smoke_a2d_r2k3_same_resolution_factorial",
        ),
    )


def _source_fingerprint(repo: Path) -> str:
    return source_manifest.fingerprint(repo, _source_paths(repo))


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


def _append_event(log: Path, payload: dict[str, Any]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(json.dumps(payload) + "\n")


def _wait_for_gpu(args: argparse.Namespace, log: Path) -> None:
    while True:
        processes = _gpu_processes(args.gpu)
        if not processes:
            return
        _append_event(
            log,
            {
                "event": "gpu_wait",
                "gpu": args.gpu,
                "processes": processes,
                "time": _now(),
            },
        )
        time.sleep(args.poll_seconds)


def _wait_for_predecessor(args: argparse.Namespace, log: Path) -> None:
    if args.predecessor_status is None:
        return
    last_observed: str | None = None
    while True:
        observed = "MISSING"
        try:
            payload = json.loads(args.predecessor_status.read_text())
            observed = str(payload.get("status", "RUNNING"))
        except (OSError, json.JSONDecodeError):
            pass
        if observed in PREDECESSOR_TERMINAL_STATES:
            _append_event(
                log,
                {
                    "event": "predecessor_complete",
                    "path": str(args.predecessor_status),
                    "status": observed,
                    "time": _now(),
                },
            )
            return
        if observed != last_observed:
            _append_event(
                log,
                {
                    "event": "predecessor_wait",
                    "path": str(args.predecessor_status),
                    "status": observed,
                    "time": _now(),
                },
            )
            last_observed = observed
        time.sleep(args.poll_seconds)


def _environment(args: argparse.Namespace, variant: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": f"{args.repo / 'src'}:{args.repo / 'scripts'}",
            "PYTHONUNBUFFERED": "1",
            "LNET_COMPILE_MODE": "default",
            "LNET_PERSISTENT_WORKERS": "1",
            "LNET_DATALOADER_WORKERS": str(args.workers),
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            # All cells share one content-addressed cache.  Separate per-cell
            # caches duplicate the same scan/PG kernels dozens of times.
            "TORCHINDUCTOR_CACHE_DIR": str(args.root / "cache" / "smoke"),
            "MALLOC_ARENA_MAX": "2",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "WANDB_MODE": environment.get("WANDB_MODE", "online"),
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_GROUP": args.wandb_group,
            "WANDB_NAME": f"{variant}-S501",
        }
    )
    return environment


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log: Path,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(json.dumps({"event": "launch", "time": _now(), "command": command}) + "\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(
            json.dumps({"event": "exit", "time": _now(), "returncode": completed.returncode}) + "\n"
        )
    return completed.returncode


def _valid_smoke(root: Path, fingerprint: str) -> bool:
    smoke = root / "evidence/cuda/smoke.json"
    marker = root / "evidence/cuda/source.sha256"
    if not smoke.exists() or not marker.exists() or marker.read_text().strip() != fingerprint:
        return False
    try:
        payload = json.loads(smoke.read_text())
        variants = payload["variants"]
        full_batch = payload["full_batch"]
        return bool(
            payload.get("device") == "cuda"
            and payload.get("status") == "passed"
            and payload.get("candidate_count") == len(runner.VARIANTS)
            and payload.get("expected_factorial_cells") == len(runner.VARIANTS)
            and set(variants) == set(runner.VARIANTS)
            and set(payload.get("compile_variants", ())) == COMPILE_VARIANTS
            and all(
                variants[name].get("status") == "passed"
                and artifacts.finite(variants[name].get("eager_loss"))
                and float(variants[name].get("checkpoint_max_abs", -1.0)) == 0.0
                and (
                    name not in COMPILE_VARIANTS
                    or artifacts.finite(variants[name].get("compiled_loss"))
                )
                for name in runner.VARIANTS
            )
            and set(full_batch) == set(MAXIMAL_VARIANTS)
            and all(
                artifacts.finite(full_batch[name].get("loss"))
                and 0.0 < float(full_batch[name].get("peak_bytes", 0.0)) < 24 * 1024**3
                and float(full_batch[name].get("steps", 0.0)) >= 10.0
                for name in MAXIMAL_VARIANTS
            )
            and all(float(value) <= 1.0e-5 for value in payload["reference_parity"].values())
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _ensure_smoke(args: argparse.Namespace, fingerprint: str, log: Path) -> None:
    if _valid_smoke(args.root, fingerprint):
        return
    _wait_for_gpu(args, log)
    smoke_root = args.root / "evidence/cuda"
    command = [
        str(args.python),
        "-u",
        str(args.repo / "scripts/smoke_a2d_r2k3_same_resolution_factorial.py"),
        "--device",
        "cuda",
        "--compile",
        "--full-batch",
        "--full-batch-size",
        str(args.batch_size),
        "--repeat-steps",
        "10",
        "--root",
        str(smoke_root),
    ]
    if (
        _run_logged(
            command,
            cwd=args.repo,
            environment=_environment(args, "smoke-all"),
            log=log,
        )
        != 0
    ):
        message = "factorial CUDA smoke failed; full training was not started"
        raise RuntimeError(message)
    smoke_root.joinpath("source.sha256").write_text(fingerprint + "\n")
    if not _valid_smoke(args.root, fingerprint):
        message = "factorial CUDA smoke evidence is incomplete"
        raise RuntimeError(message)


def _ensure_smoke_once(args: argparse.Namespace, fingerprint: str, log: Path) -> None:
    """Serialize the shared campaign preflight across both GPU queues."""
    lock_path = args.root / "smoke.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _ensure_smoke(args, fingerprint, log)


def _wandb_url(log: Path) -> str | None:
    if not log.exists():
        return None
    matches = WANDB_URL_PATTERN.findall(log.read_text(errors="replace"))
    return matches[-1] if matches else None


def _reference_state() -> dict[str, dict[str, str]]:
    old_root = Path("/home/qlab/experiments/alphabet/r2k3-same-resolution-depth-20260816/runs")
    current_root = Path(
        "/home/qlab/experiments/alphabet/r2k3-same-resolution-fullblock-20260817/runs"
    )
    references: dict[str, dict[str, str]] = {}
    for factorial_name, existing_name in runner.REFERENCE_VARIANTS.items():
        source_root = old_root if existing_name == "K128-PF15K-D1" else current_root
        references[factorial_name] = {
            "existing_variant": existing_name,
            "run_root": str(source_root / existing_name),
        }
    return references


def _campaign_contract(args: argparse.Namespace) -> dict[str, Any]:
    data_manifest, train_images, validation_images = harness._dataset_digest(args.data_root)
    loader_workers = int(args.workers)
    return {
        "seed": 501,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": 1,
        "workers": int(args.workers),
        "loader_workers": loader_workers,
        "loader_persistent_workers": loader_workers > 0,
        "precision": "bfloat16",
        "data_root": str(args.data_root.resolve()),
        "data_manifest_sha256": data_manifest,
        "train_images": train_images,
        "validation_images": validation_images,
        "python": str(args.python.resolve()),
        "wandb_project": str(args.wandb_project),
        "wandb_entity": str(args.wandb_entity),
        "wandb_group": str(args.wandb_group),
    }


def _expected_source_sha256(repo: Path) -> dict[str, str]:
    paths = {
        "same_resolution_factorial_runner": (
            repo / "scripts/run_a2d_r2k3_same_resolution_factorial_imagenet100.py"
        ),
        "r2k3_runtime": repo / "scripts/a2d_r2k3_runtime.py",
        "r2k3_capacity_factory": repo / "scripts/a2d_r2k3_capacity_factory.py",
        "same_resolution_stage": repo / "src/lnet/pac_same_resolution_depth.py",
        "factorized_complex_scan_reader": repo / "src/lnet/pac_factorized_complex_scan_reader.py",
        "gated_post_fusion": repo / "src/lnet/pac_gated_post_fusion.py",
    }
    sources = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    dependency_paths = source_manifest.dependency_paths(
        repo,
        ("run_a2d_r2k3_same_resolution_factorial_imagenet100",),
    )
    sources["r2k3_dependency_tree"] = source_manifest.fingerprint(repo, dependency_paths)
    return sources


def _validate_references(references: dict[str, dict[str, str]]) -> None:
    if references != _reference_state():
        raise RuntimeError("factorial reference mapping changed")
    invalid = [
        factorial_name
        for factorial_name, reference in references.items()
        if not artifacts.valid_run(
            Path(reference["run_root"]),
            reference["existing_variant"],
            100,
        )
    ]
    if invalid:
        raise RuntimeError(f"factorial references are incomplete or invalid: {invalid}")


def _initial_status(args: argparse.Namespace, fingerprint: str) -> dict[str, Any]:
    jobs = JOBS_BY_GPU[args.gpu]
    return {
        "schema": "lnet.a2d.r2k3.same_resolution_factorial.queue.v1",
        "created_at": _now(),
        "updated_at": _now(),
        "gpu": args.gpu,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "source_fingerprint": fingerprint,
        "campaign": _campaign_contract(args),
        "run_source_sha256": _expected_source_sha256(args.repo),
        "factorial_cells": len(runner.VARIANTS),
        "new_training_cells": len(runner.TRAIN_VARIANTS),
        "references": _reference_state(),
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


def _load_status(args: argparse.Namespace, fingerprint: str) -> dict[str, Any]:
    path = args.root / f"queue-gpu{args.gpu}.json"
    if not path.exists():
        return _initial_status(args, fingerprint)
    payload = json.loads(path.read_text())
    if payload.get("source_fingerprint") != fingerprint:
        message = "queue source fingerprint changed; refusing mixed-source resume"
        raise RuntimeError(message)
    campaign = _campaign_contract(args)
    if payload.get("campaign") != campaign:
        message = "queue campaign contract changed; refusing mixed-recipe resume"
        raise RuntimeError(message)
    if payload.get("run_source_sha256") != _expected_source_sha256(args.repo):
        message = "queue run-source contract changed; refusing mixed-source resume"
        raise RuntimeError(message)
    if set(payload.get("jobs", {})) != set(JOBS_BY_GPU[args.gpu]):
        message = "queue membership changed"
        raise RuntimeError(message)
    for state in payload["jobs"].values():
        if state["status"] == "COMPLETE" and not artifacts.valid_run(
            Path(state["run_root"]),
            state["variant"],
            args.epochs,
            campaign=campaign,
            source_sha256=_expected_source_sha256(args.repo),
        ):
            state["status"] = "QUEUED" if int(state["attempts"]) < args.max_attempts else "FAILED"
            state["last_error"] = "completed artifact validation failed during resume"
        elif state["status"] == "RUNNING":
            state["status"] = "QUEUED"
    return payload


def _write_status(args: argparse.Namespace, status: dict[str, Any]) -> None:
    status["updated_at"] = _now()
    _atomic_json(args.root / f"queue-gpu{args.gpu}.json", status)


def _run_job_once(
    args: argparse.Namespace,
    variant: str,
    status: dict[str, Any],
) -> None:
    state = status["jobs"][variant]
    run_root = Path(state["run_root"])
    log = args.root / "logs" / f"gpu{args.gpu}-{variant}.log"
    campaign = status["campaign"]
    source_sha256 = status["run_source_sha256"]
    if artifacts.valid_run(
        run_root,
        variant,
        args.epochs,
        campaign=campaign,
        source_sha256=source_sha256,
    ):
        state["status"] = "COMPLETE"
        state["ended_at"] = state["ended_at"] or _now()
        state["wandb_url"] = state["wandb_url"] or _wandb_url(log)
        _write_status(args, status)
        return

    artifacts.quarantine_invalid_completion(
        run_root,
        variant,
        args.epochs,
        campaign=campaign,
        source_sha256=source_sha256,
    )

    _wait_for_gpu(args, log)
    state["status"] = "RUNNING"
    state["attempts"] = int(state["attempts"]) + 1
    state["started_at"] = state["started_at"] or _now()
    state["last_error"] = None
    _write_status(args, status)

    command = [
        str(args.python),
        "-u",
        str(args.repo / "scripts/run_a2d_r2k3_same_resolution_factorial_imagenet100.py"),
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
    ]
    returncode = _run_logged(
        command,
        cwd=args.repo,
        environment=_environment(args, variant),
        log=log,
    )
    state["wandb_url"] = _wandb_url(log)
    state["ended_at"] = _now()
    valid_result = artifacts.valid_run(
        run_root,
        variant,
        args.epochs,
        campaign=campaign,
        source_sha256=source_sha256,
    )
    if returncode == 0 and valid_result:
        state["status"] = "COMPLETE"
    else:
        state["status"] = "FAILED"
        state["last_error"] = f"training return code {returncode}; valid_result={valid_result}"
    _write_status(args, status)


def _run_job(args: argparse.Namespace, variant: str, status: dict[str, Any]) -> None:
    state = status["jobs"][variant]
    while state["status"] != "COMPLETE" and int(state["attempts"]) < args.max_attempts:
        _run_job_once(args, variant, status)


def main() -> None:
    args = _arguments()
    args.root.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, _parse_cpus(args.cpus))
    lock_path = args.root / f"queue-gpu{args.gpu}.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            message = f"factorial GPU {args.gpu} queue is already running"
            raise RuntimeError(message) from error

        fingerprint = _source_fingerprint(args.repo)
        status = _load_status(args, fingerprint)
        _write_status(args, status)
        preflight_log = args.root / "logs" / f"gpu{args.gpu}-preflight.log"
        _ensure_smoke_once(args, fingerprint, preflight_log)
        status["preflight"] = {
            "status": "PASSED",
            "evidence": str(args.root / "evidence/cuda/smoke.json"),
            "source_fingerprint": fingerprint,
            "completed_at": _now(),
        }
        _write_status(args, status)

        _wait_for_predecessor(args, preflight_log)
        _validate_references(status["references"])
        for variant in JOBS_BY_GPU[args.gpu]:
            _run_job(args, variant, status)
        status["status"] = (
            "COMPLETE"
            if all(state["status"] == "COMPLETE" for state in status["jobs"].values())
            else "COMPLETE_WITH_FAILURES"
        )
        _write_status(args, status)


if __name__ == "__main__":
    main()
