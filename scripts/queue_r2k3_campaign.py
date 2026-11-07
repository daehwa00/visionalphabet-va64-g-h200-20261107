#!/usr/bin/env python3
"""Restart-safe one-job queue for a single R2K3 campaign candidate.

The runner is selected with ``--runner`` instead of by one shim file per
campaign.
"""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
import argparse
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import r2k3_runner_registry as registry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Protocol

    class QueueRunner(Protocol):
        VARIANT: str
        __file__: str

WANDB_URL_PATTERN = re.compile(r"https://wandb\.ai/[^\s]+/runs/[a-zA-Z0-9]+")
TERMINAL_STATES = frozenset({"COMPLETE", "FAILED"})


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    registry.add_runner_argument(parser)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--wandb-entity", default="daehwa")
    parser.add_argument("--wandb-project", default="alphabet2d-imagenet100")
    parser.add_argument("--wandb-group", default="R2K3-K128-All-CLPost-S501")
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload) + "\n")


def _result_path(root: Path, active_runner: QueueRunner) -> Path:
    return root / "run" / "results" / f"{active_runner.VARIANT}__seed501.json"


def _completed_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("completed result must be a JSON object")
    return payload


def _is_complete(root: Path, active_runner: QueueRunner, *, epochs: int = 100) -> bool:
    path = _result_path(root, active_runner)
    if not path.exists():
        return False
    try:
        payload = _completed_payload(path)
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid completed-result artifact: {path}") from error
    history = payload.get("history")
    return bool(
        isinstance(history, list)
        and history
        and isinstance(history[-1], dict)
        and history[-1].get("epoch") == epochs
    )


def _wandb_url(log: Path) -> str | None:
    if not log.exists():
        return None
    matches = WANDB_URL_PATTERN.findall(log.read_text(errors="replace"))
    return matches[-1] if matches else None


def _preflight(args: argparse.Namespace, active_runner: QueueRunner) -> dict[str, Any]:
    try:
        payload = json.loads(args.preflight.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"missing valid CUDA preflight: {args.preflight}") from error
    if payload.get("status") != "passed" or payload.get("variant") != active_runner.VARIANT:
        raise RuntimeError("CUDA preflight does not certify the CLPost candidate")
    full_batch = payload.get("full_batch")
    if not isinstance(full_batch, dict) or int(full_batch.get("steps", 0)) < 2:
        raise RuntimeError("CUDA preflight did not exercise repeated full-batch training")
    return payload


def _training_command(args: argparse.Namespace, active_runner: QueueRunner) -> list[str]:
    return [
        str(args.python),
        "-u",
        str(args.repo / "scripts" / Path(active_runner.__file__).name),
        "--root",
        str(args.root / "run"),
        "--data-root",
        str(args.data_root),
        "--variants",
        active_runner.VARIANT,
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


def _previous_state(path: Path, variant: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid queue state: {path}") from error
    if not isinstance(payload, dict) or payload.get("variant") != variant:
        raise RuntimeError(f"queue state belongs to a different variant: {path}")
    return payload


def main_for_runner(active_runner: QueueRunner, args: argparse.Namespace) -> None:
    preflight = _preflight(args, active_runner)
    args.root.mkdir(parents=True, exist_ok=True)
    status_path = args.root / "queue.json"
    event_log = args.root / "logs" / "queue.jsonl"
    training_log = args.root / "logs" / f"{active_runner.VARIANT}.log"
    state: dict[str, Any] = {
        "schema": "lnet.a2d.r2k3.all_clpost.queue.v1",
        "variant": active_runner.VARIANT,
        "gpu": args.gpu,
        "status": "QUEUED",
        "attempts": 0,
        "created_at": _now(),
        "updated_at": _now(),
        "preflight": {
            "path": str(args.preflight),
            "peak_bytes": preflight["full_batch"]["peak_bytes"],
            "steps": preflight["full_batch"]["steps"],
        },
    }
    previous = _previous_state(status_path, active_runner.VARIANT)
    if previous is not None:
        state.update(previous)
    if _is_complete(args.root, active_runner, epochs=args.epochs):
        state.update(status="COMPLETE", updated_at=_now(), wandb_url=_wandb_url(training_log))
        _atomic_json(status_path, state)
        return

    environment = os.environ.copy()
    cache_dir = args.cache_dir or args.root / "cache" / "smoke"
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": f"{args.repo / 'src'}:{args.repo / 'scripts'}",
            "TORCHINDUCTOR_CACHE_DIR": str(cache_dir),
            "TRITON_CACHE_DIR": str(cache_dir),
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_GROUP": args.wandb_group,
        }
    )
    while int(state["attempts"]) < args.max_attempts:
        while True:
            if _is_complete(args.root, active_runner, epochs=args.epochs):
                state.update(
                    status="COMPLETE",
                    updated_at=_now(),
                    wandb_url=_wandb_url(training_log),
                )
                _atomic_json(status_path, state)
                return
            processes = _gpu_processes(args.gpu)
            if not processes:
                break
            if state.get("status") != "WAITING" or state.get("gpu_processes") != processes:
                state.update(
                    status="WAITING",
                    gpu_processes=processes,
                    updated_at=_now(),
                )
                _atomic_json(status_path, state)
                _append_event(event_log, {"event": "gpu_wait", "pids": processes, "time": _now()})
            time.sleep(args.poll_seconds)

        state["attempts"] = int(state["attempts"]) + 1
        state.update(status="RUNNING", gpu_processes=[], started_at=_now(), updated_at=_now())
        _atomic_json(status_path, state)
        command = _training_command(args, active_runner)
        _append_event(event_log, {"event": "launch", "command": command, "time": _now()})
        with training_log.open("a") as stream:
            completed = subprocess.run(
                command,
                cwd=args.repo,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode == 0 and _is_complete(
            args.root,
            active_runner,
            epochs=args.epochs,
        ):
            state.update(
                status="COMPLETE",
                ended_at=_now(),
                updated_at=_now(),
                wandb_url=_wandb_url(training_log),
            )
            _atomic_json(status_path, state)
            return
        state.update(
            status="QUEUED",
            last_returncode=completed.returncode,
            updated_at=_now(),
        )
        _atomic_json(status_path, state)
        _append_event(
            event_log,
            {"event": "retry", "returncode": completed.returncode, "time": _now()},
        )
        time.sleep(args.poll_seconds)

    state.update(status="FAILED", ended_at=_now(), updated_at=_now())
    _atomic_json(status_path, state)
    raise RuntimeError("CLPost confirmation exhausted its retry budget")


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    main_for_runner(args.runner, args)


if __name__ == "__main__":
    main()
