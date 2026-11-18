"""Bounded real-data RMT-T microbatch benchmark; never writes training checkpoints."""

# Delay scientific imports so the supervisor cannot allocate GPU memory.
# Reuse the matched recipe's internal loader, optimizer and training functions.
# ruff: noqa: NPY002
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


def parse_batches(text: str) -> tuple[int, ...]:
    try:
        batches = tuple(int(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("batches must be comma-separated integers") from error
    if (
        not batches
        or len(set(batches)) != len(batches)
        or any(value not in (8, 16, 32, 64, 128) for value in batches)
    ):
        raise argparse.ArgumentTypeError("use unique batch sizes from 8,16,32,64,128")
    return batches


def state_digest(value: Any) -> str:
    import torch

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for member in item:
                update(member)
        else:
            digest.update(repr(item).encode())
        digest.update(b"\0")

    update(value)
    return digest.hexdigest()


class BoundedBatches:
    def __init__(self, iterator: Iterator[Any], count: int) -> None:
        self.iterator, self.count = iterator, count

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Any]:
        for _ in range(self.count):
            yield next(self.iterator)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def install_stop_handlers(stop_file: Path) -> None:
    def stop(_number: int, _frame: Any) -> None:
        stop_file.touch()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def trial_command(args: argparse.Namespace, batch: int, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--rmt-source",
        str(args.rmt_source),
        "--data-root",
        str(args.data_root),
        "--output",
        str(output),
        "--workers",
        str(args.workers),
        "--batches",
        str(batch),
        "--physical-batch",
        str(batch),
        "--warmup-updates",
        str(args.warmup_updates),
        "--measure-updates",
        str(args.measure_updates),
        "--mode",
        args.mode,
        "--seed",
        str(args.seed),
        "--stop-file",
        str(args.stop_file),
    ]


def wait_for_trial(command: list[str], environment: dict[str, str], timeout: float) -> int:
    """Timeout cleanup includes decode workers in the trial's own process group."""
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    deadline = time.monotonic() + timeout
    stop_file = Path(command[command.index("--stop-file") + 1]) if "--stop-file" in command else None
    stop_deadline = None
    try:
        while True:
            if stop_file is not None and stop_file.exists() and stop_deadline is None:
                stop_deadline = time.monotonic() + 30.0
            remaining = deadline - time.monotonic()
            if remaining <= 0 or (stop_deadline is not None and time.monotonic() >= stop_deadline):
                # The outer launcher owns this supervisor, but this supervisor
                # alone owns the candidate's separate process group.
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                return process.wait(timeout=min(1.0, remaining))
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


def is_oom(error: BaseException) -> bool:
    """Inductor can wrap a genuine CUDA OOM in a compiler exception."""
    import torch

    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, torch.OutOfMemoryError):
            return True
        pending.extend(
            value
            for value in (
                current.__cause__,
                current.__context__,
                getattr(current, "inner_exception", None),
            )
            if isinstance(value, BaseException)
        )
    return False


def run_candidate(args: argparse.Namespace) -> dict[str, Any]:
    import imagenet1k_runtime as runtime
    import numpy as np
    import rmt_indexed_data
    import run_rmt_matched_imagenet1k as rmt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("RMT benchmark requires the assigned CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to the single assigned GPU UUID")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    task = runtime.BaselineTask(
        phase="full",
        model_key="RMT_T3",
        seed=args.seed,
        learning_rate=0.003,
        epochs=100,
        data_root=args.data_root.resolve(),
        output_dir=args.output,
        result_path=args.output / "unused-result.json",
        checkpoint_path=args.output / "unused.pt",
        batch_size=args.physical_batch,
        workers=args.workers,
        execution=runtime.ExecutionOptions(
            compile_model=args.mode == "compiled",
            gpu_mixup=True,
            non_blocking_transfer=True,
            prefetch_factor=2,
            metric_sync_interval=100,
        ),
    )
    model = rmt.build_model(args.rmt_source, "RMT_T3").to(
        device=device,
        memory_format=torch.channels_last,
    )
    # Exact runner protocol: model seed and data/DropPath seed are independent.
    rmt.reset_data_rng(task.seed)
    optimizer = runtime._build_optimizer(model, task.learning_rate, device)
    bundle = rmt_indexed_data.build_loaders(task, device)
    initial_state = {
        "model": state_digest(model.state_dict()),
        "optimizer": state_digest(optimizer.state_dict()),
        "rng": state_digest(runtime._capture_rng(bundle, device)),
        "data_binding": rmt_indexed_data.source_binding(),
    }
    runtime._compile_model_in_place(model, task.execution)
    accumulation = task.gradient_accumulation_steps
    if (args.warmup_updates + args.measure_updates) * accumulation > len(bundle.train):
        raise ValueError("not enough real training data for the requested benchmark")
    iterator = iter(bundle.train)
    mixup = runtime._make_mixup(task)
    full_epoch_updates = math.ceil(len(bundle.train) / accumulation)
    offset = 0
    original_lr = runtime._learning_rate

    def learning_rate(value: Any, step: int, _bounded_steps: int) -> float:
        return original_lr(value, offset + step, full_epoch_updates)

    runtime._learning_rate = learning_rate
    measurements = {}
    for phase, updates in (("warmup", args.warmup_updates), ("measurement", args.measure_updates)):
        print(
            "RMT_BENCHMARK_STAGE="
            + json.dumps(
                {
                    "stage": phase,
                    "event": "begin",
                    "physical_batch": args.physical_batch,
                    "updates": updates,
                    "workers": args.workers,
                    "mode": args.mode,
                }
            ),
            flush=True,
        )
        torch.cuda.synchronize()
        if phase == "measurement":
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        metrics, _ = runtime._train_epoch(
            model,
            BoundedBatches(iterator, updates * accumulation),
            optimizer,
            mixup,
            task,
            device,
            offset,
            stop_requested=args.stop_file.is_file,
        )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        measurements[phase] = {
            "seconds": elapsed,
            "optimizer_updates": int(metrics["optimizer_steps"]),
            "examples": int(metrics["examples"]),
            "images_per_second": metrics["examples"] / elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "loss": metrics["loss"],
        }
        print(
            "RMT_BENCHMARK_STAGE="
            + json.dumps(
                {
                    "stage": phase,
                    "event": "end",
                    **measurements[phase],
                }
            ),
            flush=True,
        )
        offset += updates
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise RuntimeError("benchmark produced nonfinite parameters")
    return {
        "status": "passed",
        "model": "RMT_T3",
        "official_commit": rmt.OFFICIAL_COMMIT,
        "official_source_sha256": rmt.OFFICIAL_SHA256,
        "initial_state": initial_state,
        "physical_batch": args.physical_batch,
        "gradient_accumulation_steps": accumulation,
        "global_batch_size": 256,
        "workers": args.workers,
        "mode": args.mode,
        "compile_model": task.execution.compile_model,
        "warmup_updates": args.warmup_updates,
        "measure_updates": args.measure_updates,
        "device": torch.cuda.get_device_name(),
        "precision": "bfloat16",
        "gpu_mixup": True,
        "real_data": True,
        "loader_wait_and_augmentation_included": True,
        "warmup_excluded_from_measurement": True,
        **measurements,
    }


def select_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if any(result.get("status") not in ("passed", "oom") for result in results):
        raise ValueError("non-OOM candidate failures cannot be used for selection")
    good = [result for result in results if result.get("status") == "passed"]
    if not good:
        raise RuntimeError("all RMT benchmark configurations failed or ran out of memory")
    for result in good:
        measured = result["measurement"]
        if (
            result["physical_batch"] * result["gradient_accumulation_steps"] != 256
            or measured["optimizer_updates"] != result["measure_updates"]
            or measured["examples"] != 256 * result["measure_updates"]
            or not math.isfinite(measured["images_per_second"])
            or measured["images_per_second"] <= 0
        ):
            raise ValueError("invalid/incomplete RMT benchmark result")
    if len({json.dumps(result["initial_state"], sort_keys=True) for result in good}) != 1:
        raise ValueError("candidate model/optimizer/RNG initialization differs")
    winner = max(good, key=lambda result: result["measurement"]["images_per_second"])
    return {
        "status": "selected",
        "model": "RMT_T3",
        "selected_batch_size": winner["physical_batch"],
        "batch_size": winner["physical_batch"],
        "workers": winner["workers"],
        "mode": winner["mode"],
        "compile_model": winner["compile_model"],
        "gradient_accumulation_steps": winner["gradient_accumulation_steps"],
        "global_batch_size": 256,
        "results": results,
        "note": "Real-data steady-state diagnostic. Microbatch changes Mixup pairing, "
        "augmentation worker assignment, BatchNorm statistics and stochastic-depth draws. "
        "One measurement per configuration; no checkpoint is created or reused.",
    }


def candidate_main(args: argparse.Namespace) -> None:
    try:
        result = run_candidate(args)
    except Exception as error:
        import imagenet1k_runtime as runtime

        stopped = isinstance(error, (InterruptedError, runtime.TrainingInterrupted))
        result = {
            "status": "stopped" if stopped else ("oom" if is_oom(error) else "failed"),
            "physical_batch": args.physical_batch,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        atomic_json(args.output / "result.json", result)
        if stopped:
            raise SystemExit(130) from error
        raise
    atomic_json(args.output / "result.json", result)


def compare_candidates(args: argparse.Namespace) -> None:
    results = []
    for batch in args.batches:
        if args.stop_file.is_file():
            raise SystemExit(130)
        output = args.output / f"batch-{batch}"
        environment = os.environ.copy()
        environment.update(
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="4"
        )
        cache_base=Path(environment.get("RMT_COMPILE_CACHE_BASE",str(args.output)))
        environment["TRITON_CACHE_DIR"] = str(cache_base / f"cache-{batch}" / "triton")
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_base / f"cache-{batch}" / "inductor")
        try:
            code = wait_for_trial(
                trial_command(args, batch, output), environment, args.trial_timeout_seconds
            )
        except subprocess.TimeoutExpired as error:
            atomic_json(
                args.output / f"timeout-{batch}.json", {"status": "timeout", "batch": batch}
            )
            raise RuntimeError(f"batch {batch} exceeded its trial timeout") from error
        if args.stop_file.is_file():
            raise SystemExit(130)
        report = output / "result.json"
        if not report.is_file():
            raise RuntimeError(f"batch {batch} exited {code} without a report")
        result = json.loads(report.read_text())
        if result.get("status") == "stopped":
            raise SystemExit(130)
        if result.get("status") not in ("passed", "oom") or (
            result["status"] == "passed" and code != 0
        ):
            raise RuntimeError(f"batch {batch} failed; inspect {report}")
        results.append(result)
    selected = select_candidate(results)
    atomic_json(args.output / "selection.json", selected)
    print("RMT_BENCHMARK_SELECTION=" + json.dumps(selected), flush=True)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmt-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batches", type=parse_batches, default=parse_batches("16,32,64,128"))
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--measure-updates", type=int, default=50)
    parser.add_argument("--mode", choices=("compiled", "eager"), default="compiled")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--trial-timeout-seconds", type=float, default=900)
    parser.add_argument("--physical-batch", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        args.workers < 0
        or min(args.warmup_updates, args.measure_updates) < 1
        or (not math.isfinite(args.trial_timeout_seconds) or args.trial_timeout_seconds <= 0)
    ):
        parser.error("workers must be nonnegative; updates and timeout must be positive")
    if args.physical_batch is not None and args.physical_batch not in args.batches:
        parser.error("internal physical batch must be one of the requested batches")
    return args


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError("benchmark output exists; preserve prior attempts")
    args.output.mkdir(parents=True)
    args.stop_file = args.stop_file or args.output / "STOP"
    install_stop_handlers(args.stop_file)
    if args.physical_batch is not None:
        candidate_main(args)
    else:
        compare_candidates(args)


if __name__ == "__main__":
    main()
