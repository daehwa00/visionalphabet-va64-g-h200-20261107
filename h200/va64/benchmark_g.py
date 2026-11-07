"""Compare VA64-G microbatches using isolated, real ImageNet training processes.

The measurement includes decode/augmentation waits, host-to-device transfer,
GPU Mixup, BF16 forward/backward, metrics, and AdamW updates. Compile and loader
startup are absorbed by the warmup. Neither candidate writes a training checkpoint.
"""

# Delayed imports keep the orchestration process free of CUDA/model allocations.
# The bounded benchmark deliberately reuses the internal scientific runtime.
# ruff: noqa: ANN401, PLC0415, SLF001, NPY002, T201, S603
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
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def state_digest(value: Any) -> str:
    """Hash actual tensor/state values without serializer storage identifiers."""
    import torch

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
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
    """Expose an exact number of batches from one ongoing loader iterator."""

    def __init__(self, iterator: Iterator[Any], count: int, stop_file: Path | None) -> None:
        self.iterator, self.count, self.stop_file = iterator, count, stop_file

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Any]:
        for _ in range(self.count):
            if self.stop_file is not None and self.stop_file.exists():
                raise InterruptedError("benchmark stop requested")
            yield next(self.iterator)


def select_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [value for value in results if value.get("status") == "passed"]
    if not successful:
        raise RuntimeError("neither batch configuration completed the benchmark")
    for value in successful:
        if (
            value["physical_batch"] * value["gradient_accumulation_steps"] != 256
            or value["measurement"]["optimizer_updates"] != value["measure_updates"]
            or value["measurement"]["examples"] != value["measure_updates"] * 256
            or not math.isfinite(value["measurement"]["images_per_second"])
            or value["measurement"]["images_per_second"] <= 0
        ):
            raise ValueError("invalid benchmark measurement")
    signatures = {json.dumps(value["initial_state"], sort_keys=True) for value in successful}
    if len(signatures) != 1:
        raise ValueError("benchmark candidates did not start from identical model/optimizer/RNG")
    winner = max(successful, key=lambda value: value["measurement"]["images_per_second"])
    return {
        "status": "selected",
        "selected_batch_size": winner["physical_batch"],
        "gradient_accumulation_steps": winner["gradient_accumulation_steps"],
        "global_batch_size": 256,
        "selection_metric": "real_data_end_to_end_images_per_second_after_compile_warmup",
        "results": results,
        "training_checkpoint_modified": False,
        "training_rng_modified": False,
        "caveats": [
            "Benchmark uses disposable subprocesses; training reloads the original checkpoint/RNG.",
            "Physical batch changes Mixup pairing/draw count, worker augmentation assignment, "
            "and batch statistics.",
            "Equal global batch does not make the two stochastic trajectories "
            "numerically identical.",
            "Single measurement per configuration; differences may include system "
            "and loader variability.",
        ],
    }


def run_candidate(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915
    import imagenet1k_runtime as runtime
    import numpy as np
    import run_va64_retrain_imagenet1k as experiment
    import torch
    from run_va64_g_h200 import initialize_common_backbone

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA on the assigned H200")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")
    task = runtime.BaselineTask(
        phase="full",
        model_key="va_k64_retrain_G",
        seed=0,
        learning_rate=0.003,
        epochs=100,
        data_root=args.data_root.resolve(),
        output_dir=args.output,
        result_path=args.output / "unused-result.json",
        checkpoint_path=args.output / "unused.pt",
        batch_size=args.physical_batch,
        workers=args.workers,
        execution=runtime.ExecutionOptions(
            compile_model=True,
            gpu_mixup=True,
            non_blocking_transfer=True,
            prefetch_factor=2,
            metric_sync_interval=100,
        ),
    )
    model = initialize_common_backbone(experiment.build_model("G")).to(
        device=device,
        memory_format=torch.channels_last,
    )
    loaders = runtime._build_loaders(task, device)
    optimizer = runtime._build_optimizer(model, task.learning_rate, device)
    completed_epochs = 0
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        runtime._restore_rng(payload["rng"], loaders, device)
        completed_epochs = int(payload.get("completed_epochs", payload.get("epoch", 0)))
        del payload
    initial_state = {
        "model_sha256": state_digest(model.state_dict()),
        "optimizer_sha256": state_digest(optimizer.state_dict()),
        "rng_sha256": state_digest(runtime._capture_rng(loaders, device)),
    }
    runtime._compile_model_in_place(model, task.execution)
    mixup = runtime._make_mixup(task)
    iterator = iter(loaders.train)
    accumulation = task.gradient_accumulation_steps
    required_batches = (args.warmup_updates + args.measure_updates) * accumulation
    if required_batches > len(loaders.train):
        raise ValueError("benchmark exceeds one available training epoch")
    full_epoch_updates = math.ceil(len(loaders.train) / accumulation)
    original_lr = runtime._learning_rate
    offset = completed_epochs * full_epoch_updates

    def learning_rate(value: Any, step: int, _bounded_steps: int) -> float:
        return original_lr(value, offset + step, full_epoch_updates)

    # A bounded loader must not shorten the scientific LR schedule to 30/100 steps.
    runtime._learning_rate = learning_rate
    print(
        f"VA64_BENCHMARK_WARMUP batch={args.physical_batch} updates={args.warmup_updates}",
        flush=True,
    )
    torch.cuda.synchronize()
    started = time.monotonic()
    warmup, _ = runtime._train_epoch(
        model,
        BoundedBatches(iterator, args.warmup_updates * accumulation, args.stop_file),
        optimizer,
        mixup,
        task,
        device,
        offset,
    )
    torch.cuda.synchronize()
    warmup_seconds = time.monotonic() - started
    offset += args.warmup_updates
    print(
        f"VA64_BENCHMARK_MEASURE batch={args.physical_batch} updates={args.measure_updates}",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    measured, _ = runtime._train_epoch(
        model,
        BoundedBatches(iterator, args.measure_updates * accumulation, args.stop_file),
        optimizer,
        mixup,
        task,
        device,
        offset,
    )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise RuntimeError("benchmark produced nonfinite model parameters")
    return {
        "status": "passed",
        "physical_batch": args.physical_batch,
        "gradient_accumulation_steps": accumulation,
        "global_batch_size": 256,
        "warmup_updates": args.warmup_updates,
        "measure_updates": args.measure_updates,
        "initial_state": initial_state,
        "device": torch.cuda.get_device_name(),
        "real_data": True,
        "synthetic_input": False,
        "compiled": True,
        "precision": "bfloat16",
        "gpu_mixup": True,
        "loader_wait_and_augmentation_included": True,
        "warmup": {"seconds": warmup_seconds, **warmup},
        "measurement": {
            "seconds": elapsed,
            "optimizer_updates": int(measured["optimizer_steps"]),
            "examples": int(measured["examples"]),
            "images_per_second": measured["examples"] / elapsed,
            "updates_per_second": measured["optimizer_steps"] / elapsed,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "loss": measured["loss"],
        },
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def install_stop_handlers(stop_file: Path) -> None:
    """Keep the supervisor waiting until its CUDA child observes the stop file."""

    def stop(_signum: int, _frame: Any) -> None:
        stop_file.touch()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-updates", type=int, default=30)
    parser.add_argument("--measure-updates", type=int, default=100)
    parser.add_argument("--physical-batch", type=int, choices=(128, 256), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup_updates < 20 or args.measure_updates < 100 or args.workers < 0:
        parser.error("at least 20 warmup and 100 measured updates; workers must be nonnegative")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error("checkpoint does not exist")
    return args


def candidate_main(args: argparse.Namespace) -> None:
    try:
        result = run_candidate(args)
    except InterruptedError as error:
        atomic_json(args.output / "result.json", {"status": "stopped", "error": str(error)})
        raise SystemExit(130) from error
    except Exception as error:
        import torch

        result = {
            "status": "oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
            "physical_batch": args.physical_batch,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        atomic_json(args.output / "result.json", result)
        raise
    atomic_json(args.output / "result.json", result)
    print("VA64_BENCHMARK=" + json.dumps(result), flush=True)


def compare_candidates(args: argparse.Namespace) -> None:
    results = []
    for batch in (128, 256):
        if args.stop_file and args.stop_file.exists():
            raise SystemExit(130)
        candidate = args.output / f"batch-{batch}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--data-root",
            str(args.data_root),
            "--output",
            str(candidate),
            "--physical-batch",
            str(batch),
            "--workers",
            str(args.workers),
            "--warmup-updates",
            str(args.warmup_updates),
            "--measure-updates",
            str(args.measure_updates),
        ]
        for option, value in (("--checkpoint", args.checkpoint), ("--stop-file", args.stop_file)):
            if value is not None:
                command.extend([option, str(value)])
        # Both candidates own their compilation caches; neither warms the other's cache.
        environment = os.environ.copy()
        environment["TRITON_CACHE_DIR"] = str(args.output / f"cache-{batch}" / "triton")
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(args.output / f"cache-{batch}" / "inductor")
        process = subprocess.run(command, env=environment, check=False)
        if args.stop_file and args.stop_file.exists():
            raise SystemExit(130)
        result_path = candidate / "result.json"
        if not result_path.is_file():
            raise RuntimeError(
                f"batch {batch} exited {process.returncode} without a benchmark report",
            )
        result = json.loads(result_path.read_text())
        if result.get("status") == "stopped":
            raise SystemExit(130)
        if result.get("status") == "passed" and process.returncode != 0:
            raise RuntimeError(f"batch {batch} exited abnormally after writing its report")
        if result.get("status") not in ("passed", "oom"):
            raise RuntimeError(f"batch {batch} failed; inspect {result_path}")
        results.append(result)
    selection = select_candidate(results)
    atomic_json(args.output / "selection.json", selection)
    print("VA64_BENCHMARK_SELECTION=" + json.dumps(selection), flush=True)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError("benchmark output already exists; preserve prior attempts")
    args.output.mkdir(parents=True)
    if args.stop_file is None:
        args.stop_file = args.output / "STOP"
    install_stop_handlers(args.stop_file)
    if args.physical_batch:
        candidate_main(args)
    else:
        compare_candidates(args)


if __name__ == "__main__":
    main()
