"""Train the frozen VA64 scan-only convolution control under the paired recipe."""
from __future__ import annotations

# Runtime serialization helpers intentionally form the existing runner contract.
# pyright: reportPrivateUsage=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportAttributeAccessIssue=false
from collections.abc import Iterator
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CODE = ROOT / "artifacts/h200-va64-g-20261107/repository"
DEFAULT_CODE = FROZEN_CODE if (FROZEN_CODE / "src/lnet").is_dir() else ROOT
# Bootstrap the frozen runtime before any internal model import. --code is also
# accepted to make a standalone copied snapshot independent of this workspace.
_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--code", type=Path, default=DEFAULT_CODE)
_bootstrap_args, _ = _bootstrap.parse_known_args()
sys.path[:0] = [str(_bootstrap_args.code / "scripts"), str(_bootstrap_args.code / "src")]
import imagenet1k_runtime as runtime
import numpy as np
import torch
from torch import nn
import va64_conv_control as control

# timm Mixup deliberately consumes NumPy's legacy global RNG, as in B.
# ruff: noqa: NPY002


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_budget(path: Path) -> dict[str, Any]:
    budget = json.loads(path.read_text())
    width = budget.get("hidden_channels")
    flops = budget.get("flops_per_image")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("budget requires positive integer hidden_channels")
    if budget.get("valid") is not True or not isinstance(flops, (int, float)) or not math.isfinite(flops):
        raise ValueError("budget must contain a valid complete FLOPs audit")
    if abs(flops / control.BASELINE_FLOPS - 1) > 0.05:
        raise ValueError("convolution budget differs from B by more than 5 percent")
    if budget.get("definition") != control.definition(width):
        raise ValueError("budget model definition differs from active convolution")
    expected_source = budget.get("source_sha256", {}).get("scripts/va64_conv_control.py")
    if expected_source != sha256(Path(control.__file__)):
        raise ValueError("budget convolution source hash mismatch")
    if budget.get("unsupported_operators") != {}:
        raise ValueError("budget contains unsupported arithmetic operators")
    return budget


def source_binding(args: argparse.Namespace) -> dict[str, str]:
    manifest = json.loads(args.source_manifest.read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source manifest requires a nonempty files mapping")
    root = args.code.resolve()
    for relative, expected in files.items():
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or not source.is_file() or sha256(source) != expected:
            raise ValueError(f"source manifest mismatch: {relative}")
    # Import a complete frozen tree, never bind only the wrapper while allowing
    # model/runtime changes. The launcher records every Python source file.
    actual = {str(p.relative_to(root)) for folder in ("src", "scripts") for p in (root / folder).rglob("*.py")}
    if not actual.issubset(files):
        raise ValueError("source manifest does not cover the full Python snapshot")
    for name in ("va64_conv_control.py", "run_va64_conv_control.py"):
        path = Path(__file__).with_name(name)
        relative = str(path.resolve().relative_to(root)) if path.resolve().is_relative_to(root) else None
        if relative is None or relative not in files:
            raise ValueError("runner/control must reside inside the pinned code snapshot")
    return {
        "source_manifest": sha256(args.source_manifest),
        "entrypoint": sha256(Path(__file__)),
        "conv_implementation": sha256(Path(control.__file__)),
        "budget": sha256(args.budget),
    }


def make_task(args: argparse.Namespace) -> runtime.BaselineTask:
    budget = load_budget(args.budget)
    manifest = json.loads(args.data_manifest.read_text())
    if manifest["root"] != str(args.data_root.resolve()) or manifest["counts"] != {"train": 1281167, "val": 50000}:
        raise ValueError("ImageNet manifest does not match the full dataset")
    output = args.output_root.resolve() / "va_k64_retrain_C" / f"seed_{args.seed}"
    return runtime.BaselineTask(
        phase="full", model_key="va_k64_retrain_C", seed=args.seed,
        learning_rate=0.003, epochs=100, data_root=args.data_root.resolve(),
        output_dir=output, result_path=output / "result.json", checkpoint_path=output / "checkpoint.pt",
        source_root=args.code.resolve(), batch_size=args.batch_size, workers=args.workers,
        resume=args.resume, wandb_mode="disabled",
        execution=runtime.ExecutionOptions(
            compile_model=not args.no_compile, gpu_mixup=True, non_blocking_transfer=True,
            prefetch_factor=2, metric_sync_interval=100,
        ),
        source_sha256={
            **source_binding(args), "dataset_manifest": sha256(args.data_manifest),
            "ablation_definition": hashlib.sha256(json.dumps(control.definition(budget["hidden_channels"]), sort_keys=True).encode()).hexdigest(),
        },
    )


def reset_data_rng(seed: int) -> None:
    random.seed(seed + 1_000_000)
    np.random.seed(seed + 1_000_000)
    torch.manual_seed(seed + 1_000_000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1_000_000)


def make_builder(args: argparse.Namespace, task: runtime.BaselineTask):
    width = load_budget(args.budget)["hidden_channels"]

    def builder(key: str, _source: str | Path | None, classes: int) -> nn.Module:
        if key != task.model_key or classes != 1000:
            raise ValueError("builder/task mismatch")
        model = control.build_model(width)
        runtime._atomic_json(task.output_dir / "model-definition.json", control.definition(width))
        runtime._atomic_json(task.output_dir / "parameter-inventory.json", {
            "stored_elements": sum(p.numel() for p in model.parameters()),
            "stored_real_scalars": sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()),
            "trainable_elements": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
        reset_data_rng(task.seed)
        return model

    return builder


def memory_metrics() -> dict[str, int]:
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
    }


def smoke(args: argparse.Namespace, task: runtime.BaselineTask) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(task.seed)
    np.random.seed(task.seed)
    torch.manual_seed(task.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(task.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    model = make_builder(args, task)(task.model_key, task.source_root, 1000).to(device)
    model = model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    runtime._compile_model_in_place(model, task.execution)
    optimizer = runtime._build_optimizer(model, task.learning_rate, device)
    bundle = runtime._build_loaders(task, device)
    count = args.smoke_updates * task.gradient_accumulation_steps
    if len(bundle.train) < count:
        raise ValueError("not enough training batches for smoke")

    class FirstBatches:
        def __len__(self) -> int:
            return count

        def __iter__(self) -> Iterator[Any]:
            return itertools.islice(bundle.train, count)

    started = time.monotonic()
    metrics, updates = runtime._train_epoch(
        model, cast(Any, FirstBatches()), optimizer, runtime._make_mixup(task), task, device, 0,
        stop_requested=args.stop_file.is_file,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    if updates != args.smoke_updates or not math.isfinite(metrics["loss"]):
        raise RuntimeError("smoke updates/loss validation failed")
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("nonfinite parameters after smoke")
    checkpoint = task.output_dir / "smoke/checkpoint.pt"
    runtime._atomic_torch(checkpoint, {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": runtime._capture_rng(bundle, device), "contract_sha256": runtime._sha256_payload(runtime._contract(task, device)),
    })
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["contract_sha256"] != runtime._sha256_payload(runtime._contract(task, device)):
        raise RuntimeError("smoke checkpoint contract changed")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    runtime._restore_rng(payload["rng"], bundle, device)
    result = {"status": "smoke_passed", "updates": updates, "train": metrics,
              "seconds_including_compile": time.monotonic() - started,
              "diagnostic_only": True, **memory_metrics()}
    runtime._atomic_json(task.output_dir / "smoke/result.json", result)
    print("VA64_C_SMOKE=" + json.dumps(result), flush=True)
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    task = make_task(args)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stop_file.is_file():
        raise runtime.TrainingInterrupted("STOP present before startup")
    if args.smoke_updates:
        return smoke(args, task)
    started = time.monotonic()
    best_path = task.output_dir / "best.json"
    best = json.loads(best_path.read_text())["accuracy"] if best_path.exists() else -math.inf

    def step_callback(row: dict[str, Any]) -> None:
        if not math.isfinite(row["loss"]):
            raise RuntimeError("nonfinite training loss")
        record = {**row, "elapsed_seconds": time.monotonic() - started, **memory_metrics()}
        with (task.output_dir / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print("VA64_C_STEP=" + json.dumps(record), flush=True)

    def epoch_callback(_model: nn.Module, row: dict[str, Any]) -> None:
        nonlocal best
        accuracy = row["validation"]["accuracy"]
        if accuracy > best:
            temporary = task.output_dir / ".best.pt.tmp"
            shutil.copyfile(task.checkpoint_path, temporary)
            temporary.replace(task.output_dir / "best.pt")
            best = accuracy
            runtime._atomic_json(best_path, {"epoch": row["epoch"], "accuracy": best})
        runtime._atomic_json(task.output_dir / "latest-epoch.json", row)
        checkpoint = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
        temporary = task.output_dir / ".epochs.jsonl.tmp"
        temporary.write_text("".join(json.dumps(r) + "\n" for r in checkpoint["history"]))
        temporary.replace(task.output_dir / "epochs.jsonl")
        print("VA64_C_EPOCH=" + json.dumps(row), flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = runtime.run_task(
        task, model_builder=make_builder(args, task), epoch_callback=epoch_callback,
        step_callback=step_callback, stop_requested=args.stop_file.is_file,
        stop_without_checkpoint=True,
    )
    runtime._atomic_json(task.output_dir / "cost.json", {
        "model": "va_k64_retrain_C", "seed": args.seed, "status": result["status"],
        "process_seconds": time.monotonic() - started, **memory_metrics(),
    })
    return result


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(16, 32, 64, 128, 256), default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke_updates < 0 or (args.smoke_updates and args.resume):
        parser.error("smoke updates must be nonnegative and cannot be resumed")
    return args


if __name__ == "__main__":
    cli = arguments()
    def request_stop(_number: int, _frame: Any) -> None:
        cli.stop_file.touch()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    try:
        train(cli)
    except runtime.TrainingInterrupted:
        print("VA64_C_STOPPED_WITHOUT_VALIDATION_OR_BACKUP", flush=True)
        raise SystemExit(130)
