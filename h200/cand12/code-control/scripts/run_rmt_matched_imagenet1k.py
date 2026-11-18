"""Train pinned, unmodified official RMT with the internal VA ImageNet recipe."""

# The internal runtime intentionally exposes its serialization/loader helpers here.
# pyright: reportPrivateUsage=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shutil
import signal
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import imagenet1k_runtime as runtime
import numpy as np
import rmt_indexed_data
import torch
from torch import nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# Match timm Mixup's global NumPy stream used by the VA runs.
# ruff: noqa: NPY002
OFFICIAL_COMMIT = "32ea21417a4d82ba6481f930f94cb4696dda0a8e"
OFFICIAL_SHA256 = "8d8afd5ee29ee55f330898628935d84a0b651d89306e89ed6bcc8bf280ba4dcf"
MODELS = ("RMT_T3", "RMT_S")


def source_file(root: Path) -> Path:
    path = root.resolve() / "classfication_release" / "RMT.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != OFFICIAL_SHA256:
        raise ValueError("official RMT.py differs from the pinned upstream commit")
    return path


def build_model(root: Path, model_name: str) -> nn.Module:
    if model_name not in MODELS:
        raise ValueError("unsupported official RMT model")
    path = source_file(root)
    spec = importlib.util.spec_from_file_location("visionalphabet_official_rmt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import official RMT")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, model_name)(None)


def make_task(args: argparse.Namespace) -> runtime.BaselineTask:
    source_file(args.rmt_source)
    manifest = json.loads(args.data_manifest.read_text())
    if manifest["root"] != str(args.data_root.resolve()) or manifest["counts"] != {
        "train": 1281167,
        "val": 50000,
    }:
        raise ValueError("ImageNet manifest does not match the full dataset")
    output = args.output_root.resolve() / args.model / f"seed_{args.seed}"
    return runtime.BaselineTask(
        phase="full",
        model_key=args.model,
        seed=args.seed,
        learning_rate=0.003,
        epochs=100,
        data_root=args.data_root.resolve(),
        output_dir=output,
        result_path=output / "result.json",
        checkpoint_path=output / "checkpoint.pt",
        source_root=args.rmt_source.resolve(),
        batch_size=args.batch_size,
        workers=args.workers,
        resume=args.resume,
        execution=runtime.ExecutionOptions(
            compile_model=not args.no_compile,
            gpu_mixup=True,
            non_blocking_transfer=True,
            prefetch_factor=2,
            metric_sync_interval=100,
        ),
        source_sha256={
            "official_commit": OFFICIAL_COMMIT,
            "official_RMT.py": OFFICIAL_SHA256,
            "entrypoint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dataset_manifest": hashlib.sha256(args.data_manifest.read_bytes()).hexdigest(),
            **rmt_indexed_data.source_binding(),
        },
    )


def parameter_inventory(model: nn.Module) -> dict[str, int]:
    return {
        "stored_elements": sum(p.numel() for p in model.parameters()),
        "trainable_elements": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "stored_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }


def reset_data_rng(seed: int) -> None:
    random.seed(seed + 1_000_000)
    np.random.seed(seed + 1_000_000)
    torch.manual_seed(seed + 1_000_000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1_000_000)


def make_builder(
    args: argparse.Namespace, task: runtime.BaselineTask
) -> Callable[[str, str | Path | None, int], nn.Module]:
    def builder(key: str, _source: str | Path | None, classes: int) -> nn.Module:
        if key != task.model_key or classes != 1000:
            raise ValueError("builder/task mismatch")
        model = build_model(args.rmt_source, args.model)
        runtime._atomic_json(
            task.output_dir / "parameter-inventory.json", parameter_inventory(model)
        )
        drop_paths = sorted(
            {
                float(value)
                for m in model.modules()
                if isinstance(value := getattr(m, "drop_prob", None), (int, float))
            }
        )
        dropouts = sorted({float(m.p) for m in model.modules() if isinstance(m, nn.Dropout)})
        runtime._atomic_json(
            task.output_dir / "model-definition.json",
            {
                "model": args.model,
                "upstream_commit": OFFICIAL_COMMIT,
                "constructor": f"{args.model}(None)",
                "pretrained": False,
                "drop_path_probabilities": drop_paths,
                "dropout_probabilities": dropouts,
                "note": "Official architecture, head, dropout and stochastic depth are retained; "
                "these differ from VA. Training recipe is reused, not official RMT's recipe.",
            },
        )
        reset_data_rng(task.seed)
        return model

    return builder


def train(args: argparse.Namespace) -> dict[str, Any]:
    task = make_task(args)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_updates:
        return smoke(args, task)
    started = time.monotonic()
    state: dict[str, Any] = {}
    best_path = task.output_dir / "best.json"
    best = json.loads(best_path.read_text())["accuracy"] if best_path.exists() else -math.inf

    def optimizer_builder(
        model: nn.Module, lr: float, device: torch.device
    ) -> torch.optim.Optimizer:
        optimizer = runtime._build_optimizer(model, lr, device)
        updates = 0

        def before_step(
            opt: torch.optim.Optimizer, _args: tuple[Any, ...], _kwargs: dict[str, Any]
        ) -> None:
            nonlocal updates
            updates += 1
            if updates % 100 == 0:
                norms = [
                    p.grad.detach().float().norm()
                    for group in opt.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                norm = torch.stack(norms).norm().item()
                if not math.isfinite(norm):
                    raise RuntimeError("nonfinite gradient norm")
                state["gradient_norm"] = norm

        optimizer.register_step_pre_hook(before_step)
        return optimizer

    def step_callback(row: dict[str, Any]) -> None:
        if not math.isfinite(row["loss"]):
            raise RuntimeError("nonfinite training loss")
        record = {**row, **state, "elapsed_seconds": time.monotonic() - started, **memory_metrics()}
        with (task.output_dir / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print("RMT_TRAIN_STEP=" + json.dumps(record), flush=True)

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
        # Rewrite derived history so recovery cannot duplicate the last epoch.
        checkpoint = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
        temporary = task.output_dir / ".epochs.jsonl.tmp"
        temporary.write_text("".join(json.dumps(r) + "\n" for r in checkpoint["history"]))
        temporary.replace(task.output_dir / "epochs.jsonl")
        print("RMT_EPOCH=" + json.dumps(row), flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = runtime.run_task(
        task,
        model_builder=make_builder(args, task),
        loader_builder=rmt_indexed_data.build_loaders,
        optimizer_builder=optimizer_builder,
        epoch_callback=epoch_callback,
        step_callback=step_callback,
        stop_requested=args.stop_file.is_file,
        stop_without_checkpoint=True,
    )
    runtime._atomic_json(
        task.output_dir / "cost.json",
        {
            "model": args.model,
            "seed": args.seed,
            "status": result["status"],
            "process_seconds": time.monotonic() - started,
            **memory_metrics(),
            "training_seconds": result.get("training_seconds"),
        },
    )
    return result


def memory_metrics() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def smoke(args: argparse.Namespace, task: runtime.BaselineTask) -> dict[str, Any]:
    """Separate real-data diagnostic; never creates a resumable training checkpoint."""
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
    bundle = rmt_indexed_data.build_loaders(task, device)
    count = args.smoke_updates * task.gradient_accumulation_steps
    if len(bundle.train) < count:
        raise ValueError("not enough data for requested smoke updates")

    class FirstBatches:
        def __len__(self) -> int:
            return count

        def __iter__(self) -> Iterator[Any]:
            import itertools

            return itertools.islice(bundle.train, count)

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    metrics, steps = runtime._train_epoch(
        model,
        cast("DataLoader[Any]", cast("object", FirstBatches())),
        optimizer,
        runtime._make_mixup(task),
        task,
        device,
        0,
        stop_requested=args.stop_file.is_file,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    duration = time.monotonic() - started
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("nonfinite parameters after smoke")
    path = task.output_dir / "smoke" / "checkpoint.pt"
    runtime._atomic_torch(
        path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": runtime._capture_rng(bundle, device),
        },
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    runtime._restore_rng(payload["rng"], bundle, device)
    result = {
        "status": "smoke_passed",
        "updates": steps,
        "train": metrics,
        "seconds_including_compile": duration,
        "images_per_second_including_compile": metrics["examples"] / duration,
        "diagnostic_only": True,
        "note": "Short-loader smoke schedule; not a training checkpoint or steady-state benchmark.",
        **memory_metrics(),
    }
    runtime._atomic_json(task.output_dir / "smoke" / "result.json", result)
    print("RMT_SMOKE=" + json.dumps(result), flush=True)
    return result


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmt-source", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32, 64, 128), default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=0)
    args = parser.parse_args(argv)
    if args.smoke_updates < 0 or (args.smoke_updates and args.resume):
        parser.error("smoke updates must be nonnegative and cannot be resumed")
    return args


if __name__ == "__main__":
    cli = arguments()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cli.stop_file.touch())
    try:
        train(cli)
    except runtime.TrainingInterrupted:
        print("RMT_STOPPED_WITHOUT_VALIDATION_OR_BACKUP", flush=True)
        raise SystemExit(130)
