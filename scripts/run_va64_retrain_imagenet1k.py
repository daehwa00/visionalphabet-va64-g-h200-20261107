"""Paired scratch VA64 B/P/A ablations; final-epoch ImageNet-1K reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import a2d_r2k3_runtime as ancestry
import imagenet1k_runtime as runtime
import numpy as np  # Legacy timm Mixup consumes the global NumPy RNG.

# ruff: noqa: NPY002
import run_a2d_r2k3_k_family_wave_a_imagenet100 as anchor
import torch
from a2d_r2k3_capacity_factory import CapacityQ4OnlyAffineClassifier
from run_lnet_k128_p160_160_160_128_d2262_imagenet1k import PrimaryLogitsAdapter
from torch import nn

from lnet.complex_scan_stage import ComplexScanStage

DEFINITIONS = {
    "B": {"phase_mode": "learned", "readout_mode": "directional", "head_dim": 320},
    "P": {"phase_mode": "zero", "readout_mode": "directional", "head_dim": 320},
    "A": {"phase_mode": "learned", "readout_mode": "mean_energy", "head_dim": 80},
    "G": {"phase_mode": "learned", "readout_mode": "feature_pool", "head_dim": 640},
}
ORDER = [("B", 0), ("P", 0), ("A", 0)]


def build_model(variant: str, *, prepared: bool = True) -> nn.Module:
    if variant not in DEFINITIONS:
        raise ValueError("unknown ablation variant")
    model = anchor._build("L-K64-U125", ancestry.model_config(output_dim=1000))
    if sum(p.numel() for p in model.parameters()) != 1_562_760:
        raise RuntimeError("original K64 topology changed")
    if variant == "P":
        count = 0
        for module in model.modules():
            if isinstance(module, ComplexScanStage):
                module.phase_mode = "zero"
                for p in (module.phase_x, module.phase_y):
                    with torch.no_grad():
                        p.zero_()
                    p.requires_grad_(requires_grad=False)
                    count += p.numel()
        if count != 1920:
            raise RuntimeError(f"phase inventory changed: {count}")
    elif variant in ("A", "G"):
        # Preserve the RNG state left by B construction, including subsequent
        # stochastic training operations. No learned checkpoint is transplanted.
        with torch.random.fork_rng(devices=[]):
            head_dim = DEFINITIONS[variant]["head_dim"]
            model.descriptor_dim += head_dim - 320
            model.classifier = CapacityQ4OnlyAffineClassifier(model.descriptor_dim, head_dim, 1000)
        model.terminal.readout_mode = DEFINITIONS[variant]["readout_mode"]
    if prepared:
        model.prepare_for_compiled_training_()
    return PrimaryLogitsAdapter(model)


def parameter_inventory(model: nn.Module) -> dict:
    groups = {"backbone": [], "head": [], "phase": []}
    for name, p in model.named_parameters():
        kind = (
            "phase"
            if name.endswith((".phase_x", ".phase_y"))
            else "head"
            if ".classifier." in name
            else "backbone"
        )
        groups[kind].append(p)
    out = {}
    for kind, parameters in groups.items():
        out[kind] = {
            "stored_elements": sum(p.numel() for p in parameters),
            "stored_real_scalars": sum(
                p.numel() * (2 if p.is_complex() else 1) for p in parameters
            ),
            "trainable_elements": sum(p.numel() for p in parameters if p.requires_grad),
        }
    out["total"] = {key: sum(g[key] for g in out.values()) for key in next(iter(out.values()))}
    return out


def assert_zero_phases(model: nn.Module) -> None:
    found = 0
    for module in model.modules():
        if isinstance(module, ComplexScanStage):
            if module.phase_mode != "zero":
                raise RuntimeError("P phase mode was lost")
            for p in (module.phase_x, module.phase_y):
                if p.requires_grad or p.grad is not None or torch.count_nonzero(p).item():
                    raise RuntimeError("P phase constraint was violated")
                found += p.numel()
    if found != 1920:
        raise RuntimeError("P phase inventory was lost")


def definition_digest(variant: str) -> str:
    return hashlib.sha256(json.dumps(DEFINITIONS[variant], sort_keys=True).encode()).hexdigest()


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    return ancestry.source_fingerprint(
        root, ancestry.source_dependency_paths(root, (Path(__file__).stem,))
    )


def make_task(args: argparse.Namespace) -> runtime.BaselineTask:
    manifest = json.loads(args.data_manifest.read_text())
    if manifest["root"] != str(args.data_root.resolve()) or manifest["counts"] != {
        "train": 1281167,
        "val": 50000,
    }:
        raise ValueError("ImageNet manifest does not match the full dataset")
    key = f"va_k64_retrain_{args.variant}"
    output = args.output_root.resolve() / key / f"seed_{args.seed}"
    return runtime.BaselineTask(
        phase="full",
        model_key=key,
        seed=args.seed,
        learning_rate=0.003,
        epochs=100,
        data_root=args.data_root.resolve(),
        output_dir=output,
        result_path=output / "result.json",
        checkpoint_path=output / "checkpoint.pt",
        source_root=Path(__file__).resolve().parents[1],
        batch_size=args.batch_size,
        workers=args.workers,
        resume=args.resume,
        wandb_mode="disabled",
        execution=runtime.ExecutionOptions(
            compile_model=True,
            gpu_mixup=getattr(args, "gpu_mixup", False),
            non_blocking_transfer=True,
            prefetch_factor=2,
            metric_sync_interval=100,
        ),
        source_sha256={
            "source_closure": source_fingerprint(),
            "ablation_definition": definition_digest(args.variant),
            "dataset_manifest": hashlib.sha256(args.data_manifest.read_bytes()).hexdigest(),
        },
    )


def train(args: argparse.Namespace) -> dict:
    task = make_task(args)
    started = time.monotonic()
    state = {}
    task.output_dir.mkdir(parents=True, exist_ok=True)
    runtime._atomic_json(
        task.output_dir / "ablation.json",
        {
            "model": "VA64",
            "variant": args.variant,
            "seed": args.seed,
            "definition": DEFINITIONS[args.variant],
            "pole_widths": [80] * 4,
            "feature_widths": [64] * 4,
            "depth": [2, 2, 6, 2],
            "epochs": 100,
            "effective_batch_size": 256,
            "stop_file": str(args.stop_file),
        },
    )

    def builder(key: str, _source: str | Path | None, classes: int) -> nn.Module:
        if key != task.model_key or classes != 1000:
            raise ValueError("builder/task mismatch")
        model = build_model(args.variant)
        runtime._atomic_json(
            task.output_dir / "parameter-inventory.json", parameter_inventory(model)
        )
        # Augmentation/initial stochastic operations have seed-local streams
        # independent of RNG consumed by the shared model builder.
        random.seed(task.seed + 1_000_000)
        np.random.seed(task.seed + 1_000_000)
        torch.manual_seed(task.seed + 1_000_000)
        torch.cuda.manual_seed_all(task.seed + 1_000_000)
        return model

    def optimizer_builder(
        model: nn.Module, lr: float, device: torch.device
    ) -> torch.optim.Optimizer:
        if args.variant == "P":
            assert_zero_phases(model)
        optimizer = runtime._build_optimizer(model, lr, device)
        count = 0

        def before_step(opt: torch.optim.Optimizer, _args: tuple, _kwargs: dict) -> None:
            nonlocal count
            count += 1
            if count % 100 == 0:
                grads = [
                    p.grad
                    for group in opt.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                norm = torch.stack([g.detach().float().norm() for g in grads]).norm().item()
                if not math.isfinite(norm):
                    raise RuntimeError("nonfinite gradient norm")
                if args.variant == "P":
                    assert_zero_phases(model)
                state["gradient_norm"] = norm
                state["optimizer_state_bytes"] = sum(
                    t.numel() * t.element_size()
                    for values in opt.state.values()
                    for t in values.values()
                    if isinstance(t, torch.Tensor)
                )

        optimizer.register_step_pre_hook(before_step)
        return optimizer

    def step_callback(row: dict[str, Any]) -> None:
        if not math.isfinite(row["loss"]):
            raise RuntimeError("nonfinite training loss")
        record = {
            **row,
            **state,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        with (task.output_dir / "steps.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print("VA_RETRAIN_STEP=" + json.dumps(record), flush=True)

    best = -math.inf
    best_path = task.output_dir / "best.json"
    if best_path.exists():
        best = json.loads(best_path.read_text())["accuracy"]

    def epoch_callback(model: nn.Module, row: dict[str, Any]) -> None:
        nonlocal best
        if args.variant == "P":
            assert_zero_phases(model)
        accuracy = row["validation"]["accuracy"]
        # A managed migration may restore historical best after this closure was built.
        if best_path.exists():
            best = max(best, json.loads(best_path.read_text())["accuracy"])
        # Runtime calls this after committing its latest checkpoint, including
        # on resume to repair derived files after interrupted writes.
        if accuracy > best:
            import shutil

            temporary = task.output_dir / ".best.pt.tmp"
            shutil.copyfile(task.checkpoint_path, temporary)
            temporary.replace(task.output_dir / "best.pt")
            best = accuracy
            runtime._atomic_json(best_path, {"epoch": row["epoch"], "accuracy": best})
        runtime._atomic_json(task.output_dir / "latest-epoch.json", row)
        with (task.output_dir / "epochs.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    torch.cuda.reset_peak_memory_stats()
    result = runtime.run_task(
        task,
        model_builder=builder,
        optimizer_builder=optimizer_builder,
        epoch_callback=epoch_callback,
        step_callback=step_callback,
        stop_requested=args.stop_file.is_file,
    )
    training_seconds = result.get("training_seconds")
    if training_seconds is None:
        training_seconds = json.loads((task.output_dir / "latest-epoch.json").read_text())[
            "training_seconds"
        ]
    runtime._atomic_json(
        task.output_dir / "cost.json",
        {
            "variant": args.variant,
            "seed": args.seed,
            "process_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "status": result["status"],
            "training_seconds": training_seconds,
            "gpu_hours": training_seconds / 3600,
            "note": "GPU hours include validation and checkpointing; process time includes startup.",
        },
    )
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=DEFINITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-mixup", action="store_true")
    parser.add_argument("--stop-file", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(arguments())
