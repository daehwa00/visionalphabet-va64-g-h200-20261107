#!/usr/bin/env python3
"""Matched 200-epoch K128 baseline versus log-damping/multiscale initialization."""

# The launcher imports sibling scripts directly through PYTHONPATH. Checkpoint JSON is dynamic.
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false, reportExplicitAny=false

from __future__ import annotations

import argparse
import faulthandler
import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import imagenet1k_runtime as runtime
import run_lnet_k128_p160_160_160_128_d2262_imagenet1k as anchor
import torch
from torch import nn

if TYPE_CHECKING:
    from torchvision.datasets import ImageFolder

from lnet.pac_pole_optimization import (
    install_log_damping_,
    pole_snapshot,
    project_log_damping_,
)

CAMPAIGN = "k128-optimization-20260929"
VARIANTS = ("baseline", "improved")
EPOCHS = 200
SEED = 501
PARAMETERS = 5_083_176
_stop_state = {"requested": False}


def _campaign_id(args: argparse.Namespace) -> str:
    return str(getattr(args, "campaign_id", CAMPAIGN))


def _stop(_signum: int, _frame: object) -> None:
    _stop_state["requested"] = True


def atomic_json(path: Path, value: object) -> None:
    runtime._atomic_json(path, value)


def build_model(variant: str, classes: int = 1000) -> nn.Module:
    if variant not in VARIANTS or classes != 1000:
        raise ValueError("Expected the fixed K128 ImageNet-1K experiment")
    model = anchor._build_model(anchor.MODEL_KEY, None, classes)
    if variant == "improved":
        install_log_damping_(model, multiscale_initialization=True)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise RuntimeError("K128 parameter budget changed")
    return model


def build_optimizer(
    model: nn.Module, learning_rate: float, device: torch.device
) -> torch.optim.Optimizer:
    optimizer = runtime._build_optimizer(model, learning_rate, device)
    if any(name.endswith("log_damping_x") for name, _ in model.named_parameters()):
        def project_after_update(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
            project_log_damping_(model)

        optimizer.register_step_post_hook(
            project_after_update
        )
    return optimizer


def source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = sorted((root / "src/lnet").glob("*.py")) + sorted((root / "scripts").glob("*.py"))
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {"campaign_runtime_closure": digest.hexdigest()}


def save_poles(output: Path, model: nn.Module, epoch: int, step: int) -> None:
    path = output / "poles" / f"epoch-{epoch:04d}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    snapshot = pole_snapshot(model)
    payload = {
        "schema": "visionalphabet.pole_history.v1",
        "epoch": epoch,
        "global_step": step,
        "banks": snapshot["banks"],
        "initialization": snapshot["initialization"],
    }
    if path.is_file():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                if json.load(stream) == payload:
                    return
        except (OSError, ValueError):
            pass  # The authoritative checkpoint can repair this derived file.
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def make_task(args: argparse.Namespace, variant: str) -> runtime.BaselineTask:
    output = args.root / variant
    return runtime.BaselineTask(
        phase="full",
        model_key=f"k128_{variant}",
        seed=SEED,
        learning_rate=0.003,
        epochs=args.epochs,
        data_root=args.data_root,
        output_dir=output,
        result_path=output / "result.json",
        checkpoint_path=output / "checkpoint.pt",
        source_root=Path(__file__).resolve().parents[1],
        batch_size=args.batch_size,
        effective_batch_size=args.batch_size if getattr(args, "fast", False) else 256,
        workers=args.workers,
        wandb_mode="disabled",
        resume=(output / "checkpoint.pt").is_file(),
        source_sha256=source_identity(),
        execution=(
            _fast_execution() if getattr(args, "fast", False) else runtime.ExecutionOptions()
        ),
    )


def _fast_execution() -> runtime.ExecutionOptions:
    from k128_h200_preflight import fast_execution

    return fast_execution()


def restore_outputs(args: argparse.Namespace) -> None:
    if os.environ.get("H200_AGENT_TOKEN"):
        from k128_h200_transport import restore_backup

        for variant in VARIANTS:
            restore_backup(
                f"k128-{variant}-501", args.root / variant,
                args.root / f"{variant}-restore.tar.gz",
            )


def use_selected_batch(args: argparse.Namespace) -> None:
    if not getattr(args, "fast", False):
        return
    from k128_h200_preflight import saved_selection

    selected = saved_selection(args.root)
    if selected is None:
        raise RuntimeError("The fast campaign requires a successful batch-selection preflight")
    args.batch_size = int(selected["selected_batch"])


def preflight(args: argparse.Namespace) -> None:
    if not args.fast or not torch.cuda.is_available():
        raise RuntimeError("Fast preflight requires CUDA and --fast")
    properties = torch.cuda.get_device_properties(0)
    if args.require_h200 and (
        "H200" not in properties.name or properties.total_memory < 120 * 1024**3
    ):
        raise RuntimeError("The production preflight requires one whole H200")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    restore_outputs(args)
    from k128_h200_preflight import select_batch

    selected = select_batch(
        args.root, build_model, build_optimizer, source_identity(), args.data_root,
        _campaign_id(args),
    )
    args.smoke_batch = int(selected["selected_batch"])
    smoke(args)


def dataset_loaders(task: runtime.BaselineTask, device: torch.device) -> runtime.LoaderBundle:
    bundle = runtime._build_loaders(task, device)
    train_data = cast("ImageFolder", bundle.train.dataset)
    validation_data = cast("ImageFolder", bundle.validation.dataset)
    train_count, val_count = len(train_data), len(validation_data)
    if (train_count, val_count) != (1_281_167, 50_000):
        raise RuntimeError(f"Expected full ImageNet-1K, got train={train_count}, val={val_count}")
    fingerprint = hashlib.sha256()
    for split, dataset in (("train", train_data), ("val", validation_data)):
        for path, target in dataset.samples:
            fingerprint.update(
                f"{split}/{Path(path).relative_to(task.data_root / split)}:{target}:{Path(path).stat().st_size}\n".encode()
            )
    manifest = {
        "train_images": train_count,
        "validation_images": val_count,
        "classes": task.num_classes,
        "relative_path_label_size_sha256": fingerprint.hexdigest(),
    }
    for path in (task.output_dir.parent / "dataset.json", task.output_dir / "dataset.json"):
        if path.is_file() and json.loads(path.read_text()) != manifest:
            raise RuntimeError("Dataset manifest changed between arms or resume sessions")
        atomic_json(path, manifest)
    # Parent telemetry must see a timestamp before the first optimizer update.
    atomic_json(
        task.output_dir / "step-progress.json",
        {"epoch": 0, "global_step": 0, "stage": "data_ready"},
    )
    return bundle


def _backup(args: argparse.Namespace, variant: str, epoch: int, step: int, *, wait: bool) -> None:
    if not os.environ.get("H200_AGENT_TOKEN"):
        return
    from k128_h200_transport import backup_run

    backup_run(
        f"k128-{variant}-501",
        args.root / variant,
        epoch,
        step,
        os.environ.get("K128_CODE_SHA", ""),
        wait=wait,
    )


def run_variant(args: argparse.Namespace) -> int:
    variant = args.variant
    output = args.root / variant
    if os.environ.get("H200_AGENT_TOKEN"):
        from k128_h200_transport import restore_backup

        restore_backup(f"k128-{variant}-501", output, args.root / f"{variant}-restore.tar.gz")
    task = make_task(args, variant)
    if getattr(args, "fast", False) and args.resume_origin is not None:
        from k128_runtime_migration import migrate_checkpoint

        specification = args.resume_origin
        if not specification.is_file():
            raise RuntimeError("The optimized deployment is missing its verified resume origin")
        origins = json.loads(specification.read_text())
        expected_origin = origins.get(variant, {})
        if expected_origin and not task.resume:
            raise RuntimeError("Expected checkpoint recovery is missing; refusing a fresh restart")
        if task.resume:
            migrate_checkpoint(task, variant, torch.device("cuda"), expected_origin)
    atomic_json(
        args.root / "current.json",
        {
            "campaign_id": _campaign_id(args),
            "job": f"k128-{variant}-501",
            "variant": variant,
            "stage": "initializing",
            "output": str(output),
            "epochs": args.epochs,
        },
    )

    def builder(_key: str, _source: str | Path | None, classes: int) -> nn.Module:
        model = build_model(variant, classes)
        if not task.resume:
            save_poles(output, model, 0, 0)
        return model

    def step(record: dict[str, Any]) -> None:
        atomic_json(output / "step-progress.json", {**record, "time": time.time()})
        atomic_json(
            args.root / "current.json",
            {
                "campaign_id": _campaign_id(args),
                "job": f"k128-{variant}-501",
                "variant": variant,
                "stage": "training",
                "output": str(output),
                "epochs": args.epochs,
            },
        )
        print("K128_STEP_JSON=" + json.dumps({"variant": variant, **record}), flush=True)

    def epoch(model: nn.Module, record: dict[str, Any]) -> None:
        number, global_step = int(record["epoch"]), int(record["global_step"])
        save_poles(output, model, number, global_step)
        atomic_json(output / "checkpoint-meta.json", {"phase": "full", **record})
        atomic_json(
            output / "step-progress.json",
            {
                "epoch": number,
                "global_step": global_step,
                "loss": record["train"]["loss"],
                "time": time.time(),
            },
        )
        if (
            number == 1
            or number % 10 == 0
            or number == args.epochs
            or _stop_state["requested"]
            or (args.root / "STOP").exists()
        ):
            _backup(args, variant, number, global_step, wait=True)

    result = runtime.run_task(
        task,
        device=torch.device("cuda"),
        model_builder=builder,
        loader_builder=dataset_loaders,
        optimizer_builder=build_optimizer,
        epoch_callback=epoch,
        step_callback=step,
        stop_requested=lambda: _stop_state["requested"] or (args.root / "STOP").exists(),
    )
    _backup(args, variant, int(result["completed_epochs"]), int(result["global_step"]), wait=True)
    return 0 if result["status"] == "completed" else 130


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.smoke_device)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        if args.require_h200 and (
            "H200" not in properties.name or properties.total_memory < 120 * 1024**3
        ):
            raise RuntimeError(
                "The production experiment requires one whole H200 (GPU allocation 7)"
            )
    if args.smoke_batch < 2:
        raise ValueError("BatchNorm training smoke requires at least two images")
    results = {}
    for variant in VARIANTS:
        print(f"K128_SMOKE_STAGE={variant}:build", flush=True)
        torch.manual_seed(SEED)
        model = build_model(variant).to(device)
        model = model.to(memory_format=torch.channels_last).train()  # pyright: ignore[reportCallIssue]
        optimizer = build_optimizer(model, 0.003, device)
        if getattr(args, "fast", False):
            runtime._compile_model_in_place(model, _fast_execution())
        image = torch.randn(args.smoke_batch, 3, 224, 224, device=device).contiguous(
            memory_format=torch.channels_last
        )
        target = torch.arange(args.smoke_batch, device=device) % 1000
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.monotonic()
        losses = []
        logits: torch.Tensor | None = None
        loss: torch.Tensor | None = None
        for index in range(2):
            print(f"K128_SMOKE_STAGE={variant}:step{index + 1}:forward", flush=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = runtime._logits(model(image), args.smoke_batch, 1000)
                loss = torch.nn.functional.cross_entropy(logits.float(), target)
            print(f"K128_SMOKE_STAGE={variant}:step{index + 1}:backward", flush=True)
            loss.backward()
            if not bool(torch.isfinite(loss)) or not all(
                bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
            ):
                raise RuntimeError(f"Nonfinite smoke loss or gradient in {variant}")
            optimizer.step()
            print(f"K128_SMOKE_STAGE={variant}:step{index + 1}:updated", flush=True)
            losses.append(float(loss.detach()))
            atomic_json(
                args.root / "current.json",
                {
                    "campaign_id": _campaign_id(args),
                    "job": f"k128-{variant}-501",
                    "stage": "preflight",
                    "probe_step": index + 1,
                    "output": str(args.root / variant),
                },
            )
        model.eval()
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            expected = model(image).float().clone()
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        checkpoint = args.root / f"smoke-{variant}.pt"
        runtime._atomic_torch(checkpoint, payload)
        torch.manual_seed(SEED)
        restored = build_model(variant).to(device).eval()
        restored.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True
        )
        restored = restored.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
        if getattr(args, "fast", False):
            runtime._compile_model_in_place(restored, _fast_execution())
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            observed = restored(image).float()
        torch.testing.assert_close(observed, expected, rtol=2e-4, atol=2e-4)
        results[variant] = {
            "losses": losses,
            "batch_size": args.smoke_batch,
            "seconds": time.monotonic() - start,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "strict_reload": True,
            "logit_max_abs_error": float((observed - expected).abs().max()),
        }
        del model, restored, optimizer, payload, expected, observed, image, target, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    receipt = {
        "status": "passed",
        "device": str(device),
        "source": source_identity(),
        "variants": results,
        "compiled": bool(getattr(args, "fast", False)),
    }
    atomic_json(args.root / "smoke.json", receipt)
    print("K128_SMOKE_JSON=" + json.dumps(receipt), flush=True)
    return receipt


def campaign(args: argparse.Namespace) -> int:
    manifest = {
        "campaign_id": _campaign_id(args),
        "variants": list(VARIANTS),
        "epochs": args.epochs,
        "seed": SEED,
        "batch_size": args.batch_size,
        "effective_batch": args.batch_size if getattr(args, "fast", False) else 256,
        "learning_rate": 0.003,
        "weight_decay": 0.05,
        "damping_min": 0.01,
        "gain_gradient": "unchanged_stop_gradient",
        "source": source_identity(),
        "improvement": "projected log damping plus deterministic half-original/half-long-memory atlas",
    }
    if getattr(args, "fast", False):
        manifest["execution"] = asdict(_fast_execution())
        manifest["workers"] = args.workers
    path = args.root / "campaign.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        from k128_runtime_migration import legacy_contract

        old_source = legacy_contract("baseline")["source_sha256"]["campaign_runtime_closure"]
        old = dict(manifest)
        old.pop("execution", None)
        old.pop("workers", None)
        old.update(batch_size=128, effective_batch=256, source={"campaign_runtime_closure": old_source})
        old["campaign_id"] = "k128-optimization-20260929"
        if (
            not getattr(args, "fast", False)
            or args.resume_origin is None
            or json.loads(path.read_text()) != old
        ):
            raise RuntimeError("Campaign identity changed; use a new output directory")
        atomic_json(args.root / "campaign-history" / f"{old_source}.json", old)
    atomic_json(path, manifest)
    for variant in VARIANTS:
        if _stop_state["requested"] or (args.root / "STOP").exists():
            return 130
        command = [
            sys.executable,
            "-B",
            __file__,
            "--root",
            str(args.root),
            "--data-root",
            str(args.data_root),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--variant",
            variant,
            "--campaign-id",
            _campaign_id(args),
        ]
        if getattr(args, "fast", False):
            command.append("--fast")
            if args.resume_origin is not None:
                command.extend(["--resume-origin", str(args.resume_origin)])
        with subprocess.Popen(command) as child:
            signaled = False
            while child.poll() is None:
                if (_stop_state["requested"] or (args.root / "STOP").exists()) and not signaled:
                    child.send_signal(signal.SIGTERM)
                    signaled = True
                time.sleep(1)
            if child.returncode:
                return child.returncode
    atomic_json(
        args.root / "completed.json", {**manifest, "status": "completed", "time": time.time()}
    )
    print("K128_CAMPAIGN_COMPLETED", flush=True)
    return 0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign-id", default=CAMPAIGN)
    parser.add_argument("--data-root", type=Path, default=Path("/app/data/ImageNet-2012"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume-origin", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke-batch", type=int, default=2)
    parser.add_argument("--require-h200", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.data_root = args.data_root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = arguments()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    if args.preflight:
        preflight(args)
        return 0
    if args.smoke:
        faulthandler.dump_traceback_later(180, repeat=True)
        smoke(args)
        faulthandler.cancel_dump_traceback_later()
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Production ImageNet-1K training requires CUDA")
    use_selected_batch(args)
    return run_variant(args) if args.variant else campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
