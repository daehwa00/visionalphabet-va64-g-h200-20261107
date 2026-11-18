"""Dataset manifest and bounded GPU training/resume/cost checks for VA B/P/A."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np

# ruff: noqa: NPY002
import run_va64_retrain_imagenet1k as experiment
import torch
from imagenet1k_runtime import LoaderBundle
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def dataset_manifest(root: Path, output: Path) -> None:
    from torchvision.datasets.folder import IMG_EXTENSIONS

    root = root.resolve()
    classes = None
    counts = {}
    hashes = {}
    listing = output.with_name("dataset-files.jsonl")
    temporary = listing.with_suffix(".tmp")
    with temporary.open("w") as f:
        for split, expected in [("train", 1281167), ("val", 50000)]:
            active = sorted(p.name for p in (root / split).iterdir() if p.is_dir())
            if len(active) != 1000 or (classes is not None and active != classes):
                raise ValueError("unexpected ImageNet class mapping")
            classes = active
            digest = hashlib.sha256()
            count = 0
            for label in classes:
                paths = sorted((root / split / label).rglob("*"))
                for path in paths:
                    if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
                        line = (
                            json.dumps(
                                [path.relative_to(root).as_posix(), path.stat().st_size],
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        f.write(line)
                        digest.update(line.encode())
                        count += 1
            if count != expected:
                raise ValueError(f"{split} count {count}, expected {expected}")
            counts[split] = count
            hashes[split] = digest.hexdigest()
    payload = {
        "root": str(root),
        "counts": counts,
        "classes": classes,
        "membership_size_sha256": hashes,
        "file_list": str(listing),
        "content_hashing": False,
        "note": "Sorted relative filenames and byte sizes; not full image-content hashes.",
    }
    if output.exists() and json.loads(output.read_text()) != payload:
        temporary.unlink()
        raise ValueError("existing dataset manifest changed")
    temporary.replace(listing)
    experiment.runtime._atomic_json(output, payload)
    print(json.dumps({"dataset": counts, "hashes": hashes}), flush=True)


def benchmark(
    variant: str, batch: int, output: Path, warmup: int = 20, measured: int = 100
) -> None:
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = experiment.build_model(variant).to(device, memory_format=torch.channels_last)
    inventory = experiment.parameter_inventory(model)
    expected = 1_322_760 if variant == "A" else 1_562_760
    if inventory["total"]["stored_elements"] != expected:
        raise RuntimeError("parameter inventory mismatch")
    model.compile(mode="default", dynamic=False)
    model.train()
    optimizer = experiment.runtime._build_optimizer(model, 0.003, device)
    physical = torch.randn(batch, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
    labels = torch.arange(batch, device=device) % 1000
    # Keep the same soft-target criterion as the production runtime.
    targets = nn.functional.one_hot(labels, 1000).float() * 0.9 + 0.0001
    finite_losses = []
    grad_norms = []

    def step(index: int) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(256 // batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(physical)
                loss = -(targets * nn.functional.log_softmax(logits.float(), dim=-1)).sum(-1).mean()
            (loss / (256 // batch)).backward()
            loss_value += float(loss.detach()) / (256 // batch)
        if index in (0, 19, warmup + measured - 1):
            grads = [
                p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None
            ]
            norm = float(torch.stack(grads).norm())
            if not math.isfinite(norm):
                raise RuntimeError("nonfinite smoke gradient")
            grad_norms.append(norm)
        if not math.isfinite(loss_value):
            raise RuntimeError("nonfinite smoke loss")
        optimizer.step()
        if variant == "P" and index % 10 == 0:
            experiment.assert_zero_phases(model)
        finite_losses.append(loss_value)

    start = time.monotonic()
    for i in range(warmup):
        step(i)
        if (i + 1) % 10 == 0:
            print(f"{variant} warmup {i + 1}/{warmup}", flush=True)
    torch.cuda.synchronize()
    startup = time.monotonic() - start
    # Real serialization, optimizer/RNG restore, phase and strict shape checks.
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model(physical[:2]).clone()
    data = TensorDataset(torch.zeros(2, 3, 4, 4), torch.zeros(2, dtype=torch.long))
    loader = DataLoader(data, batch_size=2)
    bundle = LoaderBundle(
        loader, loader, torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
    )
    checkpoint = output.with_suffix(".checkpoint.pt")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": warmup,
        "rng": experiment.runtime._capture_rng(bundle, device),
        "variant": variant,
        "definition": experiment.definition_digest(variant),
    }
    experiment.runtime._atomic_torch(checkpoint, state)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if loaded["variant"] != variant or loaded["step"] != warmup:
        raise RuntimeError("smoke checkpoint mode/step mismatch")
    if loaded["definition"] != experiment.definition_digest(variant):
        raise RuntimeError("smoke checkpoint definition mismatch")
    model.load_state_dict(loaded["model"], strict=True)
    optimizer.load_state_dict(loaded["optimizer"])
    experiment.runtime._restore_rng(loaded["rng"], bundle, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        restored = model(physical[:2])
    torch.testing.assert_close(reference, restored, rtol=1e-5, atol=1e-6)
    del loaded, state, reference, restored
    model.train()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for i in range(measured):
        torch.cuda.synchronize()
        start = time.monotonic()
        step(warmup + i)
        torch.cuda.synchronize()
        durations.append(time.monotonic() - start)
        if (i + 1) % 20 == 0:
            print(f"{variant} measured {i + 1}/{measured}", flush=True)
    if variant == "P":
        experiment.assert_zero_phases(model)
    peaks = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    timings = {}
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    optimizer.param_groups.clear()
    physical = torch.empty(0, device=device)
    labels = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    for b in (1, 16):
        x = torch.randn(b, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                model(x)
            samples = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(20):
                start = time.monotonic()
                model(x)
                torch.cuda.synchronize()
                samples.append(time.monotonic() - start)
        timings[str(b)] = {
            "median_ms": 1000 * statistics.median(samples),
            "sd_ms": 1000 * statistics.stdev(samples),
            "images_per_second": b / statistics.median(samples),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    payload = {
        "status": "passed",
        "variant": variant,
        "batch_size": batch,
        "effective_batch": 256,
        "warmup_optimizer_steps": warmup,
        "measured_optimizer_steps": measured,
        "startup_and_warmup_seconds": startup,
        "median_optimizer_step_seconds": statistics.median(durations),
        "optimizer_step_sample_sd_seconds": statistics.stdev(durations),
        "train_images_per_second": 256 / statistics.median(durations),
        "estimated_100ep_training_hours": 100 * 5005 * statistics.median(durations) / 3600,
        "estimate_excludes": "validation, checkpoint I/O, data loading, startup, other contention",
        "peak_training": peaks,
        "inference": timings,
        "parameters": inventory,
        "finite_losses": finite_losses,
        "checked_gradient_norms": grad_norms,
        "checkpoint_roundtrip": "passed",
        "resume_optimizer_step": warmup + 1,
        "synthetic_input": True,
        "accuracy_result": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    experiment.runtime._atomic_json(output, payload)
    print(json.dumps(payload), flush=True)
    checkpoint.unlink()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=experiment.DEFINITIONS)
    p.add_argument("--batch-size", type=int, choices=(64, 128), default=128)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--measured", type=int, default=100)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.data_root:
        dataset_manifest(a.data_root, a.output)
    else:
        benchmark(a.variant, a.batch_size, a.output, a.warmup, a.measured)


if __name__ == "__main__":
    main()
