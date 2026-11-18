"""Measure one training epoch on ImageNet-1K, including data loading."""
import argparse
import json
from pathlib import Path
import time

from benchmark_dw_terminal_rmsnorm_training import _install_runner, runner, target_runner
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from timm.data import create_transform
from timm.data.mixup import Mixup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(256, 1024), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--loader-batch", type=int)
    args = parser.parse_args()
    loader_batch = args.loader_batch or args.batch
    if loader_batch < 1 or args.batch % loader_batch:
        parser.error("Loader batch must divide the physical batch")
    chunks_per_batch = args.batch // loader_batch
    if args.profile_updates < 0:
        parser.error("--profile-updates must be nonnegative")
    effective = 1024
    accumulation = effective // args.batch
    torch.manual_seed(501)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    _install_runner()
    recipe = json.loads(Path("contract.json").read_text())["recipe"]
    transform = create_transform(
        input_size=(3, 224, 224), is_training=True, interpolation="bicubic",
        auto_augment="rand-m9-mstd0.5-inc1", re_prob=0.25, re_mode="pixel", re_count=1,
    )
    dataset = ImageFolder(args.data / "train", transform)
    if len(dataset) != 1_281_167 or len(dataset.classes) != 1000:
        raise ValueError(f"Expected full ImageNet-1K train split, got {len(dataset)} images")
    # Keep the same sample count and order in both arms, without a partial update.
    count = len(dataset) // effective * effective
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(501)).tolist()[:count]
    loader = DataLoader(
        dataset, batch_size=loader_batch, sampler=order, num_workers=args.workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=1,
        generator=torch.Generator().manual_seed(501),
    )
    model = target_runner._build(target_runner.VARIANT, runner.runtime.model_config())
    torch.manual_seed(502)
    head = model.classifier.affine.linear
    model.classifier.affine.linear = torch.nn.Linear(head.in_features, 1000)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    model = runner.prepare._prepare_model(model.cuda().train(), recipe)
    runtime = runner.harness._build_runtime(model, recipe)
    optimizer = target_runner._build_optimizer(model, recipe)
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=0, label_smoothing=0.1, num_classes=1000)

    timings = []
    loader_wait = []
    previous_end = None
    profiling = False

    def mark():
        if profiling:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return None

    def update(batches):
        nonlocal previous_end
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            if profiling:
                before_load = time.perf_counter()
            chunks = [next(batches) for _ in range(chunks_per_batch)]
            if profiling:
                loader_wait.append((time.perf_counter() - before_load) * 1000)
            transfer_start = mark()
            if previous_end is not None:
                timings.append(("stream_gap_before_transfer", previous_end, transfer_start))
            image_chunks = [images.cuda(non_blocking=True, memory_format=torch.channels_last)
                            for images, _ in chunks]
            label_chunks = [labels.cuda(non_blocking=True) for _, labels in chunks]
            images = image_chunks[0] if chunks_per_batch == 1 else torch.cat(image_chunks)
            labels = label_chunks[0] if chunks_per_batch == 1 else torch.cat(label_chunks)
            del chunks, image_chunks, label_chunks
            transfer_end = mark()
            images, soft_targets = mixup(images, labels)
            forward_start = mark()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = runtime(images)[0]
                loss = F.cross_entropy(logits, soft_targets)
            forward_end = mark()
            (loss / accumulation).backward()
            backward_end = mark()
            if profiling:
                timings.extend([
                    ("transfer_and_layout", transfer_start, transfer_end),
                    ("mixup", transfer_end, forward_start),
                    ("forward_and_loss", forward_start, forward_end),
                    ("backward", forward_end, backward_end),
                ])
            previous_end = backward_end
        optimizer_start = mark()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer_end = mark()
        if profiling:
            timings.append(("clipping_and_optimizer", optimizer_start, optimizer_end))
        previous_end = optimizer_end
        return loss.detach()

    batches = iter(loader)
    for _ in range(8):
        update(batches)
    torch.cuda.synchronize()
    model.load_state_dict(initial)
    optimizer = target_runner._build_optimizer(model, recipe)
    np.random.seed(501)
    torch.manual_seed(501)
    torch.cuda.synchronize()
    start = time.perf_counter()
    batches = iter(loader)
    updates = count // effective
    if args.profile_updates:
        updates = min(updates, args.profile_updates)
        # Let the loader and GPU settle after restarting the timed iterator.
        for _ in range(16):
            update(batches)
        torch.cuda.synchronize()
        profiling = True
        start = time.perf_counter()
        count = updates * effective
    for step in range(updates):
        loss = update(batches)
        if step % 100 == 99:
            print(f"update={step + 1}/{updates} batch={args.batch} loss={loss.item():.4f}", flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if not loss.isfinite().item() or not all(p.isfinite().all().item() for p in model.parameters()):
        raise RuntimeError("Nonfinite training result")
    if profiling:
        stage_ms = {}
        for stage, first, last in timings:
            stage_ms[stage] = stage_ms.get(stage, 0.0) + first.elapsed_time(last)
        stage_ms = {stage: total / updates for stage, total in stage_ms.items()}
        print("IMAGENET_PROFILE=" + json.dumps(dict(
            batch=args.batch, accumulation=accumulation, workers=args.workers,
            loader_batch=loader_batch, updates=updates,
            images_s=count / seconds, wall_ms_per_update=seconds * 1000 / updates,
            loader_cpu_wait_ms_per_update=sum(loader_wait) / updates,
            cuda_ms_per_update=stage_ms,
        )), flush=True)
    print("IMAGENET_BENCH=" + json.dumps(dict(
        batch=args.batch, accumulation=accumulation, workers=args.workers,
        examples=count, updates=updates, epoch_seconds=seconds, images_s=count / seconds,
        gpu=torch.cuda.get_device_name(), peak_mib=torch.cuda.max_memory_allocated() / 2**20,
    )), flush=True)


if __name__ == "__main__":
    main()
