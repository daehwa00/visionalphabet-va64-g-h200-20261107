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
    args = parser.parse_args()
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
        dataset, batch_size=args.batch, sampler=order, num_workers=args.workers,
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

    def update(batches):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            images, labels = next(batches)
            images = images.cuda(non_blocking=True, memory_format=torch.channels_last)
            labels = labels.cuda(non_blocking=True)
            images, soft_targets = mixup(images, labels)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = runtime(images)[0]
                loss = F.cross_entropy(logits, soft_targets)
            (loss / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
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
    for step in range(updates):
        loss = update(batches)
        if step % 100 == 99:
            print(f"update={step + 1}/{updates} batch={args.batch} loss={loss.item():.4f}", flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if not loss.isfinite().item() or not all(p.isfinite().all().item() for p in model.parameters()):
        raise RuntimeError("Nonfinite training result")
    print("IMAGENET_BENCH=" + json.dumps(dict(
        batch=args.batch, accumulation=accumulation, workers=args.workers,
        examples=count, updates=updates, epoch_seconds=seconds, images_s=count / seconds,
        gpu=torch.cuda.get_device_name(), peak_mib=torch.cuda.max_memory_allocated() / 2**20,
    )), flush=True)


if __name__ == "__main__":
    main()
