"""Measure canonical VA-family inference with or without constant-folding preparation."""

from __future__ import annotations

# Canonical model builders are script-level imports, as in the training runners.
# pyright: reportImplicitRelativeImport=false
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from va128_inference import prepare_va_inference
from va_family_inference import build_va_family

if TYPE_CHECKING:
    from va128_packed_backbone import PackedVA128Backbone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(32, 48, 64, 96, 128), default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", choices=("reference", "optimized"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch, args.warmup, args.steps, args.groups) < 1:
        parser.error("batch, warmup, steps and groups must be positive")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_va_family(args.width)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in state.items()})
    model.eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    torch.manual_seed(809)
    sample = torch.randn(args.batch, 3, 224, 224, device="cuda").contiguous(
        memory_format=torch.channels_last,
    )
    torch.cuda.reset_peak_memory_stats()
    if args.implementation == "reference":
        model.cuda()
        model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            cast("PackedVA128Backbone", model.model).prepare_for_inference_()
    infer = (
        prepare_va_inference(model, sample)
        if args.implementation == "optimized"
        else torch.compile(model, mode="reduce-overhead", dynamic=False)
    )
    # The prepared engine owns its own copy; release this uncompiled source.
    del model
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.warmup - 1):
            torch.compiler.cudagraph_mark_step_begin()
            infer(sample)
        torch.compiler.cudagraph_mark_step_begin()
        logits = infer(sample)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("inference produced nonfinite logits")
        samples = []
        for _ in range(args.groups):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for _ in range(args.steps):
                torch.compiler.cudagraph_mark_step_begin()
                infer(sample)
            torch.cuda.synchronize()
            samples.append(1000 * (time.perf_counter() - begin) / args.steps)
    median = statistics.median(samples)
    result = {
        "implementation": args.implementation,
        "width": args.width,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_scope": "includes preparation, compilation, warmup and graph capture",
        "batch": args.batch,
        "milliseconds": samples,
        "median_ms": median,
        "images_per_second": 1000 * args.batch / median,
        "warmup": args.warmup,
        "steps_per_group": args.steps,
        "groups": args.groups,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": "BF16 autocast, FP32 inputs/weights; TF32 disabled",
        "checkpoint": str(args.checkpoint.resolve()),
        "parameters": parameters,
        "input": "GPU-resident seeded Gaussian 224x224; channels-last",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
