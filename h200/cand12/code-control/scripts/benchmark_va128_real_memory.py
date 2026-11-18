"""Bounded synthetic training smoke/throughput check for the real-memory redesign."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from run_va_retrain_imagenet1k import build_model, parameter_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("real-memory", "complex"), default="real-memory")
    parser.add_argument("--variant", choices=("B", "P", "A"), default="B")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.warmup, args.steps) <= 0:
        raise ValueError("batch, warmup and steps must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(809)
    torch.backends.cudnn.benchmark = True
    model = build_model(args.variant, architecture=args.architecture).cuda()
    model.to(memory_format=torch.channels_last).train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.003, fused=True
    )
    inputs = torch.randn(args.batch, 3, 224, 224, device="cuda").contiguous(
        memory_format=torch.channels_last
    )
    targets = torch.randint(0, 1000, (args.batch,), device="cuda")
    model.compile(mode="reduce-overhead", dynamic=False)

    def step() -> torch.Tensor:
        torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.float(), targets)
        loss.backward()
        optimizer.step()
        return loss

    print("warmup", args.architecture, args.variant, args.batch, flush=True)
    for _ in range(args.warmup):
        loss = step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(args.steps):
            loss = step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / args.steps)
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    finite = bool(torch.isfinite(loss)) and all(
        bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all()))
        for p in model.parameters()
    )
    result = {
        "architecture": args.architecture,
        "variant": args.variant,
        "batch": args.batch,
        "ms": samples,
        "median_ms": statistics.median(samples),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "loss": float(loss.detach()),
        "finite": finite,
        "missing_gradients": missing,
        "inventory": parameter_inventory(model),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "compile_mode": "reduce-overhead",
        "warmup": args.warmup,
        "steps_per_group": args.steps,
        "scope": "Synthetic GPU-resident BF16 forward/backward + AdamW; no data loading or accuracy claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if missing or not finite:
        raise RuntimeError("training smoke failed")


if __name__ == "__main__":
    main()
