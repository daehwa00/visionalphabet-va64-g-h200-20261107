#!/usr/bin/env python3
"""Benchmark direct full16-cell collapse against reconstruction plus collapse."""

from __future__ import annotations

# pyright: reportExplicitAny=false
import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lnet.pac_grouped_path_cffn import (
    GroupedWidelyLinear,
    grouped_cartesian_cell_cffn,
    grouped_cartesian_cffn,
)
from lnet.pac_product_scan_reference import direction_relative_cells_to_full_resolution


class _Full16Collapse(nn.Module):
    def __init__(self, modes: int, hidden: int, *, direct: bool) -> None:
        super().__init__()
        self.direct = direct
        self.input_projection = GroupedWidelyLinear(modes, 4, hidden)
        self.output_projection = GroupedWidelyLinear(modes, hidden, 1)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        if self.direct:
            return grouped_cartesian_cell_cffn(
                real,
                imag,
                input_projection=self.input_projection,
                output_projection=self.output_projection,
            )
        full_real = direction_relative_cells_to_full_resolution(real).contiguous()
        full_imag = direction_relative_cells_to_full_resolution(imag).contiguous()
        return grouped_cartesian_cffn(
            full_real,
            full_imag,
            input_projection=self.input_projection,
            output_projection=self.output_projection,
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--modes", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--sizes", type=int, nargs="+", default=(56, 28, 14))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--minimum-speedup", type=float, default=1.15)
    parser.add_argument("--maximum-memory-ratio", type=float, default=1.05)
    return parser.parse_args()


def _loss(output: tuple[Tensor, Tensor]) -> Tensor:
    return output[0].float().square().mean() + output[1].float().square().mean()


def _parity(
    modes: int,
    hidden: int,
    size: int,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(20260817 + size)
    reference = _Full16Collapse(modes, hidden, direct=False).to(device)
    candidate = _Full16Collapse(modes, hidden, direct=True).to(device)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    coarse = size // 2
    real = torch.randn(
        1,
        coarse,
        coarse,
        4,
        4,
        modes,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    imag = torch.randn_like(real, requires_grad=True)
    candidate_real = real.detach().clone().requires_grad_()
    candidate_imag = imag.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = reference(real, imag)
        actual = candidate(candidate_real, candidate_imag)
    cotangent = tuple(torch.randn_like(value) for value in actual)
    expected_tensors = (real, imag, *reference.parameters())
    actual_tensors = (candidate_real, candidate_imag, *candidate.parameters())
    expected_gradients = torch.autograd.grad(expected, expected_tensors, cotangent)
    actual_gradients = torch.autograd.grad(actual, actual_tensors, cotangent)

    def relative_l2(value: Tensor, target: Tensor) -> float:
        difference = value.detach().float() - target.detach().float()
        denominator = target.detach().float().square().sum().sqrt().clamp_min(1.0e-12)
        return float(difference.square().sum().sqrt() / denominator)

    return {
        "output_max_abs": max(
            float((value.detach().float() - target.detach().float()).abs().max())
            for value, target in zip(actual, expected, strict=True)
        ),
        "output_relative_l2": max(
            relative_l2(value, target) for value, target in zip(actual, expected, strict=True)
        ),
        "gradient_max_relative_l2": max(
            relative_l2(value, target)
            for value, target in zip(actual_gradients, expected_gradients, strict=True)
        ),
    }


def _benchmark(
    module: nn.Module,
    *,
    batch_size: int,
    modes: int,
    size: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    torch.compiler.reset()
    module = module.to(device).train()
    compiled = torch.compile(module, mode="default", fullgraph=False, dynamic=False)
    coarse = size // 2
    real = torch.randn(
        batch_size,
        coarse,
        coarse,
        4,
        4,
        modes,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    imag = torch.randn_like(real, requires_grad=True)

    def step() -> None:
        module.zero_grad(set_to_none=True)
        real.grad = None
        imag.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = _loss(compiled(real, imag))
        loss.backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        step()
    end.record()
    torch.cuda.synchronize(device)
    return {
        "milliseconds": start.elapsed_time(end) / iterations,
        "peak_bytes": float(torch.cuda.max_memory_allocated(device)),
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    if min(args.batch_size, args.modes, args.hidden, args.warmup, args.iterations) <= 0:
        raise ValueError("cell-collapse benchmark dimensions must be positive")
    if any(size <= 0 or size % 2 for size in args.sizes):
        raise ValueError("cell-collapse benchmark sizes must be positive and even")
    if not torch.cuda.is_available():
        raise RuntimeError("cell-collapse CUDA benchmark requested without CUDA")
    device = torch.device("cuda")
    results: dict[str, Any] = {}
    reference_total = 0.0
    candidate_total = 0.0
    maximum_memory_ratio = 0.0
    parity_pass = True
    for size in args.sizes:
        parity = _parity(args.modes, args.hidden, size, device)
        parity_ok = (
            parity["output_max_abs"] <= 0.125
            and parity["output_relative_l2"] <= 2.0e-2
            and parity["gradient_max_relative_l2"] <= 2.0e-2
        )
        parity_pass = parity_pass and parity_ok
        torch.manual_seed(20260817 + size)
        reference_module = _Full16Collapse(args.modes, args.hidden, direct=False)
        candidate_module = deepcopy(reference_module)
        candidate_module.direct = True
        reference = _benchmark(
            reference_module,
            batch_size=args.batch_size,
            modes=args.modes,
            size=size,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        candidate = _benchmark(
            candidate_module,
            batch_size=args.batch_size,
            modes=args.modes,
            size=size,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        reference_total += reference["milliseconds"]
        candidate_total += candidate["milliseconds"]
        memory_ratio = candidate["peak_bytes"] / max(reference["peak_bytes"], 1.0)
        maximum_memory_ratio = max(maximum_memory_ratio, memory_ratio)
        results[str(size)] = {
            "parity": parity,
            "parity_pass": parity_ok,
            "reference": reference,
            "candidate": candidate,
            "speedup": reference["milliseconds"] / candidate["milliseconds"],
            "memory_ratio": memory_ratio,
        }
    speedup = reference_total / candidate_total
    passed = (
        parity_pass
        and math.isfinite(speedup)
        and speedup >= args.minimum_speedup
        and maximum_memory_ratio <= args.maximum_memory_ratio
    )
    payload = {
        "schema": "lnet.a2d.full16_cell_collapse_benchmark.v1",
        "device": torch.cuda.get_device_name(device),
        "dtype": "bfloat16-autocast",
        "batch_size": args.batch_size,
        "modes": args.modes,
        "hidden": args.hidden,
        "sizes": args.sizes,
        "reference_total_ms": reference_total,
        "candidate_total_ms": candidate_total,
        "weighted_speedup": speedup,
        "maximum_memory_ratio": maximum_memory_ratio,
        "parity_pass": parity_pass,
        "results": results,
        "status": "passed" if passed else "failed",
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
