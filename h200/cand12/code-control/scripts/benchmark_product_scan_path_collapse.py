#!/usr/bin/env python3
"""Benchmark separate full16 scan/collapse against their fused adjoint."""

from __future__ import annotations

# pyright: reportExplicitAny=false
import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from lnet.pac_phase_gated_transition import PathOnlyCollapse
from lnet.pac_product_scan_pipeline import (
    run_product_scan_path_collapse_pipeline,
    run_product_scan_pipeline,
)
from lnet.pac_real2d_math import discrete_pole_real2d

Pole = tuple[Tensor, Tensor, Tensor, Tensor]


class _ScanPathBlock(nn.Module):
    def __init__(self, modes: int, *, fused: bool) -> None:
        super().__init__()
        self.fused = fused
        self.modes = modes
        self.path = PathOnlyCollapse(modes, path_hidden=8)
        damping = torch.linspace(0.04, 0.35, modes).view(1, 1, 1, modes)
        for axis, phase_scale in (("x", 0.7), ("y", 0.6)):
            phase = torch.linspace(0.0, phase_scale * torch.pi, modes).view(1, 1, 1, modes)
            coefficients = discrete_pole_real2d(damping, phase, 1.0)
            for name, value in zip(
                ("decay_real", "decay_imag", "gamma_real", "gamma_imag"),
                coefficients,
                strict=True,
            ):
                self.register_parameter(f"{axis}_{name}", nn.Parameter(value))

    def _pole(self, axis: str) -> Pole:
        return cast(
            "Pole",
            tuple(
                getattr(self, f"{axis}_{name}")
                for name in ("decay_real", "decay_imag", "gamma_real", "gamma_imag")
            ),
        )

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        pole_x = self._pole("x")
        pole_y = self._pole("y")
        if self.fused:
            return run_product_scan_path_collapse_pipeline(
                pole_x,
                pole_y,
                (real, imag),
                self.path.packed_parameters(),
                gain_normalization="global",
                memory_policy="recompute",
            )
        full = run_product_scan_pipeline(
            pole_x,
            pole_y,
            (real, imag),
            epilogue="full16",
            gain_normalization="global",
            memory_policy="recompute",
        )
        if not isinstance(full, tuple):
            raise TypeError("full16 benchmark did not return directional cells")
        return self.path.forward_cells(full[0], full[1])


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--modes", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--minimum-speedup", type=float, default=1.15)
    parser.add_argument("--maximum-memory-ratio", type=float, default=1.05)
    return parser.parse_args()


def _loss(output: tuple[Tensor, Tensor]) -> Tensor:
    return output[0].float().square().mean() + output[1].float().square().mean()


def _benchmark(
    module: nn.Module,
    *,
    batch_size: int,
    height: int,
    modes: int,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    torch.compiler.reset()
    module = module.cuda().train()
    compiled = torch.compile(module, mode="default", fullgraph=False, dynamic=False)
    real = torch.randn(
        batch_size,
        height,
        height,
        modes,
        device="cuda",
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
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        step()
    end.record()
    torch.cuda.synchronize()
    return {
        "milliseconds": start.elapsed_time(end) / iterations,
        "peak_bytes": float(torch.cuda.max_memory_allocated()),
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    if min(args.batch_size, args.height, args.modes, args.warmup, args.iterations) <= 0:
        raise ValueError("scan-path benchmark dimensions must be positive")
    if args.height % 2:
        raise ValueError("scan-path benchmark height must be even")
    if not torch.cuda.is_available():
        raise RuntimeError("scan-path benchmark requires CUDA")
    torch.manual_seed(20260817)
    reference_module = _ScanPathBlock(args.modes, fused=False)
    candidate_module = deepcopy(reference_module)
    candidate_module.fused = True
    reference = _benchmark(
        reference_module,
        batch_size=args.batch_size,
        height=args.height,
        modes=args.modes,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    candidate = _benchmark(
        candidate_module,
        batch_size=args.batch_size,
        height=args.height,
        modes=args.modes,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    speedup = reference["milliseconds"] / candidate["milliseconds"]
    memory_ratio = candidate["peak_bytes"] / max(reference["peak_bytes"], 1.0)
    passed = (
        math.isfinite(speedup)
        and speedup >= args.minimum_speedup
        and memory_ratio <= args.maximum_memory_ratio
    )
    payload = {
        "schema": "lnet.a2d.product_scan_path_collapse_benchmark.v1",
        "device": torch.cuda.get_device_name(),
        "dtype": "bfloat16-autocast",
        "batch_size": args.batch_size,
        "height": args.height,
        "modes": args.modes,
        "reference": reference,
        "candidate": candidate,
        "speedup": speedup,
        "memory_ratio": memory_ratio,
        "status": "passed" if passed else "failed",
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
