#!/usr/bin/env python3
"""Benchmark dense-kernel and directly factorized R2K3 reader execution."""

from __future__ import annotations

# pyright: reportExplicitAny=false
import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader


class _DenseReferenceReader(FactorizedComplexConv2dReader):
    """Execute the factorized parameterization through its synthesized kernel."""

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_input(real, imag)
        if self.input_norm is not None:
            real, imag = self.input_norm(real, imag)
        kernel_real, kernel_imag = self.unit_energy_kernel()
        packed_kernel = torch.cat(
            (
                torch.cat((kernel_real, -kernel_imag), dim=1),
                torch.cat((kernel_imag, kernel_real), dim=1),
            ),
            dim=0,
        )
        packed_input = torch.cat(
            (real.movedim(-1, 1), imag.movedim(-1, 1)),
            dim=1,
        )
        output = functional.conv2d(packed_input, packed_kernel, padding=self.padding)
        if self.match_input_rms:
            input_energy = packed_input.float().square().sum(dim=1, keepdim=True).div(
                self.input_modes
            )
            output_energy = (
                output.float()
                .reshape(
                    output.shape[0],
                    2,
                    self.output_modes,
                    output.shape[2],
                    output.shape[3],
                )
                .square()
                .sum(dim=(1, 2), keepdim=False)
                .unsqueeze(1)
                .div(self.output_modes)
            )
            output = output * torch.sqrt(
                (input_energy + self.variance_epsilon)
                / (output_energy + self.variance_epsilon)
            ).to(dtype=output.dtype)
        return (
            output[:, : self.output_modes].movedim(1, -1).contiguous(),
            output[:, self.output_modes :].movedim(1, -1).contiguous(),
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-modes", type=int, default=128)
    parser.add_argument("--output-modes", type=int, default=160)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--sizes", type=int, nargs="+", default=(56, 28, 14, 7))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--minimum-speedup", type=float, default=1.5)
    parser.add_argument("--maximum-memory-ratio", type=float, default=1.05)
    return parser.parse_args()


def _module(
    module_type: type[FactorizedComplexConv2dReader],
    input_modes: int,
    output_modes: int,
    rank: int,
    policy: str,
) -> FactorizedComplexConv2dReader:
    module = module_type(
        input_modes,
        output_modes,
        rank=rank,
        kernel_size=3,
        normalize_input=policy == "normalized",
        match_input_rms=policy == "rmsmatched",
    )
    module.initialize_semi_orthogonal_()
    return module


def _loss(output: tuple[Tensor, Tensor]) -> Tensor:
    return output[0].float().square().mean() + output[1].float().square().mean()


def _parity(
    *,
    input_modes: int,
    output_modes: int,
    rank: int,
    size: int,
    policy: str,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(20260817 + size)
    reference = _module(
        _DenseReferenceReader,
        input_modes,
        output_modes,
        rank,
        policy,
    ).to(device)
    candidate = _module(
        FactorizedComplexConv2dReader,
        input_modes,
        output_modes,
        rank,
        policy,
    ).to(device)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    real = torch.randn(
        1,
        size,
        size,
        input_modes,
        device=device,
        dtype=torch.bfloat16,
    ).requires_grad_()
    imag = torch.randn_like(real).requires_grad_()
    candidate_real = real.detach().clone().requires_grad_()
    candidate_imag = imag.detach().clone().requires_grad_()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected = reference(real, imag)
        actual = candidate(candidate_real, candidate_imag)
    expected_loss = _loss(expected)
    actual_loss = _loss(actual)
    expected_loss.backward()
    actual_loss.backward()
    if any(value.grad is None for value in (real, imag, candidate_real, candidate_imag)):
        raise RuntimeError("reader parity produced a missing input gradient")
    output_max_abs = max(
        float((expected[0].float() - actual[0].float()).abs().max()),
        float((expected[1].float() - actual[1].float()).abs().max()),
    )
    input_grad_max_abs = max(
        float((cast("Tensor", real.grad).float() - cast("Tensor", candidate_real.grad).float()).abs().max()),
        float((cast("Tensor", imag.grad).float() - cast("Tensor", candidate_imag.grad).float()).abs().max()),
    )
    parameter_grad_max_abs = 0.0
    candidate_parameters = dict(candidate.named_parameters())
    for name, parameter in reference.named_parameters():
        candidate_parameter = candidate_parameters[name]
        if parameter.grad is None or candidate_parameter.grad is None:
            raise RuntimeError(f"reader parity produced a missing parameter gradient: {name}")
        parameter_grad_max_abs = max(
            parameter_grad_max_abs,
            float((parameter.grad.float() - candidate_parameter.grad.float()).abs().max()),
        )
    return {
        "output_max_abs": output_max_abs,
        "input_grad_max_abs": input_grad_max_abs,
        "parameter_grad_max_abs": parameter_grad_max_abs,
        "loss_abs": abs(float(expected_loss) - float(actual_loss)),
    }


def _benchmark(
    module: nn.Module,
    *,
    batch_size: int,
    input_modes: int,
    size: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    torch.compiler.reset()
    module = module.to(device).train()
    compiled = torch.compile(module, mode="default", fullgraph=False, dynamic=False)
    real = torch.randn(
        batch_size,
        size,
        size,
        input_modes,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    imag = torch.randn_like(real, requires_grad=True)

    def step() -> None:
        module.zero_grad(set_to_none=True)
        real.grad = None
        imag.grad = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
    milliseconds = start.elapsed_time(end) / iterations
    peak_bytes = float(torch.cuda.max_memory_allocated(device))
    torch.cuda.empty_cache()
    torch.compiler.reset()
    return {"milliseconds": milliseconds, "peak_bytes": peak_bytes}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    if (
        min(
            args.batch_size,
            args.input_modes,
            args.output_modes,
            args.rank,
            args.warmup,
            args.iterations,
            *args.sizes,
        )
        <= 0
    ):
        raise ValueError("reader benchmark dimensions and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("reader CUDA benchmark requested without CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    results: dict[str, Any] = {}
    reference_total = 0.0
    candidate_total = 0.0
    maximum_memory_ratio = 0.0
    parity_pass = True
    for policy in ("rmsmatched", "normalized"):
        for size in args.sizes:
            key = f"{policy}-{size}"
            parity = _parity(
                input_modes=args.input_modes,
                output_modes=args.output_modes,
                rank=args.rank,
                size=size,
                policy=policy,
                device=device,
            )
            parity_ok = (
                parity["output_max_abs"] <= 0.25
                and parity["input_grad_max_abs"] <= 2.0e-3
                and parity["parameter_grad_max_abs"] <= 0.25
                and parity["loss_abs"] <= 2.0e-3
            )
            parity_pass = parity_pass and parity_ok
            torch.manual_seed(20260817 + size)
            reference = _benchmark(
                _module(
                    _DenseReferenceReader,
                    args.input_modes,
                    args.output_modes,
                    args.rank,
                    policy,
                ),
                batch_size=args.batch_size,
                input_modes=args.input_modes,
                size=size,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )
            torch.manual_seed(20260817 + size)
            candidate = _benchmark(
                _module(
                    FactorizedComplexConv2dReader,
                    args.input_modes,
                    args.output_modes,
                    args.rank,
                    policy,
                ),
                batch_size=args.batch_size,
                input_modes=args.input_modes,
                size=size,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )
            reference_total += reference["milliseconds"]
            candidate_total += candidate["milliseconds"]
            memory_ratio = candidate["peak_bytes"] / max(reference["peak_bytes"], 1.0)
            maximum_memory_ratio = max(maximum_memory_ratio, memory_ratio)
            results[key] = {
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
        "schema": "lnet.a2d.factorized_complex_scan_reader_benchmark.v1",
        "device": torch.cuda.get_device_name(device),
        "dtype": "bfloat16-autocast",
        "batch_size": args.batch_size,
        "input_modes": args.input_modes,
        "output_modes": args.output_modes,
        "rank": args.rank,
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
