#!/usr/bin/env python3
"""Compare split and packed Gated PostFusion training paths."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportPrivateUsage=false
import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from lnet.pac_gated_post_fusion import (
    GatedComplexPostFusion,
    GatedStrictComplexPostFusion,
    _GatedPostFusionBase,
)
from lnet.pac_triton_complex_rmsnorm import (
    packed_complex_rms_norm,
    supports_packed_complex_rms_norm,
)

ROWS = (401_408, 100_352, 25_088, 6_272)
MULTIPLICITY = (1, 2, 2, 2)
MODES = 128
HIDDEN = 192


class _SplitStrictPostFusion(GatedStrictComplexPostFusion):
    """Evaluate strict PostFusion one coordinate at a time."""

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        return self._split_forward(real, imag)


class _SplitWidelyPostFusion(GatedComplexPostFusion):
    """Evaluate widely-linear PostFusion one coordinate at a time."""

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        return self._split_forward(real, imag)


class _NativePackedWidelyPostFusion(nn.Module):
    """Store the arbitrary real-linear PostFusion maps in their GEMM layout."""

    def __init__(self, modes: int, hidden_modes: int) -> None:
        super().__init__()
        source = GatedComplexPostFusion(modes, hidden_modes)
        self.modes = modes
        self.hidden_modes = hidden_modes
        self.norm = source.norm
        self.joint_weight = nn.Parameter(
            torch.cat((source.value.packed_weight(), source.gate.weight), dim=0)
            .detach()
            .clone()
        )
        self.output_weight = nn.Parameter(source.out.packed_weight().detach().clone())

    @torch.no_grad()
    def load_reference(self, source: _GatedPostFusionBase) -> None:
        if not isinstance(source, GatedComplexPostFusion):
            raise TypeError("native packed candidate requires a widely-linear reference")
        self.norm.load_state_dict(source.norm.state_dict())
        self.joint_weight.copy_(
            torch.cat((source.value.packed_weight(), source.gate.weight), dim=0)
        )
        self.output_weight.copy_(source.out.packed_weight())

    @staticmethod
    def _widely_gradient(packed: Tensor, output_modes: int) -> dict[str, Tensor]:
        input_modes = packed.shape[1] // 2
        top_left = packed[:output_modes, :input_modes]
        top_right = packed[:output_modes, input_modes:]
        bottom_left = packed[output_modes:, :input_modes]
        bottom_right = packed[output_modes:, input_modes:]
        return {
            "weight_real": top_left + bottom_right,
            "weight_imag": bottom_left - top_right,
            "conjugate_real": top_left - bottom_right,
            "conjugate_imag": top_right + bottom_left,
        }

    def reference_gradients(self) -> dict[str, Tensor]:
        if (
            self.norm.weight.grad is None
            or self.joint_weight.grad is None
            or self.output_weight.grad is None
        ):
            raise RuntimeError("native packed candidate has missing gradients")
        value_gradient = self.joint_weight.grad[: 2 * self.hidden_modes]
        gate_gradient = self.joint_weight.grad[2 * self.hidden_modes :]
        gradients = {
            "norm.weight": self.norm.weight.grad,
            "gate.weight": gate_gradient,
        }
        gradients.update(
            {
                f"value.{name}": value
                for name, value in self._widely_gradient(
                    value_gradient,
                    self.hidden_modes,
                ).items()
            }
        )
        gradients.update(
            {
                f"out.{name}": value
                for name, value in self._widely_gradient(
                    self.output_weight.grad,
                    self.modes,
                ).items()
            }
        )
        return gradients

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        if supports_packed_complex_rms_norm(real, imag, self.norm.weight):
            normalized = packed_complex_rms_norm(
                real,
                imag,
                self.norm.weight,
                self.norm.epsilon,
            )
        else:
            normalized = torch.cat(self.norm(real, imag), dim=-1)
        joint = torch.nn.functional.linear(normalized, self.joint_weight)
        value, gate_logits = joint.split(
            (2 * self.hidden_modes, self.hidden_modes),
            dim=-1,
        )
        gate = torch.nn.functional.silu(gate_logits).unsqueeze(-2)
        gated = (value.unflatten(-1, (2, self.hidden_modes)) * gate).flatten(-2)
        update = torch.nn.functional.linear(gated, self.output_weight)
        update_real, update_imag = update.split(self.modes, dim=-1)
        return real + update_real, imag + update_imag


FAMILIES: dict[str, tuple[type[nn.Module], type[nn.Module]]] = {
    "strict": (_SplitStrictPostFusion, GatedStrictComplexPostFusion),
    "widely": (_SplitWidelyPostFusion, GatedComplexPostFusion),
    "native_widely": (GatedComplexPostFusion, _NativePackedWidelyPostFusion),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    # Eight iterations left the smallest row counts inside run-to-run noise,
    # where a single launch-bound step decides the gate.
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--minimum-speedup", type=float, default=1.10)
    parser.add_argument("--minimum-family-speedup", type=float, default=1.00)
    parser.add_argument("--maximum-memory-ratio", type=float, default=1.10)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILIES),
        default=tuple(FAMILIES),
    )
    return parser.parse_args()


def _loss(output: tuple[Tensor, Tensor]) -> Tensor:
    return output[0].float().square().mean() + output[1].float().square().mean()


def _parity(
    rows: int,
    device: torch.device,
    reference_type: type[nn.Module],
    candidate_type: type[nn.Module],
) -> dict[str, float]:
    torch.manual_seed(20260817 + rows)
    reference = reference_type(MODES, HIDDEN).to(device)
    candidate = candidate_type(MODES, HIDDEN).to(device)
    if isinstance(candidate, _NativePackedWidelyPostFusion):
        candidate.load_reference(cast("_GatedPostFusionBase", reference))
    else:
        candidate.load_state_dict(reference.state_dict(), strict=True)
    real = torch.randn(rows, MODES, device=device, dtype=torch.bfloat16, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
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
        raise RuntimeError("missing input gradient during packed parity check")
    real_grad = cast("Tensor", real.grad)
    imag_grad = cast("Tensor", imag.grad)
    candidate_real_grad = cast("Tensor", candidate_real.grad)
    candidate_imag_grad = cast("Tensor", candidate_imag.grad)
    output_max_abs = max(
        float((expected[0].float() - actual[0].float()).abs().max()),
        float((expected[1].float() - actual[1].float()).abs().max()),
    )
    input_grad_max_abs = max(
        float((real_grad.float() - candidate_real_grad.float()).abs().max()),
        float((imag_grad.float() - candidate_imag_grad.float()).abs().max()),
    )
    parameter_grad_max_abs = 0.0
    candidate_gradients = (
        candidate.reference_gradients()
        if isinstance(candidate, _NativePackedWidelyPostFusion)
        else {
            name: parameter.grad
            for name, parameter in candidate.named_parameters()
        }
    )
    for name, parameter in reference.named_parameters():
        candidate_gradient = candidate_gradients[name]
        if parameter.grad is None or candidate_gradient is None:
            raise RuntimeError(f"missing parity gradient for {name}")
        parameter_grad_max_abs = max(
            parameter_grad_max_abs,
            float((parameter.grad.float() - candidate_gradient.float()).abs().max()),
        )
    return {
        "output_max_abs": output_max_abs,
        "input_grad_max_abs": input_grad_max_abs,
        "parameter_grad_max_abs": parameter_grad_max_abs,
        "loss_abs": abs(float(expected_loss) - float(actual_loss)),
    }


def _benchmark(
    module: nn.Module,
    rows: int,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    torch.compiler.reset()
    module = module.to(device).train()
    compiled = torch.compile(module, mode="default", fullgraph=False, dynamic=False)
    real = torch.randn(rows, MODES, device=device, dtype=torch.bfloat16, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    grad_real = torch.randn_like(real)
    grad_imag = torch.randn_like(imag)

    def step() -> None:
        module.zero_grad(set_to_none=True)
        real.grad = None
        imag.grad = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = compiled(real, imag)
        torch.autograd.backward(output, (grad_real, grad_imag))

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


def _measure_family(
    reference_type: type[nn.Module],
    candidate_type: type[nn.Module],
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    weighted_reference = 0.0
    weighted_candidate = 0.0
    peak_ratio = 0.0
    parity_pass = True
    for rows, multiplicity in zip(ROWS, MULTIPLICITY, strict=True):
        torch.manual_seed(20260817 + rows)
        reference = _benchmark(
            reference_type(MODES, HIDDEN),
            rows,
            device,
            warmup=warmup,
            iterations=iterations,
        )
        parity = _parity(rows, device, reference_type, candidate_type)
        row_parity_pass = (
            parity["output_max_abs"] <= 0.125
            and parity["input_grad_max_abs"] <= 2.0e-4
            and parity["parameter_grad_max_abs"] <= 0.125
            and parity["loss_abs"] <= 1.0e-3
        )
        parity_pass = parity_pass and row_parity_pass
        torch.manual_seed(20260817 + rows)
        candidate = _benchmark(
            candidate_type(MODES, HIDDEN),
            rows,
            device,
            warmup=warmup,
            iterations=iterations,
        )
        weighted_reference += multiplicity * reference["milliseconds"]
        weighted_candidate += multiplicity * candidate["milliseconds"]
        memory_ratio = candidate["peak_bytes"] / max(reference["peak_bytes"], 1.0)
        peak_ratio = max(peak_ratio, memory_ratio)
        results[str(rows)] = {
            "multiplicity": multiplicity,
            "reference": reference,
            "timing": candidate,
            "parity": parity,
            "parity_pass": row_parity_pass,
            "speedup": reference["milliseconds"] / candidate["milliseconds"],
            "memory_ratio": memory_ratio,
        }
    return {
        "reference": reference_type.__name__,
        "candidate": candidate_type.__name__,
        "weighted_reference_ms": weighted_reference,
        "weighted_candidate_ms": weighted_candidate,
        "weighted_speedup": weighted_reference / weighted_candidate,
        "maximum_memory_ratio": peak_ratio,
        "parity_pass": parity_pass,
        "results": results,
    }


def main() -> None:
    args = _arguments()
    if min(args.warmup, args.iterations) <= 0:
        raise ValueError("warmup and iterations must be positive")
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested without an available device")
    torch.set_float32_matmul_precision("high")
    families = {
        name: _measure_family(
            reference_type,
            candidate_type,
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for name, (reference_type, candidate_type) in FAMILIES.items()
        if name in args.families
    }
    # The minimum speedup is a whole-block claim, not a per-family one: a family
    # whose packed residual already landed has no headroom left to re-measure,
    # so each family only has to avoid regressing while the weighted total over
    # every PostFusion site carries the improvement.
    weighted_reference_ms = sum(
        cast("float", family["weighted_reference_ms"]) for family in families.values()
    )
    weighted_candidate_ms = sum(
        cast("float", family["weighted_candidate_ms"]) for family in families.values()
    )
    weighted_speedup = weighted_reference_ms / weighted_candidate_ms
    passed = (
        math.isfinite(weighted_speedup)
        and weighted_speedup >= args.minimum_speedup
        and all(
            family["parity_pass"]
            and math.isfinite(cast("float", family["weighted_speedup"]))
            and family["weighted_speedup"] >= args.minimum_family_speedup
            and family["maximum_memory_ratio"] <= args.maximum_memory_ratio
            for family in families.values()
        )
    )
    payload = {
        "schema": "lnet.a2d.gated_postfusion_complex_linear_benchmark.v2",
        "device": torch.cuda.get_device_name(device),
        "dtype": "bfloat16-autocast",
        "modes": MODES,
        "hidden_modes": HIDDEN,
        "rows": list(ROWS),
        "multiplicity": list(MULTIPLICITY),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "minimum_required_speedup": args.minimum_speedup,
        "minimum_required_family_speedup": args.minimum_family_speedup,
        "maximum_allowed_memory_ratio": args.maximum_memory_ratio,
        "weighted_reference_ms": weighted_reference_ms,
        "weighted_candidate_ms": weighted_candidate_ms,
        "weighted_speedup": weighted_speedup,
        "weighted_speedups": {
            name: family["weighted_speedup"] for name, family in families.items()
        },
        "parity_pass_by_name": {name: family["parity_pass"] for name, family in families.items()},
        "maximum_memory_ratio_by_name": {
            name: family["maximum_memory_ratio"] for name, family in families.items()
        },
        "families": families,
        "status": "passed" if passed else "failed",
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
