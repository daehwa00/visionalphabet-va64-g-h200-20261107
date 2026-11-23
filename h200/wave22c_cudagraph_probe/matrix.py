"""Measure matrix exponential error and forward/backward time independently."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liverms_v2_opt"))
from bench_h200 import VARIANTS
from run_va_eight_ablations import build_variant

from lnet.capture_safe_orthogonal import fixed_taylor_matrix_exp, prepare_capture_safe_orthogonal_

if TYPE_CHECKING:
    from collections.abc import Callable


def polynomial(matrix: torch.Tensor, *, degree: int, scaling: int) -> torch.Tensor:
    """Evaluate Taylor terms in blocks of three powers."""
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    x = matrix * 2.0**-scaling
    x2 = x @ x
    x3 = x @ x2
    bases = (identity, x, x2)

    def block(k: int) -> torch.Tensor:
        return sum(
            (base / math.factorial(k + i) for i, base in enumerate(bases) if k + i <= degree),
            start=torch.zeros_like(identity),
        )

    result = block(degree // 3 * 3)
    for k in range(degree // 3 * 3 - 3, -1, -3):
        result = result @ x3 + block(k)
    for _ in range(scaling):
        result = result @ result
    return result


def evaluate(fn: Callable, matrix: torch.Tensor, probe: torch.Tensor) -> tuple:
    output = fn(matrix)
    (grad,) = torch.autograd.grad(output, matrix, probe)
    return output.detach(), grad


def error(output: torch.Tensor, reference: torch.Tensor) -> dict:
    return {
        "exact": torch.equal(output, reference),
        "max_abs": (output - reference).abs().max().item(),
    }


def benchmark(fn: Callable, matrix: torch.Tensor, probe: torch.Tensor, count: int) -> dict:
    for _ in range(4):
        evaluate(fn, matrix, probe)
    if matrix.is_cuda:
        torch.cuda.synchronize()
    timing = []
    if matrix.is_cuda:
        start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(count):
        start = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
        if matrix.is_cuda:
            start_event.record()
        evaluate(fn, matrix, probe)
        if matrix.is_cuda:
            end_event.record()
            end_event.synchronize()
            event_ms = start_event.elapsed_time(end_event)
        else:
            event_ms = None
        timing.append([(time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - start) * 1000, event_ms])
    return {
        "wall_ms": statistics.median(row[0] for row in timing),
        "event_ms": statistics.median(row[1] for row in timing) if matrix.is_cuda else None,
    }


def check_models(device: str) -> tuple[dict, dict]:
    info = {}
    matrices = {}
    for variant in VARIANTS:
        torch.manual_seed(501)
        native = build_variant(variant).to(device)
        candidate = build_variant(variant).to(device)
        candidate.load_state_dict(native.state_dict(), strict=True)
        keys = tuple(native.state_dict())
        params = dict(candidate.named_parameters())
        paths = prepare_capture_safe_orthogonal_(candidate)
        candidate.load_state_dict(native.state_dict(), strict=True)
        native.load_state_dict(candidate.state_dict(), strict=True)
        info[variant] = {
            "paths": paths,
            "state_keys_equal": keys == tuple(candidate.state_dict()),
            "parameter_identity_preserved": all(
                params[name] is p for name, p in candidate.named_parameters()
            ),
            "parameter_count": sum(p.numel() for p in params.values()),
            "strict_reload": True,
        }
        info[variant]["frame_parity"] = {}
        for path in paths:
            layer, _, tensor = path.rpartition(".")
            module = candidate.get_submodule(layer)
            parametrization = module.parametrizations[tensor]
            native_list = native.get_submodule(layer).parametrizations[tensor]
            original = parametrization.original.detach().requires_grad_()
            probe = torch.randn_like(original)
            parity = {}
            for precision in ("fp32", "bf16_autocast"):
                with torch.autocast(
                    device, dtype=torch.bfloat16, enabled=precision == "bf16_autocast"
                ):
                    actual, actual_grad = evaluate(parametrization[0], original, probe)
                    expected, expected_grad = evaluate(native_list[0], original, probe)
                parity[precision] = {
                    "frame_dtype": str(actual.dtype),
                    "frame": error(actual, expected),
                    "gradient": error(actual_grad, expected_grad),
                }
            info[variant]["frame_parity"][path] = parity
            original = original.detach()
            working = original if original.shape[0] >= original.shape[1] else original.mT
            rows, columns = working.shape
            lower = torch.nn.functional.pad(working.tril(), (0, rows - columns))
            matrices[f"{variant}/{path}"] = lower - lower.mT
    return info, matrices


def run(args: argparse.Namespace) -> dict:
    torch.set_float32_matmul_precision("highest")
    models, matrices = check_models(args.device)
    torch.manual_seed(501)
    random = torch.randn(64, 64, device=args.device)
    skew = random - random.mT
    for norm in (0.1, 1.0, 5.0, 10.0):
        matrices[f"stress/{norm}"] = skew * (norm / skew.norm())
    # A Frobenius norm can hide a much larger spectral norm than random matrices exhibit.
    concentrated = torch.zeros_like(skew)
    concentrated[1, 0], concentrated[0, 1] = 10 / math.sqrt(2), -10 / math.sqrt(2)
    matrices["stress/rank2-norm10"] = concentrated

    candidates = {
        "native": torch.matrix_exp,
        "horner12-scale3": fixed_taylor_matrix_exp,
        "ps6-scale3": partial(polynomial, degree=6, scaling=3),
        "ps8-scale2": partial(polynomial, degree=8, scaling=2),
        "ps12-scale3": partial(polynomial, degree=12, scaling=3),
        "ps14-scale2": partial(polynomial, degree=14, scaling=2),
    }
    results = {}
    for name, matrix in matrices.items():
        matrix.requires_grad_()
        probe = torch.randn_like(matrix)
        native = evaluate(torch.matrix_exp, matrix, probe)
        fp64 = evaluate(torch.matrix_exp, matrix.double(), probe.double())
        results[name] = {"frobenius_norm": matrix.detach().norm().item(), "variants": {}}
        for method, fn in candidates.items():
            output, grad = evaluate(fn, matrix, probe)
            results[name]["variants"][method] = {
                "output_native": error(output, native[0]),
                "grad_native": error(grad, native[1]),
                "output_fp64": error(output.double(), fp64[0]),
                "grad_fp64": error(grad.double(), fp64[1]),
            }
            if name == "stress/10.0":
                results[name]["variants"][method]["timing"] = benchmark(
                    fn, matrix, probe, args.count
                )
    return {"device": args.device, "models": models, "matrices": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args), indent=2), encoding="utf-8")
