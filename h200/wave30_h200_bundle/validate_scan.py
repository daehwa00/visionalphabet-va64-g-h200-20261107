"""Compare Wave 30A CUDA forward and adjoints against off and FP32."""

import argparse
import json
import os
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
from test_general_cuda_scans import _inputs, _pipeline, _reference_vscan

from lnet import general_cuda_scans as scans
from lnet.product_scan_reference import bidirectional_product_scan_reference

# ruff: noqa: SLF001

PRESETS = ("off", "wp_all_waves", "wp_batch2", "wp_batch2_all", "wp_vscan_opt", "auto")
SQUARE = (*((s, s, 160, c) for s in (56, 28, 14) for c in (False, True)), (7, 7, 128, False))
CASES = (*SQUARE, (7, 8, 128, False), (8, 8, 128, True), (37, 53, 160, False), (38, 54, 160, True))
DTYPES = (torch.bfloat16, torch.float16)
SCAN_PLANS = scans._scan_plans


def cached_backward(*args: object, **kwargs: bool | int) -> list[scans.ScanPlan]:
    if not (kwargs["vertical"] and kwargs["backward"]):
        return SCAN_PLANS(*args, **kwargs)
    kwargs["cache_trajectories"] = True
    length = args[0]
    plans = SCAN_PLANS(*args, **kwargs)
    return [
        p for p in plans
        if p.warps == 1 and p.cache_weights and p.cache_source and not p.shared_acc
        and p.full_trajectory == (length != 56) and (length != 56 or p.chunk == 8)
    ]


def compare(actual: torch.Tensor, expected: torch.Tensor) -> float:
    reference = expected.float()
    error = actual.float() - reference
    relative_l2 = (error.norm() / reference.norm().clamp_min(1e-6)).item()
    # Storage casts can turn a small FP32 difference into a one-ULP delta.
    eps = 1e-4 if actual.dtype == torch.float32 else 2 * torch.finfo(actual.dtype).eps
    max_error = eps * reference.abs().max().item() + 3e-5
    if not (relative_l2 <= 1e-4 and error.abs().max().item() <= max_error):
        raise ValueError((relative_l2, error.abs().max().item(), max_error))
    return relative_l2


def run(case: tuple, dtype: torch.dtype, *, interleaved: bool, forced: bool) -> Iterator[dict]:
    h, w, m, coarse = case
    torch.manual_seed(719)
    px, py, real, imag, aux, path = _inputs(h, w, m, 8, dtype, "cuda", batch=2)
    if interleaved:
        real, imag = torch.cat((real, imag), -1).detach().requires_grad_().chunk(2, -1)
    state = torch.stack(bidirectional_product_scan_reference(
        tuple(px), (real.float(), imag.float()),
    ))
    expected = (state, *_reference_vscan(py, state, aux, path, coarse))
    adjoint = torch.randn((*expected[1].shape[:-1], 2 * m), device="cuda", dtype=dtype)
    dy = adjoint.chunk(2, -1)
    if not interleaved:
        dy = tuple(x.contiguous() for x in dy)
    gradients = (torch.randn_like(state, dtype=dtype), *dy)
    targets = [real, imag, *px, *py, *path]
    reference = (*expected, *torch.autograd.grad(expected, targets, gradients))
    baseline = None
    launches = []
    dispatch = scans._dispatch

    def track(source: torch.Tensor, launch: scans.Launch, args: tuple) -> None:
        if launch.key.vertical and launch.key.backward:
            launches.append(launch)
        dispatch(source, launch, args)

    operation = partial(_pipeline, px, py, real, imag, aux, path, coarse, "cuda")
    for preset in PRESETS:
        launches.clear()
        scans._launch_for_shape.cache_clear()
        with patch.dict(os.environ, {"LNET_SCAN_W30A": preset}), patch.object(
            scans, "_dispatch", track,
        ):
            actual = operation()
            observed = (*actual, *torch.autograd.grad(actual, targets, gradients))
        if baseline is None:
            baseline = observed
        try:
            off_error = [compare(x, y) for x, y in zip(observed, baseline, strict=True)]
        except ValueError as exc:
            raise ValueError((case, dtype, interleaved, forced, preset, str(exc))) from exc
        errors = [
            ((x.float() - y.float()).norm() / y.float().norm().clamp_min(0.02)).item()
            for x, y in zip(observed, reference, strict=True)
        ]
        limit = 0.02 if dtype == torch.bfloat16 else 0.005
        if max(errors) > limit:
            raise ValueError((case, dtype, interleaved, forced, preset, errors))
        batch2 = any(launch.key.backward_batch2 for launch in launches)
        chunk14 = any(launch.key.chunk == 14 for launch in launches)
        expected_batch = preset in {"wp_batch2_all", "auto"} or (h == 56 and preset == "wp_batch2")
        expected_chunk = h == 56 and preset == "wp_vscan_opt"
        if forced and (batch2 != expected_batch or chunk14 != expected_chunk):
            raise ValueError((case, dtype, interleaved, preset, batch2, chunk14))
        yield {
            "case": case, "dtype": str(dtype), "interleaved": interleaved, "forced": forced,
            "preset": preset, "batch2": batch2, "chunk14": chunk14,
            "exact": [torch.equal(x, y) for x, y in zip(observed, baseline, strict=True)],
            "off_relative_l2": off_error, "fp32_relative_l2": errors,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with (
        args.output.open("w") as stream,
        patch.dict(os.environ, {"LNET_VSCAN_DECOUPLED": "0", "LNET_SCAN_W27A": "off"}),
    ):
        for forced, cases in ((False, CASES), (True, SQUARE)):
            scans._launch_for_shape.cache_clear()
            with patch.object(scans, "_scan_plans", cached_backward if forced else SCAN_PLANS):
                for case in cases:
                    for dtype in DTYPES:
                        for interleaved in (False, True):
                            for row in run(case, dtype, interleaved=interleaved, forced=forced):
                                print(json.dumps(row), file=stream, flush=True)


if __name__ == "__main__":
    main()
