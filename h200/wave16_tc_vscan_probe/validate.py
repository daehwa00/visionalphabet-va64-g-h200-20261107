"""Check spatial VScan against Wave 14A and differentiable PyTorch references."""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
from bench import CASES, sample

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, T201


def compare(actual: torch.Tensor, expected: torch.Tensor, options: dict) -> dict:
    a, e = actual.float(), expected.float()
    error = (a - e).abs()
    return {
        "max_abs": float(error.max()),
        "mismatched": int((~torch.isclose(a, e, **options)).sum()),
    }


def merge(total: dict, one: dict) -> None:
    total["max_abs"] = max(total.get("max_abs", 0.0), one["max_abs"])
    total["mismatched"] = total.get("mismatched", 0) + one["mismatched"]


def check(  # noqa: C901
    shape: tuple[int, int, int, int, bool], dtype: torch.dtype, variant: str,
) -> dict:
    b, h, w, m, coarse = shape
    step = 1 + int(coarse)
    states = sample((4, b, h, w, m), dtype, 0.4)
    pole = [sample((m,), torch.float32, 0.12) for _ in range(4)]
    path = [sample(size, torch.float32, 0.17) for size in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    # Unequal gain tables also exercise the epsilon clamp.
    auxiliary = [
        torch.rand((2, length, m), device="cuda").mul_(0.4).add_(0.6) for length in (w, h)
    ]
    for table in auxiliary:
        table[0, 0, 0] = 1e-3
    dy = tuple(sample((b, h // step, w // step, m), dtype, 0.1) for _ in range(2))
    dispatch = scans._dispatch
    launches = []

    def track(source: torch.Tensor, launch: scans.Launch, args: tuple) -> None:
        launches.append(launch.key.spatial)
        dispatch(source, launch, args)

    def fused(setting: str) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]:
        os.environ["LNET_VSCAN_SPATIAL"] = setting
        with torch.no_grad(), patch.object(scans, "_dispatch", track):
            out = scans._vscan(pole, states, auxiliary, path, coarse, 1e-6, precise=False)
            grad = scans.vertical_backward(pole, states, auxiliary, path, *dy, coarse)
        if not isinstance(out, tuple):
            raise TypeError("Expected separate VScan outputs")
        return out, grad

    baseline, baseline_grad = fused("")
    actual, actual_grad = fused(variant)
    if launches != [False, False, True, True]:
        raise RuntimeError(f"VScan candidate did not launch for {shape}, {dtype}: {launches}")
    fwd_tol, bwd_tol = (
        ((8e-3, 1e-3), (5e-2, 2e-2)) if dtype == torch.float16
        else ((4e-2, 5e-3), (1.2e-1, 8e-2))
    )
    fwd_opts = {"rtol": fwd_tol[0], "atol": fwd_tol[1]}
    bwd_opts = {"rtol": bwd_tol[0], "atol": bwd_tol[1]}
    errors = {"wave14_forward": [], "wave14_backward": [], "reference": {}}
    for outputs, targets, opts, label in (
        (actual, baseline, fwd_opts, "wave14_forward"),
        (actual_grad, baseline_grad, bwd_opts, "wave14_backward"),
    ):
        for a, e in zip(outputs, targets, strict=True):
            stats = {}
            for part, ref in zip(a.flatten().split(2**22), e.flatten().split(2**22), strict=True):
                merge(stats, compare(part, ref, opts))
            errors[label].append(stats)

    for ref_dtype in (dtype, torch.float32):
        param_grads = [torch.zeros_like(tensor) for tensor in (*pole, *path)]
        checks = {arm: {"forward": {}, "state_grad": {}, "params": []}
                  for arm in ("wave14", "spatial")}
        # Batch chunks keep autograd's intermediates well below the device capacity.
        batch_chunk = max(1, min(b, (2**20) // (h * w * m)))
        for begin in range(0, b, batch_chunk):
            part = slice(begin, min(b, begin + batch_chunk))
            p = [tensor.detach().requires_grad_() for tensor in pole]
            z = states[:, part].to(ref_dtype).contiguous().requires_grad_()
            weights = [tensor.detach().requires_grad_() for tensor in path]
            result = scans._reference_vertical(p, z, auxiliary, weights, coarse, 1e-6)
            gradients = torch.autograd.grad(
                result, [*p, z, *weights], tuple(d[part].to(ref_dtype) for d in dy),
            )
            for arm, outputs, grads in (
                ("wave14", baseline, baseline_grad), ("spatial", actual, actual_grad),
            ):
                for a, ref in zip(outputs, result, strict=True):
                    stat = compare(a[part], ref.detach().to(dtype), fwd_opts)
                    merge(checks[arm]["forward"], stat)
                merge(
                    checks[arm]["state_grad"],
                    compare(grads[4][:, part], gradients[4].to(dtype), bwd_opts),
                )
            for buf, grad in zip(param_grads, (*gradients[:4], *gradients[5:]), strict=True):
                buf.add_(grad)
        for arm, grads in (("wave14", baseline_grad), ("spatial", actual_grad)):
            checks[arm]["params"] = [
                compare(a, e, bwd_opts)
                for a, e in zip((*grads[:4], *grads[5:]), param_grads, strict=True)
            ]
        errors["reference"][str(ref_dtype)] = checks
    return {"shape": shape, "dtype": str(dtype), "launches": launches, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--variant", default="auto")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--cases", nargs="+", type=int, default=list(range(len(CASES))))
    args = parser.parse_args()
    os.environ["LNET_VSCAN_DECOUPLED"] = "0"
    failed = False
    torch.manual_seed(645)
    with args.out.open("w") as handle:
        for dtype in (torch.bfloat16, torch.float16):
            for index in args.cases:
                b, *rest = CASES[index]
                shape = (args.batch if b == 128 else b, *rest)
                row = check(shape, dtype, args.variant)
                print(json.dumps(row), file=handle, flush=True)
                errors = row["errors"]
                reference = [
                    stat for refs in errors["reference"].values()
                    for stat in (refs["spatial"]["forward"], refs["spatial"]["state_grad"],
                                 *refs["spatial"]["params"])
                ]
                all_stats = [*errors["wave14_forward"], *errors["wave14_backward"], *reference]
                mismatch = sum(stat["mismatched"] for stat in all_stats)
                failed |= bool(mismatch)
                print(dtype, shape, "mismatched", mismatch, flush=True)
    if failed:
        raise SystemExit("Strict reference parity gate failed")


if __name__ == "__main__":
    main()
