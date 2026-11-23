"""Compare every gradient against the FP32 scan reference at batch one."""

import json
import os
from unittest.mock import patch

import torch
from bench import SHAPES
from test_general_cuda_scans import _inputs, _reference_vscan

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, T201


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    diff = actual.float() - expected.float()
    finite = bool(actual.isfinite().all()) and bool(expected.isfinite().all())
    return {
        "finite": finite,
        "max_abs": diff.abs().max().item(),
        "relative_l2": (diff.norm() / expected.float().norm().clamp_min(1e-12)).item(),
    }


def probe(h: int, w: int, m: int, *, coarse: bool) -> None:
    torch.manual_seed(1439)
    _, py, _, _, auxiliary, path = _inputs(h, w, m, 8, torch.float32, "cuda")
    states = torch.randn((4, 1, h, w, m), device="cuda").mul_(0.1).requires_grad_()
    step = 1 + int(coarse)
    # Quantized dY and states are shared by the reference and tested kernel.
    dy = [torch.randn((1, h // step, w // step, m), device="cuda").mul_(0.1) for _ in range(2)]
    dispatch = scans._dispatch
    used = []

    def track(source: torch.Tensor, launch: scans.Launch, args: tuple) -> None:
        used.append(launch.key.mma_path)
        dispatch(source, launch, args)

    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        reference_states = states.detach().to(dtype).float().requires_grad_()
        gradients = [g.to(dtype) for g in dy]
        target = torch.autograd.grad(
            _reference_vscan(py, reference_states, auxiliary, path, coarse),
            [*py, reference_states, *path],
            tuple(g.float() for g in gradients),
        )
        for policy in ("off", "on"):
            os.environ["LNET_VSCAN_MMA"] = policy
            with patch.object(scans, "_dispatch", track):
                actual = scans.vertical_backward(
                    py, reference_states.detach().to(dtype), auxiliary, path, *gradients, coarse,
                )
            named_pairs = zip(
                ("dar", "dai", "dgr", "dgi", "ds", "dw1", "db1", "dw2", "db2"),
                actual, target, strict=True,
            )
            print(json.dumps({
                "shape": (1, h, w, m, coarse), "dtype": str(dtype), "policy": policy,
                "mma": used[-1], "against_fp32": {n: compare(a, b) for n, a, b in named_pairs},
            }), flush=True)


if __name__ == "__main__":
    for h, w, m, coarse in (*SHAPES, (37, 53, 53, False)):
        probe(h, w, m, coarse=coarse)
