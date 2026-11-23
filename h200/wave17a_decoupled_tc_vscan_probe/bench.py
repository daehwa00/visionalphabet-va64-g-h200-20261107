"""Run parity and CUDA Graph A/B for each stage of decoupled vertical scan."""

import argparse
import gc
import json
import os
import statistics
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, T201, C901, PLR0912

CASES = (
    (56, 56, 160, False), (56, 56, 160, True),
    (28, 28, 320, False), (28, 28, 320, True),
    (14, 14, 640, False), (14, 14, 640, True),
    (7, 7, 640, False), (14, 14, 160, True),
)
EDGE = ((1, 5, 7, 1), (3, 13, 15, 17), (2, 17, 19, 48), (2, 9, 11, 33))
POLICY = "LNET_VSCAN_DECOUPLED"
NAMES = ("real", "imag", "da_r", "da_i", "dg_r", "dg_i", "gs", "dWi", "dbi", "dWo", "dbo")


def inputs(
    shape: tuple[int, int, int, int, bool], dtype: torch.dtype,
) -> tuple[list[Tensor], Tensor, list[Tensor], list[Tensor], list[Tensor]]:
    b, h, w, m, coarse = shape

    def rnd(dim: tuple[int, ...], scale: float, dt: torch.dtype = torch.float32) -> Tensor:
        return torch.randn(dim, device="cuda", dtype=dt).mul_(scale)

    torch.manual_seed(431)
    states = rnd((4, b, h, w, m), 0.35, dtype)
    pole = [rnd((m,), 0.15) for _ in range(4)]
    vx = rnd((1, w, m), 0.1).abs_().add_(0.8).repeat(2, 1, 1)
    vy = rnd((2, h, m), 0.1).abs_().add_(0.8)
    path = [rnd(dim, 0.17) for dim in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    step = 1 + coarse
    dy = [rnd((b, h // step, w // step, m), 0.2, dtype) for _ in range(2)]
    return pole, states, [vx, vy], path, dy


def graph(fn: Callable, repeats: int, traffic: int = 0) -> dict:
    warm = fn()
    torch.cuda.synchronize()
    del warm
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        outputs = fn()
    capture.replay()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(repeats):
        start.record()
        for _ in range(3):
            capture.replay()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) / 3)
    del outputs, capture
    low, median, high = statistics.quantiles(times, n=4, method="inclusive")
    return {
        "raw_ms": times, "median_ms": median, "iqr_ms": high - low,
        "payload_bytes": traffic, "effective_GBps": traffic / (median * 1e6) if traffic else None,
    }


def pair_graph(
    fwd: Callable, bwd: Callable, repeats: int, traffic: tuple[int, int] = (0, 0),
) -> dict:
    forward = graph(fwd, repeats, traffic[0])
    backward = graph(bwd, repeats, traffic[1])
    return {"forward": forward, "backward": backward,
            "total_median_ms": forward["median_ms"] + backward["median_ms"]}


def stats(a: Tensor, b: Tensor) -> dict:
    a, b = a.float(), b.float()
    abs_error = (a - b).abs()
    rel_error = abs_error / b.abs().clamp_min(1e-6)
    return {
        "max_abs": abs_error.max().item(), "mean_abs": abs_error.mean().item(),
        "max_rel_floor_1e-6": rel_error.max().item(),
        "mean_rel_floor_1e-6": rel_error.mean().item(),
        "max_reference": b.abs().max().item(),
    }


def parity(
    shape: tuple[int, int, int, int, bool], dtype: torch.dtype, *, edge: bool,
) -> Iterator[dict]:
    b, h, w, m, coarse = shape
    pole, states, aux, path, dy = inputs(shape, dtype)
    args = pole, states, aux, path, coarse
    os.environ[POLICY] = "off"
    fused = (
        *scans.vertical_forward(*args), *scans.vertical_backward(*args[:4], *dy, coarse),
    )
    # Run the PyTorch reference with FP32 recurrent state and MLP operands.
    ref_pole = [p.detach().requires_grad_() for p in pole]
    ref_state = states.float().requires_grad_()
    ref_path = [p.detach().requires_grad_() for p in path]
    ref_out = scans._reference_vertical(ref_pole, ref_state, aux, ref_path, coarse, 1e-6)
    ref_grad = torch.autograd.grad(
        ref_out, [*ref_pole, ref_state, *ref_path], [g.float() for g in dy],
    )
    reference = (*ref_out, *ref_grad)
    if dtype == torch.float16:
        tolerances = ((8e-3, 1e-3), (5e-2, 2e-2))
    else:
        tolerances = ((4e-2, 5e-3), (1.2e-1, 8e-2))
    for policy in ("test_saved_tc", "chunk_tc"):
        os.environ[POLICY] = policy
        for layout in (("separate", "planar", "interleaved") if edge else ("separate",)):
            saved = None
            dy_used = dy
            if layout == "interleaved":
                dy_used = torch.cat(dy, dim=-1).split(m, dim=-1)
            if policy == "test_saved_tc":
                out, second, saved = scans._vertical_forward_saved(
                    *args, 1e-6, layout == "planar", True, layout == "separate",  # noqa: FBT003
                )
                if layout == "separate":
                    pair = out, second
                else:
                    pair = out.unbind(0) if layout == "planar" else out.split(m, dim=-1)
            elif layout == "separate":
                pair = scans.vertical_forward(*args)
            else:
                out = scans.vertical_forward_packed(*args, 1e-6, layout == "planar")
                pair = out.unbind(0) if layout == "planar" else out.split(m, dim=-1)
            for compact in ((False, True) if edge and coarse else (False,)):
                actual = (*pair, *scans.vertical_backward(
                    *args[:4], *dy_used, coarse, compact=compact, saved_features=saved,
                ))
                evidence = {}
                errors = []
                for name, expected in (("fused", fused), ("fp32", reference)):
                    targets = expected
                    if compact:
                        gs = expected[6]
                        gs = torch.stack((gs[0, ..., 1::2, :], gs[1, ..., 1::2, :],
                                          gs[2, ..., ::2, :], gs[3, ..., ::2, :]))
                        targets = (*expected[:6], gs, *expected[7:])
                    evidence[name] = {}
                    fields = zip(NAMES, actual, targets, strict=True)
                    for i, (field, value, target) in enumerate(fields):
                        evidence[name][field] = stats(value, target)
                        rtol, atol = tolerances[i >= 2]
                        try:
                            torch.testing.assert_close(
                                value.float(), target.float(), rtol=rtol, atol=atol,
                            )
                        except AssertionError:
                            errors.append(f"{name}:{field}")
                yield {"shape": [b, h, w, m, coarse], "dtype": str(dtype), "policy": policy,
                       "layout": layout, "compact": compact, "metrics": evidence, "errors": errors}


def micro(shape: tuple[int, int, int, int, bool], dtype: torch.dtype, repeats: int) -> dict:
    pole, states, aux, path, dy = inputs(shape, dtype)
    args = pole, states, aux, path, shape[-1]
    b, h, w, m, coarse = shape
    cells = b * (h // (1 + coarse)) * (w // (1 + coarse))
    features = scans._empty_features(states, cells, m, cell_major=False).uniform_(-0.5, 0.5)
    cell_major = features.transpose(0, 1).contiguous().transpose(0, 1)
    packed_features = features.permute(1, 2, 0).contiguous().permute(2, 0, 1)
    out = [torch.empty_like(g) for g in dy]
    row: dict[str, Any] = {"shape": shape, "dtype": str(dtype)}
    row["projection"] = {}
    bytes_f = features.numel() * features.element_size()
    bytes_y = sum(g.numel() * g.element_size() for g in dy)
    for label, ft in (
        ("legacy", features), ("tc_planes", features),
        ("tc_cell_major", cell_major), ("tc_packed", packed_features),
    ):
        tc = label != "legacy"
        row["projection"][label] = pair_graph(
            partial(
                scans._project_path, ft, path, *out,
                backward=False, precise=True, projection_tc=tc,
            ),
            partial(
                scans._project_path, ft, path, *dy,
                backward=True, precise=False, projection_tc=tc,
            ),
            repeats, (bytes_f + bytes_y, 2 * bytes_f + bytes_y),
        )
    row["pure"] = {}
    for label, ft in (
        ("legacy", features), ("cell_major", cell_major), ("packed", packed_features),
    ):
        cells_first, packed = label == "cell_major", label == "packed"
        row["pure"][label] = pair_graph(
            partial(
                scans._vscan, *args, 1e-6, True,  # noqa: FBT003
                pure=True, cell_major=cells_first, packed_features=packed,
            ),
            partial(
                scans._vscan, *args, 1e-6, False,  # noqa: FBT003
                pure=True, features=ft, compact=coarse,
                cell_major=cells_first, packed_features=packed,
            ),
            repeats,
        )
    del features, cell_major, packed_features, ft
    row["end_to_end"] = {}
    for policy in ("off", "test_saved", "test_saved_tc", "chunk_tc"):
        os.environ[POLICY] = policy
        saved = None
        if scans._should_save_vscan(states, coarse, 8):
            saved = scans._vertical_forward_saved(*args, 1e-6, True, True, False)[2]  # noqa: FBT003

        forward = (
            partial(scans._vertical_forward_saved, *args, 1e-6, True, True, False)  # noqa: FBT003
            if saved is not None
            else partial(scans.vertical_forward_packed, *args, 1e-6, True)  # noqa: FBT003
        )
        backward = partial(
            scans.vertical_backward, *args[:4], *dy, coarse, compact=coarse, saved_features=saved,
        )
        result = pair_graph(forward, backward, repeats)
        result["saved_features_active"] = saved is not None
        row["end_to_end"][policy] = result
        if policy == "test_saved_tc" and saved is not None:
            with patch.object(scans, "_saved_backward_chunk", return_value=b):
                row["saved_tc_backward_full_chunk"] = graph(backward, repeats)
        del saved, backward
        torch.cuda.empty_cache()
    return row


def default_batch(height: int) -> int:
    return 16 if height >= 28 else 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("parity", "micro"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--case", type=int, choices=range(len(CASES)), nargs="+")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    if args.mode == "parity" and args.edge:
        shapes = [(b, h, w, m, False) for b, h, w, m in EDGE]
        shapes.extend((b, h + 1, w + 1, m, True) for b, h, w, m in EDGE)
    else:
        shapes = [
            (args.batch or (128 if args.mode == "micro" else default_batch(h)), h, w, m, coarse)
            for index in (args.case if args.case is not None else range(len(CASES)))
            for h, w, m, coarse in (CASES[index],)
        ]
    failed = False
    with args.out.open("a", encoding="utf-8") as out:
        for shape in shapes:
            print(shape, dtype, flush=True)
            if args.mode == "parity":
                for row in parity(shape, dtype, edge=args.edge):
                    print(json.dumps(row), file=out, flush=True)
                    failed |= bool(row["errors"])
                continue
            batch = shape[0]
            while batch:
                try:
                    row = micro((batch, *shape[1:]), dtype, args.repeats)
                except torch.OutOfMemoryError:
                    batch //= 2
                else:
                    print(json.dumps(row), file=out, flush=True)
                    break
                gc.collect()
                torch.cuda.empty_cache()
            failed |= not batch
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
