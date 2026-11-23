"""Measure horizontal kernels in CUDA graphs and check exact A/B parity."""

# ruff: noqa: T201, SLF001
import argparse
import json
import os
import statistics

import torch

from lnet import general_cuda_scans as scans
from lnet.product_scan_reference import bidirectional_product_scan_reference

SHAPES = [(56, 56, 160), (28, 28, 160), (14, 14, 160), (7, 7, 128),
          (64, 96, 160), (37, 53, 192), (128, 192, 160)]


def graph_ms(call, repeats: int = 9) -> float:  # noqa: ANN001
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):
            call()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(repeats)]
    for start, stop in events:
        start.record()
        graph.replay()
        stop.record()
    events[-1][1].synchronize()
    return statistics.median(start.elapsed_time(stop) / 5 for start, stop in events)


def check_reference(
    pole: list[torch.Tensor],
    source: list[torch.Tensor],
    grad: torch.Tensor,
    outputs: list[torch.Tensor],
    *,
    compact: bool,
    causal: bool,
) -> None:
    params = [x.detach().float().requires_grad_() for x in [*pole, *source]]
    states = bidirectional_product_scan_reference(tuple(params[:4]), tuple(params[4:]))
    selected = (
        [state[:, :, int(i < 2)::2] for i, state in enumerate(states)] if compact else states
    )
    if causal:
        selected = selected[:2]
    expected = torch.autograd.grad(selected, params, tuple(grad[:len(selected)].float()))
    actual = [*outputs[1:5], *outputs[5].chunk(2, dim=-1)]
    atol, rtol = {torch.float32: (5e-5, 5e-5), torch.float16: (0.002, 0.004),
                  torch.bfloat16: (0.015, 0.02)}[source[0].dtype]
    for a, b in zip([*outputs[0], *actual], [*states, *expected], strict=True):
        torch.testing.assert_close(a.float(), b, atol=atol, rtol=rtol)


def case(  # noqa: C901
    shape: tuple, dtype: torch.dtype, *, compact: bool, measure: bool, causal: bool = False,
) -> dict:
    def sample(size: tuple) -> torch.Tensor:
        return torch.randn(size, device="cuda", dtype=dtype) * 0.2

    real, imag = sample(shape), sample(shape)
    pole = [sample((shape[-1],)).float() for _ in range(4)]
    b, h, w, m = shape
    grad = sample((4, b, h, w // 2 if compact else w, m))
    if causal:
        grad[2:].zero_()

    def forward() -> torch.Tensor:
        return scans.horizontal_forward(pole, real, imag)

    states = forward()

    def backward() -> list[torch.Tensor]:
        return scans.horizontal_backward(pole, [real, imag], states, grad, compact)

    reference = None
    result = {}
    for mode in ("off", "auto", "on"):
        os.environ["LNET_HSCAN_OPT"] = mode
        launch = scans._plan_launch(
            real, shape, backward=True, vertical=False, coarse=False, hidden=0,
            precise=False, compact=compact,
        )
        outputs = [forward(), *backward()]
        if reference is None:
            reference = outputs
        max_error = [(a.float() - b.float()).abs().max().item()
                     for a, b in zip(outputs, reference, strict=True)]
        for index, (a, b) in enumerate(zip(outputs, reference, strict=True)):
            if index in (0, 5):
                atol, rtol = 2e-6, torch.finfo(a.dtype).eps
            else:
                # Pole sums include the two low-precision state recurrences.
                atol, rtol = (0.0005, 0.0005) if dtype == torch.bfloat16 else (5e-5, 5e-5)
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        row = {"paired": launch.key.paired_hscan, "smem": launch.plan.smem,
               "warps": launch.plan.warps, "workers": launch.workers,
               "partials": launch.partials, "max_error": max_error}
        if measure and mode != "on":
            del outputs
            row["fwd_ms"] = graph_ms(forward)
            row["bwd_ms"] = graph_ms(backward)
            row["total_ms"] = row["fwd_ms"] + row["bwd_ms"]
        result[mode] = row
    if not measure:
        check_reference(pole, [real, imag], grad, reference, compact=compact, causal=causal)
    if measure:
        result["delta_pct"] = 100 * (result["auto"]["total_ms"] / result["off"]["total_ms"] - 1)
    return {"shape": shape, "dtype": str(dtype), "compact": compact, "causal": causal, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(731)
    shapes = [(3, 3, 16), (7, 7, 64), (13, 13, 128), (14, 14, 160), (28, 28, 320),
              (31, 31, 17), (56, 56, 160), (64, 64, 64), (96, 96, 16)] if args.parity else SHAPES
    if args.quick:
        shapes = SHAPES[:1]
    dtypes = [torch.bfloat16, torch.float16]
    if args.parity:
        dtypes.append(torch.float32)
    for h, w, m in shapes:
        for dtype in dtypes:
            for compact in ((False, True) if w % 2 == 0 else (False,)):
                batch = 1 if args.parity else min(args.batch, max(1, 2**26 // (h * w * m)))
                for causal in ((False, True) if args.parity else (False,)):
                    row = case((batch, h, w, m), dtype, compact=compact,
                               measure=not args.parity, causal=causal)
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
