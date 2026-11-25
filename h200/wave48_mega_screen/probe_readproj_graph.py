"""Measure L2-cold CUDA Graph read projection on a dedicated stream."""

# The probe shares the existing microbenchmark's input and parity helpers.
# ruff: noqa: ANN001, ANN201, SLF001, T201
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_readproj_micro as projection  # noqa: E402
from bench_reader_micro import l2_buffer  # noqa: E402

MODES = ("off", "dual_fwd_only", "dual_mma_triton", "dual_all", "cublas_cat_ablation")


def measure(replay, evict):
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(5)
    ]
    for start, end in events:
        start.record()
        end.record()
    events[-1][1].synchronize()
    for start, end in events:
        evict.zero_()
        start.record()
        replay()
        end.record()
    events[-1][1].synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in events)


def capture(readproj, inputs, mode, evict, stream, *, tiny):
    real, imag, weight, gradient, bias = inputs
    leaves = (real, imag, weight) if bias is None else (real, imag, weight, bias)

    def forward() -> torch.Tensor:
        return projection._fwd(readproj, inputs, mode)

    def forward_backward() -> torch.Tensor:
        for leaf in leaves:
            leaf.grad = None
        output = forward()
        output.backward(gradient)
        return output

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward_backward()
        for leaf in leaves:
            leaf.grad = None
    torch.cuda.synchronize()

    fwd = torch.cuda.CUDAGraph()
    fb = torch.cuda.CUDAGraph()
    with torch.cuda.graph(fwd, stream=stream):
        fwd_output = forward()
    with torch.cuda.graph(fb, stream=stream):
        fb_output = forward_backward()
    dx, dy, dw, *db = (leaf.grad for leaf in leaves)
    dr, di = dw.split((real.shape[1], imag.shape[1]), dim=1)

    with torch.cuda.stream(stream):
        fwd_ms = measure(fwd.replay, evict)
        fb_ms = measure(fb.replay, evict)
        row = {"fwd_ms": fwd_ms, "bwd_ms": fb_ms - fwd_ms, "total_ms": fb_ms}
        if tiny:
            row["eager_fwd_ms"] = measure(forward, evict)
        # Replays overwrite the same storage. Keep these objects alive until parity.
        fwd.replay()
        fb.replay()
    stream.synchronize()
    for leaf in leaves:
        leaf.grad = None
    return row, (fwd_output.detach(), dx, dy, dr, di, *db), fb_output.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(161)
    torch.backends.cuda.matmul.allow_tf32 = False
    evict = l2_buffer()
    stream = torch.cuda.Stream()
    readproj = projection._load_projection().readproj
    rows = []
    cases = (*projection.MODEL_CASES, (1027, 37, 51, 77, 0))
    for m, n, kr, ki, calls in cases:
        shape = (m, n, kr, ki)
        tiny = calls == 0
        for dtype in (torch.bfloat16, torch.float16):
            inputs = projection._inputs(shape, dtype, bias=tiny)
            baseline = projection._values(readproj, inputs, "off")
            fp32 = tuple(
                value.detach().float().requires_grad_(value.requires_grad)
                if value is not None
                else None
                for value in inputs
            )
            reference = projection._values(readproj, fp32, "off")
            del fp32
            for mode in MODES:
                timing, outputs, fb_output = capture(
                    readproj, inputs, mode, evict, stream, tiny=tiny
                )
                parity = projection._parity(outputs, baseline, reference, dtype)
                parity["fb_output"] = projection._parity(
                    (fb_output,),
                    baseline[:1],
                    reference[:1],
                    dtype,
                )["out"]
                passed = all(entry["passed"] for entry in parity.values())
                rows.append(
                    dict(
                        shape=shape,
                        calls=calls,
                        dtype=str(dtype),
                        mode=mode,
                        **timing,
                        parity=parity,
                        parity_passed=passed,
                    )
                )
                print(shape, dtype, mode, timing, passed, flush=True)
                del outputs, fb_output
                args.out.write_text(
                    json.dumps(
                        {"flush_bytes": evict.numel(), "rows": rows}, indent=2, allow_nan=False
                    )
                    + "\n"
                )
            del inputs, baseline, reference
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
