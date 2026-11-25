"""Measure eager and CUDA Graph read projection with cold L2."""

# Diagnostic scripts use private benchmark hooks and print live progress.
# ruff: noqa: ANN001, ANN201, ANN202, SLF001, T201
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import bench_readproj_micro as projection  # noqa: E402
from bench_reader_micro import l2_buffer  # noqa: E402

MODES = ("off", "dual_fwd_only", "dual_mma_triton", "dual_all", "cublas_cat_ablation")


def measure(run, evict, *, graph=True, repeats=3):
    """Time events on the GPU, placing eviction before the start event."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=graph)
    end = torch.cuda.Event(enable_timing=True, external=graph)
    start.record()
    end.record()
    end.synchronize()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=stream):
            start.record()
            run()
            end.record()
    samples = []
    for _ in range(repeats):
        evict.add_(1)
        if graph:
            captured.replay()
        else:
            start.record()
            run()
            end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def compare(inputs, readproj, evict):
    rows = {}
    baseline = projection._values(readproj, inputs, "off")
    for mode in MODES:

        def forward(selected=mode):
            return projection._fwd(readproj, inputs, selected)

        out = forward()

        def backward(output=out):
            x, y, w, grad, _ = inputs
            return torch.autograd.grad(output, (x, y, w), grad, retain_graph=True)

        eager = {
            op: measure(fn, evict, graph=False) for op, fn in (("fwd", forward), ("bwd", backward))
        }
        graph = {op: measure(fn, evict) for op, fn in (("fwd", forward), ("bwd", backward))}
        actual = projection._values(readproj, inputs, mode)
        parity = {
            name: projection._error(got, expected)
            for name, got, expected in zip(projection.NAMES, actual, baseline, strict=False)
        }
        passed = all(
            metric["relative_l2"] is not None and metric["relative_l2"] <= 0.008
            for metric in parity.values()
        )
        rows[mode] = {"eager": eager, "graph": graph, "parity": parity, "parity_passed": passed}
        print(mode, rows[mode], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(161)
    evict = l2_buffer()
    module = projection._load_projection()
    rows = []
    for m, n, kr, ki, calls in projection.MODEL_CASES:
        shape = (m, n, kr, ki)
        print(f"readproj {shape}", flush=True)
        inputs = projection._inputs(shape, torch.bfloat16, bias=False)
        rows.append(
            {"shape": shape, "calls": calls, "modes": compare(inputs, module.readproj, evict)}
        )
        del inputs
        torch.cuda.empty_cache()
        payload = {"flush_bytes": evict.numel(), "rows": rows}
        args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
