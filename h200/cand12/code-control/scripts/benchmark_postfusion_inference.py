"""Benchmark all remaining PostFusion fusion candidates at actual VA128 sites.

Run on the serialized experiment GPU, for example:
    PYTHONPATH=src:scripts python scripts/benchmark_postfusion_inference.py --output result.json

The weighted sum uses all eleven deployed sites: 56x56 once, 28x28 twice,
14x14 six times, and 7x7 twice. These are component timings, not an end-to-end
claim. The source model and checkpoint are never mutated.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch._inductor.config as inductor_config

from lnet.pac_gated_post_fusion import GatedComplexPostFusion
from lnet.pac_postfusion_inference import POSTFUSION_VARIANTS, FrozenGatedPostFusion

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

SPATIAL_MULTIPLICITY = {56: 1, 28: 2, 14: 6, 7: 2}


def _time_cuda_graph(function: Callable[[], Tensor], iterations: int) -> float:
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    for _ in range(10):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires the serialized CUDA experiment device")
    torch.manual_seed(911)
    dtype = torch.bfloat16 if args.input_dtype == "bf16" else torch.float32
    records = []
    for batch in args.batches:
        for spatial, multiplicity in SPATIAL_MULTIPLICITY.items():
            source = GatedComplexPostFusion(128, 192).eval().cuda().requires_grad_(requires_grad=False)
            source.norm.weight.data.uniform_(0.5, 1.5)
            packed = torch.randn(batch, spatial, spatial, 256, device="cuda", dtype=dtype)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                eager_reference = source.forward_packed(packed).clone()
                for variant in ("baseline", *args.variants):
                    module = source if variant == "baseline" else FrozenGatedPostFusion(source, variant)
                    # Bound methods share code objects across modules and shapes.
                    # Isolate each standalone measurement from Dynamo's default
                    # eight-recompilation limit; never reset a serving compiler.
                    torch.compiler.reset()
                    # Capture one outer CUDA Graph for timing; nested Inductor
                    # graph capture is disabled here to make each result comparable.
                    with inductor_config.patch(freezing=True):
                        compiled = torch.compile(
                            module.forward_packed, fullgraph=True, dynamic=False,
                            options={"freezing": True, "triton.cudagraphs": False},
                        )
                        actual = compiled(packed).clone()
                        torch.cuda.synchronize()
                        error = (actual.float() - eager_reference.float()).norm()
                        relative = (error / eager_reference.float().norm()).item()
                        maximum = (actual.float() - eager_reference.float()).abs().max().item()
                        if not torch.isfinite(actual).all() or relative > 0.01:
                            raise RuntimeError(f"{variant} B{batch} S{spatial} parity failed: {relative}")
                        latency = _time_cuda_graph(partial(compiled, packed), args.iterations)
                    record = {
                        "variant": variant, "batch": batch, "spatial": spatial,
                        "multiplicity": multiplicity, "latency_ms": latency,
                        "relative_l2_error": relative, "max_abs_error": maximum,
                    }
                    records.append(record)
                    print(json.dumps(record), flush=True)
    totals = []
    for batch in args.batches:
        baseline = sum(
            row["latency_ms"] * row["multiplicity"] for row in records
            if row["batch"] == batch and row["variant"] == "baseline"
        )
        for variant in ("baseline", *args.variants):
            total = sum(
                row["latency_ms"] * row["multiplicity"] for row in records
                if row["batch"] == batch and row["variant"] == variant
            )
            totals.append({
                "batch": batch, "variant": variant, "weighted_postfusion_ms": total,
                "speedup": baseline / total,
            })
    return {
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "input_dtype": args.input_dtype, "modes": 128, "hidden_modes": 192,
        "spatial_multiplicity": SPATIAL_MULTIPLICITY, "records": records, "totals": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--variants", nargs="+", choices=POSTFUSION_VARIANTS, default=list(POSTFUSION_VARIANTS))
    parser.add_argument("--input-dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or any(batch < 1 for batch in args.batches):
        parser.error("batches and iterations must be positive")
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
