"""CUDA component benchmark for the post-scan memory-projection boundary.

This is projection→residual(/norm) fusion, not scan→projection fusion. It keeps
the planar scan output: scan CTAs operate on independent pole tiles, but the
projection requires every pole. Run trained end-to-end evaluation separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from triton.testing import do_bench_cudagraph

from lnet.pac_complex_layers import ComplexLinear
from lnet.pac_triton_complex_rmsnorm import packed_complex_rms_norm_from_packed
from lnet.pac_triton_memory_projection_inference import memory_projection_residual_norm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--height", type=int, default=14)
    parser.add_argument("--width", type=int)
    parser.add_argument("--inputs", type=int, default=160)
    parser.add_argument("--outputs", type=int, default=128)
    parser.add_argument("--residual-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA; CPU tests do not execute it")
    width = args.width if args.width is not None else args.height
    if min(args.batch, args.height, width, args.inputs, args.outputs, args.rep_ms) <= 0:
        raise ValueError("shape and repetition arguments must be positive")
    torch.manual_seed(7281)
    memory = torch.randn(
        2, args.batch, args.height, width, args.inputs, device="cuda", dtype=torch.bfloat16,
    )
    residual = torch.randn(
        args.batch, args.height, width, 2 * args.outputs, device="cuda",
        dtype=torch.float32 if args.residual_dtype == "fp32" else torch.bfloat16,
    )
    projection = (
        ComplexLinear(args.inputs, args.outputs).cuda().eval().requires_grad_(requires_grad=False)
    )
    weight = torch.rand(args.outputs, device="cuda") + 0.2

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        merged = residual + projection(memory, planar=True)
        return merged, packed_complex_rms_norm_from_packed(merged, weight, 1e-6)

    if not args.eager:
        baseline = torch.compile(baseline, fullgraph=True)
    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = tuple(value.clone() for value in baseline())
        baseline_ms = cast("float", do_bench_cudagraph(baseline, rep=args.rep_ms))
        for normalize in (False, True):
            for block_rows in (16, 32):
                for output_tile in ((64,) if normalize else (32, 64, 128)):
                    for num_warps in (4, 8):
                        def candidate(
                            *,
                            normalize: bool = normalize, block_rows: int = block_rows,
                            output_tile: int = output_tile, num_warps: int = num_warps,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
                            merged, normalized = memory_projection_residual_norm(
                                memory, projection.weight_real, projection.weight_imag,
                                residual, weight, normalize=normalize, block_rows=block_rows,
                                output_tile=output_tile, num_warps=num_warps,
                            )
                            if not normalize:
                                normalized = packed_complex_rms_norm_from_packed(
                                    merged, weight, 1e-6,
                                )
                            return merged, normalized

                        if not args.eager:
                            # All configurations share one Python code object.
                            # Isolate them from Dynamo's per-code recompile cap.
                            torch.compiler.reset()
                            candidate = torch.compile(candidate, fullgraph=True)
                        config = {
                            "fusion": "projection_residual_norm" if normalize else "projection_residual",
                            "block_rows": block_rows, "output_tile": output_tile,
                            "num_warps": num_warps,
                        }
                        try:
                            actual = candidate()
                            errors = [
                                float((left.float() - right.float()).norm()
                                      / right.float().norm().clamp_min(1e-12))
                                for left, right in zip(actual, expected, strict=True)
                            ]
                            finite = all(bool(torch.isfinite(value).all()) for value in actual)
                            passed = finite and errors[0] <= 0.005 and errors[1] <= 0.008
                            elapsed = (
                                cast("float", do_bench_cudagraph(candidate, rep=args.rep_ms))
                                if passed else None
                            )
                            record = {
                                **config, "passed": passed, "relative_l2_errors": errors,
                                "ms": elapsed, "speedup": baseline_ms / elapsed if elapsed else None,
                            }
                        except Exception as error:  # noqa: BLE001 - report each unsupported geometry.
                            record = {**config, "passed": False, "error": str(error)}
                        records.append(record)
                        print(json.dumps(record), flush=True)
    result = {
        "scope": "projection→residual(/norm); planar scan output retained",
        "device": torch.cuda.get_device_name(),
        "shape": {key: getattr(args, key) for key in ("batch", "height", "inputs", "outputs")}
                 | {"width": width},
        "compiled": not args.eager, "residual_dtype": args.residual_dtype,
        "baseline_ms": baseline_ms,
        "trained_accuracy_evaluated": False, "candidates": records,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
