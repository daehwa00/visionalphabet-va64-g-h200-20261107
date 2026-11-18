"""Measure the small-grid scan candidate and record large-grid resource failures."""
from __future__ import annotations

import argparse
import itertools
import json
from functools import partial
from pathlib import Path

import torch
import triton
from torch import Tensor

from lnet.pac_product_scan_pipeline import (
    run_product_scan_path_collapse_pipeline_packed,
    run_product_scan_pipeline_packed,
)
from lnet.pac_triton_scan2d_inference import run_product_scan_2d_inference


def _inputs(batch: int, size: int, modes: int, *, swiglu: bool) -> tuple:
    drive = torch.randn(batch, size, size, 2 * modes, device="cuda", dtype=torch.bfloat16)
    px = tuple(torch.full((1, 1, 1, modes), x, device="cuda") for x in (0.7, 0.12, 0.8, -0.3))
    py = tuple(torch.full((1, 1, 1, modes), x, device="cuda") for x in (0.55, -0.21, 0.6, 0.2))
    value = 8
    hidden = 3 * value if swiglu else 16
    path = tuple(torch.randn(*shape, device="cuda") * 0.1
                 for shape in ((modes, hidden, 8), (modes, hidden),
                               (modes, 2, 2 * value if swiglu else hidden), (modes, 2)))
    return drive, px, py, path


def _baseline(drive: Tensor, px: tuple, py: tuple, path: tuple, epilogue: str,
              gain: str, *, swiglu: bool, planar: bool) -> Tensor:
    if epilogue == "path":
        return run_product_scan_path_collapse_pipeline_packed(
            px, py, drive, path, gain_normalization=gain,
            path_swiglu=swiglu, planar_output=planar,
        )
    return run_product_scan_pipeline_packed(
        px, py, drive, epilogue=epilogue, gain_normalization=gain,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[7, 14])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--modes", type=int, nargs="+", default=[32, 48, 64, 96, 128, 160])
    parser.add_argument("--epilogues", nargs="+", choices=["path", "descriptor", "mean_energy"],
                        default=["path", "mean_energy"])
    parser.add_argument("--block-modes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--num-warps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--gain", choices=["global", "pointwise"], default="global")
    parser.add_argument("--swiglu", action="store_true")
    parser.add_argument("--planar", action="store_true")
    parser.add_argument("--experimental-large-grid", action="store_true")
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; no GPU timing was performed")
    torch.manual_seed(8823)
    records: list[dict] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                                        "precision": "BF16", "l2_gate": 0.01,
                                        "records": records}, indent=2) + "\n")

    with torch.inference_mode():
        for batch, size, modes, epilogue in itertools.product(
            args.batches, args.sizes, args.modes, args.epilogues,
        ):
            # A sweep exceeds Dynamo's per-code shape guard cache otherwise.
            torch.compiler.reset()
            drive, px, py, path = _inputs(batch, size, modes, swiglu=args.swiglu)
            base = torch.compile(partial(_baseline, drive, px, py, path, epilogue, args.gain,
                                         swiglu=args.swiglu, planar=args.planar), fullgraph=True)
            expected = base()
            torch.cuda.synchronize()
            base_ms = triton.testing.do_bench_cudagraph(base, rep=args.rep_ms)
            for block_modes, warps in itertools.product(args.block_modes, args.num_warps):
                row = {"batch": batch, "size": size, "modes": modes, "epilogue": epilogue,
                       "gain": args.gain, "swiglu": args.swiglu, "planar": args.planar,
                       "block_modes": block_modes, "warps": warps, "baseline_ms": base_ms,
                       "horizontal_bytes_removed": 2 * 4 * drive.numel() // 2 * drive.element_size()}
                try:
                    # Baseline timing is complete; isolate the next launch geometry.
                    torch.compiler.reset()
                    candidate = partial(
                        run_product_scan_2d_inference, px, py, drive,
                        path if epilogue == "path" else None,
                        epilogue=epilogue, gain_normalization=args.gain,
                        path_swiglu=args.swiglu if epilogue == "path" else False,
                        planar_output=args.planar if epilogue == "path" else False,
                        block_modes=block_modes, num_warps=warps,
                        experimental_large_grid=args.experimental_large_grid,
                    )
                    compiled = torch.compile(candidate, fullgraph=True)
                    actual = compiled()
                    torch.cuda.synchronize()
                    relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
                    row["relative_l2"] = float(relative)
                    row["max_abs"] = float((actual.float() - expected.float()).abs().max())
                    row["valid"] = bool(torch.isfinite(actual).all() and relative <= 0.01
                                        and actual.dtype == expected.dtype
                                        and actual.shape == expected.shape)
                    if row["valid"]:
                        row["candidate_ms"] = triton.testing.do_bench_cudagraph(compiled, rep=args.rep_ms)
                        row["speedup"] = base_ms / row["candidate_ms"]
                except Exception as exc:  # noqa: BLE001 - Save rejected compile/resource failures.
                    row.update(valid=False, error=f"{type(exc).__name__}: {exc}")
                records.append(row)
                save()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
