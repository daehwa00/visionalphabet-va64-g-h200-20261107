"""Measure genuine vertical scan/path collapse -> complex projection fusion."""

from __future__ import annotations

# pyright: reportPrivateUsage=false, reportArgumentType=false, reportUnknownLambdaType=false
import argparse
import itertools
import json
from functools import partial
from pathlib import Path
from typing import cast

import torch
import triton
from torch import Tensor
from torch.library import triton_op
from torch.nn import functional

from lnet.pac_product_scan_contracts import DEFAULT_EPSILON
from lnet.pac_product_scan_normalization import static_product_scan_auxiliary
from lnet.pac_product_scan_ops import (
    PATH_COLLAPSE_FORWARD_LAUNCH_NAME,
    _launch_product_scan4_forward,
)
from lnet.pac_triton_vertical_projection_inference import (
    Auxiliary,
    Pole,
    run_vertical_scan_projection_inference,
)
from lnet.pac_triton_vertical_projection_inference import (
    Path as PathParameters,
)


@triton_op("lnet::vertical_projection_benchmark_baseline", mutates_args={})
def baseline(
    horizontal: Tensor,
    poles: list[Tensor],
    auxiliary: list[Tensor],
    path: list[Tensor],
    weight_real: Tensor,
    weight_imag: Tensor,
    kind: int,
    path_swiglu: bool,  # noqa: FBT001 - Registered operator scalar metadata.
) -> Tensor:
    """Existing fused vertical/path kernel followed by existing four GEMMs."""
    packed, _, _ = _launch_product_scan4_forward(
        *cast("Pole", tuple(poles)), *cast("Pole", tuple(horizontal.unbind(0))),
        *cast("Auxiliary", tuple(auxiliary)), DEFAULT_EPSILON, kind,
        emit_coarse=False, path_collapse=cast("PathParameters", tuple(path)),
        path_swiglu=path_swiglu, emit_descriptor=False,
        packed_output=True, planar_output=True,
        launch_name=PATH_COLLAPSE_FORWARD_LAUNCH_NAME,
    )
    real, imag = packed.unbind(0)
    wr, wi = weight_real.bfloat16(), weight_imag.bfloat16()
    return torch.cat(
        (functional.linear(real, wr) - functional.linear(imag, wi),
         functional.linear(real, wi) + functional.linear(imag, wr)), dim=-1,
    )


def make_inputs(
    batch: int, height: int, width: int, poles: int, outputs: int, *,
    device: str = "cuda", swiglu: bool = False, gain: str = "global",
) -> tuple[Tensor, Pole, Auxiliary, PathParameters, Tensor, Tensor]:
    """Finite frozen inputs with actual P160 -> M128 production widths."""
    generator = torch.Generator(device=device).manual_seed(9021)
    horizontal = torch.randn(4, batch, height, width, poles, generator=generator,
                             device=device, dtype=torch.bfloat16)
    px, py = tuple(
        tuple(torch.full((1, 1, 1, poles), coefficient, device=device)
              for coefficient in axis)
        for axis in ((0.7, 0.12, 0.8, -0.3), (0.55, -0.21, 0.6, 0.2))
    )
    kind = 1 if gain == "global" else 0
    auxiliary = static_product_scan_auxiliary(
        cast("Pole", px), cast("Pole", py), horizontal[0],
        epsilon=DEFAULT_EPSILON, gain_kind=kind,
    )
    value = 8
    hidden = 3 * value if swiglu else 16
    path = tuple(torch.randn(*shape, generator=generator, device=device) * 0.1
                 for shape in ((poles, hidden, 8), (poles, hidden),
                               (poles, 2, 2 * value if swiglu else hidden), (poles, 2)))
    wr, wi = tuple(torch.randn(outputs, poles, generator=generator, device=device)
                   * poles ** -0.5 for _ in range(2))
    return horizontal, cast("Pole", py), auxiliary, cast("PathParameters", path), wr, wi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[14])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--poles", type=int, default=160)
    parser.add_argument("--outputs", type=int, default=128)
    parser.add_argument("--pole-tiles", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--output-tiles", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--num-warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--gain", choices=["global", "pointwise"], default="global")
    parser.add_argument("--swiglu", action="store_true")
    parser.add_argument("--experimental-large-height", action="store_true")
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; no GPU timing was performed")
    records: list[dict[str, object]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(),
                                       "torch": torch.__version__, "precision": "BF16",
                                       "relative_l2_gate": 0.01, "records": records},
                                      indent=2) + "\n")

    with torch.inference_mode():
        for batch, size in itertools.product(args.batches, args.sizes):
            torch.compiler.reset()
            horizontal, py, auxiliary, path, wr, wi = make_inputs(
                batch, size, size, args.poles, args.outputs,
                swiglu=args.swiglu, gain=args.gain,
            )
            expected_fn = torch.compile(partial(
                baseline, horizontal, list(py), list(auxiliary), list(path), wr, wi,
                1 if args.gain == "global" else 0, args.swiglu,
            ), fullgraph=True)
            expected = expected_fn()
            torch.cuda.synchronize()
            baseline_ms = cast("float", triton.testing.do_bench_cudagraph(
                expected_fn, rep=args.rep_ms,
            ))
            for pole_tile, output_tile, num_warps in itertools.product(
                args.pole_tiles, args.output_tiles, args.num_warps,
            ):
                row: dict[str, object] = {"batch": batch, "height": size, "width": size,
                       "poles": args.poles, "outputs": args.outputs, "gain": args.gain,
                       "swiglu": args.swiglu, "pole_tile": pole_tile,
                       "output_tile": output_tile, "num_warps": num_warps,
                       "baseline_ms": baseline_ms,
                       "scan_repetitions": triton.cdiv(args.outputs, output_tile),
                       "path_roundtrip_bytes_removed": 2 * batch * size * size
                       * 2 * args.poles * horizontal.element_size()}
                try:
                    torch.compiler.reset()
                    candidate = torch.compile(partial(
                        run_vertical_scan_projection_inference,
                        py, horizontal, auxiliary, path, wr, wi,
                        gain_normalization=args.gain, path_swiglu=args.swiglu,
                        pole_tile=pole_tile, output_tile=output_tile, num_warps=num_warps,
                        experimental_large_height=args.experimental_large_height,
                    ), fullgraph=True)
                    actual = candidate()
                    torch.cuda.synchronize()
                    relative = ((actual.float() - expected.float()).norm()
                                / expected.float().norm().clamp_min(1e-8))
                    row.update(relative_l2=float(relative),
                               max_abs=float((actual.float() - expected.float()).abs().max()),
                               valid=bool(torch.isfinite(actual).all() and relative <= 0.01
                                          and actual.dtype == expected.dtype
                                          and actual.shape == expected.shape))
                    if row["valid"]:
                        candidate_ms = cast("float", triton.testing.do_bench_cudagraph(
                            candidate, rep=args.rep_ms,
                        ))
                        row["candidate_ms"] = candidate_ms
                        row["speedup"] = baseline_ms / candidate_ms
                except Exception as exc:  # noqa: BLE001 - Persist compile/resource failures.
                    row.update(valid=False, error=f"{type(exc).__name__}: {exc}")
                records.append(row)
                save()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
