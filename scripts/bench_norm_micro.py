"""Time the four weighted ChannelRMSNorm activation shapes of VA-17."""

from __future__ import annotations

import os
from functools import partial

import torch
from bench_stem_micro import DTYPES, arguments, cold_graph_ms, relative_errors, result_json
from torch.nn import functional

from lnet.memory_backbone import ChannelRMSNorm

# (spatial extent, channels, model call count), traced at 224x224.
SHAPES = ((56, 64, 3), (28, 96, 4), (14, 128, 12), (7, 256, 4))


def norm_calls(batch: int, side: int, width: int, dtype: torch.dtype) -> dict:
    x = torch.randn(batch, side, side, width, device="cuda", dtype=dtype, requires_grad=True)
    norm = ChannelRMSNorm(width).cuda()

    def reference() -> torch.Tensor:
        return functional.rms_norm(x, (width,), norm.weight.to(dtype), norm.epsilon)

    return {"off": reference, "fast_fwd": partial(norm, x), "parameters": (x, norm.weight)}


def main() -> int:
    parser = arguments(__doc__, "off,fast_fwd")
    args = parser.parse_args()
    modes = args.modes.split(",")
    if set(modes) - {"off", "fast_fwd"} or min(args.batch, args.iters) <= 0:
        parser.error("expected valid modes, positive batch and iterations")
    os.environ.update(LNET_NORM_W50_MODE="fast_fwd", LNET_MODEL_W36B="off", LNET_GENERAL_SCANS="1")
    torch.manual_seed(176)
    weighted = dict.fromkeys(modes, 0.0)
    timings, parity = {}, {}
    flush = torch.zeros(128 * 1024**2, device="cuda", dtype=torch.uint8) if args.flush_l2 else None
    for side, width, count in SHAPES:
        if args.parity:
            for name, dtype in DTYPES.items():
                calls = norm_calls(2, 7, width, dtype)
                ref = calls["off"]()
                dy = torch.randn_like(ref)
                expected = (ref, *torch.autograd.grad(ref, calls["parameters"], dy))
                got = calls["fast_fwd"]()
                actual = (got, *torch.autograd.grad(got, calls["parameters"], dy))
                parity[f"{name}_{width}"] = relative_errors(actual, expected)
        calls = norm_calls(args.batch, side, width, torch.bfloat16)
        timings[width] = {}
        for mode in modes:
            elapsed = cold_graph_ms(calls[mode], args.iters, flush)
            timings[width][mode] = elapsed
            weighted[mode] += count * elapsed
        del calls
    passed = all(
        max(errors) <= (0.035 if name.startswith("bf16") else 0.006)
        for name, errors in parity.items()
    )
    return result_json(args.out, {
        "status": "ok" if passed else "error", "batch": args.batch,
        "flush_l2": args.flush_l2, "bf16_ms": timings, "weighted_bf16_ms": weighted,
        "parity_rel_l2": parity,
    })


if __name__ == "__main__":
    raise SystemExit(main())
