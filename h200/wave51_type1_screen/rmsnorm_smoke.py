"""Check compiled Wave 50 fast RMSNorm on all 18 CUDA cases."""
import json
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional

from lnet.memory_backbone import ChannelRMSNorm
from lnet.model_w29c import _fast_rmsnorm


def check(width: int, dtype: torch.dtype, limit: float) -> dict:
    torch._dynamo.reset()  # noqa: SLF001
    torch.manual_seed(176)
    norm = ChannelRMSNorm(width).cuda().to(dtype)
    x = torch.randn(2, 7, 9, width, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)

    def forward(source: torch.Tensor) -> torch.Tensor:
        if dtype == torch.float32:
            return _fast_rmsnorm(source, norm.weight, norm.epsilon, True)[0]  # noqa: FBT003
        return norm(source)

    def run(y: torch.Tensor) -> tuple:
        return y, *torch.autograd.grad(y, (x, norm.weight), dy.to(y.dtype))

    expected = run(functional.rms_norm(x, (width,), norm.weight, norm.epsilon))
    actual = run(torch.compile(forward, backend="aot_eager", fullgraph=True)(x))
    error = max(
        ((got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)).item()
        for got, ref in zip(actual, expected, strict=True)
    )
    return {"width": width, "dtype": str(dtype), "error": error, "limit": limit,
            "passed": error <= limit}


if __name__ == "__main__":
    os.environ.update(LNET_NORM_W50_MODE="fast_fwd", LNET_MODEL_W36B="off")
    cases = [
        check(width, dtype, limit)
        for dtype, limit in ((torch.bfloat16, 0.035), (torch.float16, 0.006), (torch.float32, 1e-5))
        for width in (63, 64, 96, 128, 256, 272)
    ]
    passed = sum(case["passed"] for case in cases)
    Path(sys.argv[1]).write_text(json.dumps({"passed": passed, "cases": cases}, indent=2) + "\n")
    print(f"RMSNorm {passed}/{len(cases)} PASS", flush=True)  # noqa: T201
    sys.exit(int(passed != len(cases)))
