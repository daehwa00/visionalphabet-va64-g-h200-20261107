"""Sweep the NVRTC stem's resident backward worker count."""

# ruff: noqa: ANN001, ANN201, ANN202, B023, SLF001, T201
import torch
from isolated import compare

from lnet import fused_stem_ops as stem


def main():
    torch.manual_seed(501)
    occupancy = stem._worker_limit
    for dtype in (torch.bfloat16, torch.float16):
        for spatial, channels in ((112, 32), (56, 64)):
            x = torch.randn(128, spatial, spatial, channels, dtype=dtype, device="cuda")
            dy = torch.randn_like(x)
            gamma = torch.rand(channels, device="cuda")
            beta = torch.rand_like(gamma)
            bias = torch.zeros_like(gamma)

            def run(factor):
                stem._worker_limit = lambda module, device: max(
                    1,
                    int(
                        occupancy(module, device) * factor,
                    ),
                )
                return stem._backward(x, gamma, beta, bias, dy, 1e-5, False)  # noqa: FBT003

            reference = run(1)
            arms = {}
            for factor in (1, 0.5, 2, 4):
                result = run(factor)
                print(
                    "geometry parity",
                    dtype,
                    spatial,
                    channels,
                    factor,
                    [(a - b).abs().max().item() for a, b in zip(result, reference, strict=True)],
                    flush=True,
                )
                arms[str(factor)] = lambda factor=factor: run(factor)
            print("geometry ms", dtype, spatial, channels, compare(arms), flush=True)
    stem._worker_limit = occupancy


if __name__ == "__main__":
    main()
