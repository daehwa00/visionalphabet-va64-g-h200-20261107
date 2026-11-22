"""Compare fused and decoupled vertical scans in isolation."""

import argparse
import json
import os
from pathlib import Path

import torch

from lnet import general_cuda_scans as scans

CASES = (
    (1024, 56, 56, 160, False),
    (1024, 56, 56, 160, True),
    (1024, 28, 28, 160, False),
    (1024, 28, 28, 160, True),
    (1024, 14, 14, 160, False),
    (1024, 7, 7, 160, False),
    (4, 128, 192, 160, False),
    (4, 512, 768, 48, False),
)


def probe(shape: tuple, hidden: int) -> dict:
    b, h, w, m, coarse = shape
    dtype = torch.bfloat16
    oh, ow = (h // 2, w // 2) if coarse else (h, w)

    def sample(size: tuple, storage: torch.dtype, scale: float) -> torch.Tensor:
        return torch.randn(size, device="cuda", dtype=storage).mul_(scale)

    torch.manual_seed(731)
    states = sample((4, b, h, w, m), dtype, 0.4)
    pole = [sample((m,), torch.float32, 0.12) for _ in range(4)]
    auxiliary = [torch.ones((2, length, m), device="cuda") for length in (w, h)]
    path = [
        sample(size, torch.float32, 0.17)
        for size in ((m, hidden, 8), (m, hidden), (m, 2, hidden), (m, 2))
    ]
    dy = [sample((b, oh, ow, m), dtype, 0.1) for _ in range(2)]

    def step() -> tuple:
        out = scans.vertical_forward(pole, states, auxiliary, path, coarse)
        grads = scans.vertical_backward(pole, states, auxiliary, path, *dy, coarse)
        return (*out, *grads)

    result = {"shape": shape, "hidden": hidden}
    for label, policy in (("fused", "0"), ("decoupled", "1")):
        os.environ["LNET_VSCAN_DECOUPLED"] = policy
        values = step()
        if label == "fused":
            # Keep small samples for parity so measured allocation excludes baseline outputs.
            expected = [value.flatten()[::max(1, value.numel() // 8192)].float()
                        for value in values]
        else:
            samples = [value.flatten()[::max(1, value.numel() // 8192)].float()
                       for value in values]
            result["parity"] = [
                {"max_abs": (a - e).abs().max().item(), "max_reference": e.abs().max().item()}
                for a, e in zip(samples, expected, strict=True)
            ]
        del values
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            step()
        stop.record()
        stop.synchronize()
        result[label] = {
            "ms": start.elapsed_time(stop) / 5,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "chunk": scans._decoupled_chunk(states, hidden, coarse),  # noqa: SLF001
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for shape in CASES:
        row = probe(shape, 8)
        with args.out.open("a", encoding="utf-8") as file:
            print(json.dumps(row), file=file)
        print(json.dumps(row), flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
