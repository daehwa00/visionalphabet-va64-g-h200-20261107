"""Screen NR4 launch geometries while holding the recurrence and buffers fixed."""

# ruff: noqa: T201
import json

import torch
import triton
from isolated import capture, measure

from lnet.terminal_imaginary_correlation import _nr4_correlation_kernel


def sweep() -> None:
    batch, height, width, modes = 128, 7, 7, 128
    shape = batch, height, width, modes
    states = tuple(torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    pole = tuple(torch.full((modes,), v, device="cuda") for v in (0.7, 0.05, 0.5, 0.1))
    columns = width * modes
    partials = torch.empty((4, batch, columns), device="cuda")
    grad_pole = torch.empty((8, batch, columns), device="cuda")
    upstream = torch.randn((4, batch, modes), device="cuda")
    grad_source = torch.empty((4, *shape), device="cuda", dtype=states[0].dtype)
    rows = triton.next_power_of_2(height)
    arms = ((32, 4), (16, 2), (16, 4), (32, 2), (64, 4), (64, 8))
    for backward in (False, True):
        reference = None
        for block_cols, warps in arms:
            grid = (triton.cdiv(columns, block_cols), batch, 2)

            def launch(
                *, grid: tuple[int, int, int] = grid, block_cols: int = block_cols,
                backward: bool = backward, warps: int = warps,
            ) -> None:
                _nr4_correlation_kernel[grid](
                    *states, *pole, partials, upstream, grad_source, grad_pole,
                    height, width, modes, states[0].stride(), rows, block_cols,
                    False, backward, enable_fp_fusion=False, num_warps=warps,  # noqa: FBT003
                )

            launch()
            outputs = (grad_source, grad_pole) if backward else (partials,)
            if reference is None:
                reference = tuple(t.clone() for t in outputs)
            exact = all(torch.equal(a, b) for a, b in zip(reference, outputs, strict=True))
            print(json.dumps({
                "backward": backward, "cols": block_cols, "warps": warps,
                "exact": exact, "graph_ms": measure(capture(launch)),
            }), flush=True)


if __name__ == "__main__":
    torch.manual_seed(77)
    sweep()
