"""Screen NR4 tile geometry while preserving reduction order."""

# ruff: noqa: ANN001,ANN201,ANN202,B023,FBT003,SLF001,T201
import torch
import triton
from isolated import compare

from lnet import terminal_imaginary_correlation as nr4


def launch(tiles, warps):
    def tuned(states, pole, partials, upstream, grad_source, grad_pole, *, normalize, backward):
        batch, height, width, modes = states[0].shape
        rows = triton.next_power_of_2(height)
        cols = max(4, min(tiles, 512 // rows)) if tiles == 32 else tiles
        grid = (triton.cdiv(width * modes, cols), batch, 2)
        nr4.wrap_triton(nr4._nr4_correlation_kernel)[grid](
            *states,
            *(p.contiguous() for p in pole),
            partials,
            upstream,
            grad_source,
            grad_pole,
            height,
            width,
            modes,
            states[0].stride(),
            rows,
            cols,
            normalize,
            backward,
            enable_fp_fusion=False,
            num_warps=warps,
        )

    return tuned


def main():
    torch.manual_seed(501)
    baseline = nr4._nr4_launch
    choices = {
        "base": baseline,
        "16x4": launch(16, 4),
        "32x4": launch(32, 4),
        "64x4": launch(64, 4),
        "32x8": launch(32, 8),
        "16x2": launch(16, 2),
    }
    for dtype in (torch.bfloat16, torch.float16):
        for height, width, modes in ((7, 7, 128), (7, 7, 127)):
            states = [
                torch.randn(128, height, width, modes, dtype=dtype, device="cuda") for _ in range(4)
            ]
            pole = [0.2 * torch.randn(modes, device="cuda") for _ in range(4)]
            ref = nr4._nr4_correlation(states, pole, False)
            dy = torch.randn_like(ref)
            ref_grad = nr4._nr4_correlation_backward(states, pole, dy)

            def run(impl):
                nr4._nr4_launch = impl
                return nr4._nr4_correlation(states, pole, False), *nr4._nr4_correlation_backward(
                    states,
                    pole,
                    dy,
                )

            arms = {}
            for name, impl in choices.items():
                try:
                    result = run(impl)
                    errors = [
                        (a - b).abs().max().item()
                        for a, b in zip(result, (ref, *ref_grad), strict=True)
                    ]
                    print("nr4 parity", dtype, height, width, modes, name, errors, flush=True)
                    arms[name] = lambda impl=impl: run(impl)
                except RuntimeError as exc:
                    print(name, exc, flush=True)
            print("nr4 ms", dtype, height, width, modes, compare(arms), flush=True)
            nr4._nr4_launch = baseline


if __name__ == "__main__":
    main()
