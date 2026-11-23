"""Measure the existing NR4 and standalone post-fusion paths."""

# ruff: noqa: T201
import json

import torch
import triton
from isolated import capture, measure

from lnet.gated_post_fusion import GatedStrictComplexPostFusion
from lnet.terminal_imaginary_correlation import _nr4_correlation_kernel


def correlation() -> None:
    batch, height, width, modes = 128, 7, 7, 128
    shape = (batch, height, width, modes)
    states = tuple(torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    pole = tuple(torch.zeros(modes, device="cuda") for _ in range(4))
    for parameter, value in zip(pole, (0.7, 0.05, 0.5, 0.1), strict=True):
        parameter.fill_(value)
    columns = width * modes
    partials = torch.empty((4, batch, columns), device="cuda")
    grad_pole = torch.empty((8, batch, columns), device="cuda")
    upstream = torch.ones((4, batch, modes), device="cuda")
    grad_source = torch.empty((4, *shape), device="cuda", dtype=states[0].dtype)
    rows = triton.next_power_of_2(height)
    block_cols = max(4, min(32, 512 // rows))
    grid = (triton.cdiv(columns, block_cols), batch, 2)
    results = {}
    for backward in (False, True):
        def launch(*, backward: bool = backward) -> None:
            _nr4_correlation_kernel[grid](
                *states, *pole, partials, upstream, grad_source, grad_pole,
                height, width, modes, states[0].stride(), rows, block_cols,
                False, backward, enable_fp_fusion=False,  # noqa: FBT003
            )
        payload = sum(state.nbytes for state in states) + sum(p.nbytes for p in pole)
        payload += grad_source.nbytes + grad_pole.nbytes + upstream.nbytes if backward else (
            partials.nbytes
        )
        time = measure(capture(launch))
        results["backward" if backward else "forward"] = {
            "ms": time, "logical_bytes": payload, "effective_gbps": payload / time / 1e6,
        }
    print(json.dumps({"nr4": results, "shape": shape, "dtype": str(states[0].dtype)}))


def postfusion() -> None:
    modes = 128
    module = GatedStrictComplexPostFusion(modes, modes).cuda()
    x = torch.randn((128, 7, 7, 2 * modes), device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    compiled = torch.compile(module, fullgraph=True)

    def forward() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return compiled(x)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = forward()
    inputs = (x, *module.parameters())
    with torch.cuda.stream(stream):
        _ = torch.autograd.grad(y, inputs, dy, retain_graph=True)
    torch.cuda.current_stream().wait_stream(stream)

    def backward() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(y, inputs, dy, retain_graph=True)

    fwd = measure(capture(forward))
    bwd = measure(capture(backward, stream))
    fwd_bytes = x.nbytes + y.nbytes + sum(p.nbytes for p in module.parameters())
    bwd_bytes = dy.nbytes + sum(t.nbytes for t in inputs)
    print(json.dumps({
        "standalone_postfusion": {
            "forward_ms": fwd, "backward_ms": bwd,
            "forward_io_gbps": fwd_bytes / fwd / 1e6,
            "backward_io_gbps": bwd_bytes / bwd / 1e6,
        },
    }))


if __name__ == "__main__":
    correlation()
    postfusion()
