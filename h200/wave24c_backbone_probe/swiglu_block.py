"""Check the activation candidate inside the real compiled residual block."""

# ruff: noqa: ANN001, ANN201, ANN202, B023, FBT003, T201
import copy

import torch
import triton
from isolated import compare, swiglu
from torch.library import triton_op, wrap_triton

from lnet.memory_backbone import SwiGLU


def launch(x, dy, out, backward):
    half = x.shape[-1] // 2
    count = x.numel() // 2
    wrap_triton(swiglu)[(triton.cdiv(count, 1024),)](
        x,
        dy,
        out,
        count,
        half,
        backward,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton_op("wave24c_probe::activation_backward", mutates_args=())
def backward(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    return launch(x, dy.contiguous(), torch.empty_like(x), True)


@triton_op("wave24c_probe::activation", mutates_args=())
def activation(x: torch.Tensor) -> torch.Tensor:
    return launch(x, x, x.new_empty((*x.shape[:-1], x.shape[-1] // 2)), False)


def setup_context(ctx, inputs, output):  # noqa: ARG001
    ctx.save_for_backward(*inputs)


def differentiate(ctx, dy):
    return backward(*ctx.saved_tensors, dy)


activation.register_autograd(differentiate, setup_context=setup_context)


class ExperimentalSwiGLU(SwiGLU):
    def forward(self, x):
        return x + self.output(activation(self.value_gate(self.norm(x))))


def main():
    for dtype in (torch.bfloat16, torch.float16):
        torch.manual_seed(501)
        ref = SwiGLU(256, 384).cuda()
        test = ExperimentalSwiGLU(256, 384).cuda()
        test.load_state_dict(copy.deepcopy(ref.state_dict()))
        x = torch.randn(6272, 256, device="cuda", dtype=dtype, requires_grad=True)
        dy = torch.randn_like(x)
        modules = {"baseline": ref, "candidate": test}
        compiled = {name: torch.compile(model, fullgraph=True) for name, model in modules.items()}

        def forward(name):
            with torch.autocast("cuda", dtype=dtype):
                return compiled[name](x)

        def step(name):
            result = forward(name)
            return result, *torch.autograd.grad(result, (x, *modules[name].parameters()), dy)

        parity = [
            (a - b).abs().max().item()
            for a, b in zip(step("baseline"), step("candidate"), strict=True)
        ]
        print("swiglu block parity", dtype, parity, flush=True)
        for label, impl in (("fw", forward), ("step", step)):
            print(
                "swiglu block ms",
                dtype,
                label,
                compare({name: lambda name=name: impl(name) for name in modules}),
                flush=True,
            )


if __name__ == "__main__":
    main()
