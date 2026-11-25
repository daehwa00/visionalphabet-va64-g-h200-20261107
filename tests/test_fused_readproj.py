"""Compatibility and first-order training checks for the read projection."""

import pytest
import torch
from run_va_eight_ablations import build_variant
from torch._dynamo import lookup_backend

from lnet.complex_layers import ComplexLinear
from lnet.fused_readproj_ops import MODES, readproj
from lnet.memory_backbone import MemoryReadProjection


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_readproj_cpu(mode: str, dtype: torch.dtype, bias: bool) -> None:
    rng = torch.Generator().manual_seed(15)
    shapes = ((3, 7, 13), (3, 7, 19), (11, 13), (11, 19))
    if bias:
        shapes += ((11,),)
    fp32 = tuple(torch.randn(shape, generator=rng) for shape in shapes)
    inputs = tuple(t.to(dtype).requires_grad_() for t in fp32)
    g = torch.randn((3, 7, 11), generator=rng).to(dtype)

    def run(args, grad, selected):
        x, y, w, v, *optional_bias = args
        b = optional_bias[0] if optional_bias else None
        out = readproj(x, y, w, v, b, selected)
        return (out, *torch.autograd.grad(out, args, grad))

    baseline = run(inputs, g, "off")
    reference = run(tuple(t.detach().float().requires_grad_() for t in inputs), g.float(), "off")
    values = run(inputs, g, mode)
    for actual, expected, fp32_reference in zip(values, baseline, reference, strict=True):
        assert actual.isfinite().all()
        eps = torch.finfo(dtype).eps
        torch.testing.assert_close(actual, expected, rtol=4 * eps, atol=12 * eps)
        torch.testing.assert_close(actual.float(), fp32_reference, rtol=4 * eps, atol=12 * eps)


@pytest.mark.parametrize("mode", MODES)
def test_projection_checkpoint_and_autocast(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = MemoryReadProjection(ComplexLinear(13, 7))
    actual = MemoryReadProjection(ComplexLinear(13, 7))
    state = baseline.state_dict()
    assert {key: tensor.shape for key, tensor in state.items()} == {"weight": (14, 26)}
    actual.load_state_dict(state, strict=True)
    rng = torch.Generator().manual_seed(46)
    args = tuple(torch.randn((3, 5, 13), generator=rng).requires_grad_() for _ in range(2))
    monkeypatch.setenv("LNET_READPROJ_OPT", "off")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reference = baseline(*args)
    monkeypatch.setenv("LNET_READPROJ_OPT", mode)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = actual(*args)
    torch.testing.assert_close(
        result,
        reference,
        atol=0.03125,
        rtol=0.008,
    )
    for left, right in zip(
        torch.autograd.grad(result.sum(), (*args, actual.weight)),
        torch.autograd.grad(reference.sum(), (*args, baseline.weight)),
        strict=True,
    ):
        torch.testing.assert_close(left, right, atol=0.03125, rtol=0.008)


def test_full_model_aot_eager_readproj(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = build_variant("17-p1-nr4-1024d-v3")
    model = build_variant("17-p1-nr4-1024d-v3")
    model.load_state_dict(reference.state_dict(), strict=True)
    sample = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(46))
    monkeypatch.setenv("LNET_READPROJ_OPT", "off")
    eager = reference(sample)[0]
    eager.square().mean().backward()
    monkeypatch.setenv("LNET_READPROJ_OPT", "dual_all")
    forward_ops = []

    def aot_eager(graph, example_inputs):
        forward_ops.extend(
            node
            for node in graph.graph.nodes
            if node.target is torch.ops.lnet.readproj_forward.default
        )
        return lookup_backend("aot_eager")(graph, example_inputs)

    compiled = torch.compile(model, backend=aot_eager, fullgraph=True)
    actual = compiled(sample)[0]
    actual.square().mean().backward()
    assert len(forward_ops) == 10
    torch.testing.assert_close(actual, eager)
    grads = [
        (p.grad, q.grad)
        for p, q in zip(model.parameters(), reference.parameters(), strict=True)
        if p.grad is not None or q.grad is not None
    ]
    assert len(grads) == 193
    for got, want in grads:
        assert got is not None
        assert want is not None
        assert got.isfinite().all()
        assert want.isfinite().all()
        torch.testing.assert_close(got, want, rtol=3e-4, atol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_compiled_and_captured_training() -> None:
    rng = torch.Generator(device="cuda").manual_seed(161)
    shapes = ((1057, 17), (1057, 33), (31, 17), (31, 33), (31,))
    args = tuple(
        torch.randn(shape, generator=rng, device="cuda", dtype=torch.float16).requires_grad_()
        for shape in shapes
    )
    g = torch.randn((1057, 31), generator=rng, device="cuda", dtype=torch.float16)

    def forward(x, y, w, v, bias):
        return readproj(x, y, w, v, bias, "dual_all")

    compiled = torch.compile(forward, fullgraph=True)
    reference = readproj(*args, "off")
    expected = (reference, *torch.autograd.grad(reference, args, g))

    def train():
        result = compiled(*args)
        return (result, *torch.autograd.grad(result, args, g))

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        train()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values = train()
    graph.replay()
    for got, want in zip(values, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0.005, atol=0.032)
