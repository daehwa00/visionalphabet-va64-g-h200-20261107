from __future__ import annotations

import run_alphabet_ablations as runner
import torch
from alphabet_model_builder import model_config
from torch.nn import functional

from alphabet.depthwise_memory_reader import DepthwiseMemoryReader
from alphabet.memory_backbone import FactorizedMemoryReader


def test_model_inventory_and_preserved_initialization() -> None:
    torch.manual_seed(501)
    first = runner._build(runner.VARIANT, model_config())
    torch.manual_seed(501)
    new = runner._build(runner.VARIANT, model_config())
    readers = [
        (name, m) for name, m in new.named_modules() if isinstance(m, DepthwiseMemoryReader)
    ]
    assert len(readers) == 12
    assert not any(isinstance(m, FactorizedMemoryReader) for m in new.modules())
    first_state = first.state_dict()
    newstate = new.state_dict()
    terminal_norm_key = "backbone.terminal.pole_input_projection.input_norm.weight"
    assert terminal_norm_key in newstate
    for name, value in newstate.items():
        torch.testing.assert_close(value, first_state[name], rtol=0, atol=0)
    torch.testing.assert_close(
        newstate[terminal_norm_key],
        newstate["backbone.same_resolution_blocks.7.reader.input_norm.weight"],
        rtol=0,
        atol=0,
    )
    assert sum(p.numel() for p in new.parameters()) == 2_389_796


def test_effective_dense_kernel_values_and_gradients() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.VARIANT, model_config())
    reader = model.backbone.same_resolution_blocks["56"].reader
    x = torch.randn(2, 5, 6, reader.input_channels, requires_grad=True)
    actual = reader(x)
    source = reader.input_norm(x)
    kernel = reader.point_weight[:, :, None, None] * reader.depthwise_weight[:, 0][None]
    kernel = (
        kernel
        * kernel.square().sum((1, 2, 3), keepdim=True).clamp_min(reader.variance_epsilon).rsqrt()
    )
    expected = functional.conv2d(source.movedim(-1, 1), kernel, padding=1).movedim(1, -1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    params = (x, reader.point_weight, reader.depthwise_weight)
    a = torch.autograd.grad(actual.square().mean(), params, retain_graph=True)
    b = torch.autograd.grad(expected.square().mean(), params)
    for aa, bb in zip(a, b, strict=True):
        torch.testing.assert_close(aa, bb, atol=2e-6, rtol=2e-5)


def test_terminal_input_norm_and_output_layout() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.VARIANT, model_config())
    r = model.backbone.terminal.pole_input_projection
    stage4_norm = model.backbone.same_resolution_blocks["7"].reader.input_norm
    assert r.input_norm is not None
    assert stage4_norm is not None
    assert r.input_norm is not stage4_norm
    assert not r.match_input_rms
    assert r.input_norm.epsilon == stage4_norm.epsilon
    torch.testing.assert_close(r.input_norm.weight, stage4_norm.weight, rtol=0, atol=0)
    x = torch.randn(2, 7, 7, r.input_channels, requires_grad=True)
    y = r(x)
    scaled = r(x * 3)
    assert y.shape == (2, 7, 7, 2 * r.output_modes)
    torch.testing.assert_close(y, scaled, rtol=2e-5, atol=2e-5)
    y.square().mean().backward()
    assert r.input_norm.weight.grad is not None
    assert torch.isfinite(r.input_norm.weight.grad).all()
    assert r.input_norm.weight.grad.abs().sum() > 0
