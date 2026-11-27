"""Check packed read routing and training numerics at both scan call sites."""

import copy

import pytest
import torch
from alphabet_model_builder import build_dw_terminal_rmsnorm
from torch._inductor import config as inductor_config

from alphabet.complex_layers import ComplexLinear
from alphabet.memory_backbone import MemoryReadProjection


class _CustomProjection(MemoryReadProjection):
    def forward_planes(self, first, second):
        return super().forward_planes(first, second) + 1


def test_packed_projection_matches_planar_gemms(monkeypatch) -> None:
    projection = MemoryReadProjection(ComplexLinear(8, 4))
    generator = torch.Generator().manual_seed(19)
    source = torch.randn((2, 3, 4, 16), generator=generator, requires_grad=True)
    linear = torch.nn.functional.linear
    widths = []

    def record_linear(inputs, weight):
        widths.append(weight.shape[-1])
        return linear(inputs, weight)

    monkeypatch.setattr(torch.nn.functional, "linear", record_linear)
    packed = projection(source)
    planar = projection(*source.split(8, dim=-1))
    assert widths == [16, 8, 8]
    torch.testing.assert_close(packed, planar)
    gradient = torch.randn(packed.shape, generator=generator)
    params = (source, projection.weight)
    packed_grad = torch.autograd.grad(packed, params, gradient)
    planar_grad = torch.autograd.grad(planar, params, gradient)
    for actual, expected in zip(packed_grad, planar_grad, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.fixture(scope="module")
def topology() -> torch.nn.Module:
    return build_dw_terminal_rmsnorm(factorized_reader=True).backbone


@pytest.mark.parametrize("coarse", [False, True], ids=["same-resolution", "coarse"])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required",
    ))],
)
def test_planar_projection_fullgraph_training(topology, device, coarse, monkeypatch) -> None:
    torch.compiler.reset()
    monkeypatch.delenv("ALPHABET_GENERAL_SCAN_REF_FWD", raising=False)
    monkeypatch.setenv("ALPHABET_FUSED_FACTORIZED_READER", "1")
    if coarse:
        module = copy.deepcopy(topology.stage1).to(device)
        projection = module.augmented.memory_projection
        channels = module.pole_input_projection.input_channels
    else:
        module = copy.deepcopy(topology.same_resolution_blocks["56"]).to(device)
        projection = module.memory_projection
        channels = module.reader.input_channels
    assert type(projection) is MemoryReadProjection

    generator = torch.Generator().manual_seed(1901)
    height, width = (14, 14) if device == "cuda" else (4, 6)
    shape = (2, height, width, channels)
    inputs = torch.randn(shape, generator=generator).to(device).requires_grad_()

    def forward(source):
        if coarse:
            return module(source, emit_descriptor=False)[0]
        return module(source)

    def run(runtime):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            result = runtime(inputs)
        gradient = torch.randn(result.shape, generator=generator).to(result)
        derivatives = torch.autograd.grad(result, (inputs, projection.weight), gradient)
        return result, derivatives

    calls = []

    def observe(_module, args, kwargs):
        calls.append((len(args), kwargs.get("planar", False), args[0].is_contiguous()))

    with inductor_config.patch(emulate_precision_casts=False):
        hook = projection.register_forward_pre_hook(observe, with_kwargs=True)
        monkeypatch.setenv("ALPHABET_GENERAL_SCANS", "0")
        seed_state = generator.get_state()
        expected, expected_grad = run(forward)
        monkeypatch.setenv("ALPHABET_GENERAL_SCANS", "1")
        generator.set_state(seed_state)
        eager, eager_grad = run(forward)
        hook.remove()
        assert calls == [(2, False, True), (2, False, True)]
        compiled = torch.compile(
            forward, fullgraph=True, backend="inductor" if device == "cuda" else "aot_eager",
        )
        generator.set_state(seed_state)
        actual, actual_grad = run(compiled)

    _assert_training_close((actual, *actual_grad), (expected, *expected_grad), device, 0.04)
    _assert_training_close((actual, *actual_grad), (eager, *eager_grad), device, 0.025)


def _assert_training_close(actual, expected, device, bf16_tolerance) -> None:
    for value, reference in zip(actual, expected, strict=True):
        if device == "cuda":
            # Compiled and eager BF16 scan accumulation can round differently.
            relative_error = (value.float() - reference.float()).norm() / reference.float().norm()
            assert relative_error.item() < bf16_tolerance
        else:
            torch.testing.assert_close(value, reference, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("coarse", [False, True], ids=["same-resolution", "coarse"])
@pytest.mark.parametrize("projection_type", [ComplexLinear, _CustomProjection])
def test_other_projections_keep_planar_call(topology, coarse, projection_type, monkeypatch) -> None:
    if coarse:
        module = copy.deepcopy(topology.stage1)
        owner = module.augmented
        reader = module.pole_input_projection
    else:
        module = copy.deepcopy(topology.same_resolution_blocks["56"])
        owner = module
        reader = module.reader
    output_channels, input_channels = owner.memory_projection.weight.shape
    legacy = ComplexLinear(input_channels // 2, output_channels // 2)
    owner.memory_projection = (
        legacy if projection_type is ComplexLinear else projection_type(legacy)
    )
    calls = []
    hook = owner.memory_projection.register_forward_pre_hook(
        lambda _, args: calls.append(len(args)),
    )
    inputs = torch.randn(1, 4, 6, reader.input_channels)
    kwargs = {"emit_descriptor": False} if coarse else {}
    with torch.no_grad(), inductor_config.patch(emulate_precision_casts=False):
        monkeypatch.delenv("ALPHABET_GENERAL_SCAN_REF_FWD", raising=False)
        monkeypatch.setenv("ALPHABET_GENERAL_SCANS", "0")
        expected = module(inputs, **kwargs)
        monkeypatch.setenv("ALPHABET_GENERAL_SCANS", "1")
        actual = module(inputs, **kwargs)
    hook.remove()
    assert calls == [2, 2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
