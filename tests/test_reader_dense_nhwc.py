from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from bench_reader_e2e_micro import SHAPES, Shape, make_reader
from torch._functorch import config as aot_config
from torch.nn import functional
from torch.utils._pytree import tree_leaves
from va_model_builder import build_dw_terminal_rmsnorm

from lnet.factorized_reader_kernels import real_packed_spatial
from lnet.memory_backbone import FactorizedMemoryReader, reader_roofline_costs
from lnet.memory_training import enable_memory_training_


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads() -> Iterator[None]:
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


def reference(reader, x, *, factorized) -> torch.Tensor:
    source = reader.input_norm(x) if reader.input_norm is not None else x
    full, inverse = reader._kernel_and_scale()
    if factorized:
        projected = functional.linear(source, reader.point_weight.transpose(0, 1).flatten(0, 1))
        spatial = reader.spatial_weight.reshape(
            reader.output_modes, 2, reader.rank, reader.kernel_size, reader.kernel_size
        )
        weight = (spatial * inverse).transpose(0, 1).to(projected.dtype).contiguous()
        output = real_packed_spatial(projected, weight, component_first=True)
    else:
        kernel = (full * inverse).transpose(0, 1).flatten(0, 1)
        output = functional.conv2d(source.movedim(-1, 1), kernel, padding=reader.padding)
        output = output.movedim(1, -1).contiguous()
    if reader.match_input_rms:
        epsilon = reader.variance_epsilon / 2
        input_energy = source.float().square().mean(-1, keepdim=True)
        n = reader.rms_reference_modes
        output_energy = output.unflatten(-1, (2, reader.output_modes))[..., :n]
        output_energy = output_energy.float().square().mean((-2, -1)).unsqueeze(-1)
        output = output * ((input_energy + epsilon) / (output_energy + epsilon)).sqrt().to(output)
    return output


def evaluate(reader, x, dy, fn, dtype) -> tuple[torch.Tensor, ...]:
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        output = fn(x)
    gradients = torch.autograd.grad(output, (x, *reader.parameters()), dy)
    return output, *gradients


CASES = [
    *((shape, True, False) for shape in SHAPES[:5]),
    (SHAPES[5], True, False),
    (SHAPES[6], False, False),
    (Shape(2, 3, 7, 7, 5, 3, 5), True, True),
    (Shape(2, 3, 7, 7, 5, 3, 1), False, True),
]


@pytest.mark.parametrize(("shape", "normalize", "match"), CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_output_and_all_gradients(shape, normalize, match, dtype, monkeypatch) -> None:
    torch.manual_seed(168)
    reader = make_reader(shape, torch.device("cpu"), normalize=normalize, match=match)
    x = torch.randn(2, 5, 7, shape.channels).to(dtype).float().requires_grad_()
    dy = torch.randn(2, 5, 7, 2 * shape.modes).to(dtype).float()
    fp32 = evaluate(reader, x, dy, lambda s: reference(reader, s, factorized=False), torch.float32)
    fact = evaluate(reader, x, dy, lambda s: reference(reader, s, factorized=True), dtype)
    results = {}
    for mode in ("off", "dense_nhwc", "dense_nhwc_bmm"):
        monkeypatch.setenv("LNET_READER_EXEC_MODE", mode)
        results[mode] = evaluate(reader, x, dy, reader, dtype)
        assert results[mode][0].is_contiguous()
        for value, dense, factorized in zip(results[mode], fp32, fact, strict=True):
            for expected in (dense, factorized):
                diff = value.float() - expected.float()
                bound = 0.045 if dtype == torch.bfloat16 else 1e-5
                assert torch.isfinite(value).all()
                assert diff.norm() <= bound * expected.float().norm().clamp_min(1e-6)
                assert diff.abs().max() <= bound * expected.abs().max().float().clamp_min(1e-6)

    kernel = reader.synthesized_kernel()
    bmm = reader._synthesized_kernel_bmm()
    assert (kernel - bmm).abs().max() < 1e-6
    kernel_grad = torch.randn_like(kernel)
    weights = reader.point_weight, reader.spatial_weight
    torch.testing.assert_close(
        torch.autograd.grad(kernel, weights, kernel_grad),
        torch.autograd.grad(bmm, weights, kernel_grad),
        atol=1e-6, rtol=1e-6,
    )


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_channels_last_views(monkeypatch, strided, dtype) -> None:
    reader = make_reader(SHAPES[5], torch.device("cpu"), normalize=False)
    x = torch.randn(2, 6, 7, 74 if strided else 37).to(dtype)
    if strided:
        x = x[..., ::2]
    original = functional.conv2d
    pointers = {}

    def check_conv(source, kernel, *, padding):
        assert source.is_contiguous(memory_format=torch.channels_last)
        assert kernel.is_contiguous(memory_format=torch.channels_last)
        assert kernel.dtype == source.dtype == dtype
        assert (source.data_ptr() == x.data_ptr()) == (not strided)
        output = original(source, kernel, padding=padding)
        assert output.is_contiguous(memory_format=torch.channels_last)
        pointers["output"] = output.data_ptr()
        return output

    monkeypatch.setenv("LNET_READER_EXEC_MODE", "dense_nhwc")
    monkeypatch.setattr(functional, "conv2d", check_conv)
    y = reader(x)
    assert y.is_contiguous()
    assert y.data_ptr() == pointers["output"]


def calculate(shape, dtype, sm) -> tuple[float, float]:
    return reader_roofline_costs(
        shape.batch, shape.height, shape.width, shape.channels, shape.modes, shape.rank,
        shape.kernel, dtype, *sm,
    )


@pytest.mark.parametrize("shape", SHAPES[:5])
def test_hopper_selects_dense_for_full_model_shapes(shape) -> None:
    dense, factorized = calculate(shape, torch.bfloat16, (9, 0, 132))
    assert dense < factorized


@pytest.mark.parametrize(
    ("shape", "dtype", "sm", "expected"),
    [
        (SHAPES[0], torch.float16, (7, 0, 80), True),
        (SHAPES[0], torch.bfloat16, (7, 0, 80), False),
        (SHAPES[0], torch.bfloat16, (8, 0, 108), True),
        (SHAPES[0], torch.bfloat16, (8, 9, 128), True),
        (SHAPES[0], torch.float32, (8, 0, 108), False),
        (SHAPES[0], torch.bfloat16, (11, 0, 132), True),
        (SHAPES[-1], torch.bfloat16, (9, 0, 132), False),
        (Shape(8, 32, 32, 512, 160, 1, 11), torch.bfloat16, (9, 0, 132), False),
        (Shape(8, 32, 32, 128, 160, 4, 1), torch.float32, (8, 0, 108), True),
        (Shape(1, 1, 1, 64, 160, 4, 3), torch.bfloat16, (9, 0, 132), False),
    ],
)
def test_roofline_fallbacks(shape, dtype, sm, expected) -> None:
    dense, factorized = calculate(shape, dtype, sm)
    assert (dense < factorized) == expected


@pytest.mark.parametrize(
    ("shape", "expected"), [(SHAPES[0], "dense_nhwc"), (SHAPES[-1], "factorized_triton")]
)
def test_auto_roofline_dispatch_uses_runtime_properties(shape, expected, monkeypatch) -> None:
    reader = make_reader(shape, torch.device("cpu"))
    reader.execution_mode = "dense"
    source = SimpleNamespace(
        is_cuda=True, device=torch.device("cuda:1"), dtype=torch.bfloat16,
        shape=(shape.batch, shape.height, shape.width, shape.channels),
    )

    def properties(index):
        assert index == 1
        return SimpleNamespace(major=9, minor=0, multi_processor_count=132)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setenv("LNET_READER_EXEC_MODE", "auto_roofline")
    assert reader._reader_execution_mode(source) == expected

    def dispatch(value):
        return value + (1 if reader._reader_execution_mode(source) == "dense_nhwc" else 2)

    compiled_dispatch = torch.compile(dispatch, backend="aot_eager", fullgraph=True)
    assert compiled_dispatch(torch.zeros(())) == (1 if expected == "dense_nhwc" else 2)
    monkeypatch.setenv("LNET_READER_EXEC_MODE", "off")
    assert reader._reader_execution_mode(source) == "dense"


@pytest.mark.parametrize("mode", ["off", "dense_nhwc", "dense_nhwc_bmm", "auto_roofline"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reader_fullgraph(mode, dtype, monkeypatch) -> None:
    monkeypatch.setenv("LNET_READER_EXEC_MODE", mode)
    torch.manual_seed(168)
    reader = make_reader(SHAPES[5], torch.device("cpu"))
    x = torch.randn(2, 5, 7, 37, requires_grad=True)
    dy = torch.randn(2, 5, 7, 58)
    compiled = torch.compile(reader, backend="aot_eager", fullgraph=True)
    eager = evaluate(reader, x, dy, reader, dtype)
    with aot_config.patch("backward_pass_autocast", "off"):
        actual = evaluate(reader, x, dy, compiled, dtype)
    torch.testing.assert_close(actual, eager)


def test_default_and_off_are_bit_identical(monkeypatch) -> None:
    reader = make_reader(SHAPES[5], torch.device("cpu"))
    x = torch.randn(2, 5, 7, 37, requires_grad=True)
    dy = torch.randn(2, 5, 7, 58)
    monkeypatch.delenv("LNET_READER_EXEC_MODE", raising=False)
    default = evaluate(reader, x, dy, reader, torch.float32)
    for mode in ("", "off"):
        monkeypatch.setenv("LNET_READER_EXEC_MODE", mode)
        actual = evaluate(reader, x, dy, reader, torch.float32)
        torch.testing.assert_close(actual, default, rtol=0, atol=0)


def test_ablation17_full_model_aot_eager(monkeypatch) -> None:
    torch.manual_seed(168)
    model = build_dw_terminal_rmsnorm(factorized_reader=True).train()
    parameters = tuple(model.parameters())
    before = {key: value.shape for key, value in model.state_dict().items()}
    readers = [module for module in model.modules() if isinstance(module, FactorizedMemoryReader)]
    assert len(parameters) == 193
    assert len(before) == 197  # Includes four persistent buffers.
    buffers = [(buffer, buffer.clone()) for buffer in model.buffers()]
    assert sum(p.numel() for p in parameters) == 2_988_260
    assert len(readers) == 12
    x = torch.randn(1, 3, 224, 224)
    reference = None
    for mode in ("off", "dense_nhwc", "dense_nhwc_bmm"):
        monkeypatch.setenv("LNET_READER_EXEC_MODE", mode)
        assert enable_memory_training_(model) == {"readers": len(readers)}
        assert all(reader.execution_mode == "factorized_triton" for reader in readers)
        assert {key: value.shape for key, value in model.state_dict().items()} == before
        for buffer, initial in buffers:
            buffer.copy_(initial)
        compiled = torch.compile(model, backend="aot_eager")
        logits = tuple(leaf for leaf in tree_leaves(compiled(x)) if isinstance(leaf, torch.Tensor))
        loss = logits[0].square().mean()
        gradients = torch.autograd.grad(loss, parameters)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        result = tuple(t.detach() for t in (*logits, *gradients))
        if reference is None:
            reference = result
        else:
            torch.testing.assert_close(result, reference, rtol=2e-3, atol=2e-5)
