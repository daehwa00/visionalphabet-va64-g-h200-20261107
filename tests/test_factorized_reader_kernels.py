"""Value, gradient and compilation coverage for the packed factorized reader."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from lnet import factorized_reader_kernels as kernels
from lnet.factorized_complex_scan_reader import FactorizedComplexConv2dReader

if TYPE_CHECKING:
    from collections.abc import Iterator

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("rank", "kernel", "modes"), list(product((1, 2, 4), (1, 3, 5), (48, 160)))
)
def test_gram_energy_and_gradients(rank: int, kernel: int, modes: int) -> None:
    torch.manual_seed(82)
    reader = FactorizedComplexConv2dReader(19, modes, rank=rank, kernel_size=kernel)
    with torch.no_grad():
        for parameter in reader.parameters():
            parameter.normal_(std=0.1)
    parameters = tuple(reader.parameters())
    energy = kernels.gram_kernel_energy(*parameters)
    real, imag = reader.synthesized_kernel()
    expected = (real.square() + imag.square()).sum((1, 2, 3))
    torch.testing.assert_close(energy, expected, rtol=0, atol=9.9e-7)
    probe = torch.linspace(-1, 1, modes)
    gradients = torch.autograd.grad(energy, parameters, probe)
    expected_gradients = torch.autograd.grad(expected, parameters, probe)
    for actual, reference in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=9.9e-7)


def _reader_pair(
    in_modes: int,
    modes: int,
    rank: int,
    kernel: int,
    *,
    rms: bool,
    device: str,
) -> tuple[FactorizedComplexConv2dReader, FactorizedComplexConv2dReader]:
    reader = FactorizedComplexConv2dReader(
        in_modes,
        modes,
        rank=rank,
        kernel_size=kernel,
        normalize_input=rms,
        match_input_rms=rms,
        rms_reference_modes=max(1, modes // 2),
        execution_mode="factorized_triton",
    ).to(device)
    with torch.no_grad():
        for parameter in reader.parameters():
            parameter.normal_(mean=0.2, std=0.15)
    reference = deepcopy(reader)
    reference.execution_mode = "factorized"
    return reader, reference


def _check_gradients(output, reference, inputs, reference_inputs, *, rtol, atol) -> None:
    probe = torch.randn_like(output, dtype=torch.float32) / output.shape[-1]
    gradients = torch.autograd.grad(output.float(), inputs, probe)
    expected_gradients = torch.autograd.grad(reference.float(), reference_inputs, probe)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("shape", "rank", "kernel", "modes"),
    [((1, 1, 1), 1, 1, 1), ((2, 3, 5), 2, 3, 48), ((2, 5, 1), 4, 5, 5)],
)
@pytest.mark.parametrize("rms", [False, True])
def test_cpu_reader_apis(shape, rank: int, kernel: int, modes: int, *, rms: bool) -> None:
    torch.manual_seed(17)
    reader, reference = _reader_pair(3, modes, rank, kernel, rms=rms, device="cpu")
    x = torch.randn((*shape, 6), requires_grad=True)
    y = x.detach().clone().requires_grad_()
    output = reader(x)
    expected = reference.forward_packed(y)
    assert output.is_contiguous()
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
    separate = reader(*x.chunk(2, dim=-1))
    torch.testing.assert_close(torch.cat(separate, dim=-1), output, rtol=2e-5, atol=2e-6)
    _check_gradients(
        output,
        expected,
        (x, *reader.parameters()),
        (y, *reference.parameters()),
        rtol=3e-4,
        atol=3e-6,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_reduced_precision(dtype: torch.dtype) -> None:
    reader, _ = _reader_pair(3, 7, 2, 3, rms=True, device="cpu")
    x = torch.randn(2, 3, 5, 6, dtype=dtype, requires_grad=True)
    output = reader(x)
    assert output.dtype == dtype
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(
        t.grad is not None and torch.isfinite(t.grad).all() for t in (x, *reader.parameters())
    )


def test_fake_custom_op_and_autograd_metadata() -> None:
    with FakeTensorMode():
        projected = torch.empty(2, 3, 7, 2 * 2 * 5, dtype=torch.float16, requires_grad=True)
        weights = tuple(torch.empty(5, 2, 3, 3, requires_grad=True) for _ in range(2))
        output = kernels._packed_spatial(projected, *weights)
        assert output.shape == (2, 3, 7, 10)
        assert output.is_contiguous()
        assert output.dtype == projected.dtype
        gradients = torch.autograd.grad(output, (projected, *weights), torch.empty_like(output))
        for gradient, original in zip(gradients, (projected, *weights), strict=True):
            assert gradient.shape == original.shape
            assert gradient.dtype == original.dtype
            assert gradient.is_contiguous()


def test_compile_cpu_dynamic_fullgraph_forward_and_backward() -> None:
    torch.manual_seed(37)
    reader, reference = _reader_pair(3, 5, 2, 3, rms=True, device="cpu")
    compiled = torch.compile(reader, fullgraph=True, dynamic=True)
    for height, width in ((3, 7), (5, 9)):
        x = torch.randn(2, height, width, 6, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        output = compiled(x)
        expected = reference(y)
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
        _check_gradients(
            output,
            expected,
            (x, *reader.parameters()),
            (y, *reference.parameters()),
            rtol=3e-4,
            atol=3e-6,
        )


@pytest.fixture
def cudnn_full_precision() -> Iterator[None]:
    with torch.backends.cudnn.flags(allow_tf32=False):
        yield


@requires_cuda
@pytest.mark.usefixtures("cudnn_full_precision")
@pytest.mark.parametrize(
    ("shape", "modes", "rank", "kernel", "dtype"),
    [
        ((1, 1, 1), 1, 1, 1, torch.float32),
        ((2, 1, 7), 9, 3, 5, torch.float16),
        ((2, 7, 1), 48, 2, 3, torch.bfloat16),
        ((2, 9, 11), 80, 4, 7, torch.float32),
        ((1, 17, 25), 96, 1, 5, torch.bfloat16),
        ((1, 128, 192), 128, 2, 3, torch.float16),
        ((1, 200, 334), 160, 2, 3, torch.bfloat16),
    ],
)
def test_cuda_spatial_matches_fp32_reference(shape, modes, rank, kernel, dtype) -> None:
    torch.manual_seed(123)
    projected = torch.randn(
        (*shape, 2 * rank * modes), device="cuda", dtype=dtype, requires_grad=True
    )
    spatial = tuple(
        (torch.randn(modes, rank, kernel, kernel, device="cuda") * 0.15).requires_grad_()
        for _ in range(2)
    )
    weights = tuple(weight.detach().clone().requires_grad_() for weight in spatial)
    ref_projected = projected.detach().clone().requires_grad_()
    output = kernels.packed_spatial(projected, *spatial)
    expected = kernels._spatial_reference(ref_projected, *weights)
    value_atol, value_rtol = {
        torch.float32: (8e-6, 3e-5),
        torch.float16: (4e-5, 1e-3),
        torch.bfloat16: (5e-5, 8e-3),
    }[dtype]
    assert output.is_contiguous()
    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, rtol=value_rtol, atol=value_atol)
    _check_gradients(
        output,
        expected,
        (projected, *spatial),
        (ref_projected, *weights),
        rtol=max(value_rtol, 1e-3),
        atol=value_atol,
    )


@requires_cuda
@pytest.mark.usefixtures("cudnn_full_precision")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_reader_fullgraph_autocast_and_gradients(dtype: torch.dtype) -> None:
    torch.manual_seed(72)
    reader, reference = _reader_pair(13, 48, 2, 3, rms=True, device="cuda")
    compiled = torch.compile(reader, fullgraph=True, dynamic=True)
    for height, width in ((7, 9), (5, 11)):
        x = torch.randn(2, height, width, 26, device="cuda", requires_grad=True)
        y = x.detach().clone().requires_grad_()
        with torch.autocast(
            "cuda",
            enabled=dtype != torch.float32,
            dtype=dtype if dtype != torch.float32 else torch.bfloat16,
        ):
            output = compiled(x)
        expected = reference(y)
        atol, rtol = {
            torch.float32: (2e-5, 3e-4),
            torch.float16: (2e-3, 1e-2),
            torch.bfloat16: (1.6e-2, 5e-2),
        }[dtype]
        assert output.dtype == dtype
        torch.testing.assert_close(output.float(), expected, rtol=rtol, atol=atol)
        _check_gradients(
            output,
            expected,
            (x, *reader.parameters()),
            (y, *reference.parameters()),
            rtol=max(rtol, 1e-3),
            atol=atol,
        )


@requires_cuda
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "inductor"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [True, False], ids=["default", "opt-out"])
def test_free_real_memory_reader_matches_grouped_cuda(
    compiled: bool, dtype: torch.dtype, fused: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lnet.memory_backbone import FactorizedMemoryReader  # noqa: PLC0415

    torch.manual_seed(972)
    legacy = FactorizedComplexConv2dReader(
        13, 48, rank=2, kernel_size=3, execution_mode="factorized_triton",
    )
    reader = FactorizedMemoryReader(legacy).cuda()
    reference = deepcopy(reader)
    reference.execution_mode = "factorized"
    with torch.no_grad():
        # Break the input factors' original complex symmetry.
        reader.point_weight.normal_(std=0.2)
        reader.spatial_weight.normal_(std=0.15)
    reference.load_state_dict(reader.state_dict())
    if fused:
        monkeypatch.delenv("LNET_FUSED_FACTORIZED_READER", raising=False)
    else:
        monkeypatch.setenv("LNET_FUSED_FACTORIZED_READER", "0")
    run = torch.compile(reader, fullgraph=True) if compiled else reader
    x = torch.randn(2, 7, 11, 26, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()

    def unexpected_synthesis() -> None:
        pytest.fail("Fused real Reader called synthesized_kernel")

    def unexpected_fused_spatial(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Opted-out Reader used fused spatial convolution")

    if fused:
        monkeypatch.setattr(reader, "synthesized_kernel", unexpected_synthesis)
    else:
        monkeypatch.setattr("lnet.memory_backbone.real_packed_spatial", unexpected_fused_spatial)
    autocast_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=dtype != torch.float32):
        expected = reference(y)
        output = run(x)
    atol, rtol = {
        torch.float32: (5e-6, 3e-5),
        torch.float16: (8e-4, 5e-3),
        torch.bfloat16: (4e-3, 3e-2),
    }[dtype]
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    _check_gradients(
        output, expected, (x, *reader.parameters()), (y, *reference.parameters()),
        rtol=max(rtol, 5e-5), atol=max(atol, 1e-5),
    )


@requires_cuda
def test_real_packed_spatial_dense_rectangle() -> None:
    torch.manual_seed(112)
    p = torch.randn(4, 512, 768, 17, device="cuda", dtype=torch.bfloat16).requires_grad_()
    w = (torch.randn(17, 2, 1, 3, 3, device="cuda", dtype=p.dtype) * 0.15).requires_grad_()
    out = kernels.real_packed_spatial(p, w)
    ref = kernels._spatial_reference(p, *w.unbind(1), strict_complex=False)
    torch.testing.assert_close(out, ref, rtol=8e-3, atol=2e-4)
    _check_gradients(out, ref, (p, w), (p, w), rtol=8e-3, atol=1e-2)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize(
    ("modes", "channels", "rank", "kernel", "packed"),
    [(17, 5, 1, 1, True), (80, 19, 2, 3, False), (128, 64, 4, 5, True), (160, 7, 4, 3, False)],
)
def test_real_gram_energy(device, modes, channels, rank, kernel, packed) -> None:
    torch.manual_seed(716)
    point = (torch.randn(modes, rank, channels, device=device) * 0.07).requires_grad_()
    spatial = torch.randn(modes, 2, rank, kernel, kernel, device=device) * 0.07
    spatial = (spatial.flatten(0, 1) if packed else spatial).requires_grad_()
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = kernels.real_gram_kernel_energy(point, spatial)
    with torch.autocast(device, enabled=False):
        dense = torch.einsum(
            "morij,mrc->mocij", spatial.reshape(modes, 2, rank, kernel, kernel), point,
        )
        expected = dense.square().sum((2, 3, 4))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=9.9e-7)
    probe = torch.randn_like(actual)
    params = point, spatial
    for grad, ref in zip(
        torch.autograd.grad(actual, params, probe),
        torch.autograd.grad(expected, params, probe),
        strict=True,
    ):
        torch.testing.assert_close(grad, ref, rtol=0, atol=9.9e-7)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize(
    ("shape", "modes", "rank", "kernel", "dtype"),
    [
        ((2, 5, 11), 17, 1, 1, torch.float32),
        ((2, 7, 13), 17, 4, 3, torch.bfloat16),
        ((2, 9, 14), 35, 4, 3, torch.bfloat16),
        ((2, 56, 56), 35, 4, 3, torch.bfloat16),
        ((1, 19, 27), 80, 2, 5, torch.float16),
        ((1, 7, 23), 128, 1, 3, torch.bfloat16),
        ((2, 9, 17), 160, 2, 3, torch.bfloat16),
    ],
)
def test_real_packed_spatial(device, shape, modes, rank, kernel, dtype) -> None:
    torch.manual_seed(915)
    p = torch.randn(*shape, rank * modes, device=device, dtype=dtype).requires_grad_()
    w = (
        torch.randn(modes, 2, rank, kernel, kernel, device=device, dtype=dtype) * 0.15
    ).requires_grad_()
    actual = kernels.real_packed_spatial(p, w)
    ref = kernels._spatial_reference(p, *w.unbind(1), strict_complex=False)
    torch.testing.assert_close(actual, ref, rtol=8e-3, atol=8e-5)
    _check_gradients(actual, ref, (p, w), (p, w), rtol=8e-3, atol=8e-5)
    probe = torch.randn_like(actual)
    grads = torch.autograd.grad(kernels.real_packed_spatial(p, w), (p, w), probe)
    repeat = torch.autograd.grad(kernels.real_packed_spatial(p, w), (p, w), probe)
    for first, second in zip(grads, repeat, strict=True):
        assert torch.equal(first, second)


@requires_cuda
@pytest.mark.usefixtures("cudnn_full_precision")
def test_packed_spatial_backward_accepts_fp32_adjoint_for_bf16_reader() -> None:
    torch.manual_seed(814)
    p = torch.randn(2, 5, 7, 34, device="cuda", dtype=torch.bfloat16)
    w = tuple(torch.randn(17, 2, 3, 3, device="cuda", dtype=p.dtype) for _ in range(2))
    adjoint = torch.randn(2, 5, 7, 34, device="cuda")
    dx, dw = kernels._packed_spatial_backward(adjoint, p, *w, strict_complex=False)
    p_ref, w0, w1 = (x.float().requires_grad_() for x in (p, *w))
    output = kernels._spatial_reference(p_ref, w0, w1, strict_complex=False)
    dx_ref, dw0, dw1 = torch.autograd.grad(output, (p_ref, w0, w1), adjoint)
    torch.testing.assert_close(dx, dx_ref.to(dx.dtype), rtol=8e-3, atol=1e-5)
    torch.testing.assert_close(dw, torch.stack((dw0, dw1)), rtol=3e-5, atol=2e-5)


@requires_cuda
@pytest.mark.usefixtures("cudnn_full_precision")
@pytest.mark.parametrize("strict", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize(
    ("dtype", "rank", "kernel"),
    [
        (torch.float32, 1, 1),
        (torch.float16, 2, 3),
        (torch.bfloat16, 3, 3),
        (torch.float32, 4, 3),
        (torch.float16, 5, 5),
        (torch.bfloat16, 8, 3),
    ],
)
def test_cuda_spatial_uses_nvrtc(dtype, rank, kernel, strict, monkeypatch) -> None:
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("native BF16 requires SM80 or newer")
    torch.manual_seed(513)
    modes = 35
    p = torch.randn(
        2, 5, 9, (2 if strict else 1) * rank * modes, device="cuda", dtype=dtype,
    ).requires_grad_()
    w = (torch.randn(modes, 2, rank, kernel, kernel, device="cuda", dtype=dtype) * 0.1)
    w.requires_grad_()
    ref_p, ref_w = (tensor.detach().float().requires_grad_() for tensor in (p, w))
    expected = kernels._spatial_reference(ref_p, *ref_w.unbind(1), strict_complex=strict).to(dtype)

    def unexpected_reference(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Eligible reader fell back to convolution")

    monkeypatch.setattr(kernels, "_spatial_reference", unexpected_reference)
    monkeypatch.setattr(kernels, "_spatial_reference_backward", unexpected_reference)
    output = (
        kernels.packed_spatial(p, *w.unbind(1)) if strict else kernels.real_packed_spatial(p, w)
    )
    rtol, atol = {
        torch.float32: (4e-5, 5e-6),
        torch.float16: (2e-3, 1e-4),
        torch.bfloat16: (8e-3, 2e-4),
    }[dtype]
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    _check_gradients(output, expected, (p, w), (ref_p, ref_w), rtol=rtol, atol=atol)


@requires_cuda
@pytest.mark.usefixtures("cudnn_full_precision")
@pytest.mark.parametrize("strict", [False, True], ids=["real", "complex"])
def test_spatial_uses_pytorch_when_no_cuda_kernel(strict, monkeypatch) -> None:
    torch.manual_seed(416)
    p = torch.randn(2, 3, 5, (2 if strict else 1) * 2 * 7, device="cuda").requires_grad_()
    w = torch.randn(7, 2, 2, 3, 3, device="cuda").requires_grad_()
    ref_p, ref_w = (tensor.detach().clone().requires_grad_() for tensor in (p, w))
    reference = kernels._spatial_reference(ref_p, *ref_w.unbind(1), strict_complex=strict)
    monkeypatch.setattr(kernels, "_use_reader_cuda", lambda *_args, **_kwargs: False)
    output = (
        kernels.packed_spatial(p, *w.unbind(1)) if strict else kernels.real_packed_spatial(p, w)
    )
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    _check_gradients(output, reference, (p, w), (ref_p, ref_w), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize(
    ("rank", "kernel", "dtype", "eps"),
    [
        (1, 1, torch.float32, 1e-8),
        (3, 3, torch.float32, 0.2),
        (4, 3, torch.float32, 1e-8),
        (5, 5, torch.float16, 1e-8),
        (8, 5, torch.bfloat16, 1e-8),
        (9, 1, torch.float32, 1e-8),
    ],
)
def test_real_gram_normalize_spatial(device, rank, kernel, dtype, eps) -> None:
    torch.manual_seed(617)
    modes = 13
    point = (torch.randn(modes, rank, 67, device=device) * 0.1).requires_grad_()
    spatial = (torch.randn(2 * modes, rank, kernel, kernel, device=device) * 0.1).requires_grad_()
    energy = kernels.real_gram_kernel_energy(point, spatial)
    inv = energy.clamp_min(eps).rsqrt()[..., None, None, None]
    expected = (spatial.unflatten(0, (modes, 2)) * inv).transpose(0, 1).to(dtype)
    actual = kernels.real_gram_normalize_spatial(point, spatial, eps, dtype)
    assert actual.is_contiguous()
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    probe = torch.randn_like(expected)
    # A singleton filter's spatial gradient cancels two reciprocal-scale terms.
    gradient_atol = 2e-5 if rank == 1 and kernel == 1 else 3e-6
    for grad, ref in zip(
        torch.autograd.grad(actual, (point, spatial), probe),
        torch.autograd.grad(expected, (point, spatial), probe),
        strict=True,
    ):
        torch.testing.assert_close(grad, ref, rtol=2e-5, atol=gradient_atol)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("broadcast_probe", [False, True])
def test_gram_normalize_strided_inputs(dtype, broadcast_probe) -> None:
    point = torch.randn(4, 23, 6, device="cuda", dtype=dtype)[:, :, 1::2].transpose(1, 2)
    spatial = torch.randn(2, 4, 3, 3, 6, device="cuda", dtype=dtype)[..., 1::2]
    spatial = spatial.permute(1, 0, 2, 4, 3)
    point.requires_grad_()
    spatial.requires_grad_()
    eps = 1e-8
    actual = kernels.real_gram_normalize_spatial(point, spatial, eps, dtype)
    contiguous = kernels.real_gram_normalize_spatial(
        point.contiguous(), spatial.contiguous(), eps, dtype,
    )
    probe = (
        torch.randn(2, 4, 3, 1, 1, device="cuda", dtype=dtype).expand_as(actual)
        if broadcast_probe
        else torch.randn(4, 2, 3, 3, 3, device="cuda", dtype=dtype).transpose(0, 1)
    )
    torch.testing.assert_close(actual, contiguous, rtol=0, atol=0)
    strided = torch.autograd.grad(actual, (point, spatial), probe)
    expected = torch.autograd.grad(contiguous, (point, spatial), probe.contiguous())
    torch.testing.assert_close(strided, expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("operation", ["real", "strict", "gram"])
def test_reader_cuda_graph_replay(operation) -> None:
    modes, rank = 7, 2
    channels = modes * rank * (2 if operation == "strict" else 1)
    shape = (modes, rank, 23) if operation == "gram" else (2, 4, 5, channels)
    first = torch.randn(shape, device="cuda", requires_grad=True)
    filters = torch.randn(modes, 2, rank, 3, 3, device="cuda", requires_grad=True)

    def forward_backward():
        if operation == "gram":
            output = kernels.real_gram_normalize_spatial(first, filters, 1e-8, torch.float32)
        elif operation == "strict":
            output = kernels.packed_spatial(first, *filters.unbind(1))
        else:
            output = kernels.real_packed_spatial(first, filters)
        return output, *torch.autograd.grad(output.square().mean(), (first, filters))

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        forward_backward()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward_backward()
    for _ in range(2):
        with torch.no_grad():
            first.add_(0.125)
            filters.mul_(0.75)
        graph.replay()
        torch.testing.assert_close(captured, forward_backward())
