"""Parity, layout handling, and capture for the opt-in Wave 29C kernels."""

import pytest
import torch
from torch import Tensor
from torch.nn import functional

from lnet.memory_backbone import ChannelRMSNorm
from lnet.model_w29c import _row_pitch, model_w29c_enabled, packed_swiglu_vec2d

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(("mode", "features"), [
    ("off", ()),
    ("swiglu_vec2d", ("swiglu_vec2d",)),
    ("rmsnorm_bwd", ("rmsnorm_bwd",)),
    ("rmsnorm_full", ("rmsnorm_bwd", "rmsnorm_full")),
    ("combined_bwd", ("swiglu_vec2d", "rmsnorm_bwd", "nr4")),
    ("combined_full", ("swiglu_vec2d", "rmsnorm_bwd", "rmsnorm_full", "nr4")),
])
def test_presets(mode: str, features: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W29C", mode)
    keys = ("swiglu_vec2d", "rmsnorm_bwd", "rmsnorm_full", "nr4")
    assert tuple(key for key in keys if model_w29c_enabled(torch.device("cuda"), key)) == features


def test_row_pitch() -> None:
    tensor = torch.empty(3, 5, 17)
    assert _row_pitch(tensor) == 17
    assert _row_pitch(tensor[..., :9]) == 17
    assert _row_pitch(tensor.transpose(0, 1)) == 0
    assert _row_pitch(tensor[:, ::2, :]) == 0
    assert _row_pitch(tensor[..., :1].expand_as(tensor)) == 0


def _output_and_grads(
    output: Tensor, grad: Tensor, params: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    return output, *torch.autograd.grad(output, params, grad)


def _assert_close(
    actual: tuple[Tensor, ...], expected: tuple[Tensor, ...], dtype: torch.dtype,
) -> None:
    for got, want in zip(actual, expected, strict=True):
        error = (got.float() - want.float()).norm()
        assert error <= 2 * torch.finfo(dtype).eps * want.float().norm() + 1e-6


def _compare(output: Tensor, ref: Tensor, grad: Tensor, params: tuple[Tensor, ...]) -> None:
    _assert_close(
        _output_and_grads(output, grad, params), _output_and_grads(ref, grad, params), ref.dtype,
    )


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [65, 96, 144, 192, 384, 520])
def test_swiglu_vec2d(dtype: torch.dtype, width: int) -> None:
    packed = torch.randn(3, 11, 2 * width, device="cuda", dtype=dtype, requires_grad=True)
    value, gate = packed.chunk(2, -1)
    ref = value * functional.silu(gate)
    _compare(packed_swiglu_vec2d(packed), ref, torch.randn_like(ref), (packed,))


@CUDA
@pytest.mark.parametrize("mode", ["rmsnorm_bwd", "rmsnorm_full"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [64, 65, 96, 128, 256])
@pytest.mark.parametrize("layout", ["dense", "pitched", "transposed", "broadcast_grad"])
def test_channel_rmsnorm(
    mode: str, dtype: torch.dtype, width: int, layout: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LNET_MODEL_W29C", mode)
    torch.manual_seed(502)
    norm = ChannelRMSNorm(width).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        norm.weight.normal_()
    pitch = width + 16 if layout == "pitched" else width
    x = torch.randn(3, 179, pitch, device="cuda", dtype=dtype)[..., :width]
    grad = torch.randn(3, 179, pitch, device="cuda", dtype=dtype)[..., :width]
    if layout == "transposed":
        x = x.transpose(0, 1)
    if layout == "broadcast_grad":
        grad = grad[..., :1].expand_as(x)
    if layout == "transposed":
        grad = grad.transpose(0, 1)
    x.requires_grad_()
    ref = functional.rms_norm(x, (width,), norm.weight, norm.epsilon)
    _compare(norm(x), ref, grad, (x, norm.weight))


@CUDA
@pytest.mark.parametrize("mode", ["combined_bwd", "combined_full"])
def test_compile_fullgraph(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W29C", mode)
    torch.manual_seed(503)
    norm = ChannelRMSNorm(96).cuda()
    packed = torch.randn(29, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def run(x: Tensor) -> Tensor:
        return norm(packed_swiglu_vec2d(x))

    eager = run(packed)
    compiled = torch.compile(run, fullgraph=True)(packed)
    upstream = torch.randn_like(eager)
    params = packed, norm.weight
    actual = _output_and_grads(compiled, upstream, params)
    expected = _output_and_grads(eager, upstream, params)
    # The scale gradient is stored as FP32 after low-precision activation math.
    # Check at activation precision and independently against FP32 arithmetic.
    _assert_close(actual, expected, packed.dtype)

    packed_fp32 = packed.detach().float().requires_grad_()
    weight_fp32 = norm.weight.detach().float().requires_grad_()
    value, gate = packed_fp32.chunk(2, -1)
    reference = functional.rms_norm(value * functional.silu(gate), (96,), weight_fp32, norm.epsilon)
    fp32 = _output_and_grads(reference, upstream.float(), (packed_fp32, weight_fp32))
    _assert_close(expected, fp32, packed.dtype)
    _assert_close(actual, fp32, packed.dtype)
