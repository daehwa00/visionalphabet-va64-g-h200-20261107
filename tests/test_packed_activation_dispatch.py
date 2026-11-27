"""Numerical checks for the Wave 28C model kernels."""

import pytest
import torch
from torch import Tensor
from torch.nn import functional

from alphabet.memory_backbone import SwiGLU
from alphabet.packed_activation_dispatch import packed_activation_dispatch_enabled, packed_swiglu
from alphabet.terminal_imaginary_correlation import _nonredundant_four_directions_axis

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("capability", [(7, 5), (8, 6), (8, 9), (9, 0), (10, 0)])
def test_auto_dispatch(capability: tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPHABET_MODEL_KERNEL_POLICY", raising=False)
    monkeypatch.delenv("ALPHABET_PACKED_ACTIVATION_FEATURES", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    assert not packed_activation_dispatch_enabled(torch.device("cpu"), "nr4")
    device = torch.device("cuda")
    assert packed_activation_dispatch_enabled(device, "nr4") == (capability >= (8, 0))
    assert not packed_activation_dispatch_enabled(device, "swiglu")
    monkeypatch.setenv("ALPHABET_PACKED_ACTIVATION_FEATURES", "combined")
    assert packed_activation_dispatch_enabled(device, "swiglu") == (capability >= (8, 0))
    monkeypatch.setenv("ALPHABET_MODEL_KERNEL_POLICY", "off")
    assert not packed_activation_dispatch_enabled(device, "swiglu")
    assert not packed_activation_dispatch_enabled(device, "nr4")


def _outputs_and_grads(output: Tensor, upstream: Tensor, *inputs: Tensor) -> tuple[Tensor, ...]:
    return output, *torch.autograd.grad(output, inputs, upstream)


def _near_fp32(
    actual: tuple[Tensor, ...], expected: tuple[Tensor, ...], dtype: torch.dtype,
) -> None:
    for got, want in zip(actual, expected, strict=True):
        error = (got.float() - want.float()).norm()
        assert error <= 2 * torch.finfo(dtype).eps * want.float().norm() + 1e-6


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [96, 144, 192, 384])
def test_packed_swiglu(dtype: torch.dtype, hidden: int) -> None:
    torch.manual_seed(501)
    joint = torch.randn(515, 2 * hidden, device="cuda", dtype=dtype, requires_grad=True)
    upstream = torch.randn(515, hidden, device="cuda", dtype=dtype)
    actual = _outputs_and_grads(packed_swiglu(joint), upstream, joint)

    def reference(packed: Tensor) -> Tensor:
        value, gate = packed.chunk(2, dim=-1)
        return value * functional.silu(gate)

    expected = _outputs_and_grads(reference(joint), upstream, joint)
    torch.testing.assert_close(actual, expected, atol=2 * torch.finfo(dtype).eps, rtol=0)
    joint_fp32 = joint.detach().float().requires_grad_()
    reference_fp32 = _outputs_and_grads(reference(joint_fp32), upstream.float(), joint_fp32)
    _near_fp32(actual, reference_fp32, dtype)


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_swiglu_projection_parity(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(501)
    layer = SwiGLU(64, 96).cuda()
    inputs = torch.randn(2, 11, 64, device="cuda", dtype=dtype, requires_grad=True)
    upstream = torch.randn_like(inputs)
    monkeypatch.setenv("ALPHABET_PACKED_ACTIVATION_FEATURES", "off")
    with torch.autocast("cuda", dtype=dtype):
        expected = layer(inputs)
    expected_grads = _outputs_and_grads(expected, upstream, inputs, *layer.parameters())
    monkeypatch.setenv("ALPHABET_PACKED_ACTIVATION_FEATURES", "swiglu")
    with torch.autocast("cuda", dtype=dtype):
        actual = layer(inputs)
    actual_grads = _outputs_and_grads(actual, upstream, inputs, *layer.parameters())
    _near_fp32(actual_grads, expected_grads, dtype)


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("normalize", [False, True])
def test_nr4_warp_tile(dtype: torch.dtype, normalize: bool) -> None:
    torch.manual_seed(501)
    states = tuple(
        torch.randn(3, 7, 7, 128, device="cuda", dtype=dtype).requires_grad_() for _ in range(4)
    )
    pole = tuple(
        (torch.randn(128, device="cuda") * factor).requires_grad_()
        for factor in (0.2, 0.2, 0.6, 0.3)
    )

    def evaluate(values: tuple, *, fused: bool, warp_tile: bool) -> Tensor:
        return torch.stack(
            _nonredundant_four_directions_axis(
                pole, values, normalize=normalize, fused=fused, warp_tile=warp_tile
            ),
        )

    actual = evaluate(states, fused=True, warp_tile=True)
    upstream = torch.randn_like(actual)
    grads = _outputs_and_grads(actual, upstream, *states, *pole)
    baseline = evaluate(states, fused=True, warp_tile=False)
    torch.testing.assert_close(grads, _outputs_and_grads(baseline, upstream, *states, *pole))

    states_fp32 = tuple(state.detach().float().requires_grad_() for state in states)
    reference = evaluate(states_fp32, fused=False, warp_tile=False)
    _near_fp32(grads, _outputs_and_grads(reference, upstream, *states_fp32, *pole), dtype)
