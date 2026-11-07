# pyright: reportMissingImports=false, reportPrivateUsage=false

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import Tensor
from torch.library import opcheck

from lnet.pac_recurrence import (
    RecurrenceBackend,
    RecurrenceDirection,
    real2d_loop_recurrence,
    recurrence_real2d_directional,
    recurrence_real2d_state_variance_directional,
)
from lnet.pac_triton_recurrence import triton_fused_recurrence
from lnet.pac_triton_recurrence_op import (
    _select_recurrence_num_warps,
    pac_triton_recurrence_op,
    pac_triton_recurrence_opaque_op,
    pac_triton_state_variance_recurrence_op,
)

GradTuple = tuple[Tensor, Tensor, Tensor, Tensor]


def test_recurrence_num_warps_default_is_portable_across_shape_and_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LNET_PAC_RECURRENCE_NUM_WARPS", raising=False)
    monkeypatch.delenv("LNET_PAC_RECURRENCE_FORWARD_NUM_WARPS", raising=False)
    monkeypatch.delenv("LNET_PAC_RECURRENCE_BACKWARD_NUM_WARPS", raising=False)
    device = torch.device("cuda:0")

    for batch, n_steps, modes in ((1, 16, 4), (32, 127, 16), (64, 512, 32)):
        for backward in (False, True):
            assert (
                _select_recurrence_num_warps(
                    batch=batch,
                    n_steps=n_steps,
                    modes=modes,
                    device=device,
                    backward=backward,
                )
                == 4
            )


def test_recurrence_num_warps_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    monkeypatch.setenv("LNET_PAC_RECURRENCE_NUM_WARPS", "2")
    monkeypatch.setenv("LNET_PAC_RECURRENCE_BACKWARD_NUM_WARPS", "1")

    assert (
        _select_recurrence_num_warps(
            batch=2,
            n_steps=17,
            modes=8,
            device=device,
            backward=False,
        )
        == 2
    )
    assert (
        _select_recurrence_num_warps(
            batch=2,
            n_steps=17,
            modes=8,
            device=device,
            backward=True,
        )
        == 1
    )

    monkeypatch.setenv("LNET_PAC_RECURRENCE_FORWARD_NUM_WARPS", "3")
    with pytest.raises(ValueError, match="FORWARD_NUM_WARPS must be one of"):
        _select_recurrence_num_warps(
            batch=2,
            n_steps=17,
            modes=8,
            device=device,
            backward=False,
        )


def _make_inputs(
    n_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(991 + n_steps)
    shape = (2, n_steps, 3)
    decay_real_leaf = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    decay_imag_leaf = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    input_real = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    input_imag = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    decay_real = (0.8 * torch.sigmoid(decay_real_leaf)).detach().requires_grad_()
    decay_imag = (0.1 * torch.tanh(decay_imag_leaf)).detach().requires_grad_()
    return decay_real, decay_imag, input_real, input_imag


@pytest.mark.parametrize("backend", ["real2d_loop", "triton_fused", "auto"])
def test_directional_recurrence_rejects_unknown_direction(
    backend: RecurrenceBackend,
) -> None:
    inputs = _make_inputs(5, device=torch.device("cpu"))
    invalid_direction = cast("RecurrenceDirection", cast("object", "sideways"))

    with pytest.raises(ValueError, match="unsupported recurrence direction"):
        recurrence_real2d_directional(*inputs, backend, invalid_direction)


@pytest.mark.parametrize("backend", ["real2d_loop", "triton_fused", "auto"])
def test_directional_state_variance_recurrence_rejects_unknown_direction(
    backend: RecurrenceBackend,
) -> None:
    inputs = _make_inputs(5, device=torch.device("cpu"))
    variance_decay = torch.full_like(inputs[0], 0.75)
    variance_input = torch.ones_like(inputs[0])
    invalid_direction = cast("RecurrenceDirection", cast("object", "sideways"))

    with pytest.raises(ValueError, match="unsupported recurrence direction"):
        recurrence_real2d_state_variance_directional(
            *inputs,
            variance_decay,
            variance_input,
            backend,
            invalid_direction,
        )


def _loss(outputs: tuple[Tensor, Tensor]) -> Tensor:
    states_real, states_imag = outputs
    return states_real.square().mean() + 0.7 * states_imag.square().mean()


def _outputs_and_grads(
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    *,
    reverse: bool,
) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
    copied = tuple(tensor.detach().clone().requires_grad_() for tensor in tensors)
    outputs = pac_triton_recurrence_op(*copied, reverse=reverse)
    grads = cast("GradTuple", torch.autograd.grad(_loss(outputs), copied))
    return outputs[0].detach(), outputs[1].detach(), grads


def _reference_outputs_and_grads(
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    *,
    reverse: bool,
) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
    copied = tuple(tensor.detach().clone().requires_grad_() for tensor in tensors)
    if reverse:
        oriented = tuple(torch.flip(tensor, dims=(1,)) for tensor in copied)
        states_real, states_imag = real2d_loop_recurrence(*oriented)
        outputs = (torch.flip(states_real, dims=(1,)), torch.flip(states_imag, dims=(1,)))
    else:
        outputs = real2d_loop_recurrence(*copied)
    grads = cast("GradTuple", torch.autograd.grad(_loss(outputs), copied))
    return outputs[0].detach(), outputs[1].detach(), grads


@pytest.mark.parametrize("n_steps", [1, 13])
@pytest.mark.parametrize("reverse", [False, True])
def test_pac_recurrence_op_cpu_matches_reference(
    n_steps: int,
    reverse: bool,
) -> None:
    tensors = _make_inputs(n_steps, device=torch.device("cpu"))
    actual_real, actual_imag, actual_grads = _outputs_and_grads(tensors, reverse=reverse)
    expected_real, expected_imag, expected_grads = _reference_outputs_and_grads(
        tensors,
        reverse=reverse,
    )

    assert torch.allclose(actual_real, expected_real, atol=2.0e-6, rtol=2.0e-6)
    assert torch.allclose(actual_imag, expected_imag, atol=2.0e-6, rtol=2.0e-6)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(actual_grad, expected_grad, atol=2.0e-6, rtol=2.0e-6)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("reverse", [False, True])
def test_state_variance_fusion_matches_two_recurrences(
    device_name: str,
    reverse: bool,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused state/variance recurrence")
    device = torch.device(device_name)
    tensors = _make_inputs(13, device=device)
    variance_decay = tensors[0].detach().square().add(tensors[1].detach().square())
    variance_input = torch.rand_like(variance_decay)

    actual_inputs = cast(
        "GradTuple",
        tuple(tensor.detach().clone().requires_grad_() for tensor in tensors),
    )
    actual = pac_triton_state_variance_recurrence_op(
        *actual_inputs,
        variance_decay,
        variance_input,
        reverse=reverse,
    )
    actual_grads = torch.autograd.grad(_loss(actual[:2]), actual_inputs)

    expected_inputs = cast(
        "GradTuple",
        tuple(tensor.detach().clone().requires_grad_() for tensor in tensors),
    )
    expected_state = pac_triton_recurrence_op(*expected_inputs, reverse=reverse)
    expected_variance, _ = pac_triton_recurrence_op(
        variance_decay,
        torch.zeros_like(variance_decay),
        variance_input,
        torch.zeros_like(variance_input),
        reverse=reverse,
    )
    expected_grads = torch.autograd.grad(_loss(expected_state), expected_inputs)

    for actual_output, expected_output in zip(
        actual,
        (*expected_state, expected_variance),
        strict=True,
    ):
        assert torch.allclose(actual_output, expected_output, atol=2.0e-6, rtol=2.0e-6)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(actual_grad, expected_grad, atol=2.0e-6, rtol=2.0e-6)


@pytest.mark.parametrize("n_steps", [1, 13])
@pytest.mark.parametrize("reverse", [False, True])
def test_pac_recurrence_op_cuda_matches_existing_triton(
    n_steps: int,
    reverse: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton recurrence op parity")
    tensors = _make_inputs(n_steps, device=torch.device("cuda"))
    actual_real, actual_imag, actual_grads = _outputs_and_grads(tensors, reverse=reverse)

    copied = tuple(tensor.detach().clone().requires_grad_() for tensor in tensors)
    expected = triton_fused_recurrence(*copied, reverse=reverse)
    expected_grads = torch.autograd.grad(_loss(expected), copied)

    assert torch.allclose(actual_real, expected[0], atol=2.0e-6, rtol=2.0e-6)
    assert torch.allclose(actual_imag, expected[1], atol=2.0e-6, rtol=2.0e-6)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(actual_grad, expected_grad, atol=2.0e-6, rtol=2.0e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pac_recurrence_op_low_precision_cpu_matches_reference(dtype: torch.dtype) -> None:
    tensors = _make_inputs(13, device=torch.device("cpu"), dtype=dtype)
    actual_real, actual_imag, actual_grads = _outputs_and_grads(tensors, reverse=True)
    expected_real, expected_imag, expected_grads = _reference_outputs_and_grads(
        tensors,
        reverse=True,
    )

    assert actual_real.dtype is dtype
    assert actual_imag.dtype is dtype
    assert torch.allclose(actual_real.float(), expected_real.float(), atol=8.0e-3, rtol=8.0e-3)
    assert torch.allclose(actual_imag.float(), expected_imag.float(), atol=8.0e-3, rtol=8.0e-3)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(
            actual_grad.float(),
            expected_grad.float(),
            atol=8.0e-3,
            rtol=8.0e-3,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pac_recurrence_op_low_precision_cuda_matches_existing_kernel(
    dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for low-precision Triton recurrence parity")
    tensors = _make_inputs(13, device=torch.device("cuda"), dtype=dtype)
    actual_real, actual_imag, actual_grads = _outputs_and_grads(tensors, reverse=True)
    copied = tuple(tensor.detach().clone().requires_grad_() for tensor in tensors)
    expected = triton_fused_recurrence(*copied, reverse=True)
    expected_grads = torch.autograd.grad(_loss(expected), copied)

    assert torch.allclose(actual_real.float(), expected[0].float(), atol=8.0e-3, rtol=8.0e-3)
    assert torch.allclose(actual_imag.float(), expected[1].float(), atol=8.0e-3, rtol=8.0e-3)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.allclose(
            actual_grad.float(),
            expected_grad.float(),
            atol=8.0e-3,
            rtol=8.0e-3,
        )


def test_pac_recurrence_op_opcheck() -> None:
    tensors = tuple(
        tensor.detach().requires_grad_() for tensor in _make_inputs(3, device=torch.device("cpu"))
    )
    results = opcheck(
        torch.ops.lnet.pac_real2d_recurrence.default,
        (*tensors, False),
    )
    assert set(results.values()) == {"SUCCESS"}


def test_pac_recurrence_op_fullgraph_compile() -> None:
    tensors = _make_inputs(13, device=torch.device("cpu"))

    def compiled_target(
        decay_real: Tensor,
        decay_imag: Tensor,
        input_real: Tensor,
        input_imag: Tensor,
    ) -> Tensor:
        states_real, states_imag = pac_triton_recurrence_op(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=True,
        )
        return states_real + states_imag

    compiled = torch.compile(compiled_target, fullgraph=True)
    actual = compiled(*tensors)
    expected_real, expected_imag, _expected_grads = _reference_outputs_and_grads(
        tensors,
        reverse=True,
    )
    assert torch.allclose(actual, expected_real + expected_imag, atol=2.0e-6, rtol=2.0e-6)


@pytest.mark.parametrize("reverse", [False, True])
def test_opaque_recurrence_preserves_fullgraph_training(reverse: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for opaque recurrence training")
    tensors = _make_inputs(13, device=torch.device("cuda"))
    expected_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in tensors)
    expected = pac_triton_recurrence_op(*expected_inputs, reverse=reverse)
    expected_grads = torch.autograd.grad(_loss(expected), expected_inputs)

    def compiled_target(
        decay_real: Tensor,
        decay_imag: Tensor,
        input_real: Tensor,
        input_imag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return pac_triton_recurrence_opaque_op(
            decay_real,
            decay_imag,
            input_real,
            input_imag,
            reverse=reverse,
        )

    actual_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in tensors)
    compiled = torch.compile(
        compiled_target,
        fullgraph=True,
        options={"triton.cudagraphs": False},
    )
    actual = compiled(*actual_inputs)
    actual_grads = torch.autograd.grad(_loss(actual), actual_inputs)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_grads, expected_grads, atol=2.0e-6, rtol=2.0e-6)


def test_opaque_recurrence_fake_matches_contiguous_output_stride() -> None:
    batch, steps, modes = 2, 7, 5

    def inputs() -> tuple[Tensor, Tensor]:
        real = torch.randn(batch, modes, steps).transpose(1, 2).detach().requires_grad_()
        imag = torch.randn(batch, modes, steps).transpose(1, 2).detach().requires_grad_()
        assert real.stride() == (steps * modes, 1, steps)
        return real, imag

    def target(input_real: Tensor, input_imag: Tensor) -> Tensor:
        shape = (1, 1, modes)
        decay_real = torch.full(shape, 0.8).expand_as(input_real)
        decay_imag = torch.full(shape, 0.1).expand_as(input_imag)
        states = pac_triton_recurrence_opaque_op(
            decay_real, decay_imag, input_real, input_imag
        )
        assert states[0].is_contiguous()
        assert states[1].is_contiguous()
        return _loss(states)

    expected_inputs = inputs()
    expected_loss = target(*expected_inputs)
    expected_grads = torch.autograd.grad(expected_loss, expected_inputs)
    actual_inputs = tuple(
        tensor.detach().clone(memory_format=torch.preserve_format).requires_grad_()
        for tensor in expected_inputs
    )
    compiled_loss = torch.compile(target, fullgraph=True, dynamic=False)(*actual_inputs)
    actual_grads = torch.autograd.grad(compiled_loss, actual_inputs)
    torch.testing.assert_close(compiled_loss, expected_loss)
    torch.testing.assert_close(actual_grads, expected_grads)

    shape = (1, 1, modes)
    real, imag = inputs()
    decay_real = torch.full(shape, 0.8).expand_as(real)
    decay_imag = torch.full(shape, 0.1).expand_as(imag)
    result = opcheck(
        torch.ops.lnet.pac_real2d_recurrence_opaque.default,
        (decay_real, decay_imag, real, imag, False),
        raise_exception=True,
    )
    assert all(value == "SUCCESS" for value in result.values())
