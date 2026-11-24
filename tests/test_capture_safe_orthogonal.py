from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import orthogonal

from lnet.capture_safe_orthogonal import (
    CaptureSafeMatrixExpOrthogonal,
    fixed_taylor_matrix_exp,
    prepare_capture_safe_orthogonal_,
)


def _orthogonal_linear(rows: int, columns: int, *, device: str = "cpu") -> nn.Linear:
    torch.manual_seed(19)
    layer = nn.Linear(columns, rows, bias=False, device=device)
    nn.init.orthogonal_(layer.weight)
    orthogonal(layer, "weight", orthogonal_map="matrix_exp", use_trivialization=True)
    return layer


def _gram_error(frame: Tensor) -> Tensor:
    rows, columns = frame.shape
    if rows >= columns:
        gram = frame.mT @ frame
        identity = torch.eye(columns, dtype=frame.dtype, device=frame.device)
    else:
        gram = frame @ frame.mT
        identity = torch.eye(rows, dtype=frame.dtype, device=frame.device)
    return (gram - identity).abs().max()


@pytest.mark.parametrize(("rows", "columns"), [(32, 32), (64, 32), (32, 64)])
def test_capture_safe_map_preserves_frame_and_first_order_gradient(rows: int, columns: int) -> None:
    exact = _orthogonal_linear(rows, columns)
    original = exact.parametrizations.weight.original
    with torch.no_grad():
        original.add_(0.1 * torch.randn_like(original))
    candidate = copy.deepcopy(exact)
    replaced = prepare_capture_safe_orthogonal_(candidate)

    probe = torch.randn(rows, columns)
    exact_frame = exact.weight
    candidate_frame = candidate.weight
    exact_loss = (exact_frame * probe).mean()
    candidate_loss = (candidate_frame * probe).mean()
    exact_loss.backward()
    candidate_loss.backward()

    assert replaced == ("weight",)
    assert isinstance(candidate.parametrizations.weight[0], CaptureSafeMatrixExpOrthogonal)
    torch.testing.assert_close(candidate_frame, exact_frame, atol=5.0e-6, rtol=5.0e-6)
    torch.testing.assert_close(
        candidate.parametrizations.weight.original.grad,
        exact.parametrizations.weight.original.grad,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert _gram_error(candidate_frame) <= 5.0e-6


def test_preparation_preserves_parameter_identity_names_count_and_state_dict() -> None:
    layer = _orthogonal_linear(64, 32)
    parameter = layer.parametrizations.weight.original
    base = layer.parametrizations.weight[0].base
    parameter_names = tuple(name for name, _ in layer.named_parameters())
    state_keys = tuple(layer.state_dict())
    state = copy.deepcopy(layer.state_dict())

    assert prepare_capture_safe_orthogonal_(layer) == ("weight",)

    assert prepare_capture_safe_orthogonal_(layer) == ()
    assert layer.parametrizations.weight.original is parameter
    assert layer.parametrizations.weight[0].base is base
    assert tuple(name for name, _ in layer.named_parameters()) == parameter_names
    assert tuple(layer.state_dict()) == state_keys
    assert sum(item.numel() for item in layer.parameters()) == parameter.numel()

    native = _orthogonal_linear(64, 32)
    native.load_state_dict(layer.state_dict(), strict=True)
    layer.load_state_dict(state, strict=True)
    torch.testing.assert_close(layer.weight, native.weight, atol=2.0e-6, rtol=2.0e-6)


def test_recursive_preparation_reports_only_matrix_exp_paths() -> None:
    model = nn.ModuleDict(
        {
            "first": _orthogonal_linear(32, 32),
            "nested": nn.Sequential(_orthogonal_linear(64, 32)),
            "plain": nn.Linear(4, 4),
        }
    )

    replaced = prepare_capture_safe_orthogonal_(model)

    assert replaced == ("first.weight", "nested.0.weight")


@pytest.mark.parametrize(("rows", "columns"), [(32, 32), (64, 32)])
def test_seventy_five_adamw_steps_track_native_matrix_exp(rows: int, columns: int) -> None:
    exact = _orthogonal_linear(rows, columns)
    candidate = copy.deepcopy(exact)
    prepare_capture_safe_orthogonal_(candidate)
    generator = torch.Generator().manual_seed(23)
    inputs = torch.randn(8, columns, generator=generator)
    targets = torch.randn(8, rows, generator=generator)
    exact_optimizer = torch.optim.AdamW(exact.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    candidate_optimizer = torch.optim.AdamW(candidate.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    maximum_output_error = 0.0
    maximum_gradient_error = 0.0
    maximum_parameter_error = 0.0

    for _ in range(75):
        exact_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)
        exact_output = exact(inputs)
        candidate_output = candidate(inputs)
        exact_loss = (exact_output - targets).square().mean()
        candidate_loss = (candidate_output - targets).square().mean()
        exact_loss.backward()
        candidate_loss.backward()
        maximum_output_error = max(
            maximum_output_error,
            float((candidate_output - exact_output).abs().max().detach()),
        )
        maximum_gradient_error = max(
            maximum_gradient_error,
            float(
                (
                    candidate.parametrizations.weight.original.grad
                    - exact.parametrizations.weight.original.grad
                )
                .abs()
                .max()
                .detach()
            ),
        )
        exact_optimizer.step()
        candidate_optimizer.step()
        maximum_parameter_error = max(
            maximum_parameter_error,
            float(
                (
                    candidate.parametrizations.weight.original
                    - exact.parametrizations.weight.original
                )
                .abs()
                .max()
                .detach()
            ),
        )

    assert maximum_output_error <= 2.0e-5
    assert maximum_gradient_error <= 2.0e-5
    assert maximum_parameter_error <= 2.0e-5
    torch.testing.assert_close(candidate.weight, exact.weight, atol=2.0e-5, rtol=2.0e-5)
    assert _gram_error(candidate.weight) <= 5.0e-6


def test_capture_safe_map_supports_torch_compile_fullgraph() -> None:
    layer = _orthogonal_linear(64, 32)
    prepare_capture_safe_orthogonal_(layer)
    inputs = torch.randn(2, 32, requires_grad=True)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)

    output = compiled(inputs)
    output.square().mean().backward()

    assert output.shape == (2, 64)
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("compute_dtype", [None, torch.float64])
def test_capture_safe_map_captures_forward_and_backward_in_cuda_graph(
    compute_dtype: torch.dtype | None,
) -> None:
    layer = _orthogonal_linear(64, 32, device="cuda")
    prepare_capture_safe_orthogonal_(layer, compute_dtype=compute_dtype)
    static_inputs = torch.randn(8, 32, device="cuda")
    original = layer.parametrizations.weight.original

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            layer(static_inputs).square().mean().backward()
            assert original.grad is not None
            original.grad.zero_()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = layer(static_inputs)
        static_output.square().mean().backward()
    assert original.grad is not None
    original.grad.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(static_output).all()
    assert static_output.dtype == torch.float32
    assert original.grad is not None
    assert torch.isfinite(original.grad).all()


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
)])
@pytest.mark.parametrize("norm", [0.1, 1.0, 5.0, 10.0, "rank2"])
def test_ps14_fp64_reference_and_autocast(device: str, norm: float | str) -> None:
    generator = torch.Generator().manual_seed(501)
    random = torch.randn(64, 64, generator=generator)
    if norm == "rank2":
        skew = torch.zeros_like(random)
        skew[1, 0], skew[0, 1] = 10 / 2**0.5, -10 / 2**0.5
    else:
        skew = random - random.mT
        skew = skew * (norm / skew.norm())
    matrix = skew.to(device).requires_grad_()
    probe = torch.randn(64, 64, generator=generator).to(device)
    double = matrix.detach().double().requires_grad_()
    expected = torch.matrix_exp(double)
    (expected_grad,) = torch.autograd.grad(expected, double, probe.double())
    actual = fixed_taylor_matrix_exp(matrix)
    (actual_grad,) = torch.autograd.grad(actual, matrix, probe)
    with torch.autocast(device, dtype=torch.bfloat16):
        autocast = fixed_taylor_matrix_exp(matrix)
    (autocast_grad,) = torch.autograd.grad(autocast, matrix, probe)

    torch.testing.assert_close(autocast, actual, atol=0, rtol=0)
    torch.testing.assert_close(autocast_grad, actual_grad, atol=0, rtol=0)
    torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(actual_grad.double(), expected_grad, atol=1e-6, rtol=0)
    assert _gram_error(actual) < 1e-6


def test_ps14_gradcheck() -> None:
    torch.manual_seed(501)
    matrix = torch.randn(4, 4, dtype=torch.float64)
    skew = (matrix - matrix.mT).requires_grad_()
    assert torch.autograd.gradcheck(fixed_taylor_matrix_exp, skew)


@pytest.mark.parametrize(("policy", "auto", "enabled"), [
    ("auto", True, True), ("auto", False, False), ("on", False, True), ("off", True, False),
])
def test_capture_safe_toggle(
    monkeypatch: pytest.MonkeyPatch, policy: str, auto: bool, enabled: bool,
) -> None:
    monkeypatch.setenv("LNET_CAPTURE_SAFE_ORTHOGONAL", policy)
    layer = _orthogonal_linear(8, 8)
    assert bool(prepare_capture_safe_orthogonal_(layer, auto=auto)) is enabled


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
)])
def test_ps14_compiled_autocast_keeps_backward_in_fp32(device: str) -> None:
    torch.manual_seed(501)
    matrix = torch.randn(64, 64, device=device) / 64
    skew = (matrix - matrix.mT).requires_grad_()
    probe = torch.randn_like(skew)
    compiled = torch.compile(fixed_taylor_matrix_exp, fullgraph=True)
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = compiled(skew)
    (actual_grad,) = torch.autograd.grad(actual, skew, probe)
    double = skew.detach().double().requires_grad_()
    expected = torch.matrix_exp(double)
    (expected_grad,) = torch.autograd.grad(expected, double, probe.double())
    torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(actual_grad.double(), expected_grad, atol=1e-6, rtol=0)
