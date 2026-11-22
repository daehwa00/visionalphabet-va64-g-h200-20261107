# pyright: reportMissingImports=false
from __future__ import annotations

import copy
from typing import cast

import pytest
import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import orthogonal

from lnet.complex_scan import (
    ComplexScanBackbone,
    ComplexScanConfig,
    bake_parametrizations_,
    untraced_parametrizations_,
)

_CUDA_AVAILABLE = torch.cuda.is_available()


def _orthogonal_linear() -> nn.Linear:
    torch.manual_seed(11)
    linear = nn.Linear(8, 8, bias=False)
    nn.init.orthogonal_(linear.weight)
    orthogonal(linear, "weight", orthogonal_map="matrix_exp", use_trivialization=True)
    return linear


def _model(device: str = "cpu") -> ComplexScanBackbone:
    torch.manual_seed(501)
    config = ComplexScanConfig(output_dim=10, stem_strides=(2, 2))
    return ComplexScanBackbone(config).to(device).eval()


def _analysis(model: ComplexScanBackbone) -> nn.Module:
    analysis = model.analysis
    assert analysis is not None
    return analysis


def test_baking_preserves_the_realized_weight_exactly() -> None:
    linear = _orthogonal_linear()
    realized = linear.weight.detach().clone()
    assert bake_parametrizations_(linear) == 1
    assert not parametrize.is_parametrized(linear)
    torch.testing.assert_close(linear.weight, realized, rtol=0.0, atol=0.0)


def test_baking_is_idempotent() -> None:
    linear = _orthogonal_linear()
    assert bake_parametrizations_(linear) == 1
    assert bake_parametrizations_(linear) == 0


def test_baking_a_copy_leaves_its_siblings_usable() -> None:
    # ``parametrize`` injects a class that ``deepcopy`` shares between copies,
    # so a naive removal would delete the managing property for every sibling.
    original = _orthogonal_linear()
    realized = original.weight.detach().clone()
    duplicate = copy.deepcopy(original)

    bake_parametrizations_(duplicate)

    assert parametrize.is_parametrized(original, "weight")
    torch.testing.assert_close(original.weight, realized, rtol=0.0, atol=0.0)
    torch.testing.assert_close(duplicate.weight, realized, rtol=0.0, atol=0.0)


def test_baked_weight_stays_orthogonal() -> None:
    linear = _orthogonal_linear()
    bake_parametrizations_(linear)
    weight = linear.weight.detach()
    gram = weight @ weight.transpose(0, 1)
    torch.testing.assert_close(gram, torch.eye(weight.shape[0]), rtol=0.0, atol=1.0e-5)


def test_prepare_for_inference_is_bit_identical_and_switches_to_eval() -> None:
    model = _model()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        reference = model(inputs)

    prepared = _model().prepare_for_inference_()
    with torch.no_grad():
        observed = prepared(inputs)

    torch.testing.assert_close(observed, reference, rtol=0.0, atol=0.0)
    assert not prepared.training
    assert not parametrize.is_parametrized(_analysis(prepared), "weight")


def test_prepare_for_inference_renames_the_parametrization_state_dict_entries() -> None:
    model = _model()
    before = set(model.state_dict().keys())
    prepared = model.prepare_for_inference_()
    after = set(prepared.state_dict().keys())

    assert before - after == {
        "analysis.parametrizations.weight.0.base",
        "analysis.parametrizations.weight.original",
    }
    assert after - before == {"analysis.weight"}


def test_an_unprepared_model_still_trains_through_the_parametrization() -> None:
    model = _model().train()
    logits = model(torch.randn(2, 3, 32, 32))
    logits.square().mean().backward()
    parametrizations = cast("nn.ModuleDict", _analysis(model).parametrizations)
    base = cast("Tensor", parametrizations["weight"].original)
    assert base.grad is not None
    assert bool(torch.isfinite(base.grad).all())


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA Graph capture requires CUDA")
def test_prepared_model_cuda_graph_preserves_outputs_within_roundoff() -> None:
    # The orthogonal ``matrix_exp`` parametrization synchronizes with the host
    # to pick its Pade order, which invalidates capture.  Baking the weight is
    # what makes the whole forward graphable, so a regression here would
    # silently remove the largest inference win rather than fail loudly.
    prepared = _model(device="cuda").prepare_for_inference_()
    static = torch.randn(2, 3, 32, 32, device="cuda")

    with torch.no_grad():
        expected = prepared(static).clone()

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                prepared(static)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = prepared(static)

    graph.replay()
    torch.cuda.synchronize()
    # Eager and captured CUDA launches may choose a different FP32 accumulation
    # order. Keep the contract at roundoff scale rather than claiming bit identity.
    torch.testing.assert_close(captured, expected, rtol=2.0e-6, atol=1.0e-7)

    # Replay must observe in-place updates to the captured input buffer.
    with torch.no_grad():
        static.copy_(torch.randn_like(static))
        refreshed = prepared(static).clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, refreshed, rtol=2.0e-6, atol=1.0e-7)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="inference preparation timing requires CUDA")
def test_prepared_model_matches_eager_cuda_within_roundoff() -> None:
    model = _model(device="cuda")
    inputs = torch.randn(4, 3, 32, 32, device="cuda")
    with torch.no_grad():
        reference = model(inputs)
    prepared = _model(device="cuda").prepare_for_inference_()
    with torch.no_grad():
        observed = prepared(inputs)
    # Baking changes the CUDA execution path, not the represented model; permit
    # only the small reduction-order error observed between the two launches.
    torch.testing.assert_close(observed, reference, rtol=2.0e-6, atol=1.0e-7)


def test_untraced_parametrizations_keeps_the_constraint_and_is_idempotent() -> None:
    linear = _orthogonal_linear()
    realized = linear.weight.detach().clone()

    assert untraced_parametrizations_(linear) == 1
    assert untraced_parametrizations_(linear) == 0

    # Only Dynamo tracing changes; the parametrization and its value stay.
    assert parametrize.is_parametrized(linear, "weight")
    torch.testing.assert_close(linear.weight, realized, rtol=0.0, atol=0.0)


def test_prepare_for_compiled_training_changes_no_value_or_gradient() -> None:
    inputs = torch.randn(2, 3, 32, 32)

    def loss_and_grads(*, prepared: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model = _model().train()
        if prepared:
            model.prepare_for_compiled_training_()
        model.zero_grad(set_to_none=True)
        loss = model(inputs).square().mean()
        loss.backward()
        return loss.detach(), {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

    reference_loss, reference_grads = loss_and_grads(prepared=False)
    observed_loss, observed_grads = loss_and_grads(prepared=True)

    torch.testing.assert_close(observed_loss, reference_loss, rtol=0.0, atol=0.0)
    assert set(reference_grads) == set(observed_grads)
    for name, reference in reference_grads.items():
        torch.testing.assert_close(observed_grads[name], reference, rtol=0.0, atol=0.0)


def test_prepare_for_compiled_training_leaves_the_model_trainable() -> None:
    model = _model().train().prepare_for_compiled_training_()

    assert parametrize.is_parametrized(_analysis(model), "weight")
    assert set(model.state_dict().keys()) == set(_model().state_dict().keys())

    model(torch.randn(2, 3, 32, 32)).square().mean().backward()
    parametrizations = cast("nn.ModuleDict", _analysis(model).parametrizations)
    base = cast("Tensor", parametrizations["weight"].original)
    assert base.grad is not None
    assert bool(torch.isfinite(base.grad).all())
