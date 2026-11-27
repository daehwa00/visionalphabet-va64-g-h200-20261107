"""Compare direct factor execution with the retained dense-kernel reference."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from alphabet.factorized_complex_scan_reader import FactorizedComplexConv2dReader


@pytest.mark.parametrize(("rank", "kernel"), [(1, 1), (2, 3), (3, 5)])
@pytest.mark.parametrize(
    ("normalize", "rms"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_factor_execution_preserves_values_and_all_gradients(
    rank: int,
    kernel: int,
    *,
    normalize: bool,
    rms: bool,
) -> None:
    torch.manual_seed(809)
    dense = FactorizedComplexConv2dReader(
        3,
        4,
        rank=rank,
        kernel_size=kernel,
        normalize_input=normalize,
        match_input_rms=rms,
        rms_reference_modes=2,
    )
    with torch.no_grad():
        for p in dense.parameters():
            p.normal_(0, 0.2)
    fact = deepcopy(dense)
    fact.execution_mode = "factorized"
    xr = torch.randn(2, 6, 7, 3, requires_grad=True)
    xi = torch.randn_like(xr, requires_grad=True)
    yr = xr.detach().clone().requires_grad_()
    yi = xi.detach().clone().requires_grad_()
    a = dense(xr, xi)
    b = fact(yr, yi)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=3e-5, atol=3e-6)
    probes = [torch.randn_like(v) for v in a]
    sum((v * p + 0.1 * v.square()).mean() for v, p in zip(a, probes, strict=True)).backward()
    sum((v * p + 0.1 * v.square()).mean() for v, p in zip(b, probes, strict=True)).backward()
    grad_pairs = [
        (xr.grad, yr.grad),
        (xi.grad, yi.grad),
        *[(p.grad, q.grad) for p, q in zip(dense.parameters(), fact.parameters(), strict=True)],
    ]
    for x, y in grad_pairs:
        assert x is not None
        assert y is not None
        torch.testing.assert_close(x, y, rtol=2e-4, atol=3e-6)
    assert dense.state_dict().keys() == fact.state_dict().keys()


def test_factor_execution_zero_input_has_finite_gradients_and_same_checkpoint() -> None:
    dense = FactorizedComplexConv2dReader(3, 4, normalize_input=True, match_input_rms=True)
    fact = FactorizedComplexConv2dReader(
        3,
        4,
        normalize_input=True,
        match_input_rms=True,
        execution_mode="factorized",
    )
    fact.load_state_dict(dense.state_dict(), strict=True)
    x = torch.zeros(2, 4, 5, 3, requires_grad=True)
    real, imag = fact(x, x)
    assert not real.count_nonzero()
    assert not imag.count_nonzero()
    (real.sum() + imag.sum()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fact.parameters())
    with pytest.raises(ValueError, match="execution mode"):
        FactorizedComplexConv2dReader(3, 4, execution_mode="bad")
