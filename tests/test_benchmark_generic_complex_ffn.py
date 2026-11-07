from __future__ import annotations

from typing import TYPE_CHECKING

import benchmark_generic_complex_ffn as benchmark
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


def _count_forward_calls(
    module: torch.nn.Module,
) -> tuple[list[int], Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    calls: list[int] = []
    original = module.forward

    def tracked(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(1)
        return original(real, imag)

    return calls, tracked


def test_projection_reference_stays_on_projection_graph_during_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = benchmark.ResidualCFFN(4, 7, layout="projection")
    input_calls, tracked_input = _count_forward_calls(module.input)
    output_calls, tracked_output = _count_forward_calls(module.output)
    monkeypatch.setattr(module.input, "forward", tracked_input)
    monkeypatch.setattr(module.output, "forward", tracked_output)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    real = torch.randn(2, 3, 4)
    imag = torch.randn_like(real)
    module(real, imag)

    assert input_calls == [1]
    assert output_calls == [1]


def test_measurement_order_is_abba() -> None:
    order = tuple(role for round_index in range(4) for role in benchmark._paired_order(round_index))

    assert order == (
        "projection",
        "packed",
        "packed",
        "projection",
        "projection",
        "packed",
        "packed",
        "projection",
    )
