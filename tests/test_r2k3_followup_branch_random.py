from __future__ import annotations

# pyright: reportMissingImports=false
import torch
from run_r2k3_followup_branch_random import (
    _condition_id,
    _hierarchical_control_class_ci,
    scaled_memory_carry,
)
from torch import Tensor, nn


class _Transition(nn.Module):
    def forward(
        self,
        memory_real: Tensor,
        memory_imag: Tensor,
        carry_real: Tensor,
        carry_imag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return memory_real + carry_real, memory_imag + carry_imag


class _SameResolutionBlock(nn.Module):
    @staticmethod
    def collapsed_memory(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        return 2 * real, 2 * imag

    @staticmethod
    def post_fusion(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        return real, imag

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        memory_real, memory_imag = self.collapsed_memory(real, imag)
        return self.post_fusion(real + memory_real, imag + memory_imag)


class _Stage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.augmented = _Transition()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = _Stage()
        self.stage2 = _Stage()
        self.stage3 = _Stage()
        self.same_resolution_blocks = nn.ModuleDict({"14": _SameResolutionBlock()})
        self.extra_same_resolution_blocks: dict[str, nn.ModuleList] = {}


def test_scaled_memory_carry_scales_both_merge_types_and_restores() -> None:
    model = _Model()
    value = torch.tensor(3.0)
    block = model.same_resolution_blocks["14"]
    original_forward = block.forward
    with scaled_memory_carry(model, memory_scale=0.25, carry_scale=0.5):
        main = model.stage1.augmented(value, value, value, value)
        same = block(value, value)
        torch.testing.assert_close(main[0], torch.tensor(2.25))
        torch.testing.assert_close(same[0], torch.tensor(3.0))
    assert block.forward == original_forward
    torch.testing.assert_close(block(value, value)[0], torch.tensor(9.0))


def test_hierarchical_control_class_ci_is_reproducible() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    controls = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    first = _hierarchical_control_class_ci(
        target,
        controls,
        labels,
        repeats=200,
        seed=17,
    )
    second = _hierarchical_control_class_ci(
        target,
        controls,
        labels,
        repeats=200,
        seed=17,
    )
    assert first == second
    assert first[0] <= 0 <= first[1]


def test_branch_condition_ids_are_stable() -> None:
    assert _condition_id(0.25, 1.0) == "memory_0p25__carry_1p00"
    assert _condition_id(0.0, 0.75).endswith("carry_0p75")
