"""Opt-in frozen PostFusion execution; installed only on inference snapshots."""

from __future__ import annotations

# Buffers are initialized through register_buffer, which the type checker cannot infer.
# pyright: reportUninitializedInstanceVariable=false
from typing import TYPE_CHECKING, Literal, cast, override

import torch
from torch import Tensor, nn
from torch.nn import functional

from .complex_scan_transitions import complex_rms_unit
from .pac_gated_post_fusion import GatedComplexPostFusion, GatedStrictComplexPostFusion
from .pac_triton_complex_rmsnorm import packed_complex_rms_norm_from_packed
from .pac_triton_postfusion_inference import (
    full_postfusion_inference,
    joint_gated_inference,
    output_residual_inference,
)

if TYPE_CHECKING:
    from collections.abc import Collection

PostFusionVariant = Literal["joint_gated", "gate_output", "full"]
POSTFUSION_VARIANTS: tuple[PostFusionVariant, ...] = ("joint_gated", "gate_output", "full")


class _FrozenComplexRMSNorm(nn.Module):
    """Retain the norm attribute contract without trainable parameters."""

    weight: Tensor

    def __init__(self, weight: Tensor, epsilon: float) -> None:
        super().__init__()
        self.register_buffer("weight", weight.detach().float().clone())
        self.epsilon = epsilon

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        real, imag = complex_rms_unit(real, imag, self.epsilon)
        weight = self.weight.to(real.dtype)
        return real * weight, imag * weight


class FrozenGatedPostFusion(nn.Module):
    """Snapshot of the exact packed maps, supporting the Cartesian SiLU gate.

    The source must be an evaluation module. Derived weights are detached BF16
    buffers and normalization retains the original FP32 gain. Rebuild the
    snapshot after loading weights. CPU execution provides a BF16 equation
    oracle; CUDA dispatches the forward-only fusion kernels.
    """

    joint_weight: Tensor
    out_weight: Tensor

    def __init__(
        self, source: GatedComplexPostFusion | GatedStrictComplexPostFusion,
        variant: PostFusionVariant,
    ) -> None:
        super().__init__()
        if type(source) not in (GatedComplexPostFusion, GatedStrictComplexPostFusion):
            raise TypeError("only the exact Cartesian PostFusion classes can be frozen")
        if source.training:
            raise ValueError("PostFusion snapshots require an evaluation module")
        if variant not in POSTFUSION_VARIANTS:
            raise ValueError("unknown PostFusion fusion variant")
        if any(module._forward_hooks or module._forward_pre_hooks for module in source.modules()):
            raise ValueError("PostFusion snapshots cannot bypass module hooks")
        self.modes = source.modes
        self.hidden_modes = source.hidden_modes
        self.epsilon = source.norm.epsilon
        self.norm = _FrozenComplexRMSNorm(source.norm.weight, self.epsilon)
        self.variant = variant
        with torch.no_grad():
            joint = torch.cat((source.value.packed_weight(), source.gate.weight), dim=0)
            self.register_buffer("joint_weight", joint.detach().to(torch.bfloat16).contiguous())
            self.register_buffer(
                "out_weight", source.out.packed_weight().detach().to(torch.bfloat16).contiguous(),
            )
        self.eval()

    @property
    def norm_weight(self) -> Tensor:
        return self.norm.weight

    @override
    def train(self, mode: bool = True) -> FrozenGatedPostFusion:
        if mode:
            raise RuntimeError("frozen PostFusion supports inference only")
        return super().train(mode=False)

    def forward(self, real: Tensor, imag: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        if imag is None:
            return self.forward_packed(real)
        if real.shape != imag.shape or real.shape[-1] != self.modes:
            raise ValueError("PostFusion Cartesian coordinates have incompatible shapes")
        output = self.forward_packed(torch.cat((real, imag), dim=-1))
        output_real, output_imag = output.split(self.modes, dim=-1)
        return output_real, output_imag

    def forward_residual_packed(self, left: Tensor, right: Tensor) -> Tensor:
        if left.shape != right.shape:
            raise ValueError("PostFusion residual inputs have incompatible shapes")
        return self.forward_packed(left + right)

    def forward_packed(self, packed: Tensor) -> Tensor:
        if (packed.ndim < 1 or packed.shape[-1] != 2 * self.modes or packed.numel() == 0
                or packed.dtype not in (torch.float32, torch.bfloat16)):
            raise ValueError("frozen PostFusion requires nonempty FP32 or BF16 packed rows")
        if packed.requires_grad and torch.is_grad_enabled():
            raise RuntimeError("frozen PostFusion kernels do not support backward")
        packed = packed.contiguous()
        normalized = packed_complex_rms_norm_from_packed(
            packed, self.norm_weight, self.epsilon,
        ).to(torch.bfloat16)
        return self._fused_update(normalized, packed, add_residual=True)

    def _packed_update(self, normalized: Tensor) -> Tensor:
        """Keep the projection-fusion integration boundary used by packed callers."""
        normalized = normalized.to(torch.bfloat16).contiguous()
        return self._fused_update(normalized, normalized, add_residual=False)

    def _fused_update(self, normalized: Tensor, packed: Tensor, *, add_residual: bool) -> Tensor:
        if not packed.is_cuda:
            update = self._reference_update(normalized)
            return packed + update if add_residual else update
        if self.variant == "full":
            return full_postfusion_inference(
                normalized, self.joint_weight, self.out_weight, packed, add_residual,
            )
        if self.variant == "joint_gated":
            gated = joint_gated_inference(normalized, self.joint_weight)
            return output_residual_inference(
                gated, self.out_weight, packed, fuse_gate=False, add_residual=add_residual,
            )
        joint = functional.linear(normalized, self.joint_weight)
        return output_residual_inference(
            joint, self.out_weight, packed, fuse_gate=True, add_residual=add_residual,
        )

    def _reference_update(self, normalized: Tensor) -> Tensor:
        joint = functional.linear(normalized, self.joint_weight)
        value, gate_logits = joint.split((2 * self.hidden_modes, self.hidden_modes), dim=-1)
        gate = functional.silu(gate_logits)
        gated = (value.unflatten(-1, (2, self.hidden_modes)) * gate.unsqueeze(-2)).flatten(-2)
        return functional.linear(gated, self.out_weight)


def enable_postfusion_inference(
    model: nn.Module, variant: PostFusionVariant = "joint_gated",
    *, sites: Collection[str] | None = None,
) -> tuple[str, ...]:
    """Replace supported PostFusion children of an isolated eval model in place.

    ``sites`` optionally limits replacement to measured module paths. Exact type
    checks intentionally exclude magnitude gates and any custom subclass.
    The returned paths make the installed scope explicit to benchmark drivers.
    """
    if model.training:
        raise ValueError("install PostFusion fusion on an isolated evaluation snapshot")
    if variant not in POSTFUSION_VARIANTS:
        raise ValueError("unknown PostFusion fusion variant")
    requested = None if sites is None else set(sites)
    replacements: list[tuple[str, FrozenGatedPostFusion]] = []
    for name, module in model.named_modules():
        if (name and type(module) in (GatedComplexPostFusion, GatedStrictComplexPostFusion)
                and (requested is None or name in requested)):
            source = cast("GatedComplexPostFusion | GatedStrictComplexPostFusion", module)
            replacements.append((name, FrozenGatedPostFusion(source, variant)))
    installed = tuple(name for name, _ in replacements)
    if requested is not None and requested.difference(installed):
        raise ValueError("requested PostFusion sites are missing or unsupported")
    for name, replacement in replacements:
        model.set_submodule(name, replacement)
    return installed


__all__ = [
    "POSTFUSION_VARIANTS", "FrozenGatedPostFusion", "PostFusionVariant",
    "enable_postfusion_inference",
]
