"""VA128 execution using one real NHW[real|imag] tensor between blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from run_a2d_r2k3_stage_allocation_screen_imagenet100 import RepeatedSameResolutionBackbone
from torch import Tensor, nn

from lnet.complex_scan_transitions import _dtype_aligned_rms_norm

# The canonical campaign builders use script-level imports and private preparation APIs.
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false


if TYPE_CHECKING:
    from lnet.complex_scan_types import ComplexField

ClassifierOutput = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class PackedVA128Backbone(RepeatedSameResolutionBackbone):
    """Same modules/state_dict as VA128, with a packed CUDA execution graph."""

    def _packed_initial(self, inputs: Tensor) -> Tensor:
        if self.precomplex_fc is None or self.analysis is None or self.input_norm is None:
            raise RuntimeError("packed VA128 requires the canonical real stem")
        transformed = self.precomplex_fc(self.stem(inputs))
        return self.analysis(_dtype_aligned_rms_norm(transformed, self.input_norm))

    def _packed_apply_at(self, resolution: int, state: Tensor) -> Tensor:
        key = str(resolution)
        state = self.same_resolution_blocks[key](state)
        if key in self.extra_same_resolution_blocks:
            for block in cast("nn.ModuleList", self.extra_same_resolution_blocks[key]):
                state = block(state)
        return state

    def _packed_descriptor(self, inputs: Tensor, *, full: bool) -> Tensor:
        state = self._packed_initial(inputs)
        descriptors = []
        for resolution, stage in ((56, self.stage1), (28, self.stage2), (14, self.stage3)):
            state = self._packed_apply_at(resolution, state)
            state, descriptor = stage(state, emit_descriptor=full)
            if full:
                descriptors.append(descriptor)
        state = self._packed_apply_at(7, state)
        _, terminal = self.terminal(state, emit_descriptor=True)
        if not full:
            return terminal
        return torch.cat((*descriptors, terminal), dim=-1)

    def raw_descriptor_reference(self, inputs: Tensor) -> Tensor:
        """Original split-coordinate oracle, including unused early descriptors."""
        return super().raw_descriptor(inputs)

    def packed_features(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        state = self._packed_apply_at(56, self._packed_initial(inputs))
        state2, _ = self.stage1(state, emit_descriptor=False)
        state2 = self._packed_apply_at(28, state2)
        state3, _ = self.stage2(state2, emit_descriptor=False)
        state3 = self._packed_apply_at(14, state3)
        state4, _ = self.stage3(state3, emit_descriptor=False)
        return state2, state3, self._packed_apply_at(7, state4)

    def complex_features(self, inputs: Tensor) -> tuple[ComplexField, ComplexField, ComplexField]:
        if not inputs.is_cuda:
            return super().complex_features(inputs)
        # Compatibility coordinates are views, not separately allocated fields.
        return cast(
            "tuple[ComplexField, ComplexField, ComplexField]",
            tuple(state.chunk(2, dim=-1) for state in self.packed_features(inputs)),
        )

    def forward_reference(self, inputs: Tensor) -> ClassifierOutput:
        descriptor = self.raw_descriptor_reference(inputs)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float())

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        if not inputs.is_cuda:
            return self.raw_descriptor_reference(inputs)
        return self._packed_descriptor(inputs, full=True)

    # The campaign head returns five logits tensors; the base annotation predates it.
    def forward(self, inputs: Tensor) -> ClassifierOutput:  # pyright: ignore[reportIncompatibleMethodOverride]
        if not inputs.is_cuda:
            return self.forward_reference(inputs)
        terminal = self._packed_descriptor(inputs, full=False)
        with torch.autocast(device_type=terminal.device.type, enabled=False):
            return self.classifier(terminal.float(), terminal_only=True)


def enable_packed_va128_(model: nn.Module) -> nn.Module:
    """Install the fixed packed execution graph without wrapping/renaming state."""
    if type(model) is not RepeatedSameResolutionBackbone:
        raise TypeError("packed VA128 requires the canonical repeated-resolution backbone")
    if model._compiled_call_impl is not None:
        raise ValueError("install packed VA128 before compiling the model")
    for name in ("stage1", "stage2", "stage3", "terminal"):
        stage = getattr(model, name)
        if any(
            (
                stage.use_cccn_shortcut,
                stage.use_pole_aligned_shortcut,
                stage.use_zero_gated_pole_aligned_residual,
            )
        ):
            raise ValueError("packed VA128 does not change research-only shortcut variants")
        if stage.post_transition_ffn is not None:
            raise ValueError("packed VA128 requires its canonical PostFusion transition")
        if stage.output_modes is not None and (
            stage.carry_basis != "s2d"
            or not hasattr(stage.quadrant_path_mode_combiner, "forward_real_packed")
            or not hasattr(stage.augmented, "forward_reduced_packed")
        ):
            raise ValueError("packed VA128 requires its canonical scan/carry transition")
    for stage in (model.stage1, model.stage2, model.stage3):
        stage._fuse_coarse_path = True
    model.__class__ = PackedVA128Backbone
    return model
