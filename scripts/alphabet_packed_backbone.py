"""Packed CUDA execution backbone for VA-Ablation-10-dw-terminal-rmsnorm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

from alphabet.complex_scan import ComplexScanBackbone
from alphabet.complex_scan_transitions import _dtype_aligned_rms_norm

if TYPE_CHECKING:
    from alphabet.complex_scan_types import ComplexField
    from alphabet.same_resolution_depth import SameResolutionPoleScanBlock

ClassifierOutput = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class SameResolutionFactorialBackbone(ComplexScanBackbone):
    """Insert same-resolution pole scan stages at 56/28/14/7 resolutions."""

    def __init__(
        self,
        source: ComplexScanBackbone,
        blocks: dict[int, SameResolutionPoleScanBlock],
    ) -> None:
        nn.Module.__init__(self)
        self.config = source.config
        self.stem = source.stem
        self.input_norm = source.input_norm
        self.precomplex_fc = source.precomplex_fc
        self.analysis = source.analysis
        self.stage1 = source.stage1
        self.stage2 = source.stage2
        self.stage3 = cast("Any", source).stage3
        self.terminal = source.terminal
        self.same_resolution_blocks = nn.ModuleDict(
            {str(resolution): block for resolution, block in blocks.items()}
        )
        self.descriptor_dim = source.descriptor_dim
        self.classifier = source.classifier

    def _apply_at(self, resolution: int, state: ComplexField) -> ComplexField:
        key = str(resolution)
        if key not in self.same_resolution_blocks:
            return state
        return cast("ComplexField", self.same_resolution_blocks[key](*state))

    def complex_features(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: Tensor,
    ) -> tuple[ComplexField, ComplexField, ComplexField]:
        excitation = self._initial_excitation(inputs)
        excitation = self._apply_at(56, excitation)
        state2, _ = self.stage1(*excitation)
        state2 = self._require_state(state2)
        state2 = self._apply_at(28, state2)
        state3, _ = self.stage2(*state2)
        state3 = self._require_state(state3)
        state3 = self._apply_at(14, state3)
        state4, _ = self.stage3(*state3)
        state4 = self._require_state(state4)
        state4 = self._apply_at(7, state4)
        return state2, state3, state4

    def raw_descriptor(self, inputs: Tensor) -> Tensor:
        excitation = self._initial_excitation(inputs)
        excitation = self._apply_at(56, excitation)
        state2, descriptor1 = self.stage1(*excitation)
        state2 = self._require_state(state2)
        state2 = self._apply_at(28, state2)
        state3, descriptor2 = self.stage2(*state2)
        state3 = self._require_state(state3)
        state3 = self._apply_at(14, state3)
        state4, descriptor3 = self.stage3(*state3)
        state4 = self._require_state(state4)
        state4 = self._apply_at(7, state4)
        _, descriptor4 = self.terminal(*state4)
        descriptor = torch.cat((descriptor1, descriptor2, descriptor3, descriptor4), dim=-1)
        if descriptor.shape[-1] != self.descriptor_dim:
            raise RuntimeError("same-resolution placement changed the established Raw-Q width")
        return descriptor


class RepeatedSameResolutionBackbone(SameResolutionFactorialBackbone):
    """Append an ordered sequence of independently initialized same-resolution blocks."""

    def __init__(
        self,
        source: ComplexScanBackbone,
        extras: dict[int, tuple[SameResolutionPoleScanBlock, ...]],
    ) -> None:
        existing = dict(source.same_resolution_blocks.items())
        super().__init__(source, existing)
        self.extra_same_resolution_blocks = nn.ModuleDict(
            {
                str(resolution): nn.ModuleList(blocks)
                for resolution, blocks in extras.items()
            }
        )

    def _apply_at(self, resolution: int, state: ComplexField) -> ComplexField:
        state = super()._apply_at(resolution, state)
        key = str(resolution)
        if key not in self.extra_same_resolution_blocks:
            return state
        blocks = cast("nn.ModuleList", self.extra_same_resolution_blocks[key])
        for block in blocks:
            state = cast("ComplexField", block(*state))
        return state


class PackedVA128Backbone(RepeatedSameResolutionBackbone):
    """Same modules and state_dict as RepeatedSameResolutionBackbone, with packed CUDA execution."""

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

    def forward(self, inputs: Tensor) -> ClassifierOutput:  # pyright: ignore[reportIncompatibleMethodOverride]
        if not inputs.is_cuda:
            return self.forward_reference(inputs)
        terminal = self._packed_descriptor(inputs, full=False)
        with torch.autocast(device_type=terminal.device.type, enabled=False):
            return self.classifier(terminal.float(), terminal_only=True)


def enable_packed_va128_(model: nn.Module) -> nn.Module:
    """Install packed CUDA execution on RepeatedSameResolutionBackbone in place."""
    if type(model) is not RepeatedSameResolutionBackbone:
        raise TypeError("packed VA128 requires RepeatedSameResolutionBackbone")
    if model._compiled_call_impl is not None:
        raise ValueError("install packed VA128 before compiling the model")
    for stage in (model.stage1, model.stage2, model.stage3):
        stage._fuse_coarse_path = True
    model.__class__ = PackedVA128Backbone
    return model
