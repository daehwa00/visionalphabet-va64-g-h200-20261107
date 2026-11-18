"""Shared real depthwise spatial filtering followed by complex-coordinate writes."""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from .real_memory_backbone import RealFactorizedMemoryReader


class DepthwiseMemoryReader(nn.Module):
    def __init__(self, source: RealFactorizedMemoryReader) -> None:
        super().__init__()
        self.input_channels = source.input_channels
        self.output_modes = source.output_modes
        self.padding = source.padding
        self.variance_epsilon = source.variance_epsilon
        self.match_input_rms = source.match_input_rms
        self.rms_reference_modes = source.rms_reference_modes
        self.input_norm = copy.deepcopy(source.input_norm)
        self.depthwise_weight = nn.Parameter(torch.empty(self.input_channels, 1, 3, 3))
        self.point_weight = nn.Parameter(torch.empty(2 * self.output_modes, self.input_channels))
        nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.point_weight, a=math.sqrt(5))

    def forward(self, source: Tensor) -> Tensor:
        if self.input_norm is not None:
            source = self.input_norm(source)
        # Preserve the baseline's per-output effective-kernel L2 normalization.
        with torch.autocast(device_type=source.device.type, enabled=False):
            spatial_energy = self.depthwise_weight.float().square().sum((1, 2, 3))
            energy = self.point_weight.float().square() @ spatial_energy
            scale = torch.rsqrt(energy.clamp_min(self.variance_epsilon))[:, None]
            point = self.point_weight.float() * scale
        spatial = functional.conv2d(
            source.movedim(-1, 1),
            self.depthwise_weight,
            padding=self.padding,
            groups=self.input_channels,
        )
        output = functional.linear(spatial.movedim(1, -1), point).contiguous()
        if self.match_input_rms:
            input_energy = source.float().square().mean(-1, keepdim=True)
            selected = output.unflatten(-1, (2, self.output_modes))[..., :self.rms_reference_modes]
            output_energy = selected.float().square().mean((-1, -2)).unsqueeze(-1)
            epsilon = self.variance_epsilon / 2
            rms_scale = torch.sqrt((input_energy + epsilon) / (output_energy + epsilon))
            output = output * rms_scale.to(output.dtype)
        return output


def replace_readers_[ModuleT: nn.Module](model: ModuleT) -> ModuleT:
    replacements = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if isinstance(child, RealFactorizedMemoryReader)
    ]
    with torch.random.fork_rng(devices=[]):
        for parent, name, child in replacements:
            setattr(parent, name, DepthwiseMemoryReader(child))
    if len(replacements) != 12:
        raise RuntimeError(f"Expected 12 Readers, found {len(replacements)}")
    # Attach inventory dynamically to match the training-optimization contract.
    model._training_optimization_inventory = {"depthwise_readers": len(replacements)}  # type: ignore[attr-defined]  # noqa: SLF001
    return model
