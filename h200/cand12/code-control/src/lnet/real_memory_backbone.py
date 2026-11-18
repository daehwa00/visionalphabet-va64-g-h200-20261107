"""Topology-preserving real feature processing around the existing complex memory."""

from __future__ import annotations

# The packed backbone is the established internal execution boundary.
# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
import math
from typing import Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .pac_complex_layers import ComplexLinear
from .pac_d4_path_cffn import d4_grouped_path_collapse_packed
from .pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from .pac_gated_post_fusion import GatedComplexPostFusion
from .pac_phase_gated_transition import PathOnlyCollapse


class _PackedTopology(Protocol):
    classifier: nn.Module

    def _packed_initial(self, inputs: Tensor) -> Tensor: ...
    def _packed_apply_at(self, resolution: int, state: Tensor) -> Tensor: ...
    def _packed_descriptor(self, inputs: Tensor, *, full: bool) -> Tensor: ...


class RealRMSNorm(nn.Module):
    """Ordinary real RMSNorm with one independently learned scale per channel."""

    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        if width <= 0 or epsilon <= 0:
            raise ValueError("invalid real RMSNorm dimensions or epsilon")
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        inverse = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.epsilon).to(x.dtype)
        return x * inverse * self.weight.to(x.dtype)


class RealSwiGLU(nn.Module):
    """Real value/gate/output projections; one independent gate per hidden channel."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.width = width
        self.hidden = hidden
        self.norm = RealRMSNorm(width)
        self.value_gate = nn.Linear(width, 2 * hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)

    @classmethod
    def from_legacy(cls, source: GatedComplexPostFusion) -> RealSwiGLU:
        """Convert WL matrices exactly, then untie the two formerly shared gates.

        Gate rows start with identical values so conversion preserves the
        initial function. They are independent parameters thereafter.
        """
        width, hidden = 2 * source.modes, 2 * source.hidden_modes
        # Construction must not perturb later initialization/augmentation RNG.
        with torch.random.fork_rng(devices=[]):
            result = cls(width, hidden)
        result.norm = RealRMSNorm(width, source.norm.epsilon / 2.0)
        result.to(device=source.gate.weight.device, dtype=source.gate.weight.dtype)
        with torch.no_grad():
            result.norm.weight.copy_(source.norm.weight.repeat(2) / math.sqrt(2))
            result.value_gate.weight[:hidden].copy_(source.value.packed_weight())
            result.value_gate.weight[hidden:].copy_(source.gate.weight.repeat(2, 1))
            result.output.weight.copy_(source.out.packed_weight())
        result.train(source.training)
        return result

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.value_gate(self.norm(x)).chunk(2, dim=-1)
        return x + self.output(value * functional.silu(gate))

    def forward_residual_packed(self, left: Tensor, right: Tensor) -> Tensor:
        return self(left + right)


class RealPathCollapse(nn.Module):
    """The existing path CFFN with free real matrices instead of WL coordinates.

    Hidden width, activation, biases, direction order and fused scan contract
    stay unchanged. This is an equivalent matrix reparameterization, not a
    second SwiGLU replacement or a new direction-compression architecture.
    """

    def __init__(self, source: PathOnlyCollapse) -> None:
        super().__init__()
        self.modes = source.modes
        self.path_count = source.path_count
        self.output_paths = source.output_paths
        self.input_modes = source.input_modes
        weights = source.packed_parameters()
        self.input_weight = nn.Parameter(weights[0].detach().clone())
        self.input_bias = nn.Parameter(weights[1].detach().clone())
        self.output_weight = nn.Parameter(weights[2].detach().clone())
        self.output_bias = nn.Parameter(weights[3].detach().clone())
        self.train(source.training)

    def packed_parameters(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.input_weight, self.input_bias, self.output_weight, self.output_bias

    def forward_real_packed(self, source: Tensor) -> Tensor:
        return d4_grouped_path_collapse_packed(source, *self.packed_parameters())

    def forward(self, source: Tensor) -> Tensor:
        return self.forward_real_packed(source)


class RealFactorizedMemoryReader(nn.Module):
    """Free real write map with the original per-pole latent width and spatial support."""

    def __init__(self, source: FactorizedComplexConv2dReader) -> None:
        super().__init__()
        self.input_channels = 2 * source.input_modes
        self.output_modes = source.output_modes
        self.rank = 2 * source.rank
        self.kernel_size = source.kernel_size
        self.padding = source.padding
        self.variance_epsilon = source.variance_epsilon
        self.match_input_rms = source.match_input_rms
        self.rms_reference_modes = source.rms_reference_modes
        self.execution_mode = source.execution_mode
        pr, pi = source.point_weight_real, source.point_weight_imag
        sr, si = source.spatial_weight_real, source.spatial_weight_imag
        self.point_weight = nn.Parameter(
            torch.cat(
                (
                    torch.cat((pr, -pi), dim=-1),
                    torch.cat((pi, pr), dim=-1),
                ),
                dim=1,
            )
            .detach()
            .clone()
        )
        self.spatial_weight = nn.Parameter(
            torch.stack(
                (
                    torch.cat((sr, -si), dim=1),
                    torch.cat((si, sr), dim=1),
                ),
                dim=1,
            )
            .flatten(0, 1)
            .detach()
            .clone()
        )
        self.input_norm: RealRMSNorm | None = None
        if source.input_norm is not None:
            self.input_norm = RealRMSNorm(self.input_channels, source.input_norm.epsilon / 2)
            self.input_norm.to(device=pr.device, dtype=pr.dtype)
            with torch.no_grad():
                self.input_norm.weight.copy_(source.input_norm.weight.repeat(2) / math.sqrt(2))
        self.train(source.training)

    def synthesized_kernel(self) -> Tensor:
        # Row normalization remains FP32 under BF16 training.
        with torch.autocast(device_type=self.point_weight.device.type, enabled=False):
            return torch.einsum(
                "porij,prc->pocij",
                self.spatial_weight.float().reshape(
                    self.output_modes, 2, self.rank, self.kernel_size, self.kernel_size
                ),
                self.point_weight.float(),
            )

    def _kernel_and_scale(self) -> tuple[Tensor, Tensor]:
        full = self.synthesized_kernel()
        inverse = torch.rsqrt(
            full.square().sum((2, 3, 4), keepdim=True).clamp_min(self.variance_epsilon)
        )
        return full, inverse

    def forward(self, source: Tensor) -> Tensor:
        if source.ndim != 4 or source.shape[-1] != self.input_channels:
            raise ValueError("real Reader requires NHWC features of the configured width")
        if self.input_norm is not None:
            source = self.input_norm(source)
        full, inverse = self._kernel_and_scale()
        if not source.is_cuda or self.execution_mode == "dense":
            kernel = (
                (full * inverse)
                .permute(1, 0, 2, 3, 4)
                .reshape(
                    2 * self.output_modes, self.input_channels, self.kernel_size, self.kernel_size
                )
            )
            output = (
                functional.conv2d(source.movedim(-1, 1), kernel, padding=self.padding)
                .movedim(1, -1)
                .contiguous()
            )
        else:
            projected = functional.linear(source, self.point_weight.flatten(0, 1))
            spatial = (
                (
                    self.spatial_weight.float().reshape(
                        self.output_modes, 2, self.rank, self.kernel_size, self.kernel_size
                    )
                    * inverse
                )
                .to(projected.dtype)
                .contiguous()
            )
            if self.execution_mode == "factorized_triton":
                from .pac_triton_real_spatial import real_factorized_spatial  # noqa: PLC0415

                if getattr(self, "_real_training_autotune", False):
                    from .pac_triton_real_training import real_training_spatial  # noqa: PLC0415

                    output = real_training_spatial(
                        projected.contiguous(),
                        spatial,
                        getattr(self, "_real_training_vector_gradient", False),
                    )
                else:
                    output = real_factorized_spatial(projected.contiguous(), spatial)
            elif self.execution_mode == "factorized":
                grouped = functional.conv2d(
                    projected.movedim(-1, 1),
                    spatial.reshape(
                        2 * self.output_modes, self.rank, self.kernel_size, self.kernel_size
                    ),
                    padding=self.padding,
                    groups=self.output_modes,
                )
                n, _, h, w = grouped.shape
                output = (
                    grouped.reshape(n, self.output_modes, 2, h, w)
                    .permute(0, 3, 4, 2, 1)
                    .flatten(-2)
                    .contiguous()
                )
            else:
                raise ValueError("unknown real Reader execution mode")
        if self.match_input_rms:
            # This is energy matching across real channels; the factor of two
            # is absorbed into epsilon, retaining the previous initial function.
            input_energy = source.float().square().mean(-1, keepdim=True)
            selected = output.unflatten(-1, (2, self.output_modes))[..., : self.rms_reference_modes]
            output_energy = selected.float().square().mean((-1, -2)).unsqueeze(-1)
            epsilon = self.variance_epsilon / 2
            output = output * torch.sqrt((input_energy + epsilon) / (output_energy + epsilon)).to(
                output.dtype
            )
        return output


class RealMemoryProjection(nn.Module):
    """An unrestricted real matrix reading the memory coordinates into features."""

    def __init__(self, source: ComplexLinear) -> None:
        super().__init__()
        self.weight = nn.Parameter(source.packed_weight().detach().clone())

    def forward(self, source: Tensor, *, planar: bool = False) -> Tensor:
        if not planar:
            return functional.linear(source, self.weight)
        first, second = source.unbind(0)
        width = self.weight.shape[-1] // 2
        return functional.linear(first, self.weight[:, :width]) + functional.linear(
            second, self.weight[:, width:]
        )

    def forward_packed(self, source: Tensor) -> Tensor:
        return self(source)


def _real_replacement(child: nn.Module) -> nn.Module | None:
    if type(child) is GatedComplexPostFusion:
        return RealSwiGLU.from_legacy(child)
    if type(child) is PathOnlyCollapse:
        return RealPathCollapse(child)
    if type(child) is FactorizedComplexConv2dReader:
        return RealFactorizedMemoryReader(child)
    if type(child) is ComplexLinear:
        return RealMemoryProjection(child)
    return None


class RealMemoryBackbone(nn.Module):
    """Real feature processing with complex semantics confined to memory recurrence.

    The supplied packed backbone is consumed. Reader and read matrices, feature
    normalization scales, and carry channels become independent real parameters.
    Widths, per-pole latent rank, spatial support, path topology, stem and head
    are unchanged. Initialization copies the original constrained matrices.
    """

    architecture = "va128-real-memory-swiglu-v3-free-io"

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        if not callable(getattr(backbone, "_packed_descriptor", None)):
            raise TypeError("real-memory conversion requires the prepared packed VA128 topology")
        replacements = []
        for parent in backbone.modules():
            for name, child in parent.named_children():
                replacement = _real_replacement(child)
                if replacement is not None:
                    replacements.append((parent, name, replacement))
        if sum(isinstance(m, RealSwiGLU) for _, _, m in replacements) != 11:
            raise ValueError("VA128 conversion requires eleven original gated WL feature blocks")
        for parent, name, replacement in replacements:
            setattr(parent, name, replacement)
        for name in ("stage1", "stage2", "stage3"):
            transition = getattr(backbone, name).augmented
            transition.carry_weight = nn.Parameter(transition.carry_weight.detach().repeat(2, 1))
        self.backbone = backbone

    def forward_features(self, inputs: Tensor) -> Tensor:
        # The packed path is a single real feature vector on CPU and GPU.
        # Coordinate interpretation is local to memory write/scan/read calls.
        topology = cast("_PackedTopology", cast("object", self.backbone))
        state = topology._packed_initial(inputs)
        for resolution, name in ((56, "stage1"), (28, "stage2"), (14, "stage3")):
            state = topology._packed_apply_at(resolution, state)
            state, _ = getattr(self.backbone, name)(state, emit_descriptor=False)
        return topology._packed_apply_at(7, state)

    def forward(self, inputs: Tensor) -> Tensor:
        topology = cast("_PackedTopology", cast("object", self.backbone))
        descriptor = topology._packed_descriptor(inputs, full=False)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return topology.classifier(descriptor.float(), terminal_only=True)[0]
