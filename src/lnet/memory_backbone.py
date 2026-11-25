"""Topology-preserving real feature processing around the existing complex memory."""

from __future__ import annotations

# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
import math
import os
from typing import Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from .complex_layers import ComplexLinear
from .factorized_complex_scan_reader import FactorizedComplexConv2dReader
from .factorized_reader_kernels import real_gram_kernel_energy, real_packed_spatial
from .gated_post_fusion import GatedComplexPostFusion
from .model_w28c import model_w28c_enabled, packed_swiglu
from .model_w29c import (
    _CompiledChannelRMSNorm,
    channel_rmsnorm,
    model_w29c_enabled,
    packed_swiglu_vec2d,
    rmsnorm_eligible,
)
from .model_w36b import inductor_norms_enabled
from .phase_gated_transition import PathOnlyCollapse
from .product_scan_reference import d4_grouped_path_collapse_packed_reference

_FUSED_FACTORIZED_READER_ENV = "LNET_FUSED_FACTORIZED_READER"
_GENERAL_SCANS_ENV = "LNET_GENERAL_SCANS"
_READER_EXEC_MODE_ENV = "LNET_READER_EXEC_MODE"

_MAX_FUSED_GRAM_RANK = 8

# Approximate peak tensor-core and scalar FP32 FLOP/byte by SM architecture.
_READER_RIDGE_POINTS = {
    (7, 0): (140.0, 17.0),
    (7, 5): (200.0, 13.0),
    (8, 0): (156.0, 10.0),
    (8, 6): (185.0, 45.0),
    (8, 9): (325.0, 81.0),
    (9, 0): (206.0, 14.0),
    (10, 0): (280.0, 10.0),
}


def reader_roofline_costs(
    batch: int,
    height: int,
    width: int,
    channels: int,
    modes: int,
    rank: int,
    kernel: int,
    dtype: torch.dtype,
    major: int,
    minor: int,
    multiprocessors: int,
) -> tuple[float, float]:
    """Return dense and factorized train costs in equivalent HBM bytes per pixel."""
    outputs = 2 * modes
    latent = rank * modes
    taps = kernel**2
    dense_flops = 6 * channels * outputs * taps
    point_flops = 6 * channels * latent
    spatial_flops = 12 * latent * taps
    tensor_cores = (dtype == torch.bfloat16 and major >= 8) or (
        dtype == torch.float16 and major >= 7
    )
    if not tensor_cores:
        factorized_flops = point_flops + spatial_flops
        if kernel == 1 or dense_flops <= factorized_flops:
            return float(dense_flops), math.inf
        return math.inf, float(factorized_flops)

    fallback = (156.0, 14.0) if major >= 9 else (140.0, 17.0)
    tensor_ridge, scalar_ridge = _READER_RIDGE_POINTS.get((major, minor), fallback)
    tiles = ((batch * height * width + 63) // 64) * ((outputs + 63) // 64)
    utilization = min(1.0, max(1, tiles) / max(1, multiprocessors))
    alignment = 1.0 if channels % 8 == outputs % 8 == 0 else 0.5
    dense_ridge = tensor_ridge * 0.65 * alignment * utilization

    traffic = 3 * dtype.itemsize
    dense = max(traffic * (channels + outputs), dense_flops / dense_ridge)
    point = max(traffic * (channels + latent), point_flops / tensor_ridge)
    # Neighborhood rereads and partial spatial weight gradients lower effective bandwidth.
    spatial = max(traffic * (latent + outputs) / 0.18, spatial_flops / scalar_ridge)
    return dense, point + spatial


@torch.compiler.assume_constant_result
def _reader_cuda_properties(index: int | None) -> tuple[int, int, int]:
    props = torch.cuda.get_device_properties(index)
    return props.major, props.minor, props.multi_processor_count


def _fused_factorized_reader_enabled() -> bool:
    return os.environ.get(_FUSED_FACTORIZED_READER_ENV) != "0"


class _PackedTopology(Protocol):
    classifier: nn.Module

    def _packed_initial(self, inputs: Tensor) -> Tensor: ...
    def _packed_apply_at(self, resolution: int, state: Tensor) -> Tensor: ...
    def _packed_descriptor(self, inputs: Tensor, *, full: bool) -> Tensor: ...


class ChannelRMSNorm(nn.Module):
    """Ordinary real RMSNorm with one independently learned scale per channel."""

    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        if width <= 0 or epsilon <= 0:
            raise ValueError("invalid real RMSNorm dimensions or epsilon")
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        if inductor_norms_enabled():
            return functional.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.epsilon)
        if x.is_cuda and (
            _fused_factorized_reader_enabled()
            or os.environ.get(_GENERAL_SCANS_ENV) != "0"
        ):
            weight = self.weight.to(x.dtype)
            wave29 = (
                x.dtype in (torch.bfloat16, torch.float16)
                and x.shape[-1] == weight.numel()
                and x.numel() > 0
                and model_w29c_enabled(x.device, "rmsnorm_bwd")
            )
            if (
                wave29 and model_w29c_enabled(x.device, "rmsnorm_full")
                and rmsnorm_eligible(x, weight)
            ):
                save_inverse = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
                return channel_rmsnorm(x, weight, self.epsilon, save_inverse)[0]
            if wave29 and torch.is_grad_enabled():
                return _CompiledChannelRMSNorm.apply(x, weight, self.epsilon, True)  # noqa: FBT003
            if (
                torch.compiler.is_compiling()
                and torch.is_grad_enabled()
                and x.dtype in (torch.bfloat16, torch.float16)
                and x.shape[-1] == weight.numel()
                and x.numel() > 0
            ):
                return _CompiledChannelRMSNorm.apply(x, weight, self.epsilon, False)  # noqa: FBT003
            return functional.rms_norm(x, (x.shape[-1],), weight, self.epsilon)
        inverse = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.epsilon).to(x.dtype)
        return x * inverse * self.weight.to(x.dtype)


class SwiGLU(nn.Module):
    """Real value/gate/output projections; one independent gate per hidden channel."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.width = width
        self.hidden = hidden
        self.norm = ChannelRMSNorm(width)
        self.value_gate = nn.Linear(width, 2 * hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)

    @classmethod
    def from_legacy(cls, source: GatedComplexPostFusion) -> SwiGLU:
        """Convert WL matrices exactly, then untie the two formerly shared gates.

        Gate rows start with identical values so conversion preserves the
        initial function. They are independent parameters thereafter.
        """
        width, hidden = 2 * source.modes, 2 * source.hidden_modes
        # Construction must not perturb later initialization/augmentation RNG.
        with torch.random.fork_rng(devices=[]):
            result = cls(width, hidden)
        result.norm = ChannelRMSNorm(width, source.norm.epsilon / 2.0)
        result.to(device=source.gate.weight.device, dtype=source.gate.weight.dtype)
        with torch.no_grad():
            result.norm.weight.copy_(source.norm.weight.repeat(2) / math.sqrt(2))
            result.value_gate.weight[:hidden].copy_(source.value.packed_weight())
            result.value_gate.weight[hidden:].copy_(source.gate.weight.repeat(2, 1))
            result.output.weight.copy_(source.out.packed_weight())
        result.train(source.training)
        return result

    def forward(self, x: Tensor) -> Tensor:
        joint = self.value_gate(self.norm(x))
        if (
            joint.is_cuda
            and joint.dtype in (torch.bfloat16, torch.float16)
            and joint.is_contiguous()
            and 0 < joint.numel() < 2**31
        ):
            if model_w29c_enabled(joint.device, "swiglu_vec2d"):
                return x + self.output(packed_swiglu_vec2d(joint))
            if self.hidden % 16 == 0 and model_w28c_enabled(joint.device, "swiglu"):
                return x + self.output(packed_swiglu(joint))
        value, gate = joint.chunk(2, dim=-1)
        return x + self.output(value * functional.silu(gate))

    def forward_residual_packed(self, left: Tensor, right: Tensor) -> Tensor:
        return self(left + right)


class PathCollapse(nn.Module):
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
        return d4_grouped_path_collapse_packed_reference(source, *self.packed_parameters())

    def forward(self, source: Tensor) -> Tensor:
        return self.forward_real_packed(source)


class FactorizedMemoryReader(nn.Module):
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
        self.input_norm: ChannelRMSNorm | None = None
        if source.input_norm is not None:
            self.input_norm = ChannelRMSNorm(self.input_channels, source.input_norm.epsilon / 2)
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

    def _synthesized_kernel_bmm(self) -> Tensor:
        modes, rank, size = self.output_modes, self.rank, self.kernel_size
        with torch.autocast(device_type=self.point_weight.device.type, enabled=False):
            spatial = self.spatial_weight.float().reshape(modes, 2, rank, size, size)
            spatial = spatial.permute(0, 1, 3, 4, 2).reshape(modes, 2 * size**2, rank)
            full = torch.bmm(spatial, self.point_weight.float())
            return full.reshape(modes, 2, size, size, self.input_channels).movedim(-1, 2)

    def _kernel_and_scale(self, *, bmm: bool = False) -> tuple[Tensor, Tensor]:
        full = self._synthesized_kernel_bmm() if bmm else self.synthesized_kernel()
        inverse = torch.rsqrt(
            full.square().sum((2, 3, 4), keepdim=True).clamp_min(self.variance_epsilon)
        )
        return full, inverse

    @staticmethod
    def _dense_compute_dtype(source: Tensor) -> torch.dtype:
        if source.is_cuda and source.dtype == torch.float32 and torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
        return source.dtype

    def _normalized_kernel(self, *, bmm: bool = False) -> Tensor:
        full, inverse = self._kernel_and_scale(bmm=bmm)
        return (full * inverse).permute(1, 0, 2, 3, 4).reshape(
            2 * self.output_modes, self.input_channels, self.kernel_size, self.kernel_size
        )

    def _dense_nhwc(self, source: Tensor, *, bmm: bool) -> Tensor:
        dtype = self._dense_compute_dtype(source)
        source_in = source.to(dtype=dtype, memory_format=torch.contiguous_format).contiguous()
        source_cl = source_in.movedim(-1, 1)
        kernel_cl = self._normalized_kernel(bmm=bmm).to(
            dtype=dtype, memory_format=torch.channels_last
        )
        with torch.autocast(device_type="cuda", enabled=False):
            output = functional.conv2d(source_cl, kernel_cl, padding=self.padding).movedim(1, -1)
        return output if output.is_contiguous() else output.contiguous()

    def _reader_execution_mode(self, source: Tensor) -> str:
        mode = os.environ.get(_READER_EXEC_MODE_ENV, "off")
        if mode in {"", "off"}:
            return self.execution_mode
        if mode == "auto_roofline":
            if not source.is_cuda:
                return self.execution_mode
            dense, factorized = reader_roofline_costs(
                *source.shape[:3], self.input_channels, self.output_modes, self.rank,
                self.kernel_size, self._dense_compute_dtype(source),
                *_reader_cuda_properties(source.device.index),
            )
            return "dense_nhwc" if dense < factorized else "factorized_triton"
        if mode not in {"dense_nhwc", "dense_nhwc_bmm"}:
            raise ValueError(f"unknown LNET_READER_EXEC_MODE: {mode!r}")
        return mode

    def forward(self, source: Tensor) -> Tensor:
        if source.ndim != 4 or source.shape[-1] != self.input_channels:
            raise ValueError("real Reader requires NHWC features of the configured width")
        if self.input_norm is not None:
            source = self.input_norm(source)
        mode = self._reader_execution_mode(source)
        if mode in {"dense_nhwc", "dense_nhwc_bmm"}:
            output = self._dense_nhwc(source, bmm=mode == "dense_nhwc_bmm")
        elif not source.is_cuda or mode == "dense":
            kernel = self._normalized_kernel()
            output = (
                functional.conv2d(source.movedim(-1, 1), kernel, padding=self.padding)
                .movedim(1, -1)
                .contiguous()
            )
        else:
            fused = (
                mode == "factorized_triton"
                and _fused_factorized_reader_enabled()
            )
            point = self.point_weight.transpose(0, 1) if fused else self.point_weight
            projected = functional.linear(source, point.flatten(0, 1))
            if fused:
                if self.rank <= _MAX_FUSED_GRAM_RANK:
                    from .factorized_reader_kernels import (  # noqa: PLC0415
                        real_gram_normalize_spatial,
                    )

                    spatial = real_gram_normalize_spatial(
                        self.point_weight, self.spatial_weight, self.variance_epsilon,
                        projected.dtype,
                    )
                else:
                    energy = real_gram_kernel_energy(self.point_weight, self.spatial_weight)
                    inverse = energy.clamp_min(self.variance_epsilon).rsqrt()
                    shape = self.output_modes, 2, self.rank, self.kernel_size, self.kernel_size
                    spatial = (
                        self.spatial_weight.float().reshape(shape) * inverse[..., None, None, None]
                    ).transpose(0, 1).to(projected.dtype).contiguous()
                output = real_packed_spatial(projected, spatial, component_first=True)
            elif mode in {"factorized", "factorized_triton"}:
                _, inverse = self._kernel_and_scale()
                spatial = self.spatial_weight.float().reshape(
                    self.output_modes, 2, self.rank, self.kernel_size, self.kernel_size
                )
                spatial = (spatial * inverse).to(projected.dtype).contiguous()
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


class MemoryReadProjection(nn.Module):
    """An unrestricted real matrix reading the memory coordinates into features."""

    def __init__(self, source: ComplexLinear) -> None:
        super().__init__()
        self.weight = nn.Parameter(source.packed_weight().detach().clone())

    def forward_planes(self, first: Tensor, second: Tensor) -> Tensor:
        """Project separate contiguous real and imaginary planes into features."""
        width = self.weight.shape[-1] // 2
        return functional.linear(first, self.weight[:, :width]) + functional.linear(
            second, self.weight[:, width:]
        )

    def forward(
        self,
        source: Tensor,
        imag: Tensor | None = None,
        *,
        planar: bool = False,
    ) -> Tensor:
        if imag is not None:
            return self.forward_planes(source, imag)
        if not planar:
            return functional.linear(source, self.weight)
        first, second = source.unbind(0)
        return self.forward_planes(first, second)

    def forward_packed(self, source: Tensor) -> Tensor:
        return self(source)


def _real_replacement(child: nn.Module) -> nn.Module | None:
    if type(child) is GatedComplexPostFusion:
        return SwiGLU.from_legacy(child)
    if type(child) is PathOnlyCollapse:
        return PathCollapse(child)
    if type(child) is FactorizedComplexConv2dReader:
        return FactorizedMemoryReader(child)
    if type(child) is ComplexLinear:
        return MemoryReadProjection(child)
    return None


class MemoryBackbone(nn.Module):
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
        if sum(isinstance(m, SwiGLU) for _, _, m in replacements) != 11:
            raise ValueError("VA128 conversion requires eleven original gated WL feature blocks")
        for parent, name, replacement in replacements:
            setattr(parent, name, replacement)
        for name in ("stage1", "stage2", "stage3"):
            transition = getattr(backbone, name).augmented
            transition.carry_weight = nn.Parameter(transition.carry_weight.detach().repeat(2, 1))
        self.backbone = backbone

    def forward_features(self, inputs: Tensor) -> Tensor:
        return self.forward_feature_scales(inputs)[1]

    def forward_feature_scales(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        topology = cast("_PackedTopology", cast("object", self.backbone))
        state = topology._packed_initial(inputs)
        for resolution, name in ((56, "stage1"), (28, "stage2")):
            state = topology._packed_apply_at(resolution, state)
            state, _ = getattr(self.backbone, name)(state, emit_descriptor=False)
        state_s3 = topology._packed_apply_at(14, state)
        state_s4, _ = self.backbone.stage3(state_s3, emit_descriptor=False)
        return state_s3, topology._packed_apply_at(7, state_s4)

    def forward(self, inputs: Tensor) -> Tensor:
        topology = cast("_PackedTopology", cast("object", self.backbone))
        descriptor = topology._packed_descriptor(inputs, full=False)
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return topology.classifier(descriptor.float(), terminal_only=True)[0]
