"""VA64 channel-preserving scan-only complex convolution control.

Preserves all M complex modes and four path slots. No hidden-width compression
or FLOPs target; all other B modules remain unchanged.
"""
from __future__ import annotations

# Snapshot builders are imported after the runner establishes its module path.
# pyright: reportImplicitRelativeImport=false, reportExplicitAny=false

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from lnet.complex_scan_stage import ComplexScanStage
    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock
    from lnet.complex_scan_types import ComplexCarryBasis

SCHEMA = "visionalphabet.va64_channel_preserving_conv.v2"
BASELINE_FLOPS = 5_125_594_962
POLE_PARAMETERS = (
    "damping_logits_x", "damping_logits_y", "log_damping_x", "log_damping_y",
    "phase_x", "phase_y",
)


class FourPathConvolution(nn.Module):
    """Four independent complex depthwise 3x3 kernels, retaining every mode.

    Laplace scan is complex-linear and mode-separable; this replacement keeps
    those properties. Only spatial propagation becomes a learned local kernel.
    No bottleneck, channel projection, bias, extra activation or normalization.
    """

    def __init__(self, modes: int) -> None:
        super().__init__()
        if modes <= 0:
            raise ValueError("convolution modes must be positive")
        self.modes = modes
        self.kernel_real = nn.Conv2d(modes, 4 * modes, 3, padding=1, groups=modes, bias=False)
        self.kernel_imag = nn.Conv2d(modes, 4 * modes, 3, padding=1, groups=modes, bias=False)
        # E[|output|^2] = E[|input|^2] for independent, unit-power input.
        for layer in (self.kernel_real, self.kernel_imag):
            nn.init.normal_(layer.weight, mean=0.0, std=(1.0 / 18.0) ** 0.5)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.modes:
            raise ValueError("convolution requires matching NHWM fields")
        real_nchw = real.permute(0, 3, 1, 2)
        imag_nchw = imag.permute(0, 3, 1, 2)
        output_real = self.kernel_real(real_nchw) - self.kernel_imag(imag_nchw)
        output_imag = self.kernel_real(imag_nchw) + self.kernel_imag(real_nchw)
        batch, _, height, width = output_real.shape
        def unpack(value: Tensor) -> Tensor:
            return value.reshape(batch, self.modes, 4, height, width).permute(0, 3, 4, 2, 1)
        return unpack(output_real), unpack(output_imag)

    def analytic_flops(self, height: int, width: int, batch: int = 1) -> int:
        # Four real convolutions at MAC=2, plus real subtraction/imag addition.
        return batch * height * width * self.modes * 4 * (4 * 9 * 2 + 2)


def directional_energy(real: Tensor, imag: Tensor) -> Tensor:
    """B's directional FP32 abs² -> spatial mean -> log1p, direction-major."""
    if real.shape != imag.shape or real.ndim != 5 or real.shape[-2] != 4:
        raise ValueError("energy requires NHW-4-M fields")
    return torch.log1p((real.float().square() + imag.float().square()).mean((1, 2))).flatten(1)


def coarse_endpoints(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
    """Identical B coarse endpoint order: ++, -+, +-, --."""
    if real.shape != imag.shape or real.ndim != 5 or real.shape[-2] != 4:
        raise ValueError("coarse endpoints require NHW-4-M fields")
    if real.shape[1] % 2 or real.shape[2] % 2:
        raise ValueError("coarse endpoints require even spatial dimensions")
    offsets = ((1, 1), (1, 0), (0, 1), (0, 0))
    def select(value: Tensor) -> Tensor:
        return torch.stack([value[:, y::2, x::2, d, :] for d, (y, x) in enumerate(offsets)], -2)

    return select(real), select(imag)


class ConvolutionStage(nn.Module):
    """Narrow VA64 stage adapter that keeps all existing downstream modules."""

    def __init__(self, source: ComplexScanStage) -> None:
        super().__init__()
        for flag in ("use_cccn_shortcut", "use_pole_aligned_shortcut", "use_zero_gated_pole_aligned_residual"):
            if getattr(source, flag, False):
                raise ValueError("unsupported noncanonical VA64 shortcut")
        self.modes = source.modes
        self.input_modes = source.input_modes
        self.output_modes = source.output_modes
        self.pole_input_projection = source.pole_input_projection
        self.augmented = source.augmented
        self.quadrant_path_mode_combiner = source.quadrant_path_mode_combiner
        self.post_transition_ffn = source.post_transition_ffn
        self.carry_basis: ComplexCarryBasis = source.carry_basis
        if self.output_modes is not None and (
            self.augmented is None or self.carry_basis != "s2d"
            or self.quadrant_path_mode_combiner is None
            or not self.quadrant_path_mode_combiner.collapses_product_paths
            or self.quadrant_path_mode_combiner.requires_full_product_cells
        ):
            raise ValueError("convolution adapter requires the canonical VA64 carry/path topology")
        self.conv_scan = FourPathConvolution(self.modes)
        self.train(source.training)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[tuple[Tensor, Tensor] | None, Tensor]:
        from lnet.complex_scan_stage import complex_carry_coordinates

        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.input_modes:
            raise ValueError("stage requires matching NHWM excitations")
        carry = (real, imag)
        if self.pole_input_projection is not None:
            real, imag = self.pole_input_projection(real, imag)
        paths_real, paths_imag = self.conv_scan(real, imag)
        descriptor = directional_energy(paths_real, paths_imag)
        if self.output_modes is None:
            return None, descriptor
        if self.quadrant_path_mode_combiner is None or self.augmented is None:
            raise RuntimeError("nonterminal convolution stage lost path/transition module")
        coarse_real, coarse_imag = coarse_endpoints(paths_real, paths_imag)
        collapsed_real, collapsed_imag = self.quadrant_path_mode_combiner.forward_packed(coarse_real, coarse_imag)
        carry_real, carry_imag = complex_carry_coordinates(*carry, self.carry_basis)
        projected = self.augmented(
            collapsed_real.squeeze(-2), collapsed_imag.squeeze(-2), carry_real, carry_imag,
        )
        if self.post_transition_ffn is not None:
            projected = self.post_transition_ffn(*projected)
        return projected, descriptor


class ConvolutionSameResolutionBlock(nn.Module):
    """Keep B's reader, path collapse, memory projection and post-fusion."""

    def __init__(self, source: SameResolutionPoleScanBlock) -> None:
        super().__init__()
        self.modes = source.modes
        self.reader = source.reader
        self.path_collapse = source.path_collapse
        self.memory_projection = source.memory_projection
        self.post_fusion = source.post_fusion
        self.conv_scan = FourPathConvolution(source.pole_modes)
        self.train(source.training)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        paths = self.conv_scan(*self.reader(real, imag))
        memory_real, memory_imag = self.path_collapse.forward_packed(*paths)
        memory = (memory_real.squeeze(-2), memory_imag.squeeze(-2))
        if self.memory_projection is not None:
            memory = self.memory_projection(*memory)
        return self.post_fusion(real + memory[0], imag + memory[1])


def replace_scans_(model: nn.Module) -> nn.Module:
    from lnet.complex_scan_stage import ComplexScanStage
    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock

    replaced = 0

    def visit(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, SameResolutionPoleScanBlock):
                setattr(parent, name, ConvolutionSameResolutionBlock(child))
                replaced += 1
            elif isinstance(child, ComplexScanStage):
                setattr(parent, name, ConvolutionStage(child))
                replaced += 1
            else:
                visit(child)

    # B is fully constructed first. Added C parameters never perturb common B
    # tensor initialization or the stream left by B construction.
    with torch.random.fork_rng(devices=[]):
        visit(model)
    if replaced != 12:
        raise RuntimeError(f"expected all 12 VA64 scans, replaced {replaced}")
    if any(name.rsplit(".", 1)[-1] in POLE_PARAMETERS for name, _ in model.named_parameters()):
        raise RuntimeError("unused pole parameters remain in convolution control")
    return model


def build_model(*, prepared: bool = True) -> nn.Module:
    import run_va64_retrain_imagenet1k as baseline

    model = baseline.build_model("B", prepared=False)
    replace_scans_(model)
    if prepared:
        prepare = getattr(getattr(model, "model"), "prepare_for_compiled_training_")
        if not callable(prepare):
            raise TypeError("baseline backbone lost compiled training preparation")
        prepare()
    return model


def definition() -> dict[str, Any]:
    return {
        "schema": SCHEMA, "variant": "Cconv", "kernel_size": 3,
        "mixer": "four_complex_depthwise3x3_mode_preserving",
        "scan_sites": 12, "complex_modes": 80, "input_real_channels": 160,
        "output_real_channels": 640, "bottleneck": False,
        "mode_mixing": False, "complex_linear": True,
        "path_order": ["++", "-+", "+-", "--"],
        "path_semantics": "independent local convolution output slots, not causal scans",
        "readout": "directional_fp32_abs2_spatial_mean_log1p", "head_dim": 320,
        "flops_matched": False, "baseline_flops": BASELINE_FLOPS,
        "limitation": "Local support differs from global Laplace scan; costs reported without matching.",
    }
