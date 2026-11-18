"""VA64 scan-only convolution control, retaining the B reader and readout.

Each scan is replaced by an unconstrained real linear 1x1 -> depthwise 3x3 ->
1x1 convolution. Four independent output slots preserve the existing D4 path
interface; they do not impose causal or directional support. Complex inputs
are only an interface representation: the convolution sees 2M real channels.
There are no Laplace coefficients, recurrence, phase gates or pole gains.
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

SCHEMA = "visionalphabet.va64_scan_only_conv.v1"
BASELINE_FLOPS = 5_125_594_962
DEFAULT_HIDDEN_CHANNELS = 8
POLE_PARAMETERS = (
    "damping_logits_x", "damping_logits_y", "log_damping_x", "log_damping_y",
    "phase_x", "phase_y",
)


class FourPathConvolution(nn.Module):
    """Linear real bottleneck convolution emitting NHW-path-mode split pairs."""

    def __init__(self, modes: int, hidden_channels: int) -> None:
        super().__init__()
        if modes <= 0 or hidden_channels <= 0:
            raise ValueError("convolution dimensions must be positive")
        self.modes = modes
        self.hidden_channels = hidden_channels
        self.input_projection = nn.Conv2d(2 * modes, hidden_channels, 1, bias=False)
        self.spatial = nn.Conv2d(
            hidden_channels, hidden_channels, 3, padding=1,
            groups=hidden_channels, bias=False,
        )
        self.output_projection = nn.Conv2d(hidden_channels, 8 * modes, 1, bias=False)
        # A chain of three linear Kaiming-uniform layers attenuates variance by
        # 1/27. Initialize unit expected variance at each map instead. This is
        # initialization only; no runtime gain or input-dependent scaling.
        for layer in (self.input_projection, self.spatial, self.output_projection):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.modes:
            raise ValueError("convolution requires matching NHWM fields")
        values = torch.cat((real, imag), dim=-1).permute(0, 3, 1, 2)
        output = self.output_projection(self.spatial(self.input_projection(values)))
        batch, _, height, width = output.shape
        packed = output.permute(0, 2, 3, 1).reshape(batch, height, width, 4, 2, self.modes)
        return packed[..., 0, :], packed[..., 1, :]

    def analytic_flops(self, height: int, width: int, batch: int = 1) -> int:
        """MAC=2; packing and permutations cost zero; no bias/activation."""
        channels = self.hidden_channels
        return batch * height * width * 2 * channels * (10 * self.modes + 9)


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

    def __init__(self, source: ComplexScanStage, hidden_channels: int) -> None:
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
        self.conv_scan = FourPathConvolution(self.modes, hidden_channels)
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

    def __init__(self, source: SameResolutionPoleScanBlock, hidden_channels: int) -> None:
        super().__init__()
        self.modes = source.modes
        self.reader = source.reader
        self.path_collapse = source.path_collapse
        self.memory_projection = source.memory_projection
        self.post_fusion = source.post_fusion
        self.conv_scan = FourPathConvolution(source.pole_modes, hidden_channels)
        self.train(source.training)

    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        paths = self.conv_scan(*self.reader(real, imag))
        memory_real, memory_imag = self.path_collapse.forward_packed(*paths)
        memory = (memory_real.squeeze(-2), memory_imag.squeeze(-2))
        if self.memory_projection is not None:
            memory = self.memory_projection(*memory)
        return self.post_fusion(real + memory[0], imag + memory[1])


def replace_scans_(model: nn.Module, hidden_channels: int) -> nn.Module:
    from lnet.complex_scan_stage import ComplexScanStage
    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock

    replaced = 0

    def visit(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, SameResolutionPoleScanBlock):
                setattr(parent, name, ConvolutionSameResolutionBlock(child, hidden_channels))
                replaced += 1
            elif isinstance(child, ComplexScanStage):
                setattr(parent, name, ConvolutionStage(child, hidden_channels))
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


def build_model(hidden_channels: int = DEFAULT_HIDDEN_CHANNELS, *, prepared: bool = True) -> nn.Module:
    import run_va64_retrain_imagenet1k as baseline

    model = baseline.build_model("B", prepared=False)
    replace_scans_(model, hidden_channels)
    if prepared:
        prepare = getattr(getattr(model, "model"), "prepare_for_compiled_training_")
        if not callable(prepare):
            raise TypeError("baseline backbone lost compiled training preparation")
        prepare()
    return model


def definition(hidden_channels: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "variant": "C", "hidden_channels": hidden_channels,
        "mixer": "real_1x1_depthwise3x3_1x1_no_bias_no_added_activation",
        "scan_sites": 12, "input_real_channels": 160, "output_real_channels": 640,
        "path_order": ["++", "-+", "+-", "--"],
        "path_semantics": "independent convolution output slots, not causal scans",
        "readout": "directional_fp32_abs2_spatial_mean_log1p", "head_dim": 320,
        "baseline_flops": BASELINE_FLOPS, "flops_tolerance_fraction": 0.05,
        "limitation": "Compute is matched, not receptive field or parameter count; linear local convolution control.",
    }
