"""Inference-only execution boundary for Reader/horizontal-scan fusion candidates."""
from __future__ import annotations

# ruff: noqa: FBT001, FBT003
# pyright: reportPrivateUsage=false, reportImplicitRelativeImport=false
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn
from torch.library import triton_op

from lnet.complex_scan_stage import ComplexScanStage
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_packed_execution import weighted_s2d_packed
from lnet.pac_product_scan_contracts import DEFAULT_EPSILON, gain_kind
from lnet.pac_product_scan_normalization import static_product_scan_auxiliary
from lnet.pac_product_scan_ops import (
    COARSE_PATH_FORWARD_LAUNCH_NAME,
    DESCRIPTOR_FORWARD_LAUNCH_NAME,
    PATH_COLLAPSE_FORWARD_LAUNCH_NAME,
    _launch_product_scan4_forward,
)
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock

if TYPE_CHECKING:
    from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition
    from lnet.pac_phase_gated_transition import PathOnlyCollapse

Fusion = Literal["spatial_scan", "rms_scan", "scaled_scan"]
Poles = tuple[Tensor, Tensor, Tensor, Tensor]


@triton_op("lnet::inference_vertical_from_horizontal", mutates_args={})
def _vertical(
    horizontal: Tensor, poles: list[Tensor], path: list[Tensor],
    kind: int, epilogue: str, path_swiglu: bool, planar_output: bool,
) -> list[Tensor]:
    pole_x = cast("Poles", tuple(poles[:4]))
    pole_y = cast("Poles", tuple(poles[4:]))
    auxiliary = static_product_scan_auxiliary(
        pole_x, pole_y, horizontal[0], epsilon=DEFAULT_EPSILON, gain_kind=kind,
    )
    planes = cast("Poles", tuple(horizontal.unbind(0)))
    packed, _, descriptor = _launch_product_scan4_forward(
        *pole_y, *planes, *auxiliary, DEFAULT_EPSILON, kind,
        emit_coarse=False,
        path_collapse=cast("Poles", tuple(path)) if path else None,
        path_swiglu=path_swiglu,
        emit_descriptor=not path,
        packed_output=True,
        planar_output=planar_output,
        mean_direction_energy=epilogue == "mean_energy",
        coarse_path=epilogue == "coarse_path",
        launch_name=(COARSE_PATH_FORWARD_LAUNCH_NAME if epilogue == "coarse_path"
                     else PATH_COLLAPSE_FORWARD_LAUNCH_NAME if path
                     else DESCRIPTOR_FORWARD_LAUNCH_NAME),
    )
    return [packed, descriptor]


def _horizontal(reader: FactorizedComplexConv2dReader, source: Tensor, poles: Poles,
                fusion: Fusion) -> Tensor:
    from lnet.pac_triton_reader_scan_inference import (
        reader_rms_scan_inference,
        reader_scaled_scan_inference,
        reader_spatial_scan_inference,
    )

    packed = source
    if reader.input_norm is not None:
        real, imag = packed.chunk(2, dim=-1)
        energy = (real.float().square() + imag.float().square()).mean(-1, keepdim=True)
        inverse = torch.rsqrt(energy + reader.input_norm.epsilon).to(packed.dtype)
        packed = (packed.unflatten(-1, (2, reader.input_modes)) * inverse.unsqueeze(-1)
                  * reader.input_norm.weight.to(packed.dtype)).flatten(-2)
    input_energy = packed.float().square().sum(-1, keepdim=True).div(reader.input_modes)
    if fusion == "spatial_scan":
        projected = torch.nn.functional.linear(packed, reader._pointwise_kernel()).movedim(-1, 1)
        real, imag = reader._normalized_spatial_factors(projected.dtype)
        return reader_spatial_scan_inference(
            projected, real, imag, input_energy, poles,
            match_input_rms=reader.match_input_rms,
            rms_reference_modes=reader.rms_reference_modes, epsilon=reader.variance_epsilon,
        )
    spatial = reader._factorized_triton_conv_packed(packed)
    scan = reader_scaled_scan_inference if fusion == "scaled_scan" else reader_rms_scan_inference
    return scan(
        spatial, input_energy, poles, match_input_rms=reader.match_input_rms,
        rms_reference_modes=reader.rms_reference_modes, epsilon=reader.variance_epsilon,
    )


class _FusedBlock(SameResolutionPoleScanBlock):
    reader_scan_fusion: Fusion = "spatial_scan"

    def forward(self, real: Tensor, imag: Tensor | None = None) -> Tensor:
        if imag is not None or torch.is_grad_enabled():
            raise RuntimeError("fused Reader/scan is packed inference only")
        pole_x, pole_y = self.pole_scan.pole_coefficients((real.shape[0], real.shape[1], real.shape[2], self.pole_modes))
        horizontal = _horizontal(self.reader, real, pole_x, self.reader_scan_fusion)
        memory = _vertical(
            horizontal, [*pole_x, *pole_y], list(self.path_collapse.packed_parameters()),
            gain_kind(self.pole_scan.product_gain_normalization), "path",
            bool(getattr(self.path_collapse, "path_swiglu", False)),
            self.memory_projection is not None,
        )[0]
        if self.memory_projection is not None:
            memory = self.memory_projection(memory, planar=True)
        return self.post_fusion.forward_residual_packed(real, memory)


class _FusedStage(ComplexScanStage):
    reader_scan_fusion: Fusion = "spatial_scan"

    def forward(self, real: Tensor, imag: Tensor | None = None, *,
                emit_descriptor: bool = True) -> tuple[Tensor | None, Tensor]:
        if imag is not None or torch.is_grad_enabled():
            raise RuntimeError("fused Reader/scan is packed inference only")
        if self.output_modes is not None and emit_descriptor:
            raise ValueError("fused transition requires terminal-only inference")
        reader = cast("FactorizedComplexConv2dReader", self.pole_input_projection)
        pole_x, pole_y = self.pole_coefficients((real.shape[0], real.shape[1], real.shape[2], self.modes))
        horizontal = _horizontal(reader, real, pole_x, self.reader_scan_fusion)
        kind = gain_kind(self.product_gain_normalization)
        if self.output_modes is None:
            epilogue = "mean_energy" if self.readout_mode == "mean_energy" else "descriptor"
            return None, _vertical(horizontal, [*pole_x, *pole_y], [], kind,
                                   epilogue, False, False)[1]
        collapse = cast("PathOnlyCollapse", self.quadrant_path_mode_combiner)
        transition = cast("GatedPoleExcitationS2DTransition", self.augmented)
        planar = transition.memory_projection is not None
        memory = _vertical(horizontal, [*pole_x, *pole_y], list(collapse.packed_parameters()),
                           kind, "coarse_path", False, planar)[0]
        carry = weighted_s2d_packed(real, transition.carry_weight)
        state = transition(memory, carry_real=carry, memory_planar=planar)
        return state, pole_x[0].new_empty((0,))


def enable_reader_scan_fusion(model: nn.Module, fusion: Fusion) -> None:
    """Install on the private prepared inference snapshot only."""
    if fusion not in ("spatial_scan", "rms_scan", "scaled_scan"):
        raise ValueError("unknown Reader/scan fusion")
    for module in model.modules():
        if type(module) is SameResolutionPoleScanBlock:
            module.__class__ = _FusedBlock
            cast("_FusedBlock", module).reader_scan_fusion = fusion
        elif type(module) is ComplexScanStage and isinstance(
            module.pole_input_projection, FactorizedComplexConv2dReader,
        ):
            module.__class__ = _FusedStage
            cast("_FusedStage", module).reader_scan_fusion = fusion
