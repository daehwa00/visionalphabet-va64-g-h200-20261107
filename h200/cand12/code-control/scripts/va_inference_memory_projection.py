"""Compose memory-projection epilogues with existing Reader/scan candidates."""

from __future__ import annotations

# pyright: reportPrivateUsage=false, reportImplicitRelativeImport=false
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from va_inference_reader_scan import _FusedBlock, _FusedStage, _horizontal, _vertical
from va_inference_scan2d import _Scan2DBlock, _Scan2DStage

from lnet.complex_scan_stage import ComplexScanStage
from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition
from lnet.pac_packed_execution import weighted_s2d_packed
from lnet.pac_product_scan_contracts import gain_kind
from lnet.pac_product_scan_pipeline import (
    run_product_scan_coarse_path_collapse_packed,
    run_product_scan_path_collapse_pipeline_packed,
)
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock
from lnet.pac_triton_memory_projection_inference import (
    ProjectionFusion,
    memory_projection_post_fusion,
)

if TYPE_CHECKING:
    from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
    from lnet.pac_phase_gated_transition import PathOnlyCollapse


class _ProjectionBlock(_FusedBlock):
    memory_projection_fusion: ProjectionFusion = "projection_residual"
    memory_projection_block_rows: int = 16
    memory_projection_output_tile: int = 64
    memory_projection_num_warps: int = 4
    projection_reader_scan_enabled: bool = False
    projection_scan2d_enabled: bool = False
    scan2d_block_modes: int = 2
    scan2d_num_warps: int = 4
    scan2d_max_size: int = 16

    def forward(self, real: Tensor, imag: Tensor | None = None) -> Tensor:
        if imag is not None or torch.is_grad_enabled():
            raise RuntimeError("memory projection fusion requires packed inference")
        if self.projection_scan2d_enabled and max(real.shape[1:3]) <= self.scan2d_max_size:
            from lnet.pac_triton_scan2d_inference import run_product_scan_2d_inference

            drive = self.reader(real)
            pole_x, pole_y = self.pole_scan.pole_coefficients(
                (drive.shape[0], drive.shape[1], drive.shape[2], self.pole_modes),
            )
            memory = run_product_scan_2d_inference(
                pole_x, pole_y, drive, self.path_collapse.packed_parameters(),
                gain_normalization=self.pole_scan.product_gain_normalization,
                path_swiglu=bool(getattr(self.path_collapse, "path_swiglu", False)),
                planar_output=True,
                block_modes=self.scan2d_block_modes, num_warps=self.scan2d_num_warps,
                experimental_large_grid=self.scan2d_max_size > 16,
            )
        elif self.projection_reader_scan_enabled:
            pole_x, pole_y = self.pole_scan.pole_coefficients(
                (real.shape[0], real.shape[1], real.shape[2], self.pole_modes),
            )
            horizontal = _horizontal(self.reader, real, pole_x, self.reader_scan_fusion)
            memory = _vertical(
                horizontal, [*pole_x, *pole_y], list(self.path_collapse.packed_parameters()),
                gain_kind(self.pole_scan.product_gain_normalization), "path",
                path_swiglu=bool(getattr(self.path_collapse, "path_swiglu", False)),
                planar_output=True,
            )[0]
        else:
            drive = self.reader(real)
            pole_x, pole_y = self.pole_scan.pole_coefficients(
                (drive.shape[0], drive.shape[1], drive.shape[2], self.pole_modes),
            )
            memory = run_product_scan_path_collapse_pipeline_packed(
                pole_x, pole_y, drive, self.path_collapse.packed_parameters(),
                gain_normalization=self.pole_scan.product_gain_normalization,
                memory_policy=self.pole_scan.scan_memory_policy,
                path_swiglu=bool(getattr(self.path_collapse, "path_swiglu", False)),
                planar_output=True,
            )
        if self.memory_projection is None:
            raise RuntimeError("projection candidate requires a memory projection")
        return memory_projection_post_fusion(
            memory, real, self.memory_projection, self.post_fusion,
            fusion=self.memory_projection_fusion,
            block_rows=self.memory_projection_block_rows,
            output_tile=self.memory_projection_output_tile,
            num_warps=self.memory_projection_num_warps,
        )


class _ProjectionStage(_FusedStage):
    memory_projection_fusion: ProjectionFusion = "projection_residual"
    memory_projection_block_rows: int = 16
    memory_projection_output_tile: int = 64
    memory_projection_num_warps: int = 4
    projection_reader_scan_enabled: bool = False

    def forward(
        self, real: Tensor, imag: Tensor | None = None, *, emit_descriptor: bool = True,
    ) -> tuple[Tensor | None, Tensor]:
        if imag is not None or torch.is_grad_enabled() or emit_descriptor:
            raise RuntimeError("projection stage requires terminal-only packed inference")
        collapse = cast("PathOnlyCollapse", self.quadrant_path_mode_combiner)
        transition = cast("GatedPoleExcitationS2DTransition", self.augmented)
        reader = cast("FactorizedComplexConv2dReader", self.pole_input_projection)
        if self.projection_reader_scan_enabled:
            pole_x, pole_y = self.pole_coefficients(
                (real.shape[0], real.shape[1], real.shape[2], self.modes),
            )
            horizontal = _horizontal(reader, real, pole_x, self.reader_scan_fusion)
            memory = _vertical(
                horizontal, [*pole_x, *pole_y], list(collapse.packed_parameters()),
                gain_kind(self.product_gain_normalization), "coarse_path",
                path_swiglu=False, planar_output=True,
            )[0]
        else:
            drive = reader(real)
            pole_x, pole_y = self.pole_coefficients(
                (drive.shape[0], drive.shape[1], drive.shape[2], self.modes),
            )
            memory = run_product_scan_coarse_path_collapse_packed(
                pole_x, pole_y, drive, collapse.packed_parameters(),
                gain_normalization=self.product_gain_normalization,
                planar_output=True, memory_policy=self.scan_memory_policy,
            )
        carry = weighted_s2d_packed(real, transition.carry_weight)
        if transition.carry_projection is not None:
            carry = transition.carry_projection.forward_packed(carry)
        if transition.memory_projection is None:
            raise RuntimeError("projection candidate requires a memory projection")
        state = memory_projection_post_fusion(
            memory, carry, transition.memory_projection, transition.post_fusion,
            fusion=self.memory_projection_fusion,
            block_rows=self.memory_projection_block_rows,
            output_tile=self.memory_projection_output_tile,
            num_warps=self.memory_projection_num_warps,
        )
        return state, pole_x[0].new_empty((0,))


def use_inference_projection(
    model: nn.Module, fusion: ProjectionFusion, *, block_rows: int = 16,
    output_tile: int = 64, num_warps: int = 4,
) -> int:
    """Install on an inference snapshot after its Reader/scan installer.

    Returns the count of patched blocks/stages. Terminal descriptor stages and
    transitions with specialized merge equations are left to their own forward.
    Parameters, buffers and Reader fusion choices are preserved.
    """
    if fusion not in ("projection_residual", "projection_residual_norm"):
        raise ValueError("unknown memory projection fusion")
    if block_rows not in (16, 32) or output_tile not in (32, 64, 128) or num_warps not in (4, 8):
        raise ValueError("unsupported projection launch geometry")
    changed = 0
    for module in model.modules():
        if type(module) in (SameResolutionPoleScanBlock, _FusedBlock, _Scan2DBlock):
            if module.memory_projection is None:
                continue
            scan2d_enabled = type(module) is _Scan2DBlock
            reader_enabled = type(module) is _FusedBlock or (
                scan2d_enabled
                and getattr(module.scan2d_fallback, "__func__", None) is _FusedBlock.forward
            )
            cast("nn.Module", module).__class__ = _ProjectionBlock
            cast("_ProjectionBlock", module).projection_scan2d_enabled = scan2d_enabled
        elif type(module) in (ComplexScanStage, _FusedStage, _Scan2DStage):
            transition = module.augmented
            if (
                module.output_modes is None
                or type(transition) is not GatedPoleExcitationS2DTransition
                or transition.memory_projection is None
            ):
                continue
            reader_enabled = type(module) is _FusedStage or (
                type(module) is _Scan2DStage
                and getattr(module.scan2d_fallback, "__func__", None) is _FusedStage.forward
            )
            cast("nn.Module", module).__class__ = _ProjectionStage
        else:
            continue
        active = cast("_ProjectionBlock | _ProjectionStage", module)
        active.memory_projection_fusion = fusion
        active.memory_projection_block_rows = block_rows
        active.memory_projection_output_tile = output_tile
        active.memory_projection_num_warps = num_warps
        active.projection_reader_scan_enabled = reader_enabled
        changed += 1
    return changed
