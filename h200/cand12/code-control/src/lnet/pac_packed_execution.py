"""Packed-real execution boundaries for the canonical VA128 blocks."""

from __future__ import annotations

# Type-only references describe modules that dispatch here through local imports.
# pyright: reportImportCycles=false
from typing import TYPE_CHECKING, cast

from .pac_product_scan_pipeline import (
    run_product_scan_coarse_path_collapse_packed,
    run_product_scan_path_collapse_pipeline_packed,
    run_product_scan_pipeline_packed,
)

if TYPE_CHECKING:
    from torch import Tensor

    from .complex_scan_stage import ComplexScanStage
    from .pac_gated_post_fusion import GatedPoleExcitationS2DTransition
    from .pac_phase_gated_transition import PathOnlyCollapse
    from .pac_same_resolution_depth import SameResolutionPoleScanBlock


def weighted_s2d_packed(source: Tensor, weight: Tensor) -> Tensor:
    """Reduce physical 2x2 cells directly, without an expanded S2D copy.

    The reshape is a view. Inductor fuses multiplication and the four-cell
    reduction; there is no permute/contiguous materialization of 4C channels.
    """
    batch, height, width, channels = source.shape
    if height % 2 == 0 and width % 2 == 0 and weight.shape == (channels, 4):
        cells = source.reshape(batch, height // 2, 2, width // 2, 2, channels)
        coefficients = weight.transpose(0, 1).reshape(1, 1, 2, 1, 2, channels)
        return (cells * coefficients).sum(dim=(2, 4))
    modes = channels // 2
    if height % 2 or width % 2 or channels % 2 or weight.shape != (modes, 4):
        raise ValueError("packed S2D carry requires even spatial sizes and C-by-4 weights")
    cells = source.reshape(batch, height // 2, 2, width // 2, 2, 2, modes)
    coefficients = weight.transpose(0, 1).reshape(1, 1, 2, 1, 2, 1, modes)
    return (cells * coefficients).sum(dim=(2, 4)).flatten(-2)


def same_resolution_packed(block: SameResolutionPoleScanBlock, source: Tensor) -> Tensor:
    drive = block.reader(source)
    pole_x, pole_y = block.pole_scan.pole_coefficients((*drive.shape[:-1], block.pole_modes))
    memory = run_product_scan_path_collapse_pipeline_packed(
        pole_x,
        pole_y,
        drive,
        block.path_collapse.packed_parameters(),
        gain_normalization=block.pole_scan.product_gain_normalization,
        memory_policy=block.pole_scan.scan_memory_policy,
        path_swiglu=bool(getattr(block.path_collapse, "path_swiglu", False)),
        planar_output=block.memory_projection is not None,
    )
    if block.memory_projection is not None:
        memory = block.memory_projection(memory, planar=True)
    return block.post_fusion.forward_residual_packed(source, memory)


def stage_packed(
    stage: ComplexScanStage,
    source: Tensor,
    *,
    emit_descriptor: bool,
) -> tuple[Tensor | None, Tensor]:
    if stage.pole_input_projection is None:
        raise ValueError("packed stage requires a reader")
    drive = stage.pole_input_projection(source)
    pole_x, pole_y = stage.pole_coefficients((*drive.shape[:-1], stage.modes))
    if stage.output_modes is None:
        descriptor = run_product_scan_pipeline_packed(
            pole_x, pole_y, drive,
            epilogue="mean_energy" if stage.readout_mode == "mean_energy" else "descriptor",
            gain_normalization=stage.product_gain_normalization,
            memory_policy=stage.scan_memory_policy,
        )
        return None, cast("Tensor", descriptor)
    collapse = cast("PathOnlyCollapse", stage.quadrant_path_mode_combiner)
    transition = cast("GatedPoleExcitationS2DTransition", stage.augmented)
    memory_planar = False
    if (
        getattr(stage, "_fuse_coarse_path", False)
        and not emit_descriptor
    ):
        memory_planar = transition.memory_projection is not None
        memory = run_product_scan_coarse_path_collapse_packed(
            pole_x, pole_y, drive, collapse.packed_parameters(),
            gain_normalization=stage.product_gain_normalization,
            planar_output=memory_planar,
            memory_policy=stage.scan_memory_policy,
        )
        descriptor = pole_x[0].new_empty((0,))
    else:
        coarse, descriptor = run_product_scan_pipeline_packed(
            pole_x, pole_y, drive, epilogue="coarse",
            gain_normalization=stage.product_gain_normalization,
            memory_policy=stage.scan_memory_policy, emit_descriptor=emit_descriptor,
        )
        memory = collapse(coarse)
    carry = weighted_s2d_packed(source, transition.carry_weight)
    state = transition(memory, carry_real=carry, memory_planar=memory_planar)
    return state, descriptor
