"""Resolution-preserving stages built on the canonical associative D4 scan."""

from __future__ import annotations

# This module is the public residual-depth boundary around the existing compact
# pole coefficient builder; duplicating that algebra would be the worse API.
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from .complex_scan_stage import ComplexScanStage
from .pac_complex_layers import (
    ComplexLinear,
    semi_orthogonal_complex_linear_,
)
from .pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from .pac_gated_post_fusion import GatedComplexPostFusion
from .pac_phase_gated_transition import PathOnlyCollapse
from .pac_product_scan_contracts import DEFAULT_EPSILON
from .pac_product_scan_pipeline import run_product_scan_path_collapse_pipeline
from .pac_product_scan_reference import product_scan_full4_reference

if TYPE_CHECKING:
    from .complex_scan_types import ComplexField


D4_PATHS = 4
PoleCoefficients = tuple[Tensor, Tensor, Tensor, Tensor]


class SameResolutionPoleScanBlock(nn.Module):
    """Run the established pole stage without spatial coarsening.

    A coarsening stage merges collapsed pole memory with an S2D carry before
    PostFusion.  At unchanged resolution the corresponding carry is the input
    excitation itself; every other learned operation remains the same.
    """

    def __init__(
        self,
        modes: int,
        *,
        pole_modes: int | None = None,
        reader_rank: int,
        kernel_size: int,
        pole_template: ComplexScanStage,
        post_hidden: int,
    ) -> None:
        super().__init__()
        active_pole_modes = modes if pole_modes is None else pole_modes
        if min(modes, active_pole_modes, reader_rank, kernel_size, post_hidden) <= 0:
            message = "same-resolution pole stage dimensions must be positive"
            raise ValueError(message)
        self.modes = int(modes)
        self.pole_modes = int(active_pole_modes)
        self.reader = FactorizedComplexConv2dReader(
            modes,
            self.pole_modes,
            rank=reader_rank,
            kernel_size=kernel_size,
            normalize_input=True,
            match_input_rms=False,
        )
        if self.modes == self.pole_modes:
            self.reader.initialize_orthogonal_()
        else:
            self.reader.initialize_semi_orthogonal_()
        self.path_collapse = PathOnlyCollapse(self.pole_modes, path_hidden=2 * D4_PATHS)
        self.memory_projection = (
            None
            if self.pole_modes == self.modes
            else ComplexLinear(self.pole_modes, self.modes)
        )
        if self.memory_projection is not None:
            semi_orthogonal_complex_linear_(self.memory_projection)
        self.post_fusion = GatedComplexPostFusion(modes, post_hidden)
        self.pole_scan = ComplexScanStage(
            self.pole_modes,
            maximum_phase=1.0,
            output_modes=None,
            scan_memory_policy="retain",
            damping_min=pole_template.damping_min,
            damping_max=pole_template.damping_max,
        )
        self.copy_pole_initialization_(pole_template)

    @torch.no_grad()
    def copy_pole_initialization_(self, source: ComplexScanStage) -> None:
        """Clone a stage's pole atlas while retaining independent parameters."""
        if source.modes != self.pole_modes:
            message = "same-resolution pole template has an incompatible width"
            raise ValueError(message)
        self.pole_scan.damping_min = source.damping_min
        self.pole_scan.damping_max = source.damping_max
        self.pole_scan.product_gain_normalization = source.product_gain_normalization
        for name in ("damping_logits_x", "damping_logits_y", "phase_x", "phase_y"):
            getattr(self.pole_scan, name).copy_(getattr(source, name))

    def _scan_contract(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[ComplexField, PoleCoefficients, PoleCoefficients]:
        drive_real, drive_imag = self.reader(real, imag)
        if drive_real.shape != drive_imag.shape or drive_real.ndim != 4:
            message = "same-resolution reader emitted incompatible pole drives"
            raise ValueError(message)
        shape = cast("tuple[int, int, int, int]", tuple(drive_real.shape))
        pole_x, pole_y = self.pole_scan.pole_coefficients(shape)
        return (drive_real, drive_imag), pole_x, pole_y

    def collapsed_memory(self, real: Tensor, imag: Tensor) -> ComplexField:
        """Produce collapsed memory through the collapse's declared scan contract."""
        source, pole_x, pole_y = self._scan_contract(real, imag)
        packed_parameters = getattr(self.path_collapse, "packed_parameters", None)
        if callable(packed_parameters):
            active_parameters = cast(
                "tuple[Tensor, Tensor, Tensor, Tensor]",
                packed_parameters(),
            )
            memory_real, memory_imag = run_product_scan_path_collapse_pipeline(
                pole_x,
                pole_y,
                source,
                active_parameters,
                gain_normalization=self.pole_scan.product_gain_normalization,
                memory_policy=self.pole_scan.scan_memory_policy,
                path_swiglu=bool(
                    getattr(self.path_collapse, "path_swiglu", False)
                ),
            )
            return self._project_memory(
                memory_real.squeeze(-2),
                memory_imag.squeeze(-2),
            )
        # Research-only collapse modules without a packed kernel contract retain
        # one explicit, differentiable oracle path.  The canonical PathH4 block
        # always returns above through the fused production pipeline.
        paths = product_scan_full4_reference(
            pole_x,
            pole_y,
            source,
            gain_normalization=self.pole_scan.product_gain_normalization,
            epsilon=DEFAULT_EPSILON,
        )
        return self.collapse_paths(*paths)

    def collapse_paths(self, real: Tensor, imag: Tensor) -> ComplexField:
        """Apply the canonical D4 collapse and remove its singleton path axis."""
        memory_real, memory_imag = self.path_collapse.forward_packed(real, imag)
        return self._project_memory(
            memory_real.squeeze(-2),
            memory_imag.squeeze(-2),
        )

    def _project_memory(self, real: Tensor, imag: Tensor) -> ComplexField:
        if self.memory_projection is None:
            return real, imag
        return self.memory_projection(real, imag)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.ndim != 4 or real.shape[-1] != self.modes:
            message = "same-resolution pole block requires matching NHWM tensors"
            raise ValueError(message)
        memory_real, memory_imag = self.collapsed_memory(real, imag)
        return self.post_fusion(real + memory_real, imag + memory_imag)


__all__ = [
    "D4_PATHS",
    "SameResolutionPoleScanBlock",
]
