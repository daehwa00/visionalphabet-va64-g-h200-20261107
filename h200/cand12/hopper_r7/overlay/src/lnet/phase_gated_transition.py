"""Phase-Gated replacements for the mode FFNs in factorized transitions."""

from __future__ import annotations

from torch import Tensor

from .grouped_path_cffn import (
    GroupedWidelyLinear,
    grouped_cartesian_cell_cffn,
    grouped_cartesian_cffn,
    grouped_cartesian_cffn_packed,
)
from .path_cffn import D4PathModeCombiner
from .phase_gated_cffn import PhaseGatedComplexFFN

ComplexField = tuple[Tensor, Tensor]


class PhaseGatedModeResidualPathCollapse(D4PathModeCombiner):
    """Apply Phase-Gated mode mixing, then the established 4-to-1 path block."""

    collapses_product_paths = True

    def __init__(
        self,
        modes: int,
        *,
        mode_hidden: int,
        path_hidden: int = 8,
        self_gated: bool = False,
        unit_row_projections: bool = False,
        residual_scale_max: float | None = None,
    ) -> None:
        super().__init__()
        if min(modes, mode_hidden, path_hidden) <= 0:
            message = "phase-gated stage mixer dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = 4 * modes
        self.mode = PhaseGatedComplexFFN(
            modes,
            mode_hidden,
            self_gated=self_gated,
            unit_row_projections=unit_row_projections,
            residual_scale_max=residual_scale_max,
        )
        self.path_input = GroupedWidelyLinear(modes, 4, path_hidden, bias=True)
        self.path_output = GroupedWidelyLinear(modes, path_hidden, 1, bias=True)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "phase-gated stage mixer inputs have incompatible shapes"
            raise ValueError(message)
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            message = "phase-gated stage mixer requires NHW-path-mode inputs"
            raise ValueError(message)
        mixed_real, mixed_imag = self.mode(source_real, source_imag)
        return grouped_cartesian_cffn(
            mixed_real,
            mixed_imag,
            input_projection=self.path_input,
            output_projection=self.path_output,
        )


class PathOnlyCollapse(D4PathModeCombiner):
    """Keep the established nonlinear D4 collapse without a mode residual."""

    collapses_product_paths = True

    def __init__(
        self,
        modes: int | PhaseGatedModeResidualPathCollapse,
        *,
        path_hidden: int = 8,
    ) -> None:
        super().__init__()
        if isinstance(modes, PhaseGatedModeResidualPathCollapse):
            self.modes = modes.modes
            self.path_count = modes.path_count
            self.output_paths = modes.output_paths
            self.input_modes = modes.input_modes
            self.path_input = modes.path_input
            self.path_output = modes.path_output
            return
        if min(modes, path_hidden) <= 0:
            message = "path-only collapse dimensions must be positive"
            raise ValueError(message)
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = self.path_count * modes
        self.path_input = GroupedWidelyLinear(modes, 4, path_hidden, bias=True)
        self.path_output = GroupedWidelyLinear(modes, path_hidden, 1, bias=True)

    def copy_path_from(self, baseline: PhaseGatedModeResidualPathCollapse) -> None:
        """Copy the path block while deliberately omitting the PG mode block."""
        if baseline.modes != self.modes:
            message = "cannot copy path parameters across different mode counts"
            raise ValueError(message)
        self.path_input.load_state_dict(baseline.path_input.state_dict())
        self.path_output.load_state_dict(baseline.path_output.state_dict())

    def forward(self, real: Tensor, imag: Tensor | None = None) -> ComplexField | Tensor:
        if imag is None:
            return self.forward_real_packed(real)
        if real.shape != imag.shape or real.shape[-1] != self.input_modes:
            message = "path-only stage mixer inputs have incompatible shapes"
            raise ValueError(message)
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        expected = (self.path_count, self.modes)
        if (
            source_real.shape != source_imag.shape
            or source_real.ndim != 5
            or tuple(source_real.shape[-2:]) != expected
        ):
            message = "path-only stage mixer requires NHW-path-mode inputs"
            raise ValueError(message)
        return grouped_cartesian_cffn(
            source_real,
            source_imag,
            input_projection=self.path_input,
            output_projection=self.path_output,
        )

    def forward_real_packed(self, source: Tensor) -> Tensor:
        """Consume NHW-path-2mode states and return one NHW-2mode tensor."""
        return grouped_cartesian_cffn_packed(
            source,
            input_projection=self.path_input,
            output_projection=self.path_output,
        )

    def forward_cells(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        """Collapse direction-relative 2x2 cells directly at full resolution."""
        return grouped_cartesian_cell_cffn(
            source_real,
            source_imag,
            input_projection=self.path_input,
            output_projection=self.path_output,
        )

    def packed_parameters(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the packed projection contract consumed by fused scan epilogues."""
        input_bias = self.path_input.packed_bias()
        output_bias = self.path_output.packed_bias()
        if input_bias is None or output_bias is None:
            raise RuntimeError("path collapse fusion requires both projection biases")
        packed_hidden = 2 * self.path_input.output_paths
        return (
            self.path_input.packed_weight().reshape(
                self.modes,
                packed_hidden,
                2 * self.path_input.input_paths,
            ),
            input_bias.reshape(self.modes, packed_hidden),
            self.path_output.packed_weight().reshape(
                self.modes,
                2 * self.path_output.output_paths,
                packed_hidden,
            ),
            output_bias.reshape(self.modes, 2 * self.path_output.output_paths),
        )


__all__ = [
    "PathOnlyCollapse",
    "PhaseGatedModeResidualPathCollapse",
]
