"""Opt-in small-grid 2D scan installer for an isolated inference snapshot."""
from __future__ import annotations

# Existing Reader+scan subclasses remain the fallback for larger grids.
# pyright: reportPrivateUsage=false, reportImplicitRelativeImport=false
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from va_inference_reader_scan import _FusedBlock, _FusedStage

from lnet.complex_scan_stage import ComplexScanStage
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock
from lnet.pac_triton_scan2d_inference import run_product_scan_2d_inference

if TYPE_CHECKING:
    from collections.abc import Callable


class _Scan2DBlock(_FusedBlock):
    scan2d_block_modes: int = 2
    scan2d_num_warps: int = 4
    scan2d_max_size: int = 16
    scan2d_fallback: Callable[..., Tensor]  # pyright: ignore[reportUninitializedInstanceVariable]

    def forward(self, real: Tensor, imag: Tensor | None = None) -> Tensor:
        if max(real.shape[1:3]) > self.scan2d_max_size:
            return self.scan2d_fallback(real, imag)
        if imag is not None or torch.is_grad_enabled():
            raise RuntimeError("2D scan fusion is packed inference only")
        drive = self.reader(real)
        pole_x, pole_y = self.pole_scan.pole_coefficients(
            (*drive.shape[:-1], self.pole_modes),
        )
        memory = run_product_scan_2d_inference(
            pole_x, pole_y, drive, self.path_collapse.packed_parameters(),
            gain_normalization=self.pole_scan.product_gain_normalization,
            path_swiglu=bool(getattr(self.path_collapse, "path_swiglu", False)),
            planar_output=self.memory_projection is not None,
            block_modes=self.scan2d_block_modes, num_warps=self.scan2d_num_warps,
            experimental_large_grid=self.scan2d_max_size > 16,
        )
        if self.memory_projection is not None:
            memory = self.memory_projection(memory, planar=True)
        return self.post_fusion.forward_residual_packed(real, memory)


class _Scan2DStage(_FusedStage):
    scan2d_block_modes: int = 2
    scan2d_num_warps: int = 4
    scan2d_max_size: int = 16
    scan2d_fallback: Callable[..., tuple[Tensor | None, Tensor]]  # pyright: ignore[reportUninitializedInstanceVariable]

    def forward(
        self, real: Tensor, imag: Tensor | None = None, *, emit_descriptor: bool = True,
    ) -> tuple[Tensor | None, Tensor]:
        if self.output_modes is not None or max(real.shape[1:3]) > self.scan2d_max_size:
            return self.scan2d_fallback(real, imag, emit_descriptor=emit_descriptor)
        if imag is not None or torch.is_grad_enabled():
            raise RuntimeError("2D scan fusion is packed inference only")
        reader = cast("FactorizedComplexConv2dReader", self.pole_input_projection)
        drive = reader(real)
        pole_x, pole_y = self.pole_coefficients((*drive.shape[:-1], self.modes))
        epilogue = "mean_energy" if self.readout_mode == "mean_energy" else "descriptor"
        return None, run_product_scan_2d_inference(
            pole_x, pole_y, drive, epilogue=epilogue,
            gain_normalization=self.product_gain_normalization,
            block_modes=self.scan2d_block_modes, num_warps=self.scan2d_num_warps,
            experimental_large_grid=self.scan2d_max_size > 16,
        )


def enable_scan2d_fusion(
    model: nn.Module, *, block_modes: int = 2, num_warps: int = 4, max_size: int = 16,
) -> None:
    """Install after Reader/scan setup and before compile on the frozen copy.

    The spatial cap is structural, and can explicitly permit grids through 56.
    Larger shapes/stages retain their exact pre-installation forward method.
    Installation is idempotent and does not add or alter checkpoint tensors.
    """
    if block_modes not in (1, 2, 4, 8) or num_warps not in (4, 8):
        raise ValueError("2D scan requires block_modes 1/2/4/8 and num_warps 4/8")
    if type(max_size) is not int or not 1 <= max_size <= 56:
        raise ValueError("2D scan max_size must be an integer from 1 through 56")
    for module in model.modules():
        if isinstance(module, (_Scan2DBlock, _Scan2DStage)):
            continue
        if isinstance(module, SameResolutionPoleScanBlock):
            fallback = module.forward
            module.__class__ = _Scan2DBlock
            block = cast("_Scan2DBlock", module)
            block.scan2d_fallback = cast("Callable[..., Tensor]", fallback)
            block.scan2d_block_modes, block.scan2d_num_warps = block_modes, num_warps
            block.scan2d_max_size = max_size
        elif isinstance(module, ComplexScanStage) and isinstance(
            module.pole_input_projection, FactorizedComplexConv2dReader,
        ):
            fallback = module.forward
            module.__class__ = _Scan2DStage
            stage = cast("_Scan2DStage", module)
            stage.scan2d_fallback = cast("Callable[..., tuple[Tensor | None, Tensor]]", fallback)
            stage.scan2d_block_modes, stage.scan2d_num_warps = block_modes, num_warps
            stage.scan2d_max_size = max_size
