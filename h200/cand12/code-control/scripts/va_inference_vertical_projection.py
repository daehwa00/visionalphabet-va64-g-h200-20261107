"""Experimental genuine vertical scan/path/projection on inference snapshots."""
from __future__ import annotations

# pyright: reportPrivateUsage=false, reportImplicitRelativeImport=false
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn
from va_inference_reader_scan import Fusion, _FusedBlock, _horizontal

from lnet.pac_product_scan_contracts import DEFAULT_EPSILON, gain_kind
from lnet.pac_product_scan_normalization import static_product_scan_auxiliary
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock
from lnet.pac_triton_bidirectional_product_scan import _forward_planes_op
from lnet.pac_triton_vertical_projection_inference import run_vertical_scan_projection_inference

if TYPE_CHECKING:
    from collections.abc import Callable

    from lnet.complex_scan_types import ComplexField


class _VerticalProjectionBlock(SameResolutionPoleScanBlock):
    vertical_projection_fallback: Callable[[Tensor, Tensor | None], ComplexField | Tensor] | None = None
    vertical_projection_reader_fused: bool = False
    vertical_projection_reader_mode: Fusion = "rms_scan"
    vertical_projection_output_tile: int = 32
    vertical_projection_pole_tile: int = 32
    vertical_projection_warps: int = 8
    vertical_projection_max_height: int = 14

    def forward(self, real: Tensor, imag: Tensor | None = None) -> ComplexField | Tensor:
        if imag is not None or real.shape[1] > self.vertical_projection_max_height:
            if self.vertical_projection_fallback is None:
                raise RuntimeError("vertical projection installer did not preserve the original forward")
            return self.vertical_projection_fallback(real, imag)
        if torch.is_grad_enabled():
            raise RuntimeError("vertical scan/projection is inference only")
        pole_x, pole_y = self.pole_scan.pole_coefficients(
            (real.shape[0], real.shape[1], real.shape[2], self.pole_modes),
        )
        if self.vertical_projection_reader_fused:
            horizontal = _horizontal(self.reader, real, pole_x, self.vertical_projection_reader_mode)
        else:
            drive = self.reader(real)
            horizontal = _forward_planes_op(*pole_x, *drive.chunk(2, -1))
        auxiliary = static_product_scan_auxiliary(
            pole_x, pole_y, horizontal[0], epsilon=DEFAULT_EPSILON,
            gain_kind=gain_kind(self.pole_scan.product_gain_normalization),
        )
        if self.memory_projection is None:
            raise RuntimeError("vertical projection requires a complex memory projection")
        projected = run_vertical_scan_projection_inference(
            pole_y, horizontal, auxiliary, self.path_collapse.packed_parameters(),
            self.memory_projection.weight_real, self.memory_projection.weight_imag,
            gain_normalization=self.pole_scan.product_gain_normalization,
            path_swiglu=bool(getattr(self.path_collapse, "path_swiglu", False)),
            output_tile=self.vertical_projection_output_tile,
            pole_tile=self.vertical_projection_pole_tile,
            num_warps=self.vertical_projection_warps,
            experimental_large_height=self.vertical_projection_max_height > 16,
        )
        return self.post_fusion.forward_residual_packed(real, projected)


def enable_vertical_projection(
    model: nn.Module, *, output_tile: int = 32, pole_tile: int = 32,
    num_warps: int = 8, max_height: int = 14,
) -> int:
    """Replace only same-resolution projected blocks; other paths stay intact."""
    if output_tile not in (32, 64, 128) or pole_tile not in (16, 32, 64):
        raise ValueError("unsupported vertical projection tile")
    if num_warps not in (4, 8) or type(max_height) is not int or not 1 <= max_height <= 56:
        raise ValueError("unsupported vertical projection geometry")
    changed = 0
    for module in model.modules():
        if type(module) not in (SameResolutionPoleScanBlock, _FusedBlock):
            continue
        block = cast("SameResolutionPoleScanBlock", module)
        if block.memory_projection is None:
            continue
        previous = block.forward
        fused_reader = type(block) is _FusedBlock
        reader_mode = block.reader_scan_fusion if fused_reader else "rms_scan"
        block.__class__ = _VerticalProjectionBlock
        active = cast("_VerticalProjectionBlock", block)
        active.vertical_projection_fallback = previous
        active.vertical_projection_reader_fused = fused_reader
        active.vertical_projection_reader_mode = reader_mode
        active.vertical_projection_output_tile = output_tile
        active.vertical_projection_pole_tile = pole_tile
        active.vertical_projection_warps = num_warps
        active.vertical_projection_max_height = max_height
        changed += 1
    return changed
