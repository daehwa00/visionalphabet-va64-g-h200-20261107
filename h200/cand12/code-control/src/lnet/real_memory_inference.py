"""Inference-only copies and shape-autotuned real Reader spatial filtering.

Prepare after loading weights and moving the model to its final device/dtype.
The source model, parametrizations and optimizer parameter identities are untouched.
"""

from __future__ import annotations

# Private methods are the same internal Reader/parametrization boundaries used by the backbone.
# ruff: noqa: SLF001
import copy

import torch
import triton
from torch import Tensor, nn
from torch.library import triton_op, wrap_triton
from torch.nn import functional

from .complex_scan import bake_parametrizations_
from .pac_triton_real_inference import real_inference_projection, real_inference_spatial_rms
from .pac_triton_real_spatial import _validate, real_factorized_spatial
from .pac_triton_real_training import _tuned_forward
from .real_memory_backbone import RealFactorizedMemoryReader, RealMemoryProjection


@triton_op("lnet::real_inference_spatial", mutates_args={})
def real_inference_spatial(x: Tensor, weight: Tensor) -> Tensor:
    _validate(x, weight)
    n, h, w, _ = x.shape
    p, _, r, k, _ = weight.shape
    output = x.new_empty((n, h, w, 2 * p))
    wrap_triton(_tuned_forward)[
        lambda meta: (triton.cdiv(n * h * w, meta["PIXELS"]), triton.cdiv(p, meta["POLES"]))
    ](
        x, weight, output, n, h, w, p, r, k
    )
    return output


class PreparedRealReader(RealFactorizedMemoryReader):
    """Frozen normalized spatial coefficients; same BF16 rounding boundaries."""

    normalized_spatial: Tensor
    autotune_spatial: bool
    fuse_spatial_rms: bool

    def forward(self, source: Tensor) -> Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Prepared Reader requires eval and no_grad/inference_mode")
        if not source.is_cuda or source.ndim != 4 or source.shape[-1] != self.input_channels:
            raise ValueError("Prepared Reader requires CUDA NHWC input of configured width")
        if self.input_norm is not None:
            source = self.input_norm(source)
        projected = functional.linear(source, self.point_weight.flatten(0, 1))
        spatial = self.normalized_spatial.to(projected.dtype)
        if self.fuse_spatial_rms:
            return real_inference_spatial_rms(
                projected.contiguous(),
                spatial,
                source.contiguous(),
                self.rms_reference_modes,
                self.variance_epsilon / 2,
                self.match_input_rms,
            )
        if self.autotune_spatial:
            output = real_inference_spatial(projected.contiguous(), spatial)
        else:
            output = real_factorized_spatial(projected.contiguous(), spatial)
        if self.match_input_rms:
            input_energy = source.float().square().mean(-1, keepdim=True)
            selected = output.unflatten(-1, (2, self.output_modes))[..., : self.rms_reference_modes]
            output_energy = selected.float().square().mean((-1, -2)).unsqueeze(-1)
            epsilon = self.variance_epsilon / 2
            output = output * torch.sqrt((input_energy + epsilon) / (output_energy + epsilon)).to(
                output.dtype
            )
        return output


class PreparedRealProjection(RealMemoryProjection):
    def forward(self, source: Tensor, *, planar: bool = False) -> Tensor:
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Prepared projection requires eval and no_grad/inference_mode")
        precision_matches = (
            torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == source.dtype
        ) or (not torch.is_autocast_enabled("cuda") and self.weight.dtype == source.dtype)
        if planar and source.is_cuda and source.dtype == torch.bfloat16 and precision_matches:
            return real_inference_projection(source.contiguous(), self.weight.contiguous())
        return super().forward(source, planar=planar)


def prepare_real_memory_inference(
    model: nn.Module,
    *,
    autotune_spatial: bool = True,
    fuse_spatial_rms: bool = False,
    fuse_projection: bool = False,
) -> nn.Module:
    """Return an independent frozen CUDA inference copy. Never resume training from it.

    No tensor on the source model is modified. Re-run preparation after changing weights,
    dtype or device. Autotune chooses geometry per shape and tensor dtype; no
    model/batch-specific winning launch configuration is embedded in the code.
    """
    if model.training:
        raise ValueError("Call eval() before preparing an inference copy")
    readers = [m for m in model.modules() if isinstance(m, RealFactorizedMemoryReader)]
    if not readers or any(
        not m.point_weight.is_cuda or m.execution_mode != "factorized_triton" for m in readers
    ):
        raise ValueError("Expected CUDA real-memory model with factorized_triton Readers")
    result = copy.deepcopy(model)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        bake_parametrizations_(result)
        for reader in list(result.modules()):
            if isinstance(reader, RealFactorizedMemoryReader):
                _, inverse = reader._kernel_and_scale()
                spatial = (
                    reader.spatial_weight.float().reshape(
                        reader.output_modes, 2, reader.rank, reader.kernel_size, reader.kernel_size
                    )
                    * inverse
                ).contiguous()
                reader.__class__ = PreparedRealReader
                reader.register_buffer("normalized_spatial", spatial, persistent=False)
                reader.autotune_spatial = autotune_spatial
                reader.fuse_spatial_rms = fuse_spatial_rms
            elif fuse_projection and isinstance(reader, RealMemoryProjection):
                reader.__class__ = PreparedRealProjection
    result.requires_grad_(requires_grad=False)
    return result
