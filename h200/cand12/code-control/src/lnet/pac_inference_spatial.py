"""Forward-only spatial launch used by prepared VA128 inference copies."""

from __future__ import annotations

# Reuse the package-private equation and validation from the training operator.
# pyright: reportPrivateUsage=false
import triton
from torch import Tensor  # noqa: TC002 - triton_op resolves its schema at runtime.
from torch.library import triton_op, wrap_triton

from .pac_triton_factorized_spatial import _forward_kernel, _validate


@triton_op("lnet::factorized_spatial_inference", mutates_args={})
def factorized_spatial_inference(projected: Tensor, real: Tensor, imag: Tensor) -> Tensor:
    """Use the unchanged spatial equation with the measured large-grid warp count.

    The caller selects this path only on the RTX 4090 at the measured batch
    sizes. Training continues to use the original operator and its backward.
    """
    _validate(projected, real, imag)
    batch, _, height, width = projected.shape
    poles, rank, kernel, _ = real.shape
    output = projected.new_empty((batch, height, width, 2 * poles))
    wrap_triton(_forward_kernel)[(triton.cdiv(batch * height * width * poles, 256),)](
        projected,
        real,
        imag,
        output,
        output,
        batch,
        height,
        width,
        poles,
        rank,
        kernel,
        *projected.stride(),
        BLOCK=256,
        PACKED_OUTPUT=True,
        num_warps=8 if height >= 28 else 4,
    )
    return output
