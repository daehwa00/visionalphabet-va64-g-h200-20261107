"""Implicit 3x3 convolution with an optional LayerNorm/GELU epilogue."""

# ruff: noqa: ANN001, ANN201, ANN205, FBT003, N803
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from lnet.fused_stem_ops import _backward


@triton.jit
def conv3(
    X, W, BIAS, GAMMA, BETA, OUT, N: tl.constexpr, FUSED: tl.constexpr, TENSOR_CORES: tl.constexpr
):
    pixel = tl.program_id(0) * 32 + tl.arange(0, 32)
    k = tl.arange(0, 32)
    c = tl.arange(0, 32)
    iy = (pixel % (112 * 112)) // 112 * 2 - 1
    ix = (pixel % 112) * 2 - 1
    if TENSOR_CORES:
        ky, kx, kc = k // 9, k // 3 % 3, k % 3
        y, x = iy[:, None] + ky[None, :], ix[:, None] + kx[None, :]
        valid = (pixel[:, None] < N) & (k[None, :] < 27)
        source = (pixel[:, None] // (112 * 112) * 224 * 224 + y * 224 + x) * 3
        a = tl.load(
            X + source + kc[None, :], valid & (y >= 0) & (y < 224) & (x >= 0) & (x < 224), 0
        )
        b = tl.load(W + c[None, :] * 27 + (kc * 9 + ky * 3 + kx)[:, None], k[:, None] < 27, 0)
        value = tl.dot(a, b)
    else:
        value = tl.full((32, 32), 0.0, tl.float32)
        for index in tl.static_range(27):
            kc = index // 9
            y = iy + (index // 3) % 3
            x = ix + index % 3
            offset = (pixel // (112 * 112) * 224 * 224 + y * 224 + x) * 3
            mask = (pixel < N) & (y >= 0) & (y < 224) & (x >= 0) & (x < 224)
            a = tl.load(X + offset + kc, mask, 0).to(tl.float32)
            b = tl.load(W + c * 27 + index).to(tl.float32)
            value = tl.fma(a[:, None], b[None, :], value)
    value = value.to(OUT.dtype.element_ty).to(tl.float32)
    if FUSED:
        value += tl.load(BIAS + c)[None, :].to(tl.float32)
        centered = value - tl.sum(value, axis=1)[:, None] / 32
        inv = tl.rsqrt(tl.sum(centered * centered, axis=1)[:, None] / 32 + 1e-5)
        affine = centered * inv * tl.load(GAMMA + c)[None, :] + tl.load(BETA + c)[None, :]
        value = 0.5 * affine * (1.0 + libdevice.erf(affine * 0.7071067811865476))
    tl.store(OUT + pixel[:, None] * 32 + c[None, :], value, pixel[:, None] < N)


def run(source, weight, bias, gamma, beta, fused, *, tensor_cores=True):
    batch = source.shape[0]
    n = batch * 112 * 112
    out = source.new_empty((batch, 112, 112, 32))
    conv3[(triton.cdiv(n, 32),)](
        source,
        weight,
        bias,
        gamma,
        beta,
        out,
        n,
        fused,
        tensor_cores,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


class FusedStage(torch.autograd.Function):
    """Probe activation savings with recomputation and the existing VJPs."""

    @staticmethod
    def forward(ctx, source, weight, bias, gamma, beta):
        ctx.save_for_backward(source, weight, bias, gamma, beta)
        return run(source, weight, bias, gamma, beta, True).permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad):
        source, weight, bias, gamma, beta = ctx.saved_tensors
        pre_norm = run(source, weight, bias, gamma, beta, False)
        dy = grad.permute(0, 2, 3, 1).contiguous()
        dconv, dgamma, dbeta, dbias = _backward(pre_norm, gamma, beta, bias, dy, 1e-5, False)
        dx, dw, _ = torch.ops.aten.convolution_backward.default(
            dconv.permute(0, 3, 1, 2),
            source,
            weight,
            None,
            [2, 2],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [ctx.needs_input_grad[0], ctx.needs_input_grad[1], False],
        )
        return dx, dw, dbias, dgamma, dbeta
