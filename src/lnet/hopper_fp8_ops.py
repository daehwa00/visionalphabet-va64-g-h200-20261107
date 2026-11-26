"""Opt-in Hopper training operators. Weights retain their optimizer-owned dtype."""

from functools import lru_cache, partial
from importlib.util import find_spec
from math import prod
from types import MethodType

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

E4 = torch.float8_e4m3fn
E5 = torch.float8_e5m2
LIMIT = 512 * 1024**2


def scale(x, dtype):
    lo, hi = torch.aminmax(x)
    return torch.maximum(lo.abs(), hi.abs()).float().clamp_min(1e-12) / torch.finfo(dtype).max


def quantize(x, dtype):
    s = scale(x, dtype)
    limit = torch.finfo(dtype).max
    return (x.float() / s).clamp(-limit, limit).to(dtype), s


@triton.jit
def _cast_pair(X, S, Q, T, M: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i, i < M * K, 0).to(tl.float32) / tl.load(S)
    tl.store(Q + i, v, i < M * K)
    tl.store(T + (i % K) * M + i // K, v, i < M * K)


def pair(x, dtype, fused):
    x = x.contiguous()
    if not fused:
        q, s = quantize(x, dtype)
        return q, q.T.contiguous(), s
    s = scale(x, dtype)
    q = torch.empty_like(x, dtype=dtype)
    t = torch.empty((x.size(1), x.size(0)), device=x.device, dtype=dtype)
    _cast_pair[(triton.cdiv(x.numel(), 512),)](x, s, q, t, *x.shape, 512)
    return q, t, s


def mm(a, bt, sa, sb, dtype, fast):
    return torch._scaled_mm(a, bt.T, scale_a=sa, scale_b=sb, out_dtype=dtype, use_fast_accum=fast)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, fused, fast, row):
        if row:

            def rows(z):
                s = z.abs().amax(1, keepdim=True).float().clamp_min(1e-12) / 448
                return (z.float() / s).clamp(-448, 448).to(E4).contiguous(), s

            a, sa = rows(x)
            b, sb = rows(w)
            ctx.save_for_backward(x, w)
            y = mm(a, b, sa, sb.T, x.dtype, True)
        else:
            a, at, sa = pair(x, E4, fused)
            b, bt, sb = pair(w, E4, fused)
            ctx.save_for_backward(at, bt, sa, sb)
            y = mm(a, b, sa, sb, x.dtype, True)
        ctx.fast, ctx.fused, ctx.row = fast, fused, row
        return y

    @staticmethod
    def backward(ctx, dy):
        if ctx.row:
            x, w = ctx.saved_tensors
            at, sa = quantize(x.T.contiguous(), E4)
            bt, sb = quantize(w.T.contiguous(), E4)
        else:
            at, bt, sa, sb = ctx.saved_tensors
        g, gt, sg = pair(dy, E5, ctx.fused)
        return (
            mm(g, bt, sg, sb, dy.dtype, ctx.fast),
            mm(gt, at, sg, sa, dy.dtype, ctx.fast),
            None,
            None,
            None,
        )


def hopper(x):
    return x.is_cuda and torch.cuda.get_device_capability(x.device)[0] == 9


def fp8_linear(x, w, bias=None, *, mode="tensorwise", fused=True, fast=False):
    if mode not in ("tensorwise", "rowwise"):
        raise ValueError(mode)
    m, k, n = prod(x.shape[:-1]), x.shape[-1], w.shape[0]
    if not hopper(x) or min(m, k, n) == 0 or m < 64 or any(d % 16 for d in (m, k, n)):
        return F.linear(x, w, bias)
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    if dtype not in (torch.bfloat16, torch.float16):
        return F.linear(x, w, bias)
    y = _Linear.apply(x.reshape(m, k).to(dtype), w.to(dtype), fused, fast, mode == "rowwise")
    y = y.reshape(*x.shape[:-1], n)
    return y if bias is None else y + bias.to(dtype)


@triton.jit
def _patches(
    X,
    S,
    COL,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    ROWS: tl.constexpr,
    VALID: tl.constexpr,
    TRANS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    r, c = i // C, i % C
    h, w = (r // W) % H, r % W
    value = tl.load(X + i, r < VALID, 0).to(tl.float32) / tl.load(S)
    # Each fixed shift is a permutation of the spatial plane. Wrapped destinations
    # receive padding zeros, so neither atomics nor a separate memset are needed.
    for k in tl.static_range(9):
        oh, ow = h + 1 - k // 3, w + 1 - k % 3
        output_row = (r // (H * W) * H + (oh + H) % H) * W + (ow + W) % W
        padded = r >= VALID
        output_row = tl.where(padded, r, output_row)
        good = (oh >= 0) & (oh < H) & (ow >= 0) & (ow < W) & (~padded)
        if TRANS:
            dest = (k * C + c) * ROWS + output_row
        else:
            dest = output_row * (9 * C) + k * C + c
        tl.store(COL + dest, tl.where(good, value, 0.0), r < ROWS)


@triton.jit
def _fold(
    COL,
    DX,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    r = i // C
    total = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(9):
        h, w = (r // W) % H + 1 - k // 3, r % W + 1 - k % 3
        valid = (i < SIZE) & (h >= 0) & (h < H) & (w >= 0) & (w < W)
        p = ((r // (W * H) * H + h) * W + w) * 9 * C + k * C + i % C
        total += tl.load(COL + p, valid, 0).to(tl.float32)
    tl.store(DX + i, total, i < SIZE)


@torch.library.custom_op("lnet::wave60b_patches", mutates_args=())
def patches(x: torch.Tensor, s: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    b, h, w, c = x.shape
    rows = triton.cdiv(b * h * w, 16) * 16
    shape = (9 * c, rows) if transpose else (rows, 9 * c)
    col = torch.empty(shape, device=x.device, dtype=E4)
    _patches[(triton.cdiv(rows * c, 512),)](x, s, col, h, w, c, rows, b * h * w, transpose, 512)
    return col


@patches.register_fake
def _fake_patches(x, s, transpose=False):
    b, h, w, c = x.shape
    rows = ((b * h * w + 15) // 16) * 16
    return x.new_empty((9 * c, rows) if transpose else (rows, 9 * c), dtype=E4)


def chunk_size(h, w, c, n):
    # Includes padded rows, BF16 dcol, DY casts/transposes and FP32 filter reductions.
    per_row = 2 * 9 * c + 8 * n + 4 * c
    return (LIMIT - 16 * per_row - 16 * 9 * c * n) // (h * w * per_row)


class _Conv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        b, h, width, c = x.shape
        n = w.size(0)
        chunk = chunk_size(h, width, c, n)
        wf = w.permute(0, 2, 3, 1).reshape(n, -1)
        q, sw = quantize(wf.contiguous(), E4)
        sx = scale(x, E4)
        y = x.new_empty((b, h, width, n))
        for i in range(0, b, chunk):
            part = x[i : i + chunk]
            col = patches(part, sx)
            rows = part.numel() // c
            y[i : i + chunk].copy_(
                mm(col, q, sx, sw, x.dtype, True)[:rows].reshape_as(y[i : i + chunk])
            )
            del col
        ctx.save_for_backward(x, q, sx, sw)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, q, sx, sw = ctx.saved_tensors
        b, h, w, c = x.shape
        n = dy.size(-1)
        sg = scale(dy, E5)
        qt = q.T.contiguous()
        dw = torch.zeros_like(q, dtype=torch.float32)
        dx = torch.empty_like(x)
        chunk = chunk_size(h, w, c, n)
        for i in range(0, b, chunk):
            g = dy[i : i + chunk].reshape(-1, n)
            g = F.pad(g, (0, 0, 0, (-g.size(0)) % 16))
            g = (g.float() / sg).clamp(-57344, 57344).to(E5)
            dcol = mm(g, qt, sg, sw, x.dtype, False)
            out = dx[i : i + chunk]
            _fold[(triton.cdiv(out.numel(), 512),)](dcol, out, h, w, c, out.numel(), 512)
            del dcol
            xt = patches(x[i : i + chunk], sx, True)
            dw.add_(mm(g.T.contiguous(), xt, sg, sx, torch.float32, False))
            del xt
        return dx, dw.reshape(n, 3, 3, c).permute(0, 3, 1, 2).to(x.dtype)


def fp8_dense_nhwc_conv2d(x, w, padding=1, *, mode="im2col"):
    from .memory_backbone import dense_nhwc_conv2d

    if (
        not hopper(x)
        or x.dtype not in (torch.bfloat16, torch.float16)
        or w.shape[2:] != (3, 3)
        or padding != 1
        or x.size(-1) % 16
        or w.size(0) % 16
        or chunk_size(*x.shape[1:], w.size(0)) < 1
    ):
        return dense_nhwc_conv2d(x, w, padding)
    if mode == "cudnn":
        return _Frontend.apply(x, w)
    if mode != "im2col":
        raise ValueError(mode)
    return _Conv.apply(x.contiguous(), w)


def _linear_forward(self, x, mode):
    return fp8_linear(x, self.weight, self.bias, mode=mode)


def _reader_dense(self, source, *, bmm, opaque, mode):
    dtype = self._dense_compute_dtype(source)
    weight = self._normalized_kernel(bmm=bmm).to(dtype)
    return fp8_dense_nhwc_conv2d(source.to(dtype), weight, self.padding, mode=mode)


def _dense_mode(source):
    return "dense_nhwc"


def apply_fp8_module_patch(model, *, linear_mode="tensorwise", conv_mode="im2col"):
    from .memory_backbone import FactorizedMemoryReader

    if linear_mode not in (None, "tensorwise", "rowwise") or conv_mode not in (
        None,
        "im2col",
        "cudnn",
    ):
        raise ValueError("unknown FP8 patch mode")
    if (
        linear_mode == "rowwise"
        and not probe("rowwise", next(model.parameters()).device)["available"]
    ):
        linear_mode = "tensorwise"
    if conv_mode == "cudnn" and not probe("cudnn", next(model.parameters()).device)["available"]:
        conv_mode = "im2col"
    for mod in model.modules():
        if linear_mode and isinstance(mod, torch.nn.Linear):
            mod.forward = partial(MethodType(_linear_forward, mod), mode=linear_mode)
        if conv_mode and isinstance(mod, FactorizedMemoryReader):
            mod._reader_execution_mode = _dense_mode
            mod._dense_nhwc = partial(MethodType(_reader_dense, mod), mode=conv_mode)
    return model


@lru_cache(maxsize=96)
def _plan(op, device, stream, inputs, output):
    import cudnn

    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream)
    graph = cudnn.pygraph(handle=handle, compute_data_type=cudnn.data_type.FLOAT)

    def descriptor(spec):
        shape, strides, dtype = spec
        kind = {E4: "FP8_E4M3", E5: "FP8_E5M2"}[dtype]
        return graph.tensor(dim=shape, stride=strides, data_type=getattr(cudnn.data_type, kind))

    a, b = map(descriptor, inputs)
    y = getattr(graph, op)(a, b, padding=[1, 1], stride=[1, 1], dilation=[1, 1])
    y.set_output(True).set_dim(output[0]).set_stride(output[1]).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    return graph, handle, a, b, y


def _native(op, a, b, shape, s1, s2, dtype):
    out = torch.empty(
        shape, device=a.device, dtype=torch.float32, memory_format=torch.channels_last
    )
    spec = lambda t: (tuple(t.shape), t.stride(), t.dtype)
    graph, handle, ad, bd, yd = _plan(
        op, a.device, torch.cuda.current_stream(a.device).cuda_stream, (spec(a), spec(b)), spec(out)
    )
    ws = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device=a.device)
    graph.execute({ad: a, bd: b, yd: out}, ws, handle=handle)
    return out.mul_(s1 * s2).to(dtype)


class _Frontend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        a, s = quantize(x.contiguous(), E4)
        a = a.permute(0, 3, 1, 2)
        b, t = quantize(w.contiguous(memory_format=torch.channels_last), E4)
        ctx.save_for_backward(a, b, s, t)
        return _native(
            "conv_fprop", a, b, (x.size(0), w.size(0), *x.shape[1:3]), s, t, x.dtype
        ).permute(0, 2, 3, 1)

    @staticmethod
    def backward(ctx, dy):
        a, b, s, t = ctx.saved_tensors
        g, u = quantize(dy.contiguous(), E5)
        g = g.permute(0, 3, 1, 2)
        dx = _native("conv_dgrad", g, b, a.shape, u, t, dy.dtype)
        dw = _native("conv_wgrad", a, g, b.shape, s, u, dy.dtype)
        return dx.permute(0, 2, 3, 1), dw


def probe(mode, device="cuda"):
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA hidden or unavailable"}
    if mode == "cudnn" and find_spec("cudnn") is None:
        return {"available": False, "reason": "cudnn frontend package absent"}
    try:

        def sample(shape):
            return torch.randn(shape, dtype=torch.bfloat16, device=device, requires_grad=True)

        if torch.cuda.get_device_capability(device)[0] != 9:
            return {"available": False, "reason": "requires Hopper"}
        if mode == "cudnn":
            x, w = sample((2, 8, 8, 64)), sample((64, 64, 3, 3))
            y = fp8_dense_nhwc_conv2d(x, w, mode=mode)
        else:
            x, w = sample((128, 64)), sample((64, 64))
            y = fp8_linear(x, w, mode=mode)
        values = (y, *torch.autograd.grad(y, (x, w), torch.randn_like(y)))
        if not all(v.isfinite().all().item() for v in values):
            raise RuntimeError("nonfinite probe outputs")
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    return {"available": True}
