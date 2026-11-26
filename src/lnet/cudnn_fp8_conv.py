"""Opt-in Hopper 3x3 convolution. A Runner owns delayed history for one layer."""

import threading
from statistics import median

import torch
import triton
import triton.language as tl

E4 = torch.float8_e4m3fn
E5 = torch.float8_e5m2
CL = torch.channels_last
OPS = ("fprop", "dgrad", "wgrad")
SHAPES = ((56, 64, 320, 2), (28, 96, 320, 2), (14, 128, 320, 6), (7, 256, 256, 2))


@triton.jit
def triton_amax(X, M, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i, i < N, 0).to(tl.float32)
    tl.atomic_max(M, tl.max(tl.abs(v), 0).to(tl.int32, bitcast=True), sem="relaxed")


@triton.jit
def triton_cast_fp8_and_amax(
    X, Y, INV, M, N: tl.constexpr, LIMIT: tl.constexpr, TRACK: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i, i < N, 0).to(tl.float32)
    q = tl.minimum(tl.maximum(v * tl.load(INV), -LIMIT), LIMIT)
    tl.store(Y + i, q, i < N)
    if TRACK:
        tl.atomic_max(M, tl.max(tl.abs(v), 0).to(tl.int32, bitcast=True), sem="relaxed")


@triton.jit
def _update(M, S, INV, LIMIT: tl.constexpr):
    peak = tl.load(M).to(tl.float32, bitcast=True)
    scale = tl.maximum(peak, 1.0e-12) / LIMIT
    tl.store(S, scale)
    tl.store(INV, 1.0 / scale)


@triton.jit
def _scale(X, S, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i, i < N, 0).to(tl.float32) * tl.load(S), i < N)


def _output(shape, device, dtype):
    return torch.empty(shape, device=device, dtype=dtype, memory_format=CL)


def _finite(x):
    # Warmup only. Bound temporary memory even for the 1G-element output.
    return all(
        chunk.isfinite().all().item() for chunk in x.permute(0, 2, 3, 1).view(-1).split(1 << 22)
    )


class Device:
    def __init__(self, cudnn):
        self.handle = cudnn.create_handle()
        self.buffers = {}
        self.captured = set()
        self.retired = []  # Captured graphs can still reference an earlier arena.
        self.plans = {}

    def workspace(self, device, stream, size):
        # Streams may overlap. Each stream owns a high-water-mark arena.
        buf = self.buffers.get(stream)
        if buf is None or buf.numel() < size:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm workspace on capture stream first")
            if buf is not None and id(buf) in self.captured:
                self.retired.append(buf)
            buf = torch.empty(max(size, 1), dtype=torch.uint8, device=device)
            self.buffers[stream] = buf
        if torch.cuda.is_current_stream_capturing():
            self.captured.add(id(buf))
        return buf


_devices = {}
_lock = threading.RLock()


def _build(c, dev, stream, op, a, b, product, out):
    dt = c.data_type
    reasons = []
    for tier in "ABC" if op == "wgrad" else "AB":
        if tier == "C" and out.numel() * 4 > 4 * 1024**2:
            continue
        # Unscaled BF16 is safe even for worst finite FP8 operands.
        if (
            tier == "B"
            and max(a.numel(), b.numel()) * 57344.0**2 >= torch.finfo(torch.bfloat16).max
        ):
            continue
        for modes in ([c.heur_mode.A, c.heur_mode.B], [c.heur_mode.FALLBACK]):
            try:
                graph = c.pygraph(
                    handle=dev.handle,
                    io_data_type=dt.BFLOAT16,
                    intermediate_data_type=dt.FLOAT,
                    compute_data_type=dt.FLOAT,
                )

                def desc(t):
                    return graph.tensor(
                        dim=t.shape,
                        stride=t.stride(),
                        data_type=dt.FP8_E4M3 if t.dtype == E4 else dt.FP8_E5M2,
                    )

                ad, bd = desc(a), desc(b)
                conv = getattr(graph, "conv_" + op)(
                    ad, bd, padding=[1, 1], stride=[1, 1], dilation=[1, 1]
                )
                conv.set_dim(out.shape).set_stride(out.stride())
                sd = None
                yd = conv
                if tier == "A":
                    conv.set_data_type(dt.FLOAT)
                    sd = graph.tensor(dim=[1] * 4, stride=[1] * 4, data_type=dt.FLOAT)
                    yd = graph.mul(conv, sd)
                yd.set_output(True).set_dim(out.shape).set_stride(out.stride()).set_data_type(
                    dt.FLOAT if tier == "C" else dt.BFLOAT16
                )
                graph.validate()
                graph.build_operation_graph()
                graph.create_execution_plans(modes)
                graph.check_support()
                indexed = all(
                    callable(getattr(graph, n, None))
                    for n in (
                        "get_execution_plan_count",
                        "build_plan_at_index",
                        "execute_plan_at_index",
                        "get_workspace_size_plan_at_index",
                    )
                )
                count = graph.get_execution_plan_count() if indexed else 1
                dst = _output(out.shape, out.device, torch.float32) if tier == "C" else out
                pack = {ad: a, bd: b, yd: dst}
                if sd is not None:
                    pack[sd] = product
                timings = []
                for idx in range(min(count, 10)):
                    try:
                        if indexed:
                            graph.build_plan_at_index(idx)
                            size = graph.get_workspace_size_plan_at_index(idx)
                        else:
                            graph.build_plans(c.build_plan_policy.HEURISTICS_CHOICE)
                            size = graph.get_workspace_size()
                        ws = dev.workspace(out.device, stream, size)

                        def execute(pack, ws, idx=idx, indexed=indexed):
                            if indexed:
                                graph.execute_plan_at_index(pack, ws, idx, handle=dev.handle)
                            else:
                                graph.execute(pack, ws, handle=dev.handle)

                        execute(pack, ws)
                        stamps = []
                        for _ in range(3):
                            start, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            start.record()
                            execute(pack, ws)
                            end.record()
                            stamps.append((start, end))
                        end.synchronize()
                        if _finite(dst):
                            timings.append(
                                (median(x.elapsed_time(y) for x, y in stamps), idx, size, execute)
                            )
                    except (RuntimeError, ValueError) as exc:
                        reasons.append(str(exc))
                if timings:
                    ms, idx, size, run = min(timings, key=lambda t: t[0])
                    info = dict(tier=tier, index=idx, candidates=count, timed=len(timings), ms=ms)
                    return ad, bd, yd, sd, size, run, info
            except Exception as exc:
                reasons.append(str(exc))
    raise RuntimeError(f"No supported FP8 {op} plan: {reasons}")


class Runner:
    def __init__(self, mode="delayed", enabled=OPS):
        if mode not in ("dynamic", "delayed"):
            raise ValueError(mode)
        self.mode = mode
        self.enabled = frozenset(enabled)
        self.history = {}
        self.info = {}

    def quant(self, x, slot):
        dtype = E5 if slot == "dy" else E4
        limit = torch.finfo(dtype).max
        key = (x.device, torch.cuda.current_stream(x.device).cuda_stream, slot, tuple(x.shape))
        state = self.history.get(key)
        first = state is None
        if first:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm quantizer on capture stream first")
            state = (
                torch.zeros(1, device=x.device, dtype=torch.int32),
                torch.ones(1, device=x.device),
                torch.ones(1, device=x.device),
            )
            self.history[key] = state
        peak, scale, inv = state
        peak.zero_()
        n = x.numel()
        grid = (triton.cdiv(n, 1024),)
        if self.mode == "dynamic" or first:
            triton_amax[grid](x, peak, n, 1024)
            _update[(1,)](peak, scale, inv, limit)
        used = scale.clone()
        y = torch.empty_like(x, dtype=dtype)
        triton_cast_fp8_and_amax[grid](x, y, inv, peak, n, limit, self.mode == "delayed", 1024)
        if self.mode == "delayed":
            _update[(1,)](peak, scale, inv, limit)
        return y, used

    def op(self, name, qa, qb, shape):
        import cudnn

        a, sa = qa
        b, sb = qb
        product = (sa * sb).view(1, 1, 1, 1)
        out = _output(shape, a.device, torch.bfloat16)
        key = (name, tuple(a.shape), tuple(b.shape), tuple(shape))
        with _lock, torch.cuda.device(a.device):
            stream = torch.cuda.current_stream().cuda_stream
            if a.device not in _devices:
                _devices[a.device] = Device(cudnn)
            dev = _devices[a.device]
            cudnn.set_stream(dev.handle, stream)
            if key not in dev.plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm plan before capture")
                dev.plans[key] = _build(cudnn, dev, stream, name, a, b, product, out)
            ad, bd, yd, sd, size, run, info = dev.plans[key]
            self.info[name] = info
            dst = _output(shape, a.device, torch.float32) if info["tier"] == "C" else out
            pack = {ad: a, bd: b, yd: dst}
            if sd is not None:
                pack[sd] = product
            run(pack, dev.workspace(a.device, stream, size))
        if sd is None:
            _scale[(triton.cdiv(out.numel(), 1024),)](dst, product, out, out.numel(), 1024)
        return out

    def __call__(self, x, w):
        if (
            x.ndim != 4
            or w.ndim != 4
            or x.shape[-1] != w.shape[1]
            or w.shape[2:] != (3, 3)
            or any(i % 16 for i in w.shape[:2])
            or x.dtype != torch.bfloat16
            or x.dtype != w.dtype
            or not x.is_cuda
            or x.device != w.device
        ):
            raise ValueError("expected CUDA BF16 NHWC input and aligned OI33 weights")
        return _Conv.apply(x, w, self)


class _Conv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, runner):
        x = x.contiguous().permute(0, 3, 1, 2)
        w = w.contiguous(memory_format=CL)
        enabled = runner.enabled
        a = runner.quant(x, "x") if enabled & {"fprop", "wgrad"} else (None, None)
        b = runner.quant(w, "w") if enabled & {"fprop", "dgrad"} else (None, None)
        ctx.save_for_backward(x, w, *a, *b)
        ctx.runner = runner
        shape = (x.size(0), w.size(0), *x.shape[2:])
        y = (
            runner.op("fprop", a, b, shape)
            if "fprop" in enabled
            else torch.nn.functional.conv2d(x, w, padding=1)
        )
        return y.permute(0, 2, 3, 1)

    @staticmethod
    def backward(ctx, dy):
        x, w, a, sa, b, sb = ctx.saved_tensors
        r = ctx.runner
        dy = dy.contiguous().permute(0, 3, 1, 2)
        g = r.quant(dy, "dy") if r.enabled & {"dgrad", "wgrad"} else None
        dx = (
            r.op("dgrad", g, (b, sb), x.shape)
            if "dgrad" in r.enabled
            else torch.nn.grad.conv2d_input(x.shape, w, dy, padding=1)
        )
        dw = (
            r.op("wgrad", (a, sa), g, w.shape)
            if "wgrad" in r.enabled
            else torch.nn.grad.conv2d_weight(x, w.shape, dy, padding=1)
        )
        return dx.permute(0, 2, 3, 1), dw, None
