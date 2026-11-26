# ruff: noqa: ANN001, ANN201, ANN202, ARG001, E501, FBT002, RUF059, S603, S607, SLF001
"""Split-plane complex BF16 handoffs for the Wave 57 Hopper scan preset."""
import ctypes as ct
import re
import subprocess
import tempfile
from dataclasses import replace as with_fields
from functools import lru_cache
from pathlib import Path

import torch

from . import general_cuda_scans as scans

ARM = "w66_split_bf16"
HF_STREAMING = "w66_hf_stcs"


W66_HYBRID_SHAPES = {"56d", "56c", "28d", "28c"}


def hybrid_arm(shape):
    return ARM if shape in W66_HYBRID_SHAPES else "w57_lockin"


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"unexpected Wave 57 template near {old}")
    return source.replace(old, new)


def horizontal_source(key, height, variant):
    source = scans._kernel_source(key)
    begin = source.rindex("template<", 0, source.index("void horizontal("))
    end = source.index("template<bool PRECISE>", begin)
    body = source[begin:end]
    pos = body.index("{", body.index("for (I line")) + 1
    body = body[:pos] + """
            auto index=[&](int col,int width,bool neg) {
                I tile=(line/H)*((M+31)/32)+m/32;
                return ((((tile*2+neg)*width+col)*H+line%H)*32+(m&31));
            };
            auto fetch=[&](int x,bool adj,bool neg) {
                int step=1+int(adj && COMPACT);
                return reinterpret_cast<const Pair<T>*>(adj?grad:states)
                    [index(x/step,W/step,neg)].unpack();
            };
""" + body[pos:]
    if key.horizontal_vec2:
        if key.backward:
            begin_io = body.index("                    if constexpr (BACKWARD) {", pos)
            end_io = body.index("                    Complex output[2]", begin_io)
            body = body[:begin_io] + """
                    auto get=[&](int col,bool adj,bool hi,V& re,V& im){
                        int step=1+int(adj && COMPACT);
                        auto words=reinterpret_cast<const Pair<T>*>(adj?grad:states);
                        auto pair=*reinterpret_cast<const uint2*>(words+index(col/step,W/step,hi));
                        auto a=reinterpret_cast<Pair<T>*>(&pair.x)->unpack();
                        auto b=reinterpret_cast<Pair<T>*>(&pair.y)->unpack();
                        re={a.r,b.r}; im={a.i,b.i};
                    };
                    if(!COMPACT || ((x&1)==pass)) get(x,true,!pass,gr,gim);
                    if(pass && x) get(x-1,false,false,pr,pi);
                    if(pass && x+1<W) get(x+1,false,true,nr,ni);
""" + body[end_io:]
        else:
            begin_io = body.index("                    if (!BACKWARD || pass) {")
            end_io = body.index("\n                    }", begin_io) + len("\n                    }")
            pack = "auto a=Pair<T>::pack(output[0]), b=Pair<T>::pack(output[1]);"
            if height == 56:
                store = "__stcs(dst, pair);" if variant == HF_STREAMING else "*dst=pair;"
                stores = """
                    auto words=reinterpret_cast<Pair<T>*>(out);
                    auto dst=reinterpret_cast<uint2*>(words+index(x,W,pass));
                    auto pair=make_uint2(*reinterpret_cast<unsigned*>(&a),
                                         *reinterpret_cast<unsigned*>(&b));
                """ + store
            else:
                stores = """
                    unsigned lo=*reinterpret_cast<unsigned*>(&a), hi=*reinterpret_cast<unsigned*>(&b);
                    int warp_m=m-2*lane, active=min(32,(M-warp_m)/2), half=active/2;
                    bool lower=lane<half;
                    auto exchanged=__shfl_xor_sync(__activemask(),lower?hi:lo,half);
                    #pragma unroll
                    for(int phase=0;phase<2;++phase){
                        int mode=warp_m+2*(lane&(half-1))+int(!lower)+phase*active;
                        I tile=(line/H)*((M+31)/32)+mode/32;
                        I addr=((((tile*2+pass)*W+x)*H+line%H)*32+(mode&31));
                        reinterpret_cast<unsigned*>(out)[addr]=phase?(lower?exchanged:hi):(lower?lo:exchanged);
                    }
"""
            body = body[:begin_io] + pack + stores + body[end_io:]
    elif key.backward:
        def fetch(match):
            pointer, expr = match.groups()
            neg = "2" in expr
            col = "x" if pointer == "grad" else ("x+1" if neg else "x-1")
            return f"fetch({col},{str(pointer == 'grad').lower()},{str(neg).lower()})"
        body = re.sub(r"load_complex(?:_readonly)?\((states|grad),\s*([^,]+),\s*(?:grad_)?plane\)", fetch, body)
    else:
        for old, value, plane in (("base + I(x) * M", "pos", 0), ("2 * plane + base + I(x) * M", "neg", 1)):
            body = replace(body, f"store_complex(out, {old}, plane, {value});",
                           f"reinterpret_cast<Pair<T>*>(out)[index(x,W,{plane})]=Pair<T>::pack({value});")
    return source[:begin] + body + source[end:]


def vertical_source(key):
    source = scans._kernel_source(key)
    begin = source.index("    __device__ Quad<T> packed_source(int y)")
    end = source.index("    __device__ float2 gain(", begin)
    source = source[:begin] + """
    I tile;
    int xa,xb,x;
    __device__ I index(int col,int width,int y,bool neg) const {
        return ((((tile*2+neg)*width+col)*H+y)*32+(m&31));
    }
    __device__ Quad<T> packed_source(int y) const {
        auto words=reinterpret_cast<const Pair<T>*>(states);
        return {words[index(xa,W,y,false)], words[index(xb,W,y,true)]};
    }
    __device__ float4 source(int y) const {return packed_source(y).unpack();}
""" + source[end:]
    end = source.index("};", source.index("Column<T, I> c{states, vy,"))
    source = source[:end] + ",I(b)*tiles+m/32,xa,xb,x" + source[end:]
    begin = source.index("    const I row", source.index("void write_column_gradient("))
    end = source.index("    if constexpr (REVERSE)", begin)
    source = source[:begin] + """
    auto words=reinterpret_cast<Pair<T>*>(gs);
    auto ia=c.index(c.x,c.OW,y,false), ib=c.index(c.x,c.OW,y,true);
""" + source[end:]
    for slot, val in (("ia","ds_a"), ("ib","ds_b")):
        source = replace(source, f"load_complex(gs, {slot}, plane)", f"words[{slot}].unpack()")
        source = replace(source, f"store_complex(gs, {slot}, plane, {val});", f"words[{slot}]=Pair<T>::pack({val});")
    return source


def reuse_source_rows(source, key):
    """Place reverse Path injections in rows retired by the descending sweep."""
    if not (key.sgs or key.l2_single_write):
        source = scans._sgs_source(source, persistent=True)
    if key.l2_single_write or not key.sgs:
        source = replace(source, "buf + I(oh) * stride, stride",
                         "scratch + I(worker)*M+m, I(workers)*M")
        # pos_grad has its own stride when the line buffer lives in shared memory.
        source = source.replace("gs, pos_grad, g2_stride", "gs, pos_grad, pos_stride")
    source = source.replace("CACHE_SOURCE ? H : span", "H")
    begin = source.index("void column_backward(")
    end = source.index("#undef CACHED_SOURCE_AT", begin)
    body = source[begin:end]
    split = body.index("    adj_a = adj_b = {};")
    descending, ascending = body[:split], body[split:]
    # During reverse traversal each source row stays live through its positive
    # pole gradient. Only then may its injection take over the same eight bytes.
    descending = re.sub(r"g2\[I\((?:COARSE \? y / 2 : y|yo)\) \* g2_stride\] =", "auto injection =", descending)
    descending = re.sub(r"(?m)^( +)if constexpr \(full\) \{",
                        r"\1seg_src[I(y-int(COARSE))*THREADS]=injection;\n\1if constexpr (full) {", descending)
    descending = descending.replace("source_cache[(j) * THREADS].unpack()",
                                    "(CACHE_SOURCE ? source_cache[(j)*THREADS].unpack() : c.source(y0+(j)))")
    descending = re.sub(r"if constexpr \(COARSE && !CACHE_SOURCE\) seg_src\[j \* THREADS\] = Quad<T>::pack\(source\);", "", descending)
    ascending = ascending.replace("g2[I(COARSE ? y / 2 : y) * g2_stride].unpack()", "seg_src[I(y)*THREADS].unpack()")
    ascending = ascending.replace("source_cache[j * THREADS].unpack()", "c.source(y0+j)").replace("CACHED_SOURCE_AT(j)", "c.source(y)")
    ascending = ascending.replace("if constexpr (!CACHE_SOURCE) seg_src[j * THREADS] = Quad<T>::pack(s);", "")
    return source[:begin] + descending + ascending + source[end:]


def shared_bytes(launch, height):
    key = launch.key
    extra = 0
    if key.vertical and key.backward and height != 56 and not key.cache_source:
        extra = max(0, height-(key.chunk or height))*key.warps*32*8
    return launch.plan.smem + extra


def kernel_source(key, variant, width, height=None, modes=None, full_rows=False):
    source = vertical_source(key) if key.vertical else horizontal_source(key, height, variant)
    if key.vertical and key.backward and variant == ARM:
        if height != 56:
            source = reuse_source_rows(source, key)
        elif not (key.sgs or key.l2_single_write):
            source = scans._sgs_source(source, persistent=True)
            if key.cache_line:
                source = replace(source, "buf + I(oh) * stride, stride",
                                 "scratch + I(worker)*M+m, I(workers)*M")
                source = source.replace("gs, pos_grad, g2_stride", "gs, pos_grad, pos_stride")
    if height and modes and not (height == 56 and key.vertical and key.backward):
        pos = source.rindex(") {") + 3
        source = source[:pos] + f"\n H={height}; W={width}; M={modes};\n" + source[pos:]
    return source


def compile_flags(key, variant, width):
    return ["--fmad=" + str(not key.vertical or key.backward or key.vector_weights).lower()]


@lru_cache(maxsize=256)
def _function(device, key, variant, width, height, modes, full_rows, smem):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "packed.cu"
        path.write_text(kernel_source(key, variant, width, height, modes, full_rows))
        cubin = path.with_suffix(".cubin")
        subprocess.run(
            [
                "nvcc",
                "-std=c++17",
                "-arch=sm_90a",
                *compile_flags(key, variant, width),
                "-cubin",
                str(path),
                "-o",
                str(cubin),
            ],
            check=True,
            capture_output=True,
        )
        image = ct.create_string_buffer(cubin.read_bytes())
    with torch.cuda.device(device):
        driver = scans._load_cuda_driver()
        scans._ensure_context(driver, device)
        module, fn = ct.c_void_p(), ct.c_void_p()
        scans._check_cu(driver.cuModuleLoadData(ct.byref(module), image), "load packed cubin")
        scans._check_cu(
            driver.cuModuleGetFunction(ct.byref(fn), module, scans._kernel_name(key).encode()),
            "get packed function",
        )
        scans._configure_smem(driver, fn, smem)
        return driver, module, fn


def _dispatch(real, launch, variant, width, args):
    device = real.device.index or 0
    key = launch.key
    driver, module, fn = _function(
        device,
        key,
        variant,
        width,
        real.shape[1],
        real.shape[-1],
        launch.workers == real.shape[0] * real.shape[1],
        shared_bytes(launch, real.shape[1]),
    )
    grid = scans._ceil(real.shape[-1], 32 * (1 + key.horizontal_vec2)) * scans._ceil(
        launch.workers, key.warps
    )
    argv = (ct.c_void_p * len(args))(*(ct.addressof(arg) for arg in args))
    with torch.cuda.device(device):
        stream = ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        scans._check_cu(
            driver.cuLaunchKernel(
                fn, grid, 1, 1, 32 * key.warps, 1, 1, shared_bytes(launch, real.shape[1]), stream, argv, None
            ),
            "launch packed scan",
        )


def _check_source(real, variant):
    if variant not in (ARM, HF_STREAMING):
        raise ValueError("unknown packed representation")
    if real.ndim != 4 or not real.is_cuda or real.dtype != torch.bfloat16:
        raise ValueError("expected CUDA BF16 [B,H,W,M] source")
    if torch.cuda.get_device_capability(real.device) != (9, 0):
        raise ValueError("packed kernels require SM90a")


def _check_storage(real, words, width):
    b, h, _, m = real.shape
    tiles = (m + 31) // 32
    dtype = torch.uint32
    if (
        tuple(words.shape) != (b, tiles, 2, width, h, 32)
        or words.dtype != dtype
        or words.device != real.device
        or not words.is_contiguous()
    ):
        raise ValueError("invalid packed words")
def _words(real, width):
    b, h, _, m = real.shape
    return torch.empty(
        (b, (m + 31) // 32, 2, width, h, 32),
        device=real.device,
        dtype=torch.uint32,
    )


def horizontal(pole, r, i, variant=ARM, states=None, adjoints=None, compact=False):
    _check_source(r, variant)
    if (
        r.shape != i.shape
        or r.device != i.device
        or r.dtype != i.dtype
        or not scans._pole_ok(pole, r)
    ):
        raise ValueError("invalid horizontal scan inputs")
    shape = tuple(r.shape)
    b, h, w, m = shape
    backward = adjoints is not None
    if compact and w % 2:
        raise ValueError("wave57 horizontal backward requires even compact width")
    layout = scans._horizontal_layout(r, i)
    r, i = [
        t if a or z else t.contiguous()
        for t, a, z in zip((r, i), layout["interleaved"], layout["transposed"], strict=True)
    ]
    launch = scans._plan_launch(
        r,
        shape,
        backward=backward,
        vertical=False,
        coarse=False,
        hidden=0,
        precise=False,
        compact=compact,
        aligned=scans._vec2_aligned(r, i),
        **layout,
    )
    # Short rows may use a register-only Wave 57 launch which reloads inputs.
    # Use the identical cached template for the return sweep in that case.
    if not backward and not launch.key.cache_line:
        threads = launch.key.warps * 32
        smem = max(launch.plan.smem, w * threads * (8 if launch.key.horizontal_vec2 else 4))
        launch = launch._replace(key=with_fields(launch.key, cache_line=True),
                                 plan=launch.plan._replace(cache_line=True, smem=smem))
    partial = scratch = None
    if backward:
        _check_storage(r, states, w)
        _check_storage(r, adjoints, w // (1 + compact))
        q = states
        dq = adjoints
        out = r.new_empty((b, h, w, 2 * m))
        partial = r.new_empty((launch.partials, 4, m), dtype=torch.float32)
        if not launch.plan.cache_line:
            scratch = r.new_empty((w // (1 + compact), launch.workers, m, 2), dtype=torch.float32)

    else:
        q = dq = None
        out = _words(r, w)

    inputs = (*(t.float().contiguous() for t in pole), r, i, q, dq, out, partial, scratch)
    _dispatch(
        r,
        launch,
        variant,
        w,
        (
            *map(scans._ptr, inputs),
            *map(ct.c_int, (*shape, launch.workers)),

        ),
    )
    if backward:
        return [*scans._reduce_partials(partial, pole), out]
    return out


def vertical(pole, r, states, aux, path, coarse, variant=ARM, grad=None):
    _check_source(r, variant)
    b, h, w, m = r.shape
    _check_storage(r, states, w)
    logical = r.unsqueeze(0).expand(4, -1, -1, -1, -1)
    if (
        not scans._pole_ok(pole, r)
        or not scans._vertical_inputs_ok(logical, aux, path, coarse=coarse)
        or path[0].shape[1] != 8
    ):
        raise ValueError("invalid vertical scan inputs")
    q = states
    backward = grad is not None
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    gr, gi = scans._vertical_adjoints(grad, logical, (b, oh, ow, m)) if backward else (None, None)
    interleaved = (not gr.is_contiguous(), not gi.is_contiguous()) if backward else (True, True)
    # Plan with the logical four-plane numel and dtype without allocating legacy states.
    launch = scans._plan_launch(
        logical,
        (b, h, w, m),
        backward=backward,
        vertical=True,
        coarse=coarse,
        hidden=8,
        precise=not backward,
        compact=backward and coarse,
        interleaved=interleaved,
    )
    outr = outi = dq = partial = scratch = checkpoints = None
    if backward:
        dq = _words(r, ow)
        partial = r.new_empty((launch.partials, 94, m), dtype=torch.float32)
        ck = scans._checkpoint_rows(h, launch.plan.chunk)
        if launch.plan.global_checkpoints and ck:
            checkpoints = r.new_empty((ck, launch.workers, m, 4), dtype=torch.float32)
    else:
        out = r.new_empty((b, oh, ow, 2 * m))
        outr, outi = out.split(m, -1)
    if backward and h == 56:
        positive = h if launch.key.l2_single_write or not launch.key.sgs else 0
        rows = positive + (0 if launch.plan.cache_line else oh)
        if rows:
            scratch = r.new_empty((rows, launch.workers, m, 4))
    elif backward:
        if launch.key.l2_single_write or not launch.key.sgs:
            scratch = r.new_empty((h, launch.workers, m, 4))
    elif not launch.plan.cache_line:
        scratch = r.new_empty((oh, launch.workers, m, 4))
    inputs = (
        *(t.float().contiguous() for t in pole),
        q,
        *(t.float().contiguous() for t in aux),
        *(t.contiguous() for t in path),
        gr,
        gi,
        outr,
        outi,
        dq,
        partial,
        scratch,
        checkpoints,
    )
    args = (
        *map(scans._ptr, inputs),
        *map(ct.c_int, (b, h, w, m, launch.workers, 8, scans._scalar_types(path))),
        ct.c_float(1e-6),
        ct.c_longlong(0),
        ct.c_longlong(0),

    )
    _dispatch(r, launch, variant, w, args)
    if backward:
        params = scans._reduce_partials(partial, [*pole, *path], 8)
        return [*params[:4], dq, *params[4:]]
    return out


def pipeline(pole, r, i, aux, path, coarse, grad):
    states = horizontal(pole, r, i, ARM)
    out = vertical(pole, r, states, aux, path, coarse, ARM)
    v = vertical(pole, r, states, aux, path, coarse, ARM, grad)
    h = horizontal(pole, r, i, ARM, states, v[4], coarse)
    return [out, *v[:4], *v[5:], *h]
