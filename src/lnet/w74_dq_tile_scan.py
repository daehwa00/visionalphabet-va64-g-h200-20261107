# ruff: noqa: ANN001, ANN201, ANN202, S603, S607, SLF001
"""Wave 74 shared-memory positive-gradient tiles for large vertical backward scans."""

import ctypes as ct
import re
import subprocess
import tempfile
from dataclasses import replace as with_fields
from functools import lru_cache
from pathlib import Path

import torch

from . import w72_split_fp8_scan as base
from .w72_split_fp8_scan import (
    _check_source,
    _check_storage,
    _words,
    compile_flags,
    configure,
    profile_name,
    scans,
)

VARIANTS = {
    "56d": (
        "dq_tile_full_nocachew",
        "dq_tile_hybrid_46",
        "dq_tile_hybrid_44",
        "dq_tile_hybrid_52_nocachew",
        "dq_tile_full_aw4",
    ),
    "56c": ("dq_tile_full",),
    "28d": ("dq_tile_full", "dq_tile_full_fp8seg"),
    "28c": ("dq_tile_full",),
}


def tile_rows(height, arm):
    return {"dq_tile_hybrid_44": 44, "dq_tile_hybrid_46": 46, "dq_tile_hybrid_52_nocachew": 52}.get(
        arm, height
    )


def candidate_plan(launch, height, arm):
    key = launch.key
    label = f"{height}{'c' if key.coarse else 'd'}"
    if arm not in VARIANTS.get(label, ()) or key.chunk or not key.cache_source or key.warps != 4:
        raise ValueError("unsupported Wave 74 tile or launch layout")
    key = with_fields(key, cache_weights=not arm.endswith("nocachew"))
    seg_bytes = 4 if height == 56 or arm.endswith("fp8seg") else 8
    threads = 32 * key.warps
    smem = (5632 if key.cache_weights else 0) + threads * (
        height * (seg_bytes + 4) + 4 * (height // (1 + key.coarse) + tile_rows(height, arm))
    )
    return launch._replace(key=key, plan=launch.plan._replace(smem=smem))


def kernel_source(key, width, height, modes, arm):
    src = base.kernel_source(key, width, height, modes)
    if not key.cache_weights:
        # Match NativeWeights' BF16 values and grouping directly from the mode-major inputs.
        marker = "struct ProjectWeights {"
        src = src.replace(
            marker,
            """
template<class T> struct GlobalWeights {
    static constexpr bool vector_weights = true;
    const T *wi, *bi, *wo;
    __device__ float4 input_group(int group) const {
        auto bits = __ldg(reinterpret_cast<const uint2*>(wi) + group);
        const auto* v = reinterpret_cast<const T*>(&bits);
        return make_float4(read(v[0]), read(v[1]), read(v[2]), read(v[3]));
    }
    __device__ float bias(int j) const { return read(__ldg(bi + j)); }
    __device__ float output(int j) const { return read(__ldg(wo + j)); }
};
"""
            + marker,
        )
        src = src.replace(
            "if constexpr (VECTOR_WEIGHTS && !REGISTER_PATH)",
            "if constexpr (VECTOR_WEIGHTS && CACHE_WEIGHT && !REGISTER_PATH)",
        )
        src = src.replace(
            "return NativeWeights<T>{weights_shared, size, lane};",
            """if constexpr (CACHE_WEIGHT) return NativeWeights<T>{weights_shared, size, lane};
                          else return GlobalWeights<T>{static_cast<const T*>(wi) + m * 64,
                              static_cast<const T*>(bi) + m * 8,
                              static_cast<const T*>(wo) + m * 16};""",
        )
    if arm.endswith("fp8seg"):
        src = src.replace("Quad<T>* seg,", "Fp8Quad* seg,").replace(
            "auto* seg = reinterpret_cast<Quad<T>*>", "auto* seg = reinterpret_cast<Fp8Quad*>"
        )
        src = re.sub(r"(seg\[[^\]]+\] = )Quad<T>::pack", r"\1Fp8Quad::pack", src)
    rows = tile_rows(height, arm)
    src = src.replace("Quad<T> positive = {}", "E5m2Quad positive = {}")
    for field in ("a", "b"):
        src = src.replace(f"positive.{field}.unpack()", f"positive.{field}.unpack(c.inv)")
    begin = src.index("void column_adjoint(")
    end = src.index("\n}", begin) + 2
    store = f"""
        if constexpr (REVERSE)
            write_column_gradient<true, COMPACT>(c, y, ds, gs, tile[y * {32 * key.warps}]);
        else tile[y * {32 * key.warps}] = E5m2Quad::pack(ds, c.scale);
    """
    default = "write_column_gradient<REVERSE, COMPACT, REVERSE>(c, y, ds, gs);"
    if rows < height:
        store = f"if (y < {rows}) {{ {store} }} else {{ {default} }}"
    helper = src[begin:end].replace("T* gs\n", "T* gs, E5m2Quad* tile\n").replace(default, store)
    src = src[:begin] + helper + src[end:]
    pattern = r"(column_adjoint<(?:true|false), COMPACT>\([^;]+?), gs\);"
    offset = height // (1 + key.coarse)
    src, count = re.subn(pattern, rf"\1, gs, g2 + {offset} * THREADS);", src)
    if count != 5:
        raise ValueError("unexpected column adjoint schedule")
    return src


@lru_cache(maxsize=256)
def _function(device, key, width, height, modes, smem, arm):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "packed.cu"
        path.write_text(kernel_source(key, width, height, modes, arm))
        cubin = path.with_suffix(".cubin")
        subprocess.run(
            [
                "nvcc",
                "-std=c++17",
                "-arch=sm_90a",
                *compile_flags(key, width),
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
            driver.cuModuleGetFunction(ct.byref(fn), module, profile_name(key).encode()),
            "get packed function",
        )
        scans._configure_smem(driver, fn, smem)
        scale_fn = ct.c_void_p()
        if key.vertical and key.backward:
            scans._check_cu(
                driver.cuModuleGetFunction(
                    ct.byref(scale_fn), module, b"va_profile_scale__w72_split_fp8"
                ),
                "get dq scale",
            )
        return driver, module, fn, scale_fn


def _dispatch(real, launch, width, args, scale_args=(), *, arm):
    device = real.device.index or 0
    key = launch.key
    driver, _module, fn, scale_fn = _function(
        device,
        key,
        width,
        real.shape[1],
        real.shape[-1],
        launch.plan.smem,
        arm,
    )
    grid = scans._ceil(real.shape[-1], 32 * (1 + key.horizontal_vec2)) * scans._ceil(
        launch.workers, key.warps
    )
    argv = (ct.c_void_p * len(args))(*(ct.addressof(arg) for arg in args))
    with torch.cuda.device(device):
        stream = ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        if scale_args:
            scale_argv = (ct.c_void_p * len(scale_args))(*(ct.addressof(v) for v in scale_args))
            scans._check_cu(
                driver.cuLaunchKernel(scale_fn, 1, 1, 1, 256, 1, 1, 0, stream, scale_argv, None),
                "launch dq scale",
            )
        scans._check_cu(
            driver.cuLaunchKernel(
                fn,
                grid,
                1,
                1,
                32 * key.warps,
                1,
                1,
                launch.plan.smem,
                stream,
                argv,
                None,
            ),
            "launch packed scan",
        )


def plan_for(r, coarse, gr, gi, arm):
    b, h, w, m = r.shape
    logical = r.unsqueeze(0).expand(4, -1, -1, -1, -1)
    launch = scans._plan_launch(
        logical,
        (b, h, w, m),
        backward=True,
        vertical=True,
        coarse=coarse,
        hidden=8,
        precise=False,
        compact=coarse,
        interleaved=(not gr.is_contiguous(), not gi.is_contiguous()),
    )
    if h == 28:
        launch = scans._plan_w57(r.device.index or 0, launch, (b, h, w, m), "v57_bwd_seg16_2chunk")
    launch = configure(launch, h)
    return launch if arm == "w72" else candidate_plan(launch, h, arm)


def vertical(
    pole,
    r,
    states,
    aux,
    path,
    coarse,
    grad=None,
    *,
    arm,
    epsilon=1e-6,
):
    _check_source(r)
    b, h, w, m = r.shape
    _check_storage(r, states, w)
    logical = r.unsqueeze(0).expand(4, -1, -1, -1, -1)
    if (
        not scans._pole_ok(pole, r)
        or not scans._vertical_inputs_ok(logical, aux, path, coarse=coarse)
        or path[0].shape[1] != 8
    ):
        raise ValueError("invalid vertical scan inputs")
    if grad is None or w != h or m != 160:
        raise ValueError("expected a large vertical backward scan")
    if arm.endswith("nocachew") and any(t.dtype != torch.bfloat16 for t in path[:3]):
        raise ValueError("uncached Wave 74 weights require BF16 storage")
    q = states
    ow = w // (1 + coarse)
    gr, gi = scans._vertical_adjoints(grad, logical, (b, h // (1 + coarse), ow, m))
    launch = plan_for(r, coarse, gr, gi, arm)
    dq = _words(r, ow)
    partial = r.new_empty((launch.partials, 94, m), dtype=torch.float32)
    outr = outi = scratch = checkpoints = None
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
        ct.c_float(epsilon),
        ct.c_longlong(0),
        ct.c_longlong(0),
    )
    scales = r.new_empty((2,), dtype=torch.float32)
    args += (scans._ptr(scales),)
    scale_args = (
        *map(scans._ptr, (gr, gi, scales)),
        ct.c_longlong(gr.numel()),
        *map(ct.c_int, (m, gr.stride(-2), gi.stride(-2))),
    )
    _dispatch(r, launch, w, args, scale_args, arm=arm)
    params = scans._reduce_partials(partial, [*pole, *path], 8)
    return [*params[:4], dq, *params[4:], scales]
