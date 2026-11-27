# ruff: noqa: ANN001, ANN201, SLF001
"""Opt-in SM90a vertical backward register-resident bias and output weights."""

from functools import lru_cache, partial

from . import w72_split_fp8_scan as fp

REG = "w76_reg_wobi"
K49 = "w76_reg_wobi_k49"
VB_ARMS = {
    s: (REG, K49) if s == "56d" else (REG,)
    for s in ("56d", "56c", "28d", "28c", "14d", "14c", "7d")
}


def kernel_source(key, width, height, modes, *, packed=None, k49=False):
    if not (key.vertical and key.backward and key.cache_weights and key.warps == 4):
        raise ValueError("expected cached four-warp VB")
    if packed is None:
        packed = height == 56
    native = fp.with_fields(key, vertical_w55="v55_bf16x2_bf16w") if height == 7 else key
    src = fp.kernel_source(native, width, height, modes).replace(
        fp.profile_name(native, height), fp.profile_name(key, height)
    )
    subst = fp.w66.replace
    src = subst(src, "matrix < (BACKWARD ? 3 : 4)", "matrix < 1")
    src = subst(src, "weight_fields = 11 * I(size)", "weight_fields = 8 * I(size)")
    fields = "__nv_bfloat162 resident[12];" if packed else "float resident[24];"
    src = subst(src, "int size,lane;", "int size,lane; " + fields)
    for offset, base in ((8, 0), (9, 8)):
        value = f"resident[{base}+j]"
        src = subst(
            src,
            f"read(cache[({offset}*size+j)*32+lane])",
            f"((j&1) ? __high2float(resident[({base}+j)/2]) : __low2float(resident[({base}+j)/2]))"
            if packed
            else value,
        )
    init = """
            #pragma unroll
            for (int j=0; j<24; ++j) {
                const bool bias = j<8;
                const int index = bias ? m*8+j : m*16+j-8;
                result.resident[j] = active ? rounded<T>(load_scalar(
                    bias ? bi : wo, index, (types >> (bias ? 2 : 4)) & 3)) : 0.f;
            }"""
    if packed:
        init = """
            #pragma unroll
            for (int j=0; j<12; ++j) {
                const bool bias = j<4;
                const int index = bias ? m*8+2*j : m*16+2*j-8;
                const void* src = bias ? bi : wo;
                const int dtype = (types >> (bias ? 2 : 4)) & 3;
                result.resident[j] = __floats2bfloat162_rn(
                    active ? load_scalar(src,index,dtype) : 0.f,
                    active ? load_scalar(src,index+1,dtype) : 0.f);
            }"""
    src = subst(
        src,
        "return NativeWeights<T>{weights_shared, size, lane};",
        "NativeWeights<T> result{weights_shared, size, lane};" + init + "return result;",
    )
    return subst(src, "if (y < 46)", "if (y < 49)") if k49 else src


def shared_bytes(launch, height, arm):
    return (
        fp.shared_bytes(launch, height)
        - (7168 if height == 7 else 1536)
        + (1536 if arm == K49 else 0)
    )


def dispatch(real, launch, width, args, scale_args, *, arm):
    fp._dispatch(
        real,
        launch,
        width,
        args,
        scale_args,
        source_fn=source_for(arm),
        smem=shared_bytes(launch, real.shape[1], arm),
    )


@lru_cache
def source_for(arm):
    return partial(kernel_source, k49=arm == K49)


def vertical(pole, r, states, aux, path, coarse, grad, *, arm):
    label = f"{r.shape[1]}{'c' if coarse else 'd'}"
    if arm != "w74" and arm not in VB_ARMS[label]:
        raise ValueError(f"unsupported {label}/{arm}")
    run = None if arm == "w74" else partial(dispatch, arm=arm)
    return fp.vertical(pole, r, states, aux, path, coarse, grad, dispatch=run)
