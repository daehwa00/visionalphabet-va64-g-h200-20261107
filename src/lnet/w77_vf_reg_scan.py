# ruff: noqa: ANN001, ANN201, ANN202, SLF001
"""Opt-in Wave 77 vertical forward kernels."""

from functools import lru_cache, partial

import torch

from . import w72_split_fp8_scan as fp

ALL = "w77_vf_reg_allw"
CACHED = "w77_vf_reg_wobi"
GLOBAL = "w77_vf_nocachew_reg_wobi"
HALF = "fp16x2_hidden"


def arms(label):
    return (ALL, CACHED, GLOBAL) + (() if label == "56d" else (HALF,))


def kernel_source(key, width, height, modes, *, arm):
    src = fp.kernel_source(key, width, height, modes)
    if not key.vertical or key.backward or arm not in arms(f"{height}{'c' if key.coarse else 'd'}"):
        raise ValueError("unsupported VF candidate")
    if arm == HALF:
        return fp._fp16x2_hidden(src)
    # Keep the control recurrence contraction. Only replace its weight prologue.
    start = src.index("    const I weight_fields = 11 * I(size);")
    end = src.index("    auto* ckpt = seg", start)
    prologue = """
    auto path = load_register_path_weights<T,I,true>(wi, bi, wo, types, m, active);
    auto* seg = shared;
"""
    if arm != ALL:
        # The twelve packed pairs hold bi and Wo. bo already lives outside the loop.
        init = src.index(
            "        #pragma unroll", src.index("RegisterPathWeights<T, VECTOR> weights;")
        )
        stop = src.index("        #pragma unroll", init + 1)
        src = src[:start] + prologue + src[end:]
        src = src[:init] + src[stop:]
        src = (
            src.replace("pairs[44]", "pairs[12]")
            .replace("pairs[32 + ", "pairs[")
            .replace("pairs[36 + ", "pairs[4 + ")
        )
        if arm == CACHED:
            old = "    auto* seg = shared;"
            src = src.replace(
                old,
                """
    for (int i=threadIdx.x;i<32*32;i+=threads) {
        int mode=i/32, pair=i%32;
        if (mode0+mode<M) {
            PathWeights<T,I,false> source{wi,bi,wo,types,mode0+mode,8};
            reinterpret_cast<Pair<T>*>(shared)[pair*32+mode]=Pair<T>::pack(source.input_pair(pair));
        }
    }
    __syncthreads();
    path.inputs = reinterpret_cast<Pair<T>*>(shared)+lane;
    auto* seg = shared+256;
""",
            )
            member = "const Pair<T>* inputs;"
            access = "inputs[j*32].unpack()"
        else:
            src = src.replace(
                "    auto* seg = shared;",
                """
    path.inputs = reinterpret_cast<const Pair<T>*>(wi) + m*32;
    auto* seg = shared;""",
            )
            member = "const Pair<T>* inputs;"
            access = """[&] {
                unsigned word;
                asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(word) : "l"(inputs+j));
                return reinterpret_cast<const Pair<T>*>(&word)->unpack();
            }()"""
        src = src.replace("Pair<T> pairs[12];", "Pair<T> pairs[12]; " + member)
        return src.replace("return pairs[j].unpack();", f"return {access};")
    return src[:start] + prologue + src[end:]


@lru_cache
def source_for(arm):
    return partial(kernel_source, arm=arm)


def shared_bytes(launch, h, arm):
    if arm == HALF:
        return fp.shared_bytes(launch, h)
    k = launch.key
    fp8_line = h == 56 or (h == 28 and not k.coarse)
    line = k.warps * 32 * (h // (1 + k.coarse)) * (4 if fp8_line else 8)
    return line * int(k.cache_line) + (4096 if arm == CACHED else 0)


def vertical(pole, r, states, aux, path, coarse, *, arm="w74", separate=False):
    if arm == GLOBAL and path[0].dtype != torch.bfloat16:
        raise ValueError("global Wi requires BF16")

    def dispatch(real, launch, width, args, scales):
        fp._dispatch(
            real,
            launch,
            width,
            args,
            scales,
            source_fn=source_for(arm),
            smem=shared_bytes(launch, width, arm),
        )

    return fp.vertical(
        pole,
        r,
        states,
        aux,
        path,
        coarse,
        separate=separate,
        dispatch=None if arm == "w74" else dispatch,
    )
