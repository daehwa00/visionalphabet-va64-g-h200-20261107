"""Generate experimental scan specializations without changing production dispatch."""

from dataclasses import dataclass, replace

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001

SOURCE = scans._kernel_source


@dataclass(frozen=True)
class TrialKernel(scans.Kernel):
    register_batch: int = 0
    packed: bool = False
    unroll: int = 0
    cap: int = 0
    cache_vy: bool = False
    horizontal_unroll: int = 0
    horizontal_ldg: bool = False


REGISTER_WEIGHTS = r"""
template<class T, bool PACKED> struct RegisterValue { using Type = Complex; };
template<class T> struct RegisterValue<T, true> { using Type = Pair<T>; };
template<class T, bool PACKED>
struct RegisterWeights {
    using Value = typename RegisterValue<T, PACKED>::Type;
    static constexpr bool vector_weights = false;
    int size = 8;
    Value pairs[44];

    __device__ void store(int j, Complex p) {
        if constexpr (PACKED) pairs[j] = Pair<T>::pack(p);
        else pairs[j] = p;
    }
    __device__ Complex load(int j) const {
        if constexpr (PACKED) return pairs[j].unpack();
        else return pairs[j];
    }
    __device__ Complex input_pair(int j) const { return load(j); }
    __device__ float bias(int j) const {
        const auto p = load(32 + j / 2);
        return j % 2 ? p.i : p.r;
    }
    __device__ float output(int j) const {
        const auto p = load(36 + j / 2);
        return j % 2 ? p.i : p.r;
    }
};

template<class T, class I, bool PACKED>
__device__ __forceinline__ auto register_weights(
    const void* wi, const void* bi, const void* wo, int types, int m, bool active
) {
    const PathWeights<T, I, false> src{wi, bi, wo, types, m, 8};
    RegisterWeights<T, PACKED> result;
    if (active) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) result.store(j, src.input_pair(j));
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            result.store(32 + j, {src.bias(2 * j), src.bias(2 * j + 1)});
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            result.store(36 + j, {src.output(2 * j), src.output(2 * j + 1)});
    }
    return result;
}
"""


def source(key: scans.Kernel) -> str:  # noqa: C901
    cuda = SOURCE(key)
    if not isinstance(key, TrialKernel):
        return cuda
    if key.cap:
        original = f"__launch_bounds__({key.warps * 32})"
        cuda = cuda.replace(original + " void scan", f"__maxnreg__({key.cap}) void scan")
    if key.horizontal_unroll:
        a = cuda.index("void horizontal(")
        b = cuda.index("template<bool PRECISE>", a)
        directive = "#pragma unroll BACKWARD_UNROLL"
        if directive not in cuda[a:b]:
            directive = "#pragma unroll 4"
        horizontal = cuda[a:b].replace(directive, f"#pragma unroll {key.horizontal_unroll}")
        cuda = cuda[:a] + horizontal + cuda[b:]
    if key.horizontal_ldg:
        for axis in ("r", "i"):
            offset = f"s{axis}[base_{axis} + I(x) * step_{axis}]"
            cuda = cuda.replace(f"read({offset})", f"read(__ldg(&{offset}))")
    if key.cache_vy:
        cuda = cuda.replace("vy[(I(reverse) * H + y) * M + m]",
                            "vy[(int(reverse) * H + y) * 32 + (m & 31)]")
        start = cuda.index("    float* row = partials;")
        stage = """
    auto* vy_cache = reinterpret_cast<float*>(line_buf + (CACHE_LINE ? oh * threads : 0));
    for (int i = threadIdx.x; i < 2 * H * 32; i += threads) {
        const int mode = mode0 + i % 32;
        if (mode < M) vy_cache[i] = vy[I(i / 32) * M + mode];
    }
    __syncthreads();
"""
        cuda = cuda[:start] + stage + cuda[start:]
        cuda = cuda.replace("c{states, vy,", "c{states, vy_cache,")
    if key.unroll:
        start = cuda.index("float2 path_mlp(")
        end = cuda.index("for (int j = 0; j < size; ++j)", start)
        directive = cuda.rindex("#pragma unroll", start, end)
        cuda = cuda[:directive] + cuda[directive:].replace(
            "#pragma unroll", f"#pragma unroll {key.unroll}", 1,
        )
    if key.register_batch:
        start = cuda.index("template<class T, class I, bool COARSE, int WARPS")
        cuda = cuda[:start] + REGISTER_WEIGHTS + cuda[start:]
        path_start = cuda.index("    if constexpr (VECTOR_WEIGHTS", start)
        path_end = cuda.index("    constexpr int weight_itemsize", path_start)
        packed = str(key.packed).lower()
        cuda = (
            cuda[:path_start]
            + f"    const auto path = register_weights<T, I, {packed}>("
            + "wi, bi, wo, types, m, active);\n"
            + cuda[path_end:]
        )
        start = cuda.index("// The vertical recurrence depends on the source alone.")
        end = cuda.index("for (int y = remaining - 1;", start)
        part = cuda[start:end]
        for old, new in (
            ("4 * (1 + COARSE)", f"{key.register_batch} * (1 + COARSE)"),
            ("p[4][8]", f"p[{key.register_batch}][8]"),
            ("out[4]", f"out[{key.register_batch}]"),
            ("t < 4", f"t < {key.register_batch}"),
            ("PRECISE, 4>", f"PRECISE, {key.register_batch}>"),
        ):
            part = part.replace(old, new)
        cuda = cuda[:start] + part + cuda[end:]
    return cuda


def experimental(key: scans.Kernel, name: str) -> scans.Kernel:  # noqa: PLR0911
    if name == "baseline":
        return key
    trial = TrialKernel(**vars(key))
    if name.startswith(("reg", "float")):
        return replace(
            trial,
            register_batch=int(name[-1]),
            packed=name.startswith("reg"),
            cache_weights=False,
            batch_path=name[-1] != "1",
        )
    if name.startswith("hu"):
        return replace(trial, horizontal_unroll=int(name[-1]))
    if name == "hldg":
        return replace(trial, horizontal_ldg=True)
    if name == "cap192":
        return replace(trial, cap=192)
    if name == "vy":
        return replace(trial, cache_vy=True)
    if name.startswith("unroll"):
        return replace(trial, unroll=int(name[-1]))
    if name.startswith("packed"):
        return replace(trial, vector_weights=False)
    raise ValueError(name)
