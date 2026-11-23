"""Build the experimental FP32 gradient window without changing dispatch."""

# ruff: noqa: ANN201, SLF001
from unittest.mock import patch

from lnet import factorized_reader_kernels as reader


def compile_weight(*args: object, convert: bool, **kwargs: object):
    if not convert:
        return reader._reader_cuda_module(*args, **kwargs)

    source = reader._READER_CUDA
    weight = source.index("void real_weight_gradient_3x3")
    a = source.index("    using Window = ", weight)
    b = source.index("    constexpr int RankTile", a)
    end = source.index("    // Stage a subset", b)
    window = """    constexpr bool ConvertPacked = Vec == 2;
    const auto load_gradient = [](const G* ptr) {
        if constexpr (ConvertPacked) return mode_load<Vec, G, ReadOnly>(ptr);
        else return window_load<Vec, ReadOnly, ConvertWindow>(ptr);
    };
    using Window = decltype(load_gradient(grad));
    const auto gradient_values = [](Window value) {
        if constexpr (ConvertPacked) return value;
        else return window_values<Vec, G>(value);
    };
"""
    body = source[b:end].replace("window_load<Vec, ReadOnly, ConvertWindow>", "load_gradient")
    body = body.replace("window_values<Vec, G>", "gradient_values")
    with patch.object(reader, "_READER_CUDA", source[:a] + window + body + source[end:]):
        return reader._reader_cuda_module.__wrapped__(*args, **kwargs)
