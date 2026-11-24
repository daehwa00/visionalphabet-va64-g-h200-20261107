"""Wave 36 kernel attribution and baseline-relative combo selection."""

from math import isfinite
from types import MappingProxyType

from h200.liverms_v2_opt.bench_h200 import kernel_bucket

BASE = "w33_base"
COMBO = "w36_best_combo"
OFF = ("off", "off")
ARMS = {
    BASE: OFF,
    "w36a_saved_fast_proj": ("saved_fast_proj", "off"),
    "w36a_saved_fast_proj_gs1": ("saved_fast_proj_gs1", "off"),
    "w36a_hscan_bwd_hiocc": ("hscan_bwd_hiocc", "off"),
    "w36b_stem_cudnn": ("off", "stem_cudnn"),
    "w36b_capture_safe_ortho": ("off", "capture_safe_ortho"),
    "w36b_cuda_graphs": ("off", "cuda_graphs"),
    "w36b_max_autotune_gemm": ("off", "max_autotune_gemm"),
}
BUCKETS = (
    "general_vscan", "factorized_reader", "cublas_gemm", "general_hscan",
    "triton", "cudnn_conv", "fused_stem", "other",
)
EXPECTED_11 = ("pure_vscan_fwd", "pure_vscan_bwd", "project_path_fwd_reg", "project_path_bwd_k8")
PREFIXES = {
    "general_vscan": {
        **dict(zip(EXPECTED_11, EXPECTED_11, strict=True)),
        "project_path_bwd_reduce": "project_path_bwd_reduce",
        "reduce_path_mlp_partials": "project_path_bwd_reduce",
        "legacy_vscan": "legacy_vscan",
        "vscan_legacy": "legacy_vscan",
        "reduce_vscan_partials": "legacy_vscan",
        "path_mlp_": "legacy_vscan",
    },
    "general_hscan": {
        "hscan_bwd_hiocc": "hscan_bwd_hiocc",
        "legacy_hscan": "legacy_hscan",
        "hscan_legacy": "legacy_hscan",
        "reduce_hscan_partials": "legacy_hscan",
    },
    "fused_stem": {
        "stem_conv_tensor_stage0_fwd": "fused_stem_stage0_fwd",
        "stem_conv_tensor_stage0_bwd": "fused_stem_stage0_bwd",
        "stem_conv_tensor_stage0_reduce": "fused_stem_stage0_reduce",
        "stem_conv_tensor_reduce": "fused_stem_stage0_reduce",
        "conv_stem_forward": "fused_stem_stage0_fwd",
        "conv_stem_backward": "fused_stem_stage0_bwd",
        "conv_stem_reduce": "fused_stem_stage0_reduce",
        "stem_conv_tensor_stages": "fused_stem_stage0_unsplit",
        "fused_nhwc_layernorm_gelu_fwd": "fused_nhwc_layernorm_gelu_fwd",
        "fused_nhwc_layernorm_gelu_bwd": "fused_nhwc_layernorm_gelu_bwd",
        "stem_forward": "fused_nhwc_layernorm_gelu_fwd",
        "stem_backward": "fused_nhwc_layernorm_gelu_bwd",
        "stem_reduce": "fused_nhwc_layernorm_gelu_reduce",
    },
    "triton": {
        "_swiglu_vec2d_kernel": "swiglu_custom",
        "_rmsnorm_backward_kernel": "rmsnorm_bwd_custom",
        "_nr4_correlation_kernel": "nr4_correlation_custom",
        "triton_": "inductor_triton_other",
    },
}
SUBGROUPS = sorted(
    {group for names in PREFIXES.values() for group in names.values()} | {"cudnn_conv"},
)
RULES = MappingProxyType({
    "w36a_saved_fast_proj": ((("median_ms",), -0.50), (("general_vscan",), -0.50)),
    "w36a_saved_fast_proj_gs1": ((("median_ms",), -0.50), (("general_vscan",), -0.50)),
    "w36a_hscan_bwd_hiocc": ((("general_hscan",), -0.25), (("total_self_cuda_ms",), -0.20)),
    "w36b_stem_cudnn": ((("fused_stem", "cudnn_conv"), -0.30), (("total_self_cuda_ms",), -0.20)),
    "w36b_capture_safe_ortho": ((("median_ms",), -0.50), (("total_self_cuda_ms",), 0.50)),
    "w36b_cuda_graphs": ((("median_ms",), -0.50), (("total_self_cuda_ms",), 0.50)),
    "w36b_max_autotune_gemm": ((("cublas_gemm",), -0.30), (("total_self_cuda_ms",), -0.25)),
})


def classify(name: str) -> tuple[str, str | None]:
    name = name.removeprefix("void ")
    for bucket, groups in PREFIXES.items():
        for prefix, group in groups.items():
            if name.startswith(prefix):
                return bucket, group
    bucket = kernel_bucket(name)
    return (
        bucket if bucket in BUCKETS else "other",
        "cudnn_conv" if bucket == "cudnn_conv" else None,
    )


def number(value: object) -> bool:
    return type(value) in (float, int) and isfinite(value)


def valid(row: dict) -> bool:
    return (
        row.get("status") == "ok"
        and row.get("finite") is True
        and number(row.get("median_ms"))
        and number(row.get("peak_allocated_gib"))
        and 0 <= row["peak_allocated_gib"] <= 130.0
    )


def select(rows: dict) -> dict:
    base = rows[BASE]

    def delta(row: dict, keys: tuple[str, ...]) -> float:
        values = [entry.get("buckets", {}).get(key, entry.get(key))
                  for key in keys for entry in (row, base)]
        if not all(number(value) for value in values):
            return float("inf")
        return round(sum(values[::2]) - sum(values[1::2]), 9)

    def qualify(name: str) -> bool:
        row = rows[name]
        if not valid(base) or not valid(row):
            return False
        return (
            number(row.get("total_self_cuda_ms")) and number(base.get("total_self_cuda_ms"))
            and all(delta(row, keys) <= limit for keys, limit in RULES[name])
        )

    eligible = [name for name in RULES if qualify(name)]
    winners = set(eligible)
    for pair, order in (
        (("w36a_saved_fast_proj", "w36a_saved_fast_proj_gs1"), ("total_self_cuda_ms", "median_ms")),
        (("w36b_capture_safe_ortho", "w36b_cuda_graphs"), ("median_ms",)),
    ):
        candidates = [name for name in pair if name in winners]
        if candidates:
            winners.difference_update(candidates)
            winners.add(min(candidates, key=lambda name: tuple(rows[name][key] for key in order)))

    selected = [name for name in ARMS if name in winners]
    presets = [",".join(ARMS[name][axis] for name in selected if ARMS[name][axis] != "off")
               or "off" for axis in range(2)]
    return {"eligible": eligible, "winners": selected, "presets": presets}


def nograph_preset(presets: tuple[str, str], *, use_ortho: bool) -> tuple[str, str]:
    scan, model = presets
    features = [item for item in model.split(",") if item != "cuda_graphs"]
    if use_ortho:
        features.append("capture_safe_ortho")
    return scan, ",".join(features) or "off"
