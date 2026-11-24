"""Wave 37 kernel attribution and baseline-relative combo selection."""

from math import isfinite

from h200.liverms_v2_opt.bench_h200 import kernel_bucket

BASE = "w33_base"
COMBO = "w37_best_combo"
OFF = ("off", "off")
ARMS = {
    BASE: OFF,
    "w37a_cta_k2": ("cta_acc_k2", "off"),
    "w37a_cta_k4": ("cta_acc_k4", "off"),
    "w37a_cta_regw": ("cta_acc_regw", "off"),
    "w37a_cta_k8": ("cta_acc_k8_smem", "off"),
    "w37b_ortho": ("off", "capture_safe_ortho"),
    "w37b_cuda_graphs": ("off", "cuda_graphs"),
    "w37b_gemm_triton": ("off", "capture_safe_ortho,max_autotune_gemm"),
    "w37b_gemm_aten": ("off", "capture_safe_ortho,max_autotune_gemm_aten"),
    "w37b_graphs_gemm": ("off", "cuda_graphs,max_autotune_gemm"),
}
BUCKETS = (
    "general_vscan", "factorized_reader", "cublas_gemm", "general_hscan",
    "triton", "cudnn_conv", "fused_stem", "other",
)
PREFIXES = {
    "general_vscan": {
        name: name for name in (
            "vscan_bwd_cta_k2", "vscan_bwd_cta_k4", "vscan_bwd_cta_regw",
            "vscan_bwd_cta_k8", "legacy_vscan_fwd", "legacy_vscan_bwd",
        )
    } | {
        "reduce_vscan_partials": "vscan_reduce",
        "reduce_path_mlp_partials": "vscan_reduce",
        "path_mlp_forward": "legacy_vscan_fwd",
        "path_mlp_backward": "legacy_vscan_bwd",
    },
    "general_hscan": {
        "legacy_hscan": "legacy_hscan",
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

    def delta(row: dict, key: str) -> float:
        values = [entry.get(key, entry.get("buckets", {}).get(key)) for entry in (row, base)]
        if not all(number(value) for value in values):
            return float("inf")
        return round(values[0] - values[1], 9)

    eligible = [
        name for name, row in rows.items()
        if name in ARMS and name != BASE and valid(row) and number(base.get("median_ms"))
        and base.get("status") == "ok" and base.get("finite") is True
        and (
            (name.startswith("w37a_") and delta(row, "median_ms") <= -0.50
             and delta(row, "general_vscan") <= -0.50)
            or (name.startswith("w37b_") and delta(row, "median_ms") <= -0.30)
        )
    ]

    def winner(prefix: str) -> str | None:
        return min(
            (name for name in eligible if name.startswith(prefix)),
            key=lambda name: rows[name]["median_ms"], default=None,
        )

    scan, model = winner("w37a_"), winner("w37b_")
    return {
        "eligible": eligible,
        "winners": [name for name in (scan, model) if name],
        "presets": [ARMS.get(scan, OFF)[0], ARMS.get(model, OFF)[1]],
    }


def nograph_preset(presets: tuple[str, str]) -> tuple[str, str]:
    scan, model = presets
    features = ["capture_safe_ortho", *(part for part in model.split(",")
                                        if part.startswith("max_autotune_gemm"))]
    return scan, ",".join(features)
