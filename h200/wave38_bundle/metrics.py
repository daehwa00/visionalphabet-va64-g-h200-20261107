"""Wave 38 kernel attribution and baseline-relative combo selection."""

from math import isfinite

from h200.liverms_v2_opt.bench_h200 import kernel_bucket

BASE = "w33_base"
COMBO = "w38_best_combo"
OFF = ("off", "off")
ARMS = {
    BASE: OFF,
    "w38a_cta_k2": ("cta_acc_k2", "off"),
    "w38a_cta_regw": ("cta_acc_regw", "off"),
    "w38a_warp_k2": ("warp_acc_k2", "off"),
    "w38a_warp_regw": ("warp_acc_regw", "off"),
    "w38a_warp_sgs": ("warp_acc_single_gs", "off"),
    "w38a_cta_sgs": ("cta_acc_single_gs", "off"),
    "w38b_w37_ref": ("off", "cuda_graphs,max_autotune_gemm"),
    "w38b_tma_gemm": ("off", "cuda_graphs,tma_gemm"),
    "w38b_tma_pw": ("off", "cuda_graphs,tma_gemm,max_autotune_pointwise"),
    "w38b_tma_aggr": ("off", "cuda_graphs,tma_gemm,aggressive_fusion"),
    "w38b_graphs_upd": ("off", "cuda_graphs,max_autotune_gemm,compiled_update"),
}
BUCKETS = (
    "general_vscan", "factorized_reader", "cublas_gemm", "general_hscan",
    "triton", "cudnn_conv", "fused_stem", "other",
)
PREFIXES = {
    "general_vscan": {
        name: name for name in (
            "vscan_bwd_cta_k2", "vscan_bwd_cta_k4", "vscan_bwd_cta_regw",
            "vscan_bwd_cta_k8", "vscan_bwd_warp_k2", "vscan_bwd_warp_regw",
            "vscan_bwd_warp_single_gs", "vscan_bwd_cta_single_gs",
            "legacy_vscan_fwd", "legacy_vscan_bwd",
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
        "triton_tem_fused": "inductor_triton_template",
        "triton_poi_fused": "inductor_triton_pointwise",
        "triton_red_fused": "inductor_triton_reduction",
        "triton_per_fused": "inductor_triton_persistent_reduction",
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
            (name.startswith("w38a_") and delta(row, "median_ms") <= -0.30
             and delta(row, "general_vscan") <= -0.30)
            or (name.startswith("w38b_") and delta(row, "median_ms") <= -0.30)
        )
    ]

    def winner(prefix: str) -> str | None:
        return min(
            (name for name in eligible if name.startswith(prefix)),
            key=lambda name: rows[name]["median_ms"], default=None,
        )

    scan, model = winner("w38a_"), winner("w38b_")
    model_preset = ARMS.get(model, OFF)[1]
    ref = rows.get("w38b_w37_ref", {})
    update = rows.get("w38b_graphs_upd", {})
    tma = winner("w38b_tma_")
    stacked = (
        tma is not None and valid(ref) and valid(update)
        and round(ref["median_ms"] - update["median_ms"], 9) >= 0.15
        and round(ref["median_ms"] - rows[tma]["median_ms"], 9) >= 0.15
    )
    if stacked:
        model = tma
        model_preset = ARMS[model][1] + ",compiled_update"
    return {
        "stacked_update": stacked,
        "eligible": eligible,
        "winners": [name for name in (scan, model) if name],
        "presets": [ARMS.get(scan, OFF)[0], model_preset],
    }


def nograph_preset(presets: tuple[str, str]) -> tuple[str, str]:
    scan, model = presets
    features = [part for part in model.split(",") if part not in {"cuda_graphs", "compiled_update"}]
    return scan, ",".join(features) or "off"
