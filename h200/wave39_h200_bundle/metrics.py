"""Wave 39 kernel attribution and baseline-relative combo selection."""

from math import isfinite

from h200.liverms_v2_opt.bench_h200 import kernel_bucket

BASE = "w33_base"
COMBO = "w39_best_combo"
OFF = ("off", "off")
TMA_PW = "cuda_graphs,tma_gemm,max_autotune_pointwise"
ARMS = {
    BASE: OFF,
    "w39a_wmma_bwd": ("wmma_bwd", "off"),
    "w39a_wmma_fwd": ("wmma_fwd", "off"),
    "w39a_wmma_both": ("wmma_both", "off"),
    "w39a_legacy_sgs": ("legacy_sgs", "off"),
    "w39b_tma_pw": ("off", TMA_PW),
    "w39b_tma_pw_upd": ("off", TMA_PW + ",compiled_update"),
    "w39b_tma_pw_cd": ("off", TMA_PW + ",coord_descent"),
    "w39b_tma_pw_pad": ("off", TMA_PW + ",shape_padding"),
    "w39b_graphs_upd": ("off", "cuda_graphs,max_autotune_gemm,compiled_update"),
}
BUCKETS = (
    "general_vscan", "factorized_reader", "cublas_gemm", "general_hscan",
    "triton", "cudnn_conv", "fused_stem", "other",
)
PREFIXES = {
    "general_vscan": {
        name: name for name in (
            "vscan_bwd_wmma", "vscan_fwd_wmma", "vscan_bwd_legacy_sgs",
            "vscan_bwd_cta_k2", "vscan_bwd_cta_k4", "vscan_bwd_cta_regw",
            "vscan_bwd_cta_k8", "vscan_bwd_warp_k2", "vscan_bwd_warp_regw",
            "vscan_bwd_warp_single_gs", "vscan_bwd_cta_single_gs",
            "legacy_vscan_fwd", "legacy_vscan_bwd",
        )
    } | {
        "vscan_wmma_bwd": "vscan_bwd_wmma",
        "vscan_wmma_fwd": "vscan_fwd_wmma",
        "vscan_legacy_sgs": "vscan_bwd_legacy_sgs",
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
        "triton_tem_": "inductor_triton_template",
        "triton_poi_": "inductor_triton_pointwise",
        "triton_red_": "inductor_triton_reduction",
        "triton_per_": "inductor_triton_persistent_reduction",
        "triton_": "inductor_triton_other",
    },
}
SUBGROUPS = sorted(
    {group for names in PREFIXES.values() for group in names.values()}
    | {"cudnn_conv", "cublas_gemm", "scan"},
)
def classify(name: str) -> tuple[str, str | None]:
    name = name.removeprefix("void ")
    if name.split("(", 1)[0] == "scan":
        return "other", "scan"
    for bucket, groups in PREFIXES.items():
        for prefix, group in groups.items():
            if name.startswith(prefix):
                return bucket, group
    bucket = kernel_bucket(name)
    return (
        bucket if bucket in BUCKETS else "other",
        bucket if bucket in {"cudnn_conv", "cublas_gemm"} else None,
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

    eligible = []
    if valid(base):
        for name, row in rows.items():
            if name not in ARMS or name == BASE or not valid(row):
                continue
            if round(base["median_ms"] - row["median_ms"], 9) < 0.15:
                continue
            replays = row.get("cudagraph_replays") or {}
            if name.startswith("w39b_") and any(
                (replays.get(phase) or 0) < 1 for phase in ("forward", "backward")
            ):
                continue
            eligible.append(name)

    def winner(prefix: str) -> str | None:
        return min(
            (name for name in eligible if name.startswith(prefix)),
            key=lambda name: rows[name]["median_ms"], default=None,
        )

    scan, model = winner("w39a_"), winner("w39b_")
    model_preset = ARMS.get(model, OFF)[1]
    return {
        "eligible": eligible,
        "winners": [name for name in (scan, model) if name],
        "presets": [ARMS.get(scan, OFF)[0], model_preset],
        "model_update_replays": rows[model]["cudagraph_replays"].get("update")
        if model and "compiled_update" in model_preset.split(",") else None,
    }
