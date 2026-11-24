"""Check kernel attribution and Wave 36 winner selection without CUDA."""

from copy import deepcopy

import pytest

from h200.wave36_bundle.metrics import ARMS, BASE, classify, nograph_preset, select


@pytest.mark.parametrize(("symbol", "bucket", "group"), [
    ("_rmsnorm_backward_kernel", "triton", "rmsnorm_bwd_custom"),
    ("_swiglu_vec2d_kernel", "triton", "swiglu_custom"),
    ("_nr4_correlation_kernel", "triton", "nr4_correlation_custom"),
    ("pure_vscan_fwd", "general_vscan", "pure_vscan_fwd"),
    ("pure_vscan_bwd", "general_vscan", "pure_vscan_bwd"),
    ("project_path_fwd_reg", "general_vscan", "project_path_fwd_reg"),
    ("project_path_bwd_k8", "general_vscan", "project_path_bwd_k8"),
    ("reduce_path_mlp_partials", "general_vscan", "project_path_bwd_reduce"),
    ("hscan_bwd_hiocc", "general_hscan", "hscan_bwd_hiocc"),
    ("legacy_hscan", "general_hscan", "legacy_hscan"),
    ("legacy_vscan", "general_vscan", "legacy_vscan"),
    ("stem_forward", "fused_stem", "fused_nhwc_layernorm_gelu_fwd"),
    ("fused_nhwc_layernorm_gelu_bwd", "fused_stem", "fused_nhwc_layernorm_gelu_bwd"),
    ("stem_conv_tensor_stage0_fwd", "fused_stem", "fused_stem_stage0_fwd"),
    ("stem_conv_tensor_stage0_bwd", "fused_stem", "fused_stem_stage0_bwd"),
    ("stem_conv_tensor_reduce", "fused_stem", "fused_stem_stage0_reduce"),
    ("void cudnn::engines::fprop", "cudnn_conv", "cudnn_conv"),
    ("void reader_dweight<half>", "factorized_reader", None),
    ("void nvjet_hshgemm", "cublas_gemm", None),
])
def test_kernel_classifier(symbol: str, bucket: str, group: str | None) -> None:
    assert classify(symbol) == (bucket, group)


def test_select_and_fallback() -> None:
    base = {
        "status": "ok", "finite": True, "peak_allocated_gib": 129,
        "median_ms": 185, "total_self_cuda_ms": 183,
        "buckets": {"general_vscan": 80, "general_hscan": 35, "fused_stem": 14,
                    "cudnn_conv": 3, "cublas_gemm": 20},
    }
    rows = {name: deepcopy(base) for name in ARMS}
    assert select(rows) == {"eligible": [], "winners": [], "presets": ["off", "off"]}
    a, b = "w36a_saved_fast_proj", "w36a_saved_fast_proj_gs1"
    hscan, stem = "w36a_hscan_bwd_hiocc", "w36b_stem_cudnn"
    ortho, graph, gemm = "w36b_capture_safe_ortho", "w36b_cuda_graphs", "w36b_max_autotune_gemm"
    rows[a].update(median_ms=180, total_self_cuda_ms=180)
    rows[b].update(median_ms=182, total_self_cuda_ms=179, peak_allocated_gib=130)
    rows[hscan].update(total_self_cuda_ms=182.80)
    rows[stem].update(total_self_cuda_ms=182.80)
    rows[ortho].update(median_ms=184.5, total_self_cuda_ms=183.5)
    rows[graph].update(median_ms=183.5, total_self_cuda_ms=183.5)
    rows[gemm].update(total_self_cuda_ms=182.75)
    assert select(rows)["eligible"] == [ortho, graph]
    rows[a]["buckets"]["general_vscan"] = 79.5
    rows[b]["buckets"]["general_vscan"] = 79.5
    rows[hscan]["buckets"]["general_hscan"] = 34.75
    rows[stem]["buckets"]["fused_stem"] = 13.70
    rows[gemm]["buckets"]["cublas_gemm"] = 19.70
    result = select(rows)
    assert result["winners"] == [b, hscan, stem, graph, gemm]
    presets = ("saved_fast_proj_gs1,hscan_bwd_hiocc", "stem_cudnn,cuda_graphs,max_autotune_gemm")
    assert result["presets"] == list(presets)
    assert nograph_preset(presets, use_ortho=True) == (
        presets[0], "stem_cudnn,max_autotune_gemm,capture_safe_ortho",
    )
    assert nograph_preset(presets, use_ortho=False) == (presets[0], "stem_cudnn,max_autotune_gemm")
    rows[b]["peak_allocated_gib"] += 0.01
    rows[graph]["total_self_cuda_ms"] = None
    rows[hscan]["status"] = "timeout"
    rows[stem]["finite"] = False
    assert select(rows)["winners"] == [a, ortho, gemm]
    rows[BASE]["status"] = "error"
    assert select(rows)["winners"] == []
