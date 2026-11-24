"""Validate profiler names and baseline-relative eligibility."""
# ruff: noqa: S101

import pytest

from h200.wave37_bundle.metrics import BASE, classify, nograph_preset, select
from h200.wave37_bundle.run import selected_cases


@pytest.mark.parametrize(("name", "bucket", "group"), [
    ("vscan_bwd_cta_k2", "general_vscan", "vscan_bwd_cta_k2"),
    ("vscan_bwd_cta_k4", "general_vscan", "vscan_bwd_cta_k4"),
    ("vscan_bwd_cta_k8", "general_vscan", "vscan_bwd_cta_k8"),
    ("vscan_bwd_cta_regw", "general_vscan", "vscan_bwd_cta_regw"),
    ("legacy_vscan_fwd", "general_vscan", "legacy_vscan_fwd"),
    ("legacy_vscan_bwd", "general_vscan", "legacy_vscan_bwd"),
    ("reduce_vscan_partials", "general_vscan", "vscan_reduce"),
    ("legacy_hscan", "general_hscan", "legacy_hscan"),
    ("_rmsnorm_backward_kernel", "triton", "rmsnorm_bwd_custom"),
    ("_swiglu_vec2d_kernel", "triton", "swiglu_custom"),
    ("_nr4_correlation_kernel", "triton", "nr4_correlation_custom"),
])
def test_classify(name: str, bucket: str, group: str) -> None:
    assert classify("void " + name + "(float*)") == (bucket, group)


def test_auto_combo_selection() -> None:
    def row(wall: float, vscan: float | None, vram: float = 130.0) -> dict:
        return {"status": "ok", "finite": True, "peak_allocated_gib": vram,
                "median_ms": wall, "buckets": {"general_vscan": vscan}}

    rows = {
        BASE: row(10, 3),
        "w37a_cta_k2": row(9.5, 2.5),
        "w37a_cta_k4": row(9, 2.6),
        "w37a_cta_k8": row(8, 2, 130.1),
        "w37b_cuda_graphs": row(9.7, None),
        "w37b_gemm_aten": {**row(9, None), "finite": False},
    }
    result = select(rows)
    assert result["winners"] == ["w37a_cta_k2", "w37b_cuda_graphs"]
    assert result["presets"] == ["cta_acc_k2", "cuda_graphs"]
    assert nograph_preset(("cta_acc_k2", "cuda_graphs,max_autotune_gemm_aten")) == (
        "cta_acc_k2", "capture_safe_ortho,max_autotune_gemm_aten",
    )
    assert select({BASE: {"status": "error"}, **{k: v for k, v in rows.items() if k != BASE}})[
        "presets"
    ] == ["off", "off"]


def test_replay_case_with_inline_logs() -> None:
    output = (
        "tests/test_model_w36b.py::test_variant_compiled_cudagraph_replay CUDA Graph replays\n"
        "PASSED\ntests/test_model_w36b.py::test_variant_ortho_plus_max_autotune_gemm[k] FAILED\n"
        "tests/test_model_w36b.py::test_compile_policy PASSED\n"
    )
    assert selected_cases(output, "pytest_model") == ["test_variant_compiled_cudagraph_replay"]
