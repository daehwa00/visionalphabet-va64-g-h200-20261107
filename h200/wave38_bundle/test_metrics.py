"""Validate profiler names and baseline-relative eligibility."""
# ruff: noqa: S101

import pytest

from h200.wave38_bundle.arm import compact_shapes
from h200.wave38_bundle.metrics import BASE, classify, nograph_preset, select
from h200.wave38_bundle.run import selected_cases


@pytest.mark.parametrize(("name", "bucket", "group"), [
    ("vscan_bwd_cta_k2", "general_vscan", "vscan_bwd_cta_k2"),
    ("vscan_bwd_cta_k4", "general_vscan", "vscan_bwd_cta_k4"),
    ("vscan_bwd_cta_k8", "general_vscan", "vscan_bwd_cta_k8"),
    ("vscan_bwd_cta_regw", "general_vscan", "vscan_bwd_cta_regw"),
    ("vscan_bwd_warp_k2", "general_vscan", "vscan_bwd_warp_k2"),
    ("vscan_bwd_warp_regw", "general_vscan", "vscan_bwd_warp_regw"),
    ("vscan_bwd_warp_single_gs", "general_vscan", "vscan_bwd_warp_single_gs"),
    ("vscan_bwd_cta_single_gs", "general_vscan", "vscan_bwd_cta_single_gs"),
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
        "w38a_cta_k2": row(9.5, 2.7),
        "w38a_warp_k2": row(9, 2.71),
        "w38a_warp_sgs": row(8, 2, 130.1),
        "w38b_w37_ref": row(9.7, None),
        "w38b_tma_gemm": row(9.55, None),
        "w38b_graphs_upd": row(9.5, None),
        "w38b_tma_pw": {**row(9, None), "finite": False},
    }
    result = select(rows)
    assert result["winners"] == ["w38a_cta_k2", "w38b_tma_gemm"]
    assert result["presets"] == ["cta_acc_k2", "cuda_graphs,tma_gemm,compiled_update"]
    assert result["stacked_update"]
    rows["w38b_tma_gemm"] = row(9.56, None)
    assert select(rows)["winners"] == ["w38a_cta_k2", "w38b_graphs_upd"]
    assert nograph_preset(("cta_acc_k2", "cuda_graphs,tma_gemm,compiled_update")) == (
        "cta_acc_k2", "tma_gemm",
    )
    rows[BASE] = {"status": "error"}
    assert select(rows)["presets"] == ["off", "off"]


def test_scan_shape_preservation() -> None:
    plans = [{"variant": "warp_acc_single_gs", "workers": 2, "count": 3}]
    expected = [
        {"name": f"vscan_bwd_{shape}", "cuda_event_ms": 0.12346, "count": 3, "plan": plans}
        for shape in ("56_d", "56_c", "28_d", "28_c", "14_d", "14_c", "7_d")
    ]
    source = [
        {**row, "cuda_event_ms": 0.123456, "plans": row["plan"]}
        for row in expected
    ]
    assert compact_shapes([*source, {"name": "hscan_bwd_56_d"}]) == expected


def test_replay_case_with_inline_logs() -> None:
    output = (
        "tests/test_model_w36b.py::test_variant_compiled_cudagraph_replay CUDA Graph replays\n"
        "PASSED\ntests/test_model_w36b.py::test_variant_ortho_plus_max_autotune_gemm[k] FAILED\n"
        "tests/test_model_w36b.py::test_compile_policy PASSED\n"
    )
    assert selected_cases(output, "pytest_model") == ["test_variant_compiled_cudagraph_replay"]
