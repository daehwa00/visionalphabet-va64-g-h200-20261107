"""Validate profiler names and baseline-relative eligibility."""
# ruff: noqa: S101

import pytest

from h200.wave39_h200_bundle.arm import compact_shapes
from h200.wave39_h200_bundle.metrics import BASE, TMA_PW, classify, select
from h200.wave39_h200_bundle.run import selected_cases


@pytest.mark.parametrize(("name", "bucket", "group"), [
    ("vscan_bwd_cta_k2", "general_vscan", "vscan_bwd_cta_k2"),
    ("vscan_bwd_cta_k4", "general_vscan", "vscan_bwd_cta_k4"),
    ("vscan_bwd_cta_k8", "general_vscan", "vscan_bwd_cta_k8"),
    ("vscan_bwd_cta_regw", "general_vscan", "vscan_bwd_cta_regw"),
    ("vscan_bwd_warp_k2", "general_vscan", "vscan_bwd_warp_k2"),
    ("vscan_bwd_warp_regw", "general_vscan", "vscan_bwd_warp_regw"),
    ("vscan_bwd_warp_single_gs", "general_vscan", "vscan_bwd_warp_single_gs"),
    ("vscan_bwd_cta_single_gs", "general_vscan", "vscan_bwd_cta_single_gs"),
    ("vscan_wmma_bwd", "general_vscan", "vscan_bwd_wmma"),
    ("vscan_wmma_fwd", "general_vscan", "vscan_fwd_wmma"),
    ("vscan_legacy_sgs", "general_vscan", "vscan_bwd_legacy_sgs"),
    ("scan", "other", "scan"),
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
    def row(wall: float, vram: float = 130.0, forward: int = 1) -> dict:
        return {"status": "ok", "finite": True, "peak_allocated_gib": vram,
                "median_ms": wall, "cudagraph_replays": {
                    "forward": forward, "backward": 1, "update": 2,
                }}

    rows = {
        BASE: row(10),
        "w39a_wmma_bwd": row(9.85),
        "w39a_wmma_fwd": row(9.851),
        "w39a_wmma_both": row(8, vram=130.1),
        "w39a_legacy_sgs": {**row(8), "finite": False},
        "w39b_tma_pw": row(8, forward=0),
        "w39b_tma_pw_upd": row(9),
        "w39b_tma_pw_cd": row(9.1),
    }
    result = select(rows)
    assert result["winners"] == ["w39a_wmma_bwd", "w39b_tma_pw_upd"]
    assert result["presets"] == ["wmma_bwd", TMA_PW + ",compiled_update"]
    assert result["model_update_replays"] == 2
    rows[BASE] = {"status": "error"}
    assert select(rows)["presets"] == ["off", "off"]


def test_scan_shape_preservation() -> None:
    plans = [{"variant": "warp_acc_single_gs", "workers": 2, "count": 3}]
    expected = [
        {"name": f"vscan_{phase}_{shape}", "cuda_event_ms": 0.12346, "count": 3, "plan": plans}
        for shape in ("56_d", "56_c", "28_d", "28_c", "14_d", "14_c", "7_d")
        for phase in ("fwd", "bwd")
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
