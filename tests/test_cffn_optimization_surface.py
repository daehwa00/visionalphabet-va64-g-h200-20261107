from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_REMOVED_CFFN_TOKENS = (
    "grouped_coarse",
    "grouped_spatial_limit",
    "PACKED_MODE48",
    "packed_mode48",
    "LNET_DISABLE_PACKED_MODE_CFFN48",
    "LNET_DISABLE_PACKED_PATH_HANDOFF",
    "LNET_FORCE_CONTIGUOUS_PATH_CFFN8",
    "LNET_PACKED_MODE_CFFN48_MIN_ROWS",
    "LNET_PAC_FUSED_PATH_CFFN8_BLOCK_N",
    "LNET_PAC_FUSED_PATH_CFFN8_WARPS",
    "LNET_COMPLEX_FFN_FORCE_GEMM",
    "LNET_PACKED_RESIDUAL_CFFN_MAX_SPATIAL",
    "LNET_DISABLE_PACKED_RESIDUAL_CFFN",
    'LNET_CFFN_GRAPH"',
    "packed_spatial_limit",
    "_EAGER_PACKED_MAX_SPATIAL",
    "LNET_CFFN_AUTOTUNE",
    "LNET_CFFN_COMPILE_PROFILE",
    "LNET_CFFN_GRAPH_CACHE",
    "C8-Slim",
    "run_c8_slim_imagenet100",
    "launch_c8_slim_kau",
)


def _active_sources() -> list[Path]:
    return [
        *(_ROOT / "src" / "lnet").glob("*.py"),
        *(_ROOT / "scripts").glob("*.py"),
        *(_ROOT / "scripts").glob("*.sh"),
    ]


@pytest.mark.parametrize("token", _REMOVED_CFFN_TOKENS)
def test_removed_shape_specific_cffn_controls_do_not_reappear(token: str) -> None:
    offenders = [path for path in _active_sources() if token in path.read_text()]

    assert offenders == []


@pytest.mark.parametrize(
    "name",
    [
        "benchmark_fused_mode_cffn.py",
        "verify_fused_mode_cffn.py",
        "run_c8_slim_imagenet100.py",
        "launch_c8_slim_kau.sh",
        "run_a2d_m64_c8_static8_allkernels_imagenet100.py",
        "launch_m64_c8_static8_allkernels_qlab.sh",
        "launch_m64_cal_fused_qlab.sh",
        "benchmark_polepyramid_matrix_residual.py",
        "evaluate_polepyramid_matrix_residual_optimization.sh",
    ],
)
def test_stale_fused_cffn_scripts_stay_removed(name: str) -> None:
    assert not (_ROOT / "scripts" / name).exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/lnet/pac_triton_fused_path_cffn8.py",
        "src/lnet/pac_triton_pathmix_coarse8.py",
        "src/lnet/pac_triton_shared_product_recurrence.py",
        "src/lnet/pac_triton_parallel_shared_product_recurrence.py",
        "src/lnet/pac_static_eight_path_scan.py",
        "tests/test_c8_slim.py",
        "tests/test_pac_triton_fused_path_cffn8.py",
        "tests/test_pac_triton_pathmix_coarse8.py",
        "tests/test_pac_triton_shared_product_recurrence.py",
        "tests/test_pac_static_eight_path_scan.py",
        "src/lnet/pac_cffn_autotune.py",
        "src/lnet/pac_cffn_graph_policy.py",
        "tests/test_pac_cffn_autotune.py",
        "tests/test_pac_cffn_graph_policy.py",
        "tests/test_compiled_complex_ffn_autotune.py",
        "scripts/benchmark_a2d_p4_common_training.py",
        "scripts/benchmark_d4_m64_common_cffn.py",
        "scripts/evaluate_d4_m64_common_cffn_kau.sh",
        "scripts/evaluate_d4_m64_compile_auto_qlab.sh",
        "scripts/benchmark_a2d_remaining_optimization.py",
        "scripts/evaluate_a2d_remaining_optimization_qlab.sh",
    ],
)
def test_removed_c8_optimization_files_stay_removed(relative_path: str) -> None:
    assert not (_ROOT / relative_path).exists()
