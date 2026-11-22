from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "src/lnet"


def _source(name: str) -> str:
    return (PACKAGE / name).read_text()


def test_product_scan_dependencies_point_toward_the_pipeline() -> None:
    reference = _source("product_scan_reference.py")
    kernels = _source("triton_product_scan_kernels.py")
    ops = _source("product_scan_ops.py")
    pipeline = _source("product_scan_pipeline.py")
    stage = _source("same_resolution_depth.py")
    model_stage = _source("complex_scan_stage.py")

    assert "product_scan_normalization" not in reference
    assert "product_scan_ops" not in reference
    assert "torch.library" not in reference
    assert "product_scan_ops" not in kernels
    assert "product_scan_reference" not in kernels
    assert "product_scan_pipeline" not in ops
    assert "complex_scan_stage" not in ops
    assert "triton_product_scan_kernels" not in pipeline
    assert "product_scan_ops" not in stage
    assert "triton_product_scan_kernels" not in stage
    assert "product_scan_ops" not in model_stage


def test_legacy_odd_and_coarse4_modules_are_removed() -> None:
    assert not (PACKAGE / "odd_product_scan.py").exists()
    assert not (PACKAGE / "triton_product_scan_coarse4.py").exists()
    assert not (ROOT / "tests/test_triton_product_scan_coarse4.py").exists()

    production = "\n".join(
        path.read_text()
        for path in PACKAGE.glob("*.py")
        if path.name != "product_scan_reference.py"
    )
    assert "def _vertical_product_scan" not in production
    assert "normalized_odd_directional_states" not in production
