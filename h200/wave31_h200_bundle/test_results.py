"""CPU-only checks for Wave 31 selection, classification, and transport."""

# ruff: noqa: S101
import base64
import hashlib
import json
import tarfile
from pathlib import Path

from audit import classify
from results import ARMS, BASE, read, select, transport


def test_selection(tmp_path: Path) -> None:
    healthy = {
        "status": "ok", "median_ms": 185.966, "peak_allocated_gib": 85.0,
        "finite_losses": True, "finite_gradients": True,
    }
    cases = {
        BASE: {},
        "w31a_saved_fused": {"median_ms": 165.966, "peak_allocated_gib": 110.0},
        "w31a_saved_decoupled": {"median_ms": 150.0, "peak_allocated_gib": 130.001},
        "w31a_saved_decoupled_v2": {"median_ms": 149.0, "finite_gradients": False},
        "w31b_wide_fwd_only": {"median_ms": 185.9},
        "w31b_dweight_vram_split": {"median_ms": float("nan")},
        "w31b_full": {"median_ms": 180.0, "finite_losses": False},
    }
    assert set(cases) == ARMS.keys()
    for arm, override in cases.items():
        (tmp_path / f"{arm}.json").write_text(json.dumps({**healthy, **override}))
    select(tmp_path)
    result = read(tmp_path, "selection")
    assert result["winners"] == {"a": "saved_fused", "b": "off"}
    assert result["forward_arm"] == "w31a_saved_fused"


def test_profiler_names() -> None:
    expected = {
        "pure_vscan_forward_saved_pos": ("general_vscan", "pure_vscan_forward"),
        "pure_vscan_backward_saved_neg": ("general_vscan", "pure_vscan_backward"),
        "vscan_forward_saved_fused": ("general_vscan", "saved_fused_forward"),
        "vscan_backward_saved_fused": ("general_vscan", "saved_fused_backward"),
        "project_path_v2_forward": ("general_vscan", "project_path_v2_forward"),
        "project_path_v2_backward": ("general_vscan", "project_path_v2_backward"),
        "reduce_project_path_v2": ("general_vscan", "project_path_v2_reduce"),
        "reduce_path_mlp_partials": ("general_vscan", "projection_shared_reduce"),
        "reduce_vscan_partials": ("general_vscan", "legacy_vscan_reduce"),
        "reader_dweight_reduce_chunks": ("factorized_reader", "reader_reduce"),
        "dweight_reduce_parallel": ("factorized_reader", "reader_reduce"),
        "reader_gram_normalize": ("factorized_reader", "reader_gram_normalize"),
        "reader_forward": ("factorized_reader", "reader_forward"),
        "reader_dinput": ("factorized_reader", "reader_dinput"),
        "reader_dweight": ("factorized_reader", "reader_dweight"),
    }
    assert {name: classify(name) for name in expected} == expected


def test_transport(tmp_path: Path) -> None:
    data = tmp_path / "results"
    data.mkdir()
    result = {"profiler": {"major_kernel_ms": {"reader_forward": 1.2}, "kernels": []}}
    (data / "arm.json").write_text(json.dumps(result))
    transport(data, tmp_path)
    archive = tmp_path / "task113.tar.xz"
    payload = base64.b64encode(archive.read_bytes())
    assert len(payload) < 45000
    assert hashlib.sha256(base64.b64decode(payload, validate=True)).digest() == hashlib.sha256(
        archive.read_bytes(),
    ).digest()
    with tarfile.open(archive) as tar:
        extracted = tar.extractfile("arm.json")
        assert extracted is not None
        assert json.load(extracted) == {"profiler": {"major_kernel_ms": {"reader_forward": 1.2}}}
