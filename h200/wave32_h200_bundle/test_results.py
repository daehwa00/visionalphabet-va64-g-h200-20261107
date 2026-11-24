"""CPU-only checks for Wave 32 selection, classification, and transport."""

# ruff: noqa: S101, SLF001
import base64
import hashlib
import json
import re
import tarfile
from pathlib import Path

import pytest
import torch
from audit import classify
from results import ARMS, BASE, COMBO, phase_two, read, select, transport

from lnet import factorized_reader_kernels as reader
from lnet.general_cuda_scans import Kernel, _kernel_name, _kernel_source


def test_selection(tmp_path: Path) -> None:
    healthy = {
        "status": "ok", "median_ms": 185.966, "peak_allocated_gib": 85.0,
        "finite_losses": True, "finite_gradients": True,
    }
    cases = {
        BASE: {},
        "w32a_fused_tc": {"median_ms": 165.966, "peak_allocated_gib": 110.0},
        "w32a_saved_fused_tc": {"median_ms": 150.0, "peak_allocated_gib": 130.001},
        "w32a_decoupled_tc": {"median_ms": 149.0, "finite_gradients": False},
        "w32b_wide_fwd_only": {"median_ms": 185.9},
        "w32b_dweight_vec2_split": {"median_ms": float("nan")},
        "w32b_dweight_vec2_4row": {"median_ms": 181.0, "status": "oom"},
        "w32b_full": {"median_ms": 180.0, "finite_losses": False},
    }
    assert set(cases) == ARMS.keys()
    for arm, override in cases.items():
        (tmp_path / f"{arm}.json").write_text(json.dumps({**healthy, **override}))
    select(tmp_path)
    result = read(tmp_path, "selection")
    assert result["winners"] == {"a": "fused_tc", "b": "off"}
    assert result["forward_arm"] == "w32a_fused_tc"


@pytest.mark.parametrize(("a", "b", "forward"), [
    ("off", "off", None),
    ("fused_tc", "off", False),
    ("off", "full_w32b", False),
    ("fused_tc", "full_w32b", True),
])
def test_phase_two(a: str, b: str, forward: bool | None) -> None:  # noqa: FBT001
    expected = [] if forward is None else [(f"{COMBO}_rev", a, b), (f"{BASE}_rev", "off", "off")]
    if forward:
        expected.insert(0, (COMBO, a, b))
    assert phase_two(a, b) == expected


def test_option_a_limits(tmp_path: Path) -> None:
    common = {"status": "ok", "finite_losses": True, "finite_gradients": True}
    arms = {
        BASE: (200.0, 120.0),
        "w32a_fused_tc": (199.85, 130.0),
        "w32a_saved_fused_tc": (190.0, 130.001),
        "w32a_decoupled_tc": (199.851, 129.0),
        "w32b_wide_fwd_only": (198.0, 100.0),
        "w32b_dweight_vec2_split": (197.0, 110.0),
    }
    for name, (ms, gib) in arms.items():
        row = {**common, "median_ms": ms, "peak_allocated_gib": gib}
        (tmp_path / f"{name}.json").write_text(json.dumps(row))
    select(tmp_path)
    assert read(tmp_path, "selection")["winners"] == {
        "a": "fused_tc", "b": "dweight_vec2_split",
    }
    (tmp_path / f"{BASE}.json").unlink()
    select(tmp_path)
    assert read(tmp_path, "selection")["winners"] == {"a": "off", "b": "off"}


@pytest.mark.parametrize(("flags", "group"), [
    ({"saved_scan": True, "tensor_core": True, "tc_replay": True}, "fused_tc"),
    ({"saved_scan": True, "tensor_core": True}, "saved_fused_tc"),
    ({"saved_projection_cells": 3, "tensor_core": True}, "project_path_tc"),
    ({"saved_scan": True}, "pure_vscan"),
    ({"saved_scan": True, "saved_direction": 1}, "pure_vscan"),
    ({"saved_scan": True, "saved_direction": 2}, "pure_vscan"),
])
def test_scan_names(flags: dict, group: str) -> None:
    for backward, phase in ((False, "fwd"), (True, "bwd")):
        key = Kernel(torch.bfloat16, 4, backward, vertical=True, **flags)
        assert classify(_kernel_name(key)) == ("general_vscan", f"{group}_{phase}")


def test_reduction_names() -> None:
    for vertical, projection in ((True, False), (False, False), (True, True)):
        key = Kernel(
            torch.float32, 4, backward=True, vertical=vertical, reduction=True,
            projection_reduction=projection, tensor_core=True,
        )
        assert classify(_kernel_name(key)) == ("general_vscan", "tc_reduce")
        assert "void vscan_tc_reduce(" in _kernel_source(key)


def test_reader_names() -> None:
    names = set(re.findall(
        r"__global__ void (reader_\w+)\(", reader._READER_CUDA + reader._GRAM_NORMALIZE_CUDA,
    ))
    expected = {
        "reader_dweight_reduce_chunks": "reader_reduce",
        "reader_dweight_reduce_parallel": "reader_reduce",
        "reader_dweight_reduce": "reader_reduce",
        "reader_gram_normalize": "reader_gram_normalize",
        "reader_gram_normalize_backward": "reader_gram_normalize",
        "reader_forward": "reader_forward",
        "reader_dinput": "reader_dinput",
        "reader_dweight": "reader_dweight",
    }
    assert names == expected.keys()
    assert {name: classify(name) for name in names} == {
        name: ("factorized_reader", group) for name, group in expected.items()
    }


def test_transport(tmp_path: Path) -> None:
    data = tmp_path / "results"
    data.mkdir()
    result = {"profiler": {"major_kernel_ms": {"reader_forward": 1.2}, "kernels": []}}
    (data / "arm.json").write_text(json.dumps(result))
    transport(data, tmp_path)
    archive = tmp_path / "task117.tar.xz"
    payload = base64.b64encode(archive.read_bytes())
    assert len(payload) < 45000
    assert hashlib.sha256(base64.b64decode(payload, validate=True)).digest() == hashlib.sha256(
        archive.read_bytes(),
    ).digest()
    with tarfile.open(archive) as tar:
        extracted = tar.extractfile("arm.json")
        assert extracted is not None
        assert json.load(extracted) == {"profiler": {"major_kernel_ms": {"reader_forward": 1.2}}}
