"""CPU checks for Wave 34 selection, profiler names, and archive transport."""

# ruff: noqa: S101, SLF001
import base64
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch
from audit import classify
from results import ARMS, BASE, COMBO, phase_two, read, select, transport

from lnet import factorized_reader_kernels as reader
from lnet import model_w29c, scan_project_wgmma
from lnet.general_cuda_scans import Kernel, _kernel_name, _kernel_source
from lnet.terminal_imaginary_correlation import _nr4_correlation_kernel


def test_selection(tmp_path: Path) -> None:
    healthy = {
        "status": "ok", "median_ms": 200.0, "peak_allocated_gib": 85.0,
        "finite_losses": True, "finite_gradients": True,
    }
    cases = {
        BASE: {},
        "w34a_modefirst_tc": {"median_ms": 199.85, "peak_allocated_gib": 130.0},
        "w34a_modefirst_tc_gs1": {"median_ms": 190.0, "peak_allocated_gib": 130.001},
        "w34b_dweight_wpr": {"median_ms": 199.851},
        "w34b_dweight_wpr_waves": {"median_ms": float("nan")},
        "w34b_dinput_wpr": {"median_ms": 181.0, "status": "oom"},
        "w34b_full": {"median_ms": 180.0, "finite_losses": False},
        "w34c_rmsnorm_fwd_v2": {"median_ms": 195.0},
        "w34c_fused_res_norm": {"median_ms": 194.0},
        "w34c_full": {"median_ms": 190.0, "finite_gradients": False},
    }
    assert set(cases) == ARMS.keys()
    for arm, override in cases.items():
        (tmp_path / f"{arm}.json").write_text(json.dumps({**healthy, **override}))
    select(tmp_path)
    result = read(tmp_path, "selection")
    assert result["winners"] == {"a": "modefirst_tc", "b": "off", "c": "fused_res_norm"}
    assert result["forward_arm"] == COMBO
    (tmp_path / "w34c_rmsnorm_fwd_v2.json").unlink()
    (tmp_path / "w34c_fused_res_norm.json").unlink()
    select(tmp_path)
    assert read(tmp_path, "selection")["forward_arm"] == "w34a_modefirst_tc"
    (tmp_path / f"{BASE}.json").unlink()
    select(tmp_path)
    assert read(tmp_path, "selection")["winners"] == dict.fromkeys("abc", "off")


@pytest.mark.parametrize(("flags", "forward"), [
    (("off", "off", "off"), None),
    (("modefirst_tc", "off", "off"), False),
    (("off", "full_w34b", "off"), False),
    (("off", "off", "full_w34c"), False),
    (("modefirst_tc", "off", "full_w34c"), True),
    (("modefirst_tc", "full_w34b", "full_w34c"), True),
])
def test_phase_two(flags: tuple[str, str, str], forward: bool | None) -> None:  # noqa: FBT001
    expected = [] if forward is None else [(f"{COMBO}_rev", *flags), (f"{BASE}_rev", *ARMS[BASE])]
    if forward:
        expected.insert(0, (COMBO, *flags))
    assert phase_two(*flags) == expected


@pytest.mark.parametrize(("winners", "late", "extra"), [
    ((), True, []),
    ((), False, ["dynamo", "infer-fold", "c"]),
    (("w34c_full",), False, [f"{COMBO}_rev", f"{BASE}_rev", "dynamo", "infer-fold", "c"]),
    (("w34a_modefirst_tc", "w34b_full"), False, [
        COMBO, f"{COMBO}_rev", f"{BASE}_rev", "parity_a", "parity_b",
        "dynamo", "infer-fold", "a", "b", "c",
    ]),
])
def test_runner_phases(
    tmp_path: Path, winners: tuple[str, ...], late: bool, extra: list[str],  # noqa: FBT001
) -> None:
    runner = Path(__file__).parent
    plan = (runner / "run_h200_bundle.sh").read_text()
    phases = plan[plan.index("while read -r arm a b c;"):]
    for name in (BASE, *winners):
        row = {
            "status": "ok", "finite_losses": True, "finite_gradients": True,
            "median_ms": 200 - int(name != BASE), "peak_allocated_gib": 100,
        }
        (tmp_path / f"{name}.json").write_text(json.dumps(row))
    stub = """
set -euo pipefail
runner=$1 data=$2 PYTHON_BIN=$3 SECONDS=$4 variant=test
eval "$5"
train() { printf '%s\n' "$1"; }
bench() { printf '%s\n' "$2"; }
pytest_run() { printf '%s\n' "$2"; }
run_capped() { printf '%s\n' "$2"; }
"""
    combo = plan[plan.index("set_combo() {"):plan.index("train() (")]
    result = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", stub + phases, "test", str(runner), str(tmp_path),
         sys.executable, str(56 * 60 * late), combo],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines() == [*ARMS, *extra]


def test_scan_names() -> None:
    for backward, phase in ((False, "fwd"), (True, "bwd")):
        key = Kernel(torch.bfloat16, 4, backward, vertical=True, modefirst=True)
        name = _kernel_name(key)
        assert f"void {name}(" in _kernel_source(key)
        assert classify(name) == ("general_vscan", f"w34a_scan_{phase}")
    kernels = {
        scan_project_wgmma._project_forward: "w34a_proj_fwd",
        scan_project_wgmma._project_backward: "w34a_proj_bwd",
        scan_project_wgmma._transpose_channels: "w34a_transpose",
        scan_project_wgmma._reduce_projection: "w34a_reduce",
    }
    for kernel, group in kernels.items():
        assert classify(kernel.__name__) == ("general_vscan", group)
    assert classify("reduce_vscan_partials") == ("general_vscan", "legacy_vscan_reduce")
    assert classify("reduce_path_mlp_partials") == ("general_vscan", "legacy_vscan_reduce")


def test_reader_names() -> None:
    names = set(re.findall(
        r"__global__ void (reader_\w+)\(",
        reader._READER_CUDA + reader._GRAM_NORMALIZE_CUDA + reader._READER_WPR_CUDA,
    ))
    expected = {
        "reader_dweight_reduce_parallel": "reader_reduce",
        "reader_dweight_reduce": "reader_reduce",
        "reader_gram_normalize": "reader_gram_normalize",
        "reader_gram_normalize_backward": "reader_gram_normalize",
        "reader_forward": "reader_forward",
        "reader_dinput": "reader_dinput",
        "reader_dweight": "reader_dweight",
        "reader_dinput_wpr": "reader_dinput",
        "reader_dweight_wpr": "reader_dweight",
    }
    assert names == expected.keys()
    assert {name: classify(name) for name in names} == {
        name: ("factorized_reader", group) for name, group in expected.items()
    }


def test_model_names() -> None:
    groups = {
        model_w29c._swiglu_vec2d_kernel.__name__: "swiglu_custom",
        model_w29c._rmsnorm_backward_kernel.__name__: "rmsnorm_bwd_custom",
        _nr4_correlation_kernel.__name__: "nr4_correlation_custom",
        "_rmsnorm_w34c_forward_kernel": "rmsnorm_fwd_custom",
        "_fused_res_rmsnorm_forward_kernel": "fused_res_norm_fwd",
        "_residual_rmsnorm_backward_kernel": "fused_res_norm_bwd",
        "triton_poi_fused_add_0": "inductor_triton_other",
    }
    for name, group in groups.items():
        assert classify(name) == ("triton", group)


def test_transport(tmp_path: Path) -> None:
    data = tmp_path / "results"
    data.mkdir()
    expected = {"profiler": {"major_kernel_ms": {"reader_forward": 1.2}}}
    names = (*ARMS, COMBO, f"{COMBO}_rev", f"{BASE}_rev", "parity_a", "parity_b")
    for name in names:
        result = {"profiler": {**expected["profiler"], "kernels": [{"name": "drop"}] * 1000}}
        (data / f"{name}.json").write_text(json.dumps(result))
    transport(data, tmp_path)
    archive = tmp_path / "task124.tar.xz"
    payload = base64.b64encode(archive.read_bytes())
    assert len(payload) < 45000
    decoded = base64.b64decode(payload, validate=True)
    with archive.open("rb") as stream:
        assert hashlib.sha256(decoded).hexdigest() == hashlib.file_digest(
            stream, "sha256",
        ).hexdigest()
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == {f"{name}.json" for name in names}
        for member in tar:
            extracted = tar.extractfile(member)
            assert extracted is not None
            assert json.load(extracted) == expected
