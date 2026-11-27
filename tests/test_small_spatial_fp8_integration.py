"""Integration checks for the production scan and reader preset."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import bench_split_bf16_scan as bench
import pytest
import torch
from test_split_bf16_model import compile_source, offline_plans
from test_split_bf16_scan import model_scan_graph
from test_split_fp8_dispatch import check_compiled_training, check_joint_state, check_workspace

from alphabet import split_fp8_scan as fp
from alphabet.memory_backbone import FactorizedMemoryReader
from h200 import bench_h200
from h200.bench_h200 import profile_bucket


@pytest.mark.parametrize(
    ("fp8", "small", "bf16", "counts"),
    [
        (True, None, True, (11, 0, 0)),
        (True, "0", True, (4, 0, 7)),
        (False, "1", True, (0, 11, 0)),
        (False, "1", False, (0, 0, 11)),
    ],
)
def test_model(monkeypatch, fp8, small, bf16, counts) -> None:
    if small is None:
        monkeypatch.delenv("ALPHABET_SCAN_SMALL_SPATIAL_FP8", raising=False)
    else:
        monkeypatch.setenv("ALPHABET_SCAN_SMALL_SPATIAL_FP8", small)
    ops = model_scan_graph(monkeypatch, fp8=fp8, bf16=bf16)
    assert len(ops) == 11
    assert (
        tuple(
            sum(name.endswith(suffix) for name, _, _ in ops)
            for suffix in (
                "_split_fp8.default",
                "_split_bf16.default",
                ("planes_v1.default", "storage_v2.default"),
            )
        )
        == counts
    )
    assert "storage_v2" in ops[-1][0]


def test_reader(monkeypatch) -> None:
    mode = bench_h200.benchmark_arm_env("hopper_fp8_split")["ALPHABET_READER_MODE"]
    assert mode == "dense_nhwc"
    monkeypatch.setenv("ALPHABET_READER_MODE", mode)
    for h in (7, 14, 28, 56):
        assert FactorizedMemoryReader._reader_execution_mode(None, torch.empty(1, h, h, 1)) == mode


@pytest.mark.parametrize(
    ("h", "m", "epilogue"),
    [
        (56, 160, "full16"),
        (14, 160, "full16"),
        (14, 160, "coarse_path"),
        (7, 128, "full16"),
    ],
)
def test_joint(h, m, epilogue) -> None:
    check_joint_state(h != 7, h, m, epilogue)


@pytest.mark.parametrize("shape", bench.SHAPES)
def test_workspace(monkeypatch, shape) -> None:
    check_workspace(monkeypatch, shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("shape", bench.SHAPES)
def test_cuda_graph(monkeypatch, request, shape) -> None:
    _, h, _, _, coarse = shape
    check_compiled_training(monkeypatch, request, h, coarse, planes=h != 7)


def test_cubins(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHABET_SCAN_RESIDENT_BACKWARD_WEIGHTS", "0")
    monkeypatch.setattr(fp.scans, "device_limits", lambda _: (132, 227 << 10, 64))

    def audit(job):
        label, op, w, m, launch = job
        if op == "VB" and w == 28:
            launch = fp.scans._plan_vertical_segmented_backward(
                0, launch, (1024, w, w, m), "backward_segment16_two_chunks"
            )
        if launch.key.vertical:
            launch = fp.configure(launch, w)
        elif op == "HF":
            launch = fp._cached_horizontal(launch, w)
        key = launch.key
        symbol = fp.profile_name(key, w)
        folder = tmp_path / (label + op)
        folder.mkdir()
        source = fp.kernel_source(key, w, w, m)
        flags = [*fp.compile_flags(key, w), "--keep", "--keep-dir", str(folder)]
        row = compile_source(source, folder, flags, symbol)
        (ptx,) = folder.glob("*.ptx")
        entry = ptx.read_text().split(".entry " + symbol + "(", 1)[1].split(".entry ", 1)[0]
        assert "e4m3x2" in entry
        assert ("e5m2x2" in entry) == key.backward
        assert row["stack"] == row["LDL"] == row["STL"] == 0
        smem = fp.shared_bytes(launch, w)
        regs = row["regs"]
        aw = key.warps * min(
            64 // key.warps,
            65536 // (key.warps * 256 * ((regs + 7) // 8)),
            (228 << 10) // (((smem + row["static"] + 255) // 256) * 256 + 1024),
        )
        bucket = ("vscan" if key.vertical else "hscan") + ("_bwd" if key.backward else "_fwd")
        assert profile_bucket(symbol) == bucket
        if op == "VB" and w >= 28:
            assert smem == {"56d": 115200, "56c": 105984, "28d": 77312, "28c": 70144}[label]
            assert aw == (4 if label == "56d" else 8)
        if op == "VF" and label == "56d":
            assert "fma.rn.f16x2" in entry
        if op == "VB" and w <= 14:
            assert key.grad_store
            assert key.cache_line
            assert "words[ia].unpack" not in source
            assert source.count("words[ia]=") == source.count("words[ib]=") == 1
        return label, op, regs, smem, aw

    assert profile_bucket("alphabet_profile_scale__split_fp8") == "other"
    with ThreadPoolExecutor(8) as pool:
        for row in pool.map(audit, list(offline_plans(bench.SHAPES))):
            print(*row)  # noqa: T201


@pytest.mark.parametrize(
    "arm",
    ["hopper_fp8_resident", "hopper_fp8_split", "hopper_bf16_split", "bf16_compiled", "reference"],
)
@pytest.mark.parametrize("suffix", ["", "_rev"])
def test_arm_environment(monkeypatch, tmp_path, arm, suffix) -> None:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    name = arm + suffix
    for key in (
        "ALPHABET_SCAN_SPLIT_FP8",
        "ALPHABET_SCAN_SMALL_SPATIAL_FP8",
        "ALPHABET_SCAN_SPLIT_BF16",
        "ALPHABET_SCAN_RESIDENT_BACKWARD_WEIGHTS",
    ):
        monkeypatch.setenv(key, "0")
    monkeypatch.setattr(sys, "argv", ["bench", "--arm", name, "--out", str(tmp_path)])
    monkeypatch.setattr(bench_h200, "measure_arm", lambda *_args, **_kwargs: 0)
    assert bench_h200.main() == 0
    if arm in ("hopper_fp8_split", "hopper_fp8_resident"):
        assert os.environ["ALPHABET_SCAN_RESIDENT_BACKWARD_WEIGHTS"] == str(
            int(arm == "hopper_fp8_resident")
        )
        assert (
            os.environ["ALPHABET_SCAN_SPLIT_BF16"] == os.environ["ALPHABET_SCAN_SPLIT_FP8"] == "1"
        )
        assert os.environ["ALPHABET_READER_MODE"] == "dense_nhwc"
    else:
        assert "ALPHABET_SCAN_SPLIT_FP8" not in os.environ
        assert "ALPHABET_SCAN_RESIDENT_BACKWARD_WEIGHTS" not in os.environ
        assert "ALPHABET_SCAN_SMALL_SPATIAL_FP8" not in os.environ
