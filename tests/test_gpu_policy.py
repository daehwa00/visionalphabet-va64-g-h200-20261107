"""CPU coverage of policy selection, override precedence and operator routing."""

import os
import sys
from unittest.mock import MagicMock

import pytest
import torch

from alphabet import gpu_policy, product_scan_ops
from h200 import bench_h200


@pytest.mark.parametrize(
    ("capability", "reader", "scan", "tma", "split"),
    [
        ((9, 0), "dense_nhwc", "hopper_specialized", True, True),
        ((8, 9), "dense_nhwc", "optimized_backward", True, False),
        ((8, 0), "dense_nhwc", "optimized_backward", True, False),
        ((8, 6), "off", "optimized_backward", False, False),
        ((7, 5), "off", "optimized_backward", False, False),
    ],
)
def test_recommendation(capability, reader, scan, tma, split) -> None:
    env = gpu_policy.recommended_gpu_env(capability)
    expected = {
        **bench_h200.benchmark_arm_env("hopper_fp8_split"),
        "ALPHABET_SCAN_RESIDENT_BACKWARD_WEIGHTS": "1",
    }
    expected.update(
        ALPHABET_READER_MODE=reader, ALPHABET_HORIZONTAL_SCAN_MODE=scan
    )
    if not split:
        for flag in (
            "ALPHABET_SCAN_SPLIT_BF16",
            "ALPHABET_SCAN_SPLIT_FP8",
            "ALPHABET_SCAN_RESIDENT_BACKWARD_WEIGHTS",
        ):
            del expected[flag]
    if not tma:
        expected["ALPHABET_COMPILER_FEATURES"] = "cuda_graphs,max_autotune_pointwise"
    assert env == expected


def test_apply(monkeypatch) -> None:
    monkeypatch.setattr(
        os, "environ", {"ALPHABET_HORIZONTAL_SCAN_MODE": "off", "ALPHABET_SCAN_SPLIT_FP8": "1"}
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: (8, 6) if device == 2 else None
    )
    assert gpu_policy.apply_gpu_policy(2)["ALPHABET_HORIZONTAL_SCAN_MODE"] == "off"
    assert os.environ["ALPHABET_SCAN_SPLIT_FP8"] == "1"
    assert (
        gpu_policy.apply_gpu_policy(2, overwrite=True)["ALPHABET_HORIZONTAL_SCAN_MODE"]
        == "optimized_backward"
    )
    assert "ALPHABET_SCAN_SPLIT_FP8" not in os.environ
    assert gpu_policy.apply_gpu_policy("cpu") == {}
    assert gpu_policy.recommended_gpu_env(None) == {}


def test_auto_arm(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(os, "environ", {"ALPHABET_READER_MODE": "off"})
    monkeypatch.setattr(gpu_policy, "device_capability", lambda _device: (8, 9))
    monkeypatch.setattr(sys, "argv", ["bench", "--arm", "auto", "--out", str(tmp_path)])
    monkeypatch.setattr(bench_h200, "measure_arm", lambda *_args, **_kw: 0)
    assert bench_h200.main() == 0
    assert os.environ["ALPHABET_HORIZONTAL_SCAN_MODE"] == "optimized_backward"
    assert os.environ["ALPHABET_READER_MODE"] == "off"


@pytest.mark.parametrize("flag", ["ALPHABET_SCAN_SPLIT_BF16", "ALPHABET_SCAN_SPLIT_FP8"])
def test_non_hopper_guard(monkeypatch, flag) -> None:
    monkeypatch.setattr(os, "environ", {flag: "1"})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 9))
    packed = MagicMock(shape=(1, 28, 28, 320))
    packed.__getitem__.return_value = MagicMock(
        shape=(1, 28, 28, 160),
        ndim=4,
        dtype=torch.bfloat16,
        is_cuda=True,
    )
    path = [MagicMock(shape=(160, 8, 8))] * 4
    assert (
        product_scan_ops.select_split_scan_op(
            packed,
            path,
            "full16",
            0,
            swiglu=False,
            standard="standard",
            bf16="bf16",
            fp8="fp8",
        )
        == "standard"
    )
