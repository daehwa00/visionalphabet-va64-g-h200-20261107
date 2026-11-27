"""CPU coverage of policy selection, override precedence and operator routing."""

import os
import sys
from unittest.mock import MagicMock

import pytest
import torch

from h200 import bench_h200
from lnet import gpu_policy, product_scan_ops


@pytest.mark.parametrize(
    ("capability", "reader", "scan", "tma", "split"),
    [
        ((9, 0), "dense_nhwc", "h55_lockin", True, True),
        ((8, 9), "dense_nhwc", "bwd_opt", True, False),
        ((8, 0), "dense_nhwc", "bwd_opt", True, False),
        ((8, 6), "off", "bwd_opt", False, False),
        ((7, 5), "off", "bwd_opt", False, False),
    ],
)
def test_recommendation(capability, reader, scan, tma, split) -> None:
    env = gpu_policy.recommended_gpu_env(capability)
    expected = {**bench_h200.lockin_arm_env("w74_fp8_lockin"), "VA_W76_REG_WOBI": "1"}
    expected.update(
        LNET_READER_EXEC_MODE=reader, LNET_READER_DENSE_NHWC=reader, LNET_SCAN_HSCAN=scan
    )
    if not split:
        for flag in ("VA_W66_SPLIT_BF16", "VA_W72_SPLIT_FP8", "VA_W76_REG_WOBI"):
            del expected[flag]
    if not tma:
        expected["LNET_MODEL_W36B"] = "cuda_graphs,max_autotune_pointwise"
    assert env == expected


def test_apply(monkeypatch) -> None:
    monkeypatch.setattr(os, "environ", {"LNET_SCAN_HSCAN": "off", "VA_W72_SPLIT_FP8": "1"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: (8, 6) if device == 2 else None
    )
    assert gpu_policy.apply_gpu_policy(2)["LNET_SCAN_HSCAN"] == "off"
    assert os.environ["VA_W72_SPLIT_FP8"] == "1"
    assert gpu_policy.apply_gpu_policy(2, overwrite=True)["LNET_SCAN_HSCAN"] == "bwd_opt"
    assert "VA_W72_SPLIT_FP8" not in os.environ
    assert gpu_policy.apply_gpu_policy("cpu") == {}
    assert gpu_policy.recommended_gpu_env(None) == {}


def test_auto_arm(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(os, "environ", {"LNET_READER_EXEC_MODE": "off"})
    monkeypatch.setattr(gpu_policy, "device_capability", lambda _device: (8, 9))
    monkeypatch.setattr(sys, "argv", ["bench", "--arm", "auto_lockin", "--out", str(tmp_path)])
    monkeypatch.setattr(bench_h200, "measure_arm", lambda *_args, **_kw: 0)
    assert bench_h200.main() == 0
    assert os.environ["LNET_SCAN_HSCAN"] == "bwd_opt"
    assert os.environ["LNET_READER_EXEC_MODE"] == "off"


@pytest.mark.parametrize("flag", ["VA_W66_SPLIT_BF16", "VA_W72_SPLIT_FP8"])
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
