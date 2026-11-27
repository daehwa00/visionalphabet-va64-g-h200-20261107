"""CPU coverage of policy selection, override precedence and operator routing."""

import os
import sys
from unittest.mock import MagicMock

import alphabet_imagenet100_runtime as runtime
import pytest
import run_alphabet_ablations as ablations
import torch
from test_split_bf16_scan import trace_model_scans

from alphabet import gpu_policy, product_scan_ops
from h200 import bench_h200


@pytest.mark.parametrize(
    ("capability", "reader", "scan", "tma", "split"),
    [
        ((9, 0), "dense_nhwc", "hopper_specialized", True, True),
        ((10, 0), "dense_nhwc", "optimized_backward", False, False),
        ((12, 0), "dense_nhwc", "optimized_backward", False, False),
        ((8, 9), "dense_nhwc", "optimized_backward", True, False),
        ((8, 0), "dense_nhwc", "optimized_backward", True, False),
        ((8, 6), "off", "optimized_backward", False, False),
        ((7, 5), "off", "optimized_backward", False, False),
    ],
)
def test_recommendation(capability, reader, scan, tma, split) -> None:
    env = gpu_policy.recommended_gpu_env(capability, allow_fp8=False)
    expected = bench_h200.benchmark_arm_env("hopper_bf16_split")
    expected.update(
        ALPHABET_READER_MODE=reader, ALPHABET_HORIZONTAL_SCAN_MODE=scan
    )
    if not split:
        del expected["ALPHABET_SCAN_SPLIT_BF16"]
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


def test_cli_device_index(monkeypatch, tmp_path) -> None:
    capability = MagicMock(return_value=(10, 0))
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(torch.cuda, "set_device", MagicMock())
    monkeypatch.setattr(bench_h200, "measure_arm", lambda *_args, **_kw: 0)
    monkeypatch.setattr(
        sys, "argv", ["bench", "--device", "cuda:1", "--arm", "auto", "--out", str(tmp_path)]
    )
    assert bench_h200.main() == 0
    capability.assert_called_once_with(1)


@pytest.mark.parametrize(
    ("shape", "epilogue", "expected"),
    [
        ((56, 56, 32), "coarse_path", "fp8"),
        ((28, 28, 192), "full16", "fp8"),
        ((32, 48, 64), "coarse_path", "fp8"),
        ((7, 7, 64), "full16", "fp8"),
        ((7, 7, 64), "coarse_path", "standard"),
        ((32, 47, 64), "coarse_path", "standard"),
        ((32, 32, 33), "full16", "fp8"),
        ((256, 256, 64), "full16", "standard"),
    ],
)
def test_split_routing(monkeypatch, shape, epilogue, expected) -> None:
    h, w, m = shape
    monkeypatch.setattr(os, "environ", gpu_policy.recommended_gpu_env((9, 0), allow_fp8=True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    packed = MagicMock(shape=(1, h, w, 2 * m))
    packed.__getitem__.return_value = MagicMock(
        shape=(1, h, w, m), ndim=4, dtype=torch.bfloat16, is_cuda=True
    )
    assert product_scan_ops.select_split_scan_op(
        packed, [MagicMock(shape=(m, 8, 8))] * 4, epilogue, 0, swiglu=False,
        standard="standard", bf16="bf16", fp8="fp8",
    ) == expected


def test_ablation_device_index(monkeypatch, tmp_path) -> None:
    capability = MagicMock(return_value=(12, 0))
    runner = MagicMock()
    monkeypatch.setattr(os, "environ", {"ALPHABET_AUTO_GPU_POLICY": "1"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(runtime, "main", runner)
    monkeypatch.setattr(
        sys, "argv", ["train", "--device", "cuda:1", "--root", str(tmp_path),
                      "--data-root", str(tmp_path)],
    )
    ablations.main()
    capability.assert_called_once_with(1)
    assert runner.call_args.kwargs["args"].device == torch.device("cuda:1")

@pytest.mark.parametrize(
    ("device", "index"),
    [(None, None), ("cuda", None), (2, 2), (torch.device("cuda:1"), 1)],
)
def test_device_capability(monkeypatch, device, index) -> None:
    capability = MagicMock(return_value=(9, 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert gpu_policy.device_capability(device) == (9, 0)
    capability.assert_called_once_with(index)
    assert gpu_policy.apply_gpu_policy("cpu") == {}
    capability.assert_called_once_with(index)


def test_hopper_opt_in_and_reset(monkeypatch) -> None:
    monkeypatch.setattr(gpu_policy, "device_capability", lambda _: (9, 0))
    env = {"ALPHABET_ALLOW_FP8": "1"}
    fp8 = bench_h200.benchmark_arm_env("hopper_fp8_resident")
    assert gpu_policy.apply_gpu_policy(env=env) == fp8
    del env["ALPHABET_ALLOW_FP8"]
    assert gpu_policy.apply_gpu_policy(env=env, overwrite=True) == env
    assert env == bench_h200.benchmark_arm_env("hopper_bf16_split")
    assert gpu_policy.apply_gpu_policy(env=env, allow_fp8=True) == fp8


@pytest.mark.parametrize("poles", [None, (32, 64, 64, 64), (64, 128, 192, 256)])
@pytest.mark.parametrize("opt_in", [False, True])
def test_auto_hopper_graph(monkeypatch, tmp_path, poles, opt_in) -> None:
    monkeypatch.setattr(os, "environ", {"CUDA_VISIBLE_DEVICES": ""})
    if opt_in:
        monkeypatch.setenv("ALPHABET_ALLOW_FP8", "1")
    monkeypatch.setattr(gpu_policy, "device_capability", lambda _: (9, 0))
    monkeypatch.setattr(sys, "argv", ["bench", "--arm", "auto", "--out", str(tmp_path)])
    monkeypatch.setattr(bench_h200, "measure_arm", lambda *_args, **_kw: 0)
    assert bench_h200.main() == 0
    ops = trace_model_scans(monkeypatch, poles)
    suffix = "_split_fp8.default" if opt_in else "_split_bf16.default"
    assert len(ops) == 11
    assert all(name.endswith(suffix) for name, _, _ in ops)


@pytest.mark.parametrize(
    ("arm", "allow", "fp8"),
    [("auto", "0", False), ("auto", "1", True),
     ("hopper_bf16_split", "1", False), ("hopper_fp8_resident", "0", True)],
)
def test_ablation_hopper_arm(monkeypatch, tmp_path, arm, allow, fp8) -> None:
    monkeypatch.setattr(os, "environ", {"ALPHABET_ALLOW_FP8": allow, "CUDA_VISIBLE_DEVICES": ""})
    monkeypatch.setattr(gpu_policy, "device_capability", lambda _: (9, 0))
    monkeypatch.setattr(runtime, "main", MagicMock())
    monkeypatch.setattr(sys, "argv", [
        "train", "--arm", arm, "--device", "cuda:1", "--root", str(tmp_path),
        "--data-root", str(tmp_path),
    ])
    ablations.main()
    assert os.environ["ALPHABET_SCAN_SPLIT_BF16"] == "1"
    for flag in ("ALPHABET_SCAN_SPLIT_FP8", "ALPHABET_SCAN_RESIDENT_BACKWARD_WEIGHTS"):
        assert os.environ.get(flag) == ("1" if fp8 else None)
