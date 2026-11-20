"""Exercise the opt-in R12 scan ABI, numerics and release cache identities."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from lnet import r6_dispatch, retained_path_dispatch
from lnet import r11_cuda_hscan as hscan
from lnet import r11_cuda_vscan as vscan
from lnet import r12_cuda_scans as r12
from lnet.path_backward_layout import _LayoutGraphPass
from lnet.product_scan_pipeline import (
    run_product_scan_coarse_path_collapse_planes_packed,
    run_product_scan_path_collapse_planes_packed,
)

_R11_HASHES = {
    "product_scan_pipeline.py": "c23967598a703a0f2240180efd02e5166aa92860a2f3c20348c7dd7f0c17e912",
    "r11_cuda_hscan.py": "b6ccd3a7f9cfc3191aad149e3a02f30a70b968b7c6344d13a766b29341538411",
    "r11_cuda_vscan.py": "f747a4ee3e82b04e40c11297c6740538d9b20726dc50e7f6467b1861fbd89866",
    "r11_dispatch.py": "03b50aa233fae43b7a56e9f3b9a75640aafaf548ba10520d7584d59e43d6185a",
    "r6_dispatch.py": "b41213b45f136a7d9a795b4e5d4a8526e3df742b2c3c2b4493fcfb006a0e0327",
    "retained_path_dispatch.py": "e1a22352d3d57a53bf5e1c4bff1b1c51255d47248509d0206dcd53ab24398f8d",
}
_R11_FLAGS = tuple(f"LNET_R11_CUDA_{direction}" for direction in ("VFWD", "VBACK", "HFWD", "HBACK"))
_R12_FLAGS = tuple(f"LNET_R12_{variant}" for variant in ("PACKED", "PRECISE", "INDEX32"))
_LEGACY_FLAGS = (
    "LNET_R6_HBACK",
    "LNET_R6_CABWD",
    "LNET_R7_AFWD",
    "LNET_R7_DIRECT",
    "LNET_R8_HFWD",
    "LNET_R8_HBACK",
    "LNET_R9_RMS_DW",
    "LNET_R10_RMS_DW",
    "LNET_R10_VFWD",
    "LNET_R10_HFWD56_SEG4",
    "LNET_R10_RMS_DW56",
    "LNET_R10_TUNED_LAUNCH",
    "LNET_R10_SEG4PACK2",
    *_R11_FLAGS,
)


@pytest.fixture(autouse=True)
def disable_experimental_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for flag in (*_LEGACY_FLAGS, *_R12_FLAGS):
        monkeypatch.delenv(flag, raising=False)


def test_r12_opt_out_preserves_legacy_source_and_cache_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(r6_dispatch.__file__).parent
    reference = Path("/tmp/va-r12-baseline")  # noqa: S108
    for path in source.glob("*.py"):
        reconstructed = r6_dispatch._pre_r12_source(path)
        if path.name in _R11_HASHES:
            assert hashlib.sha256(reconstructed).hexdigest() == _R11_HASHES[path.name]
            if (reference / path.name).is_file():
                assert reconstructed == (reference / path.name).read_bytes()
        (tmp_path / path.name).write_bytes(reconstructed)

    spec = importlib.util.spec_from_file_location(
        "lnet.r6_dispatch_r11", tmp_path / "r6_dispatch.py"
    )
    assert spec is not None
    assert spec.loader is not None
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    previous = _LayoutGraphPass("sentinel", None)
    assert r6_dispatch.r6_graph_pass(previous) is previous
    for flag in _LEGACY_FLAGS:
        monkeypatch.setenv(flag, "1")
        assert r6_dispatch.r6_graph_pass(previous).uuid() == baseline.r6_graph_pass(previous).uuid()
        monkeypatch.delenv(flag)
    for flag in _LEGACY_FLAGS:
        monkeypatch.setenv(flag, "1")
    assert r6_dispatch.r6_graph_pass(previous).uuid() == baseline.r6_graph_pass(previous).uuid()

    for operation, kernel in enumerate(retained_path_dispatch._KERNELS):
        expected = hashlib.sha256(kernel.cache_key.encode())
        expected.update(baseline.legacy_dispatch_source(tmp_path / "retained_path_dispatch.py"))
        expected.update((tmp_path / "product_scan_pipeline.py").read_bytes())
        assert retained_path_dispatch.step_revision(operation) == expected.hexdigest()


def test_r12_flags_isolate_each_inductor_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = _LayoutGraphPass("sentinel", None)
    keys = {previous.uuid()}
    for flags in (*((f,) for f in _R12_FLAGS), _R12_FLAGS):
        for flag in _R12_FLAGS:
            monkeypatch.setenv(flag, "1" if flag in flags else "0")
        assert tuple(r6_dispatch.active_r6_flags()) == flags
        keys.add(r6_dispatch.r6_graph_pass(previous).uuid())
    assert len(keys) == 5


def test_r12_index32_checks_all_four_planes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not r12._use_index32(1)
    monkeypatch.setenv("LNET_R12_INDEX32", "1")
    assert r12._use_index32(2**29)
    assert not r12._use_index32(2**29 + 1)


def test_r12_precision_variant_changes_only_forward_epilogue() -> None:
    bias_last = r12.vertical_source(index32=False, bias_last=True)
    assert "float acc_r[2] = {};" in bias_last
    assert "float acc_i[2] = {};" in bias_last
    assert "f32_to_bf16((acc_r[0] + acc_r[1]) + bor)" in bias_last
    assert "f32_to_bf16((acc_i[0] + acc_i[1]) + boi)" in bias_last
    assert "__global__ void vscan_bwd_" not in bias_last
    assert "fast_sigmoid(float x)" in bias_last
    assert "__expf(-x)" in bias_last


def _inputs(height):
    generator = torch.Generator(device="cuda").manual_seed(421 + height)
    b, m = 2, 32

    def randn(*shape: int, dtype: torch.dtype = torch.float32):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)

    poles = []
    for _ in range(2):
        angle = 0.3 * randn(1, 1, 1, m)
        radius = 0.7 + 0.2 * randn(1, 1, 1, m).sigmoid()
        poles.extend(
            (
                radius * angle.cos(),
                radius * angle.sin(),
                0.5 + 0.1 * randn(1, 1, 1, m),
                0.1 * randn(1, 1, 1, m),
            )
        )
    return (
        *poles,
        randn(b, height, height, 2 * m, dtype=torch.bfloat16),
        0.2 * randn(m, 8, 8),
        0.1 * randn(m, 8),
        0.2 * randn(m, 2, 8),
        0.1 * randn(m, 2),
    )


def _run_pipeline(height, flags, monkeypatch, *, coarse, compiled, horizontal_cuda=True):
    for flag in _R11_FLAGS[: 4 if horizontal_cuda else 2]:
        monkeypatch.setenv(flag, "1")
    # Parameter gradients use atomic adds when split into multiple CTAs per image.
    monkeypatch.setattr(vscan, "_choose_x_chunks", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(hscan, "_choose_y_chunks", lambda *_args: 1)
    inputs = tuple(t.requires_grad_() for t in _inputs(height))
    scan = (
        run_product_scan_coarse_path_collapse_planes_packed
        if coarse
        else run_product_scan_path_collapse_planes_packed
    )

    def pipeline(*tensors: torch.Tensor):
        return scan(
            tensors[:4],
            tensors[4:8],
            tensors[8],
            tensors[9:],
            gain_normalization="pointwise",
            memory_policy="retain",
        )

    def executable():
        if not compiled:
            return pipeline
        torch._dynamo.reset_code(pipeline.__code__)
        graph_pass = r6_dispatch.r6_graph_pass(_LayoutGraphPass("r12-test", None))
        return torch.compile(
            pipeline,
            fullgraph=True,
            dynamic=False,
            options={"post_grad_custom_pre_pass": graph_pass},
        )

    expected_out = executable()(*inputs)
    cotangents = tuple(
        torch.linspace(-0.7, 0.9, t.numel(), device=t.device).reshape(t.shape).to(t.dtype)
        for t in expected_out
    )
    expected_grad = torch.autograd.grad(expected_out, inputs, cotangents)
    for flag in flags:
        monkeypatch.setenv(flag, "1")
    out = executable()(*inputs)
    grad = torch.autograd.grad(out, inputs, cotangents)
    for actual, expected in zip((*out, *grad), (*expected_out, *expected_grad), strict=True):
        assert torch.isfinite(actual).all()
        if "LNET_R12_PRECISE" in flags:
            rel_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(
                1e-8
            )
            assert rel_l2.item() < 1e-3
        elif "LNET_R12_PACKED" in flags and actual.dtype == torch.float32:
            # The Triton fallback can accumulate pole gradients with atomic adds.
            rtol = 2e-6 if horizontal_cuda else 1e-4
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=rtol)
        else:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("height", [56, 28, 14])
@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "inductor"])
def test_r12_all_flags_forward_and_backward(
    height: int,
    coarse: bool,
    compiled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_pipeline(height, _R12_FLAGS, monkeypatch, coarse=coarse, compiled=compiled)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("flag", _R12_FLAGS)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "inductor"])
def test_r12_individual_flags(flag: str, compiled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_pipeline(28, (flag,), monkeypatch, coarse=False, compiled=compiled)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("coarse", [False, True])
def test_r12_packed_vertical_with_horizontal_fallback(
    coarse: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_pipeline(
        28, (_R12_FLAGS[0],), monkeypatch, coarse=coarse, compiled=True, horizontal_cuda=False
    )
