"""Wave 34C dispatch and RMSNorm forward and backward parity."""

import pytest
import torch
from torch import Tensor
from torch.nn import functional

from lnet.memory_backbone import ChannelRMSNorm, SwiGLU
from lnet.model_w29c import model_w29c_enabled, model_w34c_mode
from lnet.model_w34c import rmsnorm_w34c

PRESETS = ("off", "rmsnorm_fwd_v2", "fused_res_norm", "full_w34c", "auto")
CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires CUDA SM 8.0 or newer",
)


def test_cpu_policy_fullgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LNET_MODEL_W34C", raising=False)
    monkeypatch.setenv("LNET_MODEL_OPT", "auto")
    assert model_w34c_mode() == "off"
    module = SwiGLU(65, 96)
    left, right = torch.randn(2, 3, 5, 65), torch.randn(2, 3, 5, 65)
    expected = module.forward_residual_packed(left, right)
    compiled = torch.compile(module.forward_residual_packed, backend="eager", fullgraph=True)
    for preset in PRESETS:
        monkeypatch.setenv("LNET_MODEL_W34C", preset)
        assert model_w34c_mode() == preset
        torch.testing.assert_close(compiled(left, right), expected, rtol=0, atol=0)


@pytest.mark.parametrize("capability", [(7, 5), (8, 6), (8, 9), (9, 0), (10, 0)])
def test_full_policy(capability: tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W29C", "off")
    monkeypatch.setenv("LNET_MODEL_W34C", "full_w34c")
    monkeypatch.setenv("LNET_MODEL_OPT", "auto")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    for feature in ("swiglu_vec2d", "rmsnorm_bwd", "nr4"):
        assert model_w29c_enabled(torch.device("cuda"), feature) == (capability >= (8, 0))
    monkeypatch.setenv("LNET_MODEL_OPT", "off")
    assert model_w34c_mode() == "off"


def _assert_parity(actual: Tensor, expected: Tensor, dtype: torch.dtype) -> None:
    error = (actual.float() - expected.float()).norm()
    assert error <= 4 * torch.finfo(dtype).eps * expected.float().norm() + 1e-6


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["dense", "pitched"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("residual", [False, True])
def test_fused_rmsnorm_parity(
    dtype: torch.dtype, layout: str, compiled: bool, residual: bool,
) -> None:
    torch.manual_seed(34)
    width = 97
    pitch = width + 19 if layout == "pitched" else width
    left, right = (
        torch.randn(2, 3, 5, pitch, device="cuda", dtype=dtype)[..., :width].requires_grad_()
        for _ in range(2)
    )
    weight = torch.randn(width, device="cuda", dtype=dtype, requires_grad=True)

    def run(x: Tensor, skip: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
        y, added, _ = rmsnorm_w34c(x, skip if residual else None, scale, 1e-6, save_inverse=True)
        return y, added

    entry = torch.compile(run, fullgraph=True) if compiled else run
    got, added = entry(left, right, weight)
    summed = left + right if residual else left
    reference = functional.rms_norm(summed, (width,), weight, 1e-6)
    params = (left, right, weight) if residual else (left, weight)
    output_grad = torch.randn_like(got)
    if residual:
        torch.testing.assert_close(added, summed, rtol=0, atol=0)
        skip_grad = torch.randn_like(added)
        actual = torch.autograd.grad((got, added), params, (output_grad, skip_grad))
        expected = torch.autograd.grad((reference, summed), params, (output_grad, skip_grad))
    else:
        actual = torch.autograd.grad(got, params, output_grad)
        expected = torch.autograd.grad(reference, params, output_grad)
    for value, target in zip((got, *actual), (reference, *expected), strict=True):
        _assert_parity(value, target, dtype)


@CUDA
@pytest.mark.parametrize("preset", PRESETS)
def test_swiglu_dispatch_parity(preset: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_OPT", "auto")
    monkeypatch.setenv("LNET_MODEL_W34C", preset)
    module = SwiGLU(96, 128).cuda()
    left, right = (torch.randn(2, 5, 7, 96, device="cuda", requires_grad=True) for _ in range(2))
    entry = torch.compile(module.forward_residual_packed, fullgraph=True)
    actual = entry(left, right)
    summed = left + right
    normalizer: ChannelRMSNorm = module.norm
    normalized = functional.rms_norm(summed, (96,), normalizer.weight, normalizer.epsilon)
    value, gate = module.value_gate(normalized).chunk(2, -1)
    expected = summed + module.output(value * functional.silu(gate))
    grad = torch.randn_like(expected)
    params = left, right, *module.parameters()
    for got, want in zip(
        (actual, *torch.autograd.grad(actual, params, grad)),
        (expected, *torch.autograd.grad(expected, params, grad)),
        strict=True,
    ):
        _assert_parity(got, want, left.dtype)
