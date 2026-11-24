"""Wave 30C preset isolation, capture, and CUDA numerical parity."""

import pytest
import torch
from torch import Tensor, nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional

from lnet.complex_scan_transitions import _dtype_aligned_rms_norm
from lnet.memory_backbone import ChannelRMSNorm
from lnet.model_w29c import _CompiledChannelRMSNorm, model_w29c_enabled
from lnet.model_w30c import model_w30c_enabled, packed_swiglu_v2, swiglu_tile

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
FEATURES = ("swiglu_v2", "rmsnorm_bwd_v2", "input_norm_v2", "nr4")


@pytest.mark.parametrize(("mode", "features"), [
    ("off", ()),
    ("swiglu_v2", ("swiglu_v2",)),
    ("rmsnorm_bwd_v2", ("rmsnorm_bwd_v2",)),
    ("input_norm_v2", ("input_norm_v2",)),
    ("combined_v2", FEATURES),
])
@pytest.mark.parametrize("base", ["off", "combined_bwd"])
def test_presets(mode: str, features: tuple, base: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W29C", base)
    monkeypatch.setenv("LNET_MODEL_W30C", mode)
    device = torch.device("cuda")
    assert tuple(key for key in FEATURES if model_w30c_enabled(device, key)) == features
    assert model_w29c_enabled(device, "nr4") == (base == "combined_bwd" or mode == "combined_v2")
    assert model_w29c_enabled(device, "swiglu_vec2d") == (base == "combined_bwd")
    assert model_w29c_enabled(device, "rmsnorm_bwd") == (base == "combined_bwd")


@pytest.mark.parametrize(("capability", "enabled"), [((8, 9), False), ((9, 0), True)])
def test_auto(capability: tuple[int, int], enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W30C", "auto")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    assert not model_w30c_enabled(torch.device("cpu"), "swiglu_v2")
    assert all(
        model_w30c_enabled(torch.device("cuda"), key) == enabled for key in FEATURES
    )


@pytest.mark.parametrize(("hidden", "cols"), [(96, 32), (144, 16), (192, 64), (384, 64)])
def test_divisor_tiles(hidden: int, cols: int) -> None:
    rows, actual_cols = swiglu_tile(hidden)
    assert actual_cols == cols
    assert hidden % actual_cols == 0
    assert rows * cols == 4096


@pytest.mark.parametrize("epsilon", [None, 1e-6])
def test_fake_fx_joint_backward(epsilon: float | None) -> None:
    def backward(x: Tensor, weight: Tensor, dy: Tensor) -> tuple[Tensor, ...]:
        activated = packed_swiglu_v2(x)
        normalized = _CompiledChannelRMSNorm.apply(
            activated, weight, epsilon, True, True,  # noqa: FBT003
        )
        return torch.autograd.grad(normalized, (x, weight), dy)

    x = torch.empty(37, 192, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.empty(96, dtype=x.dtype, requires_grad=True)
    dy = torch.empty(37, 96, dtype=x.dtype)
    graph = make_fx(backward, tracing_mode="fake")(x, weight, dy)
    launches = [node for node in graph.graph.nodes if node.op == "call_function"]
    targets = [str(node.target) for node in launches]
    assert "lnet.w30c_swiglu.default" in targets
    assert "lnet.w30c_swiglu_backward.default" in targets
    assert "lnet.w29c_rmsnorm_forward.default" not in targets
    (rms_bwd,) = [
        node for node in launches if str(node.target) == "lnet.w29c_rmsnorm_backward.default"
    ]
    assert rms_bwd.kwargs["v2"] is True


def _check_parity(
    actual: Tensor,
    native: Tensor,
    fp32: Tensor,
    grad: Tensor,
    params: tuple[Tensor, ...],
    fp32_params: tuple[Tensor, ...],
) -> None:
    mixed = (actual, *torch.autograd.grad(actual, params, grad))
    baseline = (native, *torch.autograd.grad(native, params, grad))
    reference = (fp32, *torch.autograd.grad(fp32, fp32_params, grad.float()))
    eps = torch.finfo(actual.dtype).eps
    for value, expected, ideal in zip(mixed, baseline, reference, strict=True):
        for target in (expected.float(), ideal):
            error = (value.float() - target).norm()
            assert error <= (3 * eps * target.norm()) + 1e-6


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [96, 144, 192, 384])
def test_cuda_swiglu(dtype: torch.dtype, hidden: int) -> None:
    torch.manual_seed(502)
    x = torch.randn(3, 113, 2 * hidden, device="cuda", dtype=dtype, requires_grad=True)
    fp32 = x.detach().float().requires_grad_()
    value, gate = x.chunk(2, -1)
    fp32_value, fp32_gate = fp32.chunk(2, -1)
    ref = value * functional.silu(gate)
    _check_parity(
        packed_swiglu_v2(x), ref, fp32_value * functional.silu(fp32_gate),
        torch.randn_like(ref), (x,), (fp32,),
    )


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("kind", "width"), [("stem", 64), ("channel", 96), ("channel", 256)])
@pytest.mark.parametrize("pitch_padding", [0, 16])
def test_cuda_rmsnorm(
    dtype: torch.dtype, kind: str, width: int, pitch_padding: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LNET_MODEL_W30C", "combined_v2")
    torch.manual_seed(502)
    pitch = width + pitch_padding
    x = torch.randn(3, 6011, pitch, device="cuda", dtype=dtype)[..., :width].requires_grad_()
    grad = torch.randn(3, 6011, pitch, device="cuda", dtype=dtype)[..., :width]
    if kind == "stem":
        norm = nn.RMSNorm(width, device="cuda")
        apply = _dtype_aligned_rms_norm
        epsilon = norm.eps
    else:
        norm = ChannelRMSNorm(width).cuda()
        apply = _apply_channel_norm
        epsilon = norm.epsilon
    with torch.no_grad():
        norm.weight.normal_()
    actual = apply(x, norm)
    assert actual.dtype == dtype
    expected = functional.rms_norm(x, (width,), norm.weight.to(dtype), epsilon)
    fp32_x = x.detach().float().requires_grad_()
    fp32_weight = norm.weight.detach().float().requires_grad_()
    ideal = functional.rms_norm(fp32_x, (width,), fp32_weight, epsilon)
    _check_parity(actual, expected, ideal, grad, (x, norm.weight), (fp32_x, fp32_weight))


def _apply_channel_norm(x: Tensor, norm: nn.Module) -> Tensor:
    return norm(x)


@CUDA
def test_cuda_compile_fullgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W30C", "combined_v2")
    norm = nn.RMSNorm(96, device="cuda")
    x = torch.randn(147, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def run(t: Tensor) -> Tensor:
        return _dtype_aligned_rms_norm(packed_swiglu_v2(t), norm)

    eager = run(x)
    compiled = torch.compile(run, fullgraph=True)(x)
    dy = torch.randn_like(eager)
    (compiled_dx,) = torch.autograd.grad(compiled, x, dy)
    (eager_dx,) = torch.autograd.grad(eager, x, dy)
    for actual, expected in ((compiled, eager), (compiled_dx, eager_dx)):
        error = (actual.float() - expected.float()).norm()
        assert error <= 3 * torch.finfo(expected.dtype).eps * expected.float().norm() + 1e-6
