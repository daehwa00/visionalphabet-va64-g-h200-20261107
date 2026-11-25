"""CPU checks for saved VScan trajectories and memory policy."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils._pytree import tree_leaves
from va_model_builder import build_dw_terminal_rmsnorm

from lnet import general_cuda_scans as scans
from lnet.product_scan_reference import d4_grouped_path_collapse_packed_reference


def _trajectory(source, pole, reverse, storage_dtype=None):
    decay = torch.complex(pole[0], pole[1])
    gain = torch.complex(pole[2], pole[3])
    if reverse:
        decay, gain = decay.conj(), gain.conj()
    state = torch.zeros_like(source[:, 0])
    values = []
    for row in range(source.shape[1]):
        value = source[:, -row - 1 if reverse else row]
        prev = state
        if storage_dtype is not None:
            rounded = torch.complex(state.real.to(storage_dtype).float(),
                                    state.imag.to(storage_dtype).float())
            # Keep the forward value exact while evaluating the pole derivative
            # with the rounded input to the recurrence.
            prev = rounded.detach() + (state - state.detach())
            exact = decay * state + gain * value
            surrogate = decay * prev + gain * value
            state = exact.detach() + (surrogate - surrogate.detach())
        else:
            state = decay * state + gain * value
        values.append(state)
    return torch.stack(values[::-1] if reverse else values, dim=1)


def _evaluate(inputs, coarse, dtype, saved_dtype=None):
    pole, state, gains, path = inputs
    _, _, h, w, _ = state.shape
    real, imag = [], []
    for reverse in (False, True):
        for side in (0, 1):
            xs = slice(1 - side, w // 2 * 2, 2) if coarse else slice(None)
            ys = slice(int(not reverse), h // 2 * 2, 2) if coarse else slice(None)
            signal = torch.complex(state[2 * side, :, :, xs].float(),
                                   state[2 * side + 1, :, :, xs].float())
            value = _trajectory(signal, pole, reverse, saved_dtype)[:, ys]
            scale = (gains[0][side, xs] * gains[1][int(reverse), ys, None]).rsqrt()
            real.append((value.real.to(dtype).float() * scale).to(dtype))
            imag.append((value.imag.to(dtype).float() * scale).to(dtype))
    features = torch.cat((torch.stack(real, dim=-2), torch.stack(imag, dim=-2)), dim=-1)
    return d4_grouped_path_collapse_packed_reference(features, *path)


@pytest.mark.parametrize(("h", "w", "m", "coarse"), [(17, 19, 33, False), (31, 23, 65, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_trajectory_reference(h: int, w: int, m: int, coarse: bool, dtype: torch.dtype) -> None:
    rng = torch.Generator().manual_seed(175)
    pole = [torch.full((m,), x, requires_grad=True) for x in (0.73, 0.13, 0.21, 0.09)]
    state = torch.randn((4, 2, h, w, m), generator=rng).mul(0.1).to(dtype)
    state.requires_grad_()
    tables = [1 + torch.rand((2, n, m), generator=rng) for n in (w, h)]
    path = [
        torch.randn(shape, generator=rng).mul(0.125).to(dtype).requires_grad_()
        for shape in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))
    ]
    expected = _evaluate((pole, state, tables, path), coarse, dtype)
    if not coarse or dtype != torch.float32:
        reference = scans._reference_vertical(pole, state, tables, path, coarse, 1e-6)
        tol = 1e-5 if dtype == torch.float32 else 2e-3
        torch.testing.assert_close(expected, torch.cat(reference, dim=-1), rtol=tol, atol=tol)
    seed = torch.randn(expected.shape, generator=rng).to(dtype)
    params = [*pole, state, *path]
    expected_grads = torch.autograd.grad(expected, params, seed)
    names = ("dr", "di", "gr", "gi", "states", "wi", "bi", "wo", "bo")
    fp32_params = [p.detach().float().requires_grad_() for p in params]
    full = _evaluate((fp32_params[:4], fp32_params[4], tables, fp32_params[5:]),
                     coarse, torch.float32)
    full_grads = torch.autograd.grad(full, fp32_params, seed.float())
    for storage in dict.fromkeys((torch.float32, dtype)):
        actual = _evaluate((pole, state, tables, path), coarse, dtype, storage)
        grads = torch.autograd.grad(actual, params, seed)
        for name, got, baseline, fp32 in zip(names, grads, expected_grads, full_grads, strict=True):
            assert got.isfinite().all()
            err = got.float() - baseline.float()
            rel_l2 = err.norm() / baseline.float().norm().clamp_min(1e-12)
            fp32_err = got.float() - fp32
            fp32_l2 = fp32_err.norm() / fp32.norm().clamp_min(1e-12)
            print(  # noqa: T201
                dtype, coarse, storage, name, "off", err.abs().max().item(), rel_l2.item(),
                "fp32", fp32_err.abs().max().item(), fp32_l2.item(),
            )
            if storage == torch.float32:
                torch.testing.assert_close(got, baseline, atol=0, rtol=0)
            else:
                assert rel_l2 < (1e-2 if dtype == torch.bfloat16 else 1e-3)


@pytest.mark.parametrize(
    ("total", "free", "batch", "expected"),
    [(12, 11, 1, "off"), (24, 23, 1, "off"), (80, 70, 1024, "saved_split"),
     (140, 105, 1024, "saved_split"), (140, 4, 1024, "off"),
     (140, 140, 10_000, "off")],
)
def test_auto_vram(
    monkeypatch: pytest.MonkeyPatch, total: int, free: int, batch: int, expected: str,
) -> None:
    gib = 1024**3
    monkeypatch.setenv("LNET_SCAN_VSCAN_SAVED", "auto_vram")
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _device: SimpleNamespace(total_memory=total * gib))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (free * gib, total * gib))
    states = SimpleNamespace(
        is_cuda=True, device=torch.device("cuda"), shape=(4, batch, 56, 56, 160),
        element_size=lambda: 2,
    )
    assert scans._vscan_saved_policy(states, [torch.empty(160, 8, 8)], coarse=False) == expected


def test_ablation17_cpu_fullgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(175)
    model = build_dw_terminal_rmsnorm(factorized_reader=True).train()
    params = tuple(model.parameters())
    assert len(params) == 193
    buffers = [(value, value.clone()) for value in model.buffers()]
    sample = torch.randn(1, 3, 224, 224)
    reference = None
    for mode in ("off", "saved_fused", "saved_split", "auto_vram"):
        monkeypatch.setenv("LNET_SCAN_VSCAN_SAVED", mode)
        for value, initial in buffers:
            value.copy_(initial)
        compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
        outputs = tuple(tensor for tensor in tree_leaves(compiled(sample))
                        if isinstance(tensor, torch.Tensor))
        grads = torch.autograd.grad(outputs[0].square().mean(), params)
        assert all(grad.isfinite().all() for grad in grads)
        result = tuple(value.detach() for value in (*outputs, *grads))
        if reference is None:
            reference = result
        else:
            torch.testing.assert_close(result, reference, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_saved_parity(monkeypatch, coarse, dtype) -> None:
    from bench_scan_micro import _e2e_case, make_inputs  # noqa: PLC0415

    monkeypatch.setenv("LNET_VSCAN_DECOUPLED", "off")
    monkeypatch.setenv("LNET_SCAN_VSCAN_SAVED", "off")
    inputs = make_inputs(2, 17, 19, 33, coarse, dtype)
    expected = None
    for mode in ("off", "saved_fused", "saved_split"):
        forward, backward = _e2e_case(inputs, coarse, mode)
        r, i, trajectory = forward()
        result = (r, i, *backward(trajectory))
        if expected is None:
            expected = result
        else:
            for got, want in zip(result, expected, strict=True):
                assert got.isfinite().all()
                delta = (got.float() - want.float()).norm()
                assert delta / want.float().norm().clamp_min(1e-12) < (
                    1e-2 if dtype == torch.bfloat16 else 1e-3
                )
