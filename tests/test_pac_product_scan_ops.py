from __future__ import annotations

import pytest
import torch

from lnet.pac_product_scan_contracts import (
    supports_pac_triton_product_scan_coarse4,
    supports_pac_triton_product_scan_descriptor4,
)
from lnet.pac_product_scan_normalization import (
    static_global_inverse_gain,
    static_variance_tables,
)
from lnet.pac_product_scan_ops import (
    BACKWARD_LAUNCH_NAME,
    FORWARD_LAUNCH_NAME,
    pac_triton_product_scan_coarse4,
    pac_triton_product_scan_descriptor4,
)
from lnet.pac_product_scan_reference import (
    product_scan_coarse4_reference,
    product_scan_descriptor4_reference,
    raw_product_descriptor_reference,
)


@pytest.mark.parametrize("length", [1, 6, 56])
def test_static_normalization_matches_explicit_recurrence(length: int) -> None:
    poles, _ = _inputs(torch.device("cpu"), modes=5, height=6)
    variance_x, variance_y = static_variance_tables(*poles[0], *poles[1], length, length)

    expected_tables = []
    for decay_real, decay_imag, gamma_real, gamma_imag in poles:
        decay = decay_real.detach().float().square().add(decay_imag.detach().float().square())
        gamma = gamma_real.detach().float().square().add(gamma_imag.detach().float().square())
        state = torch.zeros_like(decay)
        rows = []
        for _ in range(length):
            state = decay * state + gamma
            rows.append(state.reshape(-1))
        positive = torch.stack(rows)
        expected_tables.append(torch.stack((positive, positive.flip(0))))

    torch.testing.assert_close(variance_x, expected_tables[0])
    torch.testing.assert_close(variance_y, expected_tables[1])
    expected_gain = torch.rsqrt(
        (expected_tables[0][0].mean(0) * expected_tables[1][0].mean(0)).clamp_min(1.0e-8)
    )
    actual_gain = static_global_inverse_gain(*poles[0], *poles[1], length, length, 1.0e-8)
    torch.testing.assert_close(actual_gain, expected_gain)


def test_static_normalization_supports_distinct_axis_lengths() -> None:
    length_x, length_y = 8, 6
    poles, _ = _inputs(torch.device("cpu"), modes=5, height=length_y, width=length_x)

    variance_x, variance_y = static_variance_tables(
        *poles[0],
        *poles[1],
        length_x,
        length_y,
    )
    inverse_gain = static_global_inverse_gain(
        *poles[0],
        *poles[1],
        length_x,
        length_y,
        1.0e-8,
    )

    assert variance_x.shape == (2, length_x, 5)
    assert variance_y.shape == (2, length_y, 5)
    expected_gain = torch.rsqrt((variance_x[0].mean(0) * variance_y[0].mean(0)).clamp_min(1.0e-8))
    torch.testing.assert_close(inverse_gain, expected_gain)


def test_static_normalization_is_stable_for_near_unit_decay() -> None:
    length = 128
    modes = 3
    poles = []
    for gamma in (0.07, 0.11):
        decay_real = torch.full((1, 1, 1, modes), (1.0 - 5.0e-5) ** 0.5)
        decay_imag = torch.zeros_like(decay_real)
        gamma_real = torch.full_like(decay_real, gamma**0.5)
        gamma_imag = torch.zeros_like(decay_real)
        poles.append((decay_real, decay_imag, gamma_real, gamma_imag))

    variance_x, variance_y = static_variance_tables(*poles[0], *poles[1], length, length)
    expected_tables = []
    for decay_real, decay_imag, gamma_real, gamma_imag in poles:
        decay = decay_real.square().add(decay_imag.square()).reshape(modes)
        gamma = gamma_real.square().add(gamma_imag.square()).reshape(modes)
        state = torch.zeros_like(decay)
        rows = []
        for _ in range(length):
            state = decay * state + gamma
            rows.append(state)
        positive = torch.stack(rows)
        expected_tables.append(torch.stack((positive, positive.flip(0))))

    torch.testing.assert_close(variance_x, expected_tables[0], rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(variance_y, expected_tables[1], rtol=2.0e-5, atol=2.0e-5)
    expected_gain = torch.rsqrt(
        (expected_tables[0][0].mean(0) * expected_tables[1][0].mean(0)).clamp_min(1.0e-8)
    )
    actual_gain = static_global_inverse_gain(*poles[0], *poles[1], length, length, 1.0e-8)
    torch.testing.assert_close(actual_gain, expected_gain, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.parametrize("gain_normalization", ["pointwise", "global"])
def test_raw_descriptor_matches_direct_formula_and_has_finite_gradients(
    gain_normalization: str,
) -> None:
    generator = torch.Generator().manual_seed(29)
    states = []
    for _ in range(4):
        real = torch.randn(2, 3, 5, 7, generator=generator, requires_grad=True)
        imag = torch.randn(2, 3, 5, 7, generator=generator, requires_grad=True)
        variance = torch.rand(2, 3, 5, 7, generator=generator).add_(0.25)
        states.append((real, imag, variance))

    actual = raw_product_descriptor_reference(
        *states,
        gain_normalization=gain_normalization,  # type: ignore[arg-type]
    )
    expected_parts = []
    for real, imag, variance in states:
        active_variance = (
            variance.mean((1, 2), keepdim=True) if gain_normalization == "global" else variance
        )
        normalized_energy = real.square().add(imag.square()) / active_variance
        expected_parts.append(torch.log1p(normalized_energy.mean((1, 2))))
    expected = torch.cat(expected_parts, dim=-1)

    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    actual.square().mean().backward()
    for real, imag, _ in states:
        assert real.grad is not None
        assert torch.isfinite(real.grad).all()
        assert imag.grad is not None
        assert torch.isfinite(imag.grad).all()


def test_fused_product_scan_capability_is_explicit() -> None:
    cpu_real = torch.zeros(2, 6, 6, 4)
    assert not supports_pac_triton_product_scan_coarse4(cpu_real, cpu_real)
    assert not supports_pac_triton_product_scan_coarse4(
        torch.zeros(2, 6, 8, 4),
        torch.zeros(2, 6, 8, 4),
    )

    if torch.cuda.is_available():
        cuda_real = cpu_real.cuda()
        assert supports_pac_triton_product_scan_coarse4(cuda_real, cuda_real.clone())
        assert supports_pac_triton_product_scan_coarse4(
            cuda_real.bfloat16(),
            cuda_real.bfloat16(),
        )
        assert supports_pac_triton_product_scan_coarse4(
            cuda_real.half(),
            cuda_real.half(),
        )
        rectangular = torch.zeros(2, 6, 8, 4, device="cuda")
        assert supports_pac_triton_product_scan_coarse4(rectangular, rectangular)
        odd = torch.zeros(2, 7, 7, 4, device="cuda")
        assert not supports_pac_triton_product_scan_coarse4(odd, odd)
        assert supports_pac_triton_product_scan_descriptor4(odd, odd)
        odd_rectangular = torch.zeros(2, 7, 9, 4, device="cuda")
        assert not supports_pac_triton_product_scan_coarse4(
            odd_rectangular,
            odd_rectangular,
        )
        assert supports_pac_triton_product_scan_descriptor4(
            odd_rectangular,
            odd_rectangular,
        )
        odd_width = torch.zeros(2, 6, 7, 4, device="cuda")
        assert not supports_pac_triton_product_scan_coarse4(odd_width, odd_width)
        assert supports_pac_triton_product_scan_descriptor4(odd_width, odd_width)


@pytest.mark.parametrize(("height", "width"), [(7, 7), (7, 9), (9, 7)])
def test_descriptor_only_scan_matches_odd_reference_forward_and_backward(
    height: int,
    width: int,
) -> None:
    poles, sources = _inputs(torch.device("cpu"), height=height, width=width)
    leaves = (*poles[1], *sources[0], *sources[1])
    reference = product_scan_descriptor4_reference(
        *poles,
        *sources,
        gain_normalization="global",
    )
    actual = pac_triton_product_scan_descriptor4(
        *poles,
        *sources,
        gain_normalization="global",
    )

    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    reference.square().mean().backward(retain_graph=True)
    expected_gradients = tuple(value.grad.detach().clone() for value in leaves)
    for value in leaves:
        value.grad = None
    actual.square().mean().backward()
    for value, expected in zip(leaves, expected_gradients, strict=True):
        torch.testing.assert_close(value.grad, expected, rtol=1.0e-6, atol=1.0e-6)


def test_public_api_rejects_unknown_gain_normalization() -> None:
    poles, sources = _inputs(torch.device("cpu"))
    with pytest.raises(ValueError, match="unsupported product gain normalization"):
        pac_triton_product_scan_coarse4(
            *poles,
            *sources,
            gain_normalization="unknown",  # type: ignore[arg-type]
        )


def _inputs(
    device: torch.device,
    *,
    modes: int = 16,
    height: int = 6,
    width: int | None = None,
):
    width = height if width is None else width
    generator = torch.Generator(device=device).manual_seed(31)
    poles = []
    for _ in range(2):
        pole = tuple(
            (
                torch.randn((1, 1, 1, modes), generator=generator, device=device) * 0.03
            ).requires_grad_()
            for _ in range(4)
        )
        with torch.no_grad():
            pole[0].fill_(0.82)
            pole[2].fill_(0.18)
        poles.append(pole)
    sources = []
    for _ in range(2):
        real = torch.randn(
            (2, height, width, modes), generator=generator, device=device
        ).requires_grad_()
        imag = torch.randn(
            (2, height, width, modes), generator=generator, device=device
        ).requires_grad_()
        sources.append((real, imag))
    return tuple(poles), tuple(sources)


def _noncontiguous_sources(sources):
    return tuple(
        tuple(value.transpose(1, 2).detach().requires_grad_() for value in source)
        for source in sources
    )


def _force_vertical_grouping(monkeypatch: pytest.MonkeyPatch, block_lines: int) -> None:
    for name in (FORWARD_LAUNCH_NAME, BACKWARD_LAUNCH_NAME):
        prefix = f"LNET_LAUNCH_{name.upper()}"
        monkeypatch.setenv(f"{prefix}_BLOCK_LINES", str(block_lines))
        monkeypatch.setenv(f"{prefix}_BLOCK_MODES", "4")
        monkeypatch.setenv(f"{prefix}_WARPS", "4")


def _reference(
    poles,
    sources,
    gain_normalization="pointwise",
):
    return product_scan_coarse4_reference(
        *poles,
        sources[0],
        sources[1],
        gain_normalization=gain_normalization,
    )


def test_fused_matches_reference_forward_and_backward() -> None:
    device = torch.device("cpu")
    poles, sources = _inputs(device)
    leaves = (*poles[1], *sources[0], *sources[1])
    reference = _reference(poles, sources)
    fused = pac_triton_product_scan_coarse4(
        *poles,
        sources[0],
        sources[1],
    )
    for actual, expected in zip(fused, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    sum(value.square().mean() for value in reference).backward(retain_graph=True)
    reference_gradients = tuple(value.grad.detach().clone() for value in leaves)
    for value in leaves:
        value.grad = None
    sum(value.square().mean() for value in fused).backward()
    for actual, expected in zip((value.grad for value in leaves), reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.parametrize(("height", "width"), [(6, 8), (8, 6), (14, 22)])
def test_rectangular_coarse_scan_matches_reference_forward_and_backward(
    height: int,
    width: int,
) -> None:
    poles, sources = _inputs(torch.device("cpu"), height=height, width=width)
    leaves = (*poles[1], *sources[0], *sources[1])
    reference = _reference(poles, sources)
    actual = pac_triton_product_scan_coarse4(*poles, *sources)

    for value, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)
    grad_outputs = tuple(torch.randn_like(value) for value in actual)
    expected_gradients = torch.autograd.grad(reference, leaves, grad_outputs, retain_graph=True)
    actual_gradients = torch.autograd.grad(actual, leaves, grad_outputs)
    for value, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(value, expected, rtol=1.0e-6, atol=1.0e-6)


def test_fused_preserves_noncontiguous_source_semantics() -> None:
    poles, contiguous_sources = _inputs(torch.device("cpu"))
    sources = _noncontiguous_sources(contiguous_sources)
    assert not sources[0][0].is_contiguous()
    leaves = (*poles[1], *sources[0], *sources[1])
    reference = _reference(poles, sources)
    fused = pac_triton_product_scan_coarse4(*poles, *sources)
    for actual, expected in zip(fused, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    sum(value.square().mean() for value in reference).backward(retain_graph=True)
    expected_gradients = tuple(value.grad.detach().clone() for value in leaves)
    for value in leaves:
        value.grad = None
    sum(value.square().mean() for value in fused).backward()
    for value, expected in zip(leaves, expected_gradients, strict=True):
        torch.testing.assert_close(value.grad, expected, rtol=1.0e-6, atol=1.0e-6)


def test_global_gain_fused_matches_reference_on_cpu() -> None:
    poles, sources = _inputs(torch.device("cpu"))
    reference = _reference(poles, sources, "global")
    fused = pac_triton_product_scan_coarse4(
        *poles,
        sources[0],
        sources[1],
        gain_normalization="global",
    )

    for actual, expected in zip(fused, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_global_gain_matches_reference_forward_and_backward() -> None:
    poles, sources = _inputs(torch.device("cuda"), modes=64, height=56, width=40)
    leaves = (*poles[1], *sources[0], *sources[1])
    reference = _reference(poles, sources, "global")
    fused = pac_triton_product_scan_coarse4(
        *poles,
        sources[0],
        sources[1],
        gain_normalization="global",
    )

    for actual, expected in zip(fused[:2], reference[:2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-5)
    torch.testing.assert_close(fused[2], reference[2], rtol=2.0e-4, atol=2.0e-4)

    sum(value.square().mean() for value in reference).backward(retain_graph=True)
    expected_gradients = tuple(value.grad.detach().clone() for value in leaves)
    for value in leaves:
        value.grad = None
    sum(value.square().mean() for value in fused).backward()
    for actual, expected in zip((value.grad for value in leaves), expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3.0e-4, atol=3.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_bfloat16_storage_uses_fused_fp32_recurrence() -> None:
    poles, float_sources = _inputs(torch.device("cuda"), modes=32, height=8, width=12)
    sources = tuple(
        tuple(value.detach().bfloat16().requires_grad_() for value in source)
        for source in float_sources
    )
    reference_sources = tuple(
        tuple(value.detach().float().requires_grad_() for value in source) for source in sources
    )
    reference = _reference(poles, reference_sources)
    fused = pac_triton_product_scan_coarse4(*poles, *sources)

    assert fused[0].dtype == torch.bfloat16
    assert fused[1].dtype == torch.bfloat16
    assert fused[2].dtype == torch.float32
    for actual, expected in zip(fused[:2], reference[:2], strict=True):
        torch.testing.assert_close(actual.float(), expected, rtol=2.0e-2, atol=2.0e-2)
    torch.testing.assert_close(fused[2], reference[2], rtol=2.0e-2, atol=2.0e-2)

    sum(value.float().square().mean() for value in fused).backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in sources[0])
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in sources[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_descriptor_only_supports_odd_bfloat16_forward_and_backward() -> None:
    poles, float_sources = _inputs(torch.device("cuda"), modes=64, height=7, width=9)
    sources = tuple(
        tuple(value.detach().bfloat16().requires_grad_() for value in source)
        for source in float_sources
    )
    reference_sources = tuple(
        tuple(value.detach().float().requires_grad_() for value in source) for source in sources
    )
    reference = product_scan_descriptor4_reference(
        *poles,
        *reference_sources,
        gain_normalization="global",
    )
    actual = pac_triton_product_scan_descriptor4(
        *poles,
        *sources,
        gain_normalization="global",
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=2.0e-2, atol=2.0e-2)
    actual.square().mean().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in sources[0])
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in sources[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("block_lines", [2, 4])
def test_cuda_grouped_vertical_scan_matches_reference_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch,
    block_lines: int,
) -> None:
    _force_vertical_grouping(monkeypatch, block_lines)
    poles, sources = _inputs(torch.device("cuda"), modes=16, height=6, width=8)
    leaves = (*poles[1], *sources[0], *sources[1])
    expected = _reference(poles, sources)
    actual = pac_triton_product_scan_coarse4(*poles, *sources)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=3.0e-4, atol=3.0e-4)
    grad_outputs = tuple(torch.randn_like(value) for value in actual)
    expected_gradients = torch.autograd.grad(expected, leaves, grad_outputs)
    actual_gradients = torch.autograd.grad(actual, leaves, grad_outputs)
    for value, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(value, reference, rtol=8.0e-4, atol=8.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("height", "width"), [(6, 6), (56, 56), (16, 24), (24, 16)])
def test_cuda_fused_matches_reference(
    height: int,
    width: int,
) -> None:
    poles, sources = _inputs(torch.device("cuda"), modes=64, height=height, width=width)
    leaves = (*poles[1], *sources[0], *sources[1])
    reference = _reference(poles, sources)
    fused = pac_triton_product_scan_coarse4(
        *poles,
        sources[0],
        sources[1],
    )
    for actual, expected in zip(fused[:2], reference[:2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-5)
    torch.testing.assert_close(fused[2], reference[2], rtol=2.0e-4, atol=2.0e-4)

    sum(value.square().mean() for value in reference[:2]).backward(retain_graph=True)
    reference_gradients = tuple(value.grad.detach().clone() for value in leaves)
    for value in leaves:
        value.grad = None
    sum(value.square().mean() for value in fused[:2]).backward(retain_graph=True)
    for actual, expected in zip((value.grad for value in leaves), reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3.0e-4, atol=3.0e-4)

    for value in leaves:
        value.grad = None
    reference[2].square().mean().backward(retain_graph=True)
    descriptor_gradients = tuple(value.grad.detach().clone() for value in leaves)
    for value in leaves:
        value.grad = None
    fused[2].square().mean().backward()
    for actual, expected in zip(
        (value.grad for value in leaves), descriptor_gradients, strict=True
    ):
        relative_norm = (actual - expected).norm() / expected.norm().clamp_min(1.0e-8)
        assert float(relative_norm) < 5.0e-3
