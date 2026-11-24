"""CPU launch/dispatcher checks and opt-in CUDA numerical equivalence tests."""

import warnings
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor
from torch._inductor import config as inductor_config
from torch._subclasses.fake_tensor import FakeTensorMode

from lnet import general_cuda_scans as scans
from lnet import terminal_imaginary_correlation as terminal
from lnet.general_cuda_scans import general_scans_enabled
from lnet.product_scan_pipeline import (
    run_product_scan_coarse_path_collapse_packed,
    run_product_scan_coarse_path_collapse_planes_packed,
)
from lnet.product_scan_reference import (
    _product_coarse4_reference,
    _product_full4_reference,
    _product_scans_from_tables_reference,
    bidirectional_product_scan_reference,
    d4_grouped_path_collapse_packed_reference,
    static_variance_tables_reference,
)

# Register the local marker without changing the repository's pytest configuration.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=pytest.PytestUnknownMarkWarning)
    _cuda = pytest.mark.cuda

_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@inductor_config.patch(emulate_precision_casts=False)
def test_general_scans_default_and_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LNET_GENERAL_SCANS", raising=False)
    monkeypatch.delenv("LNET_GENERAL_SCAN_REF_FWD", raising=False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    assert general_scans_enabled()
    assert general_scans_enabled(forward=True)

    monkeypatch.setenv("LNET_GENERAL_SCAN_REF_FWD", "1")
    assert general_scans_enabled()
    assert not general_scans_enabled(forward=True)

    monkeypatch.setenv("LNET_GENERAL_SCANS", "0")
    assert not general_scans_enabled()


@pytest.mark.parametrize("kb", [48, 100, 228])
@pytest.mark.parametrize("height", [1, 7, 14, 28, 56, 64, 96, 128, 192, 193, 200, 334, 1024])
def test_smem_budget(kb: int, height: int) -> None:
    limit = kb * 1024
    for backward in (False, True):
        plans = scans._scan_plans(
            height, limit, vertical=True, backward=backward, coarse=False, hidden=8, itemsize=2
        )
        assert plans
        for plan in plans:
            threads = plan.warps * 32
            weights = 11 * 8 * 32 * 2 if plan.cache_weights else 0
            line = height * threads * 8 if plan.cache_line else 0
            acc = 94 * threads * 4 if plan.shared_acc else 0
            boundaries = 0 if plan.global_checkpoints else (height - 1) // plan.chunk
            source_rows = height if plan.cache_source else plan.chunk
            scan = (
                (plan.chunk * 16 + source_rows * 8 + boundaries * 16) * threads if backward else 0
            )
            reduction = 32 * threads * 4 if backward and plan.warps > 1 else 0
            assert plan.smem == max(reduction, weights + line + acc + scan) <= limit
    if height == 1024 and kb == 48:
        assert all(plan.global_checkpoints for plan in plans)


def test_warp_selection_and_partial_budget() -> None:
    plans = scans._scan_plans(
        56, 99 * 1024, vertical=False, backward=True, coarse=False, hidden=0, itemsize=2
    )
    assert [(plan.warps, plan.smem) for plan in plans if plan.cache_line] == [
        (4, 57344), (3, 43008), (2, 28672), (1, 14336)
    ]
    compact = scans._scan_plans(
        56, 99 * 1024, vertical=False, backward=True, coarse=False,
        hidden=0, itemsize=2, compact=True,
    )
    assert [plan.smem for plan in compact if plan.cache_line] == [
        plan.smem // 2 for plan in plans if plan.cache_line
    ]
    assert scans._workers(1, 48, 4, 4, 300) == 1
    workers = scans._workers(20_000, 200, 182, 4, 1_000)
    assert workers * 200 * 182 * 4 <= scans._DEFAULT_WORKSPACE_BYTES


@pytest.mark.parametrize(
    ("gib", "l2_mib", "budget_mib"), [(12, 6, 32), (24, 72, 64), (32, 64, 72), (140, 60, 124)]
)
def test_workspace_budget(
    monkeypatch: pytest.MonkeyPatch, gib: int, l2_mib: int, budget_mib: int
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=gib * 1024**3)
    )
    monkeypatch.setattr(scans, "_l2_size", lambda _: l2_mib * 1024**2)
    scans._scan_workspace_budget.cache_clear()
    try:
        budget = scans._scan_workspace_budget(0)
        assert budget == budget_mib * 1024**2
        workers = scans._workers(100_000, 160, 94, 1, 100_000, budget)
        assert 0 <= budget - workers * 160 * 94 * 4 < 160 * 94 * 4
    finally:
        scans._scan_workspace_budget.cache_clear()


@pytest.mark.parametrize(
    ("sms", "limit", "max_warps", "smem_per_sm"),
    [(128, 101376, 48, 102400), (132, 232448, 64, 233472)],
)
@pytest.mark.parametrize("batch", [128, 1024])
def test_adaptive_trajectory_plans(
    monkeypatch: pytest.MonkeyPatch, sms: int, limit: int, max_warps: int,
    smem_per_sm: int, batch: int,
) -> None:
    monkeypatch.setattr(scans, "device_limits", lambda _: (sms, limit, max_warps))
    monkeypatch.setattr(scans, "_scan_workspace_budget", lambda _: scans._DEFAULT_WORKSPACE_BYTES)
    monkeypatch.setattr(
        scans, "_resident", lambda _, key, smem: min(
            max_warps // key.warps,
            65536 // (240 * 32 * key.warps),
            smem_per_sm // (smem + 1024),
        ),
    )
    scans._launch_for_shape.cache_clear()
    try:
        for h in (7, 14, 28, 56):
            shape = (batch, h, h, 128 if h == 7 else 160)
            for coarse in (False, True) if h % 2 == 0 else (False,):
                launch = scans._launch_for_shape(
                    0, torch.bfloat16, shape, 4 * torch.Size(shape).numel(),
                    backward=True, vertical=True, coarse=coarse, hidden=8, precise=False,
                    compact=coarse,
                )
                assert launch is not None
                plan = launch.plan
                assert plan.smem <= limit
                assert plan.cache_source == (limit == 232448)
                assert plan.full_trajectory == (plan.cache_source and h <= 28)
                assert plan.full_trajectory == (launch.key.chunk == 0)
                if plan.full_trajectory:
                    assert plan.chunk == h
                    assert not plan.global_checkpoints
                else:
                    assert launch.key.chunk == plan.chunk
    finally:
        scans._launch_for_shape.cache_clear()


@pytest.mark.parametrize(
    ("sms", "l2_mib", "hscan_counts", "vscan_counts"),
    [
        pytest.param(128, 72, (5736, 2868, 14340), (1639, 1639, 8195), id="rtx4090"),
        pytest.param(80, 6, (3584, 1792, 8960), (1024, 1024, 5120), id="rtx3080ti"),
    ],
)
def test_launch_plan_parity(
    monkeypatch: pytest.MonkeyPatch, sms: int, l2_mib: int,
    hscan_counts: tuple[int, int, int], vscan_counts: tuple[int, int, int],
) -> None:
    # Pre-cleanup launch snapshots with deterministic occupancy for both profiles.
    monkeypatch.setattr(scans, "device_limits", lambda _: (sms, 99 * 1024, 48))
    monkeypatch.setattr(scans, "_scan_workspace_budget", lambda _: scans._DEFAULT_WORKSPACE_BYTES)
    monkeypatch.setattr(
        scans, "_horizontal_cache_budget", lambda _: l2_mib * 1024**2 * 7 // 8,
    )
    monkeypatch.setattr(
        scans, "_resident",
        lambda _, key, smem: min(8, 48 // key.warps, 100 * 1024 // max(1, smem)),
    )
    shape = (128, 56, 56, 160)
    index_bound = 16 * torch.Size(shape).numel()
    scans._launch_for_shape.cache_clear()
    try:
        hscan = scans._launch_for_shape(
            0, torch.bfloat16, shape, index_bound, backward=True, vertical=False,
            coarse=False, hidden=0, precise=True, compact=True,
        )
        vscan = scans._launch_for_shape(
            0, torch.bfloat16, shape, index_bound, backward=False, vertical=True,
            coarse=True, hidden=8, precise=True,
        )
        assert hscan is not None
        assert vscan is not None
        assert hscan.plan == scans.ScanPlan(2, 14336, cache_line=True)
        assert vscan.plan == scans.ScanPlan(1, 5632, cache_weights=True)
        for launch, counts in ((hscan, hscan_counts), (vscan, vscan_counts)):
            grid = scans._ceil(launch.workers, launch.plan.warps) * 5
            assert (launch.workers, launch.partials, grid) == counts
    finally:
        scans._launch_for_shape.cache_clear()


@pytest.mark.parametrize("shape", [(4, 128, 192, 160), (4, 96, 128, 80), (128, 56, 56, 160)])
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("coarse", [False, True])
def test_small_l2_vertical_scratch_preserves_parallelism(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, int, int, int],
    backward: bool, coarse: bool,
) -> None:
    scans._launch_for_shape.cache_clear()
    sms = 80
    cache_bytes = 6 * 1024**2 * 7 // 8
    monkeypatch.setattr(scans, "device_limits", lambda _: (sms, 99 * 1024, 48))
    monkeypatch.setattr(scans, "_scan_workspace_budget", lambda _: scans._DEFAULT_WORKSPACE_BYTES)
    monkeypatch.setattr(scans, "_resident", lambda _, _key, smem: (
        min(8 if backward else 15, (99 * 1024) // max(1, smem))
    ))
    try:
        launch = scans._launch_for_shape(
            0, torch.bfloat16, shape, 4 * torch.Size(shape).numel(),
            backward=backward, vertical=True, coarse=coarse, hidden=8, precise=True,
        )
        assert launch is not None
        assert not launch.plan.cache_line
        mode_tiles = scans._ceil(shape[-1], 32)
        assert scans._ceil(launch.workers, launch.plan.warps) * mode_tiles >= sms
        scratch = shape[-1] * launch.workers * scans._scratch_per_worker(
            launch.plan, shape[1], vertical=True, coarse=coarse, backward=backward, itemsize=2,
        )
        assert cache_bytes < scratch <= scans._DEFAULT_WORKSPACE_BYTES
    finally:
        scans._launch_for_shape.cache_clear()


def test_large_l2_vertical_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    sms, limit, max_warps = 132, 227 * 1024, 64
    extra = 60 * 1024**2
    monkeypatch.setattr(scans, "device_limits", lambda _: (sms, limit, max_warps))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=140 * 1024**3),
    )
    monkeypatch.setattr(scans, "_l2_size", lambda _: extra)
    monkeypatch.setattr(
        scans, "_resident", lambda _, key, smem: min(
            max_warps // key.warps,
            65536 // ((248 if key.backward else 64) * 32 * key.warps),
            (limit + 1024) // (smem + 1024),
        ),
    )
    scans._scan_workspace_budget.cache_clear()
    scans._launch_for_shape.cache_clear()
    try:
        assert scans._scan_workspace_budget(0) == scans._DEFAULT_WORKSPACE_BYTES + extra
        for h in (14, 28, 56):
            shape = (1024, h, h, 160)
            for backward in (False, True):
                for coarse in (False, True):
                    launch = scans._launch_for_shape(
                        0, torch.bfloat16, shape, 4 * torch.Size(shape).numel(),
                        backward=backward, vertical=True, coarse=coarse, hidden=8, precise=True,
                    )
                    assert launch is not None
                    plan = launch.plan
                    assert plan.smem <= limit
                    groups = scans._ceil(launch.workers, plan.warps)
                    assert groups * scans._ceil(shape[-1], 32) >= sms
                    assert launch.workers <= shape[0] * (h // 2 if coarse else h)
                    scratch = shape[-1] * launch.workers * scans._scratch_per_worker(
                        plan, h, vertical=True, coarse=coarse, backward=backward, itemsize=2,
                    )
                    assert scratch <= scans._DEFAULT_WORKSPACE_BYTES + extra
                    if backward:
                        assert launch.partials == groups
                        partial_bytes = launch.partials * shape[-1] * 94 * 4
                        assert partial_bytes <= scans._DEFAULT_WORKSPACE_BYTES + extra
    finally:
        scans._launch_for_shape.cache_clear()
        scans._scan_workspace_budget.cache_clear()


@pytest.mark.parametrize("backward", [False, True])
def test_wide_hscan_prefers_one_cached_warp(
    monkeypatch: pytest.MonkeyPatch, backward: bool,
) -> None:
    scans._launch_for_shape.cache_clear()
    sms = 80
    monkeypatch.setattr(scans, "device_limits", lambda _: (sms, 99 * 1024, 48))
    monkeypatch.setattr(scans, "_resident", lambda _, _key, smem: (
        min(12, (99 * 1024) // max(1, smem))
    ))
    shape = (4, 128, 192, 160)
    try:
        launch = scans._launch_for_shape(
            0, torch.bfloat16, shape, 4 * torch.Size(shape).numel(),
            backward=backward, vertical=False, coarse=False, hidden=0, precise=False,
        )
        assert launch is not None
        assert launch.plan.cache_line
        assert launch.plan.warps == 1
        assert launch.workers == shape[0] * shape[1]
    finally:
        scans._launch_for_shape.cache_clear()


def test_small_scan_spreads_ctas_over_sms(monkeypatch: pytest.MonkeyPatch) -> None:
    scans._launch_for_shape.cache_clear()
    monkeypatch.setattr(scans, "device_limits", lambda _: (128, 99 * 1024, 48))
    monkeypatch.setattr(scans, "_resident", lambda *_: 3)
    shape = (4, 16, 24, 128)
    launch = scans._launch_for_shape(
        0, torch.bfloat16, shape, 16 * torch.Size(shape).numel(),
        backward=True, vertical=True, coarse=True, hidden=8, precise=False, compact=True,
    )
    assert launch is not None
    assert launch.plan.warps == 1
    assert launch.partials == launch.workers == 48
    scans._launch_for_shape.cache_clear()


@pytest.mark.parametrize(
    ("shape", "coarse", "cache_line"),
    [
        ((4, 128, 192, 160), False, False),
        ((4, 128, 192, 160), True, False),
        ((4, 96, 128, 80), True, False),
        ((4, 48, 64, 80), True, True),
    ],
)
def test_long_vertical_forward_counts_mode_tiles(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int, int],
    coarse: bool,
    cache_line: bool,
) -> None:
    scans._launch_for_shape.cache_clear()
    monkeypatch.setattr(scans, "device_limits", lambda _: (128, 99 * 1024, 48))
    monkeypatch.setattr(scans, "_resident", lambda _, _key, smem: min(15, (99 * 1024) // smem))
    try:
        launch = scans._launch_for_shape(
            0, torch.bfloat16, shape, 16 * torch.Size(shape).numel(),
            backward=False, vertical=True, coarse=coarse, hidden=8, precise=True,
        )
        assert launch is not None
        assert launch.plan.cache_line == cache_line
        if not cache_line:
            assert launch.plan.warps == 1
    finally:
        scans._launch_for_shape.cache_clear()


@pytest.mark.parametrize(
    ("limit", "dtype", "backward", "coarse", "line", "modes", "resident", "vector"),
    [
        (227 * 1024, torch.bfloat16, False, False, False, 160, 8, True),
        (227 * 1024, torch.float16, True, True, False, 160, 8, True),
        (227 * 1024, torch.bfloat16, False, True, False, 160, 8, False),
        (227 * 1024, torch.bfloat16, True, False, True, 48, 8, False),
        (227 * 1024, torch.bfloat16, False, False, True, 160, 1, False),
        (227 * 1024, torch.float32, False, False, True, 160, 8, False),
        (99 * 1024, torch.bfloat16, False, False, True, 160, 8, False),
    ],
)
def test_vector_path_launch_policy(
    monkeypatch: pytest.MonkeyPatch, limit: int, dtype: torch.dtype,
    backward: bool, coarse: bool, line: bool, modes: int, resident: int, vector: bool,
) -> None:
    key = scans.Kernel(
        dtype, 2, backward, vertical=True, coarse=coarse, hidden8=True,
        cache_weights=True, cache_line=line, precise=not backward,
    )
    plan = scans.ScanPlan(2, 32 * 1024, cache_weights=True, cache_line=line)
    original = scans.Launch(plan, key, 2048, 1024 if backward else 2048)
    monkeypatch.setattr(scans, "device_limits", lambda _: (132, limit, 64))
    monkeypatch.setattr(
        scans, "_resident", lambda _, key, __: resident if key.vector_weights else 8,
    )
    selected = scans._with_vector_path_weights(0, original, modes=modes, hidden=8, wave_limit=8)
    assert selected is not None
    assert selected.key.vector_weights == vector
    if vector:
        assert selected.plan.smem == plan.smem + 11 * 8 * 32 * (4 - dtype.itemsize)
        assert not selected.key.precise
        assert selected.workers <= original.workers
    else:
        assert selected == original


@pytest.mark.parametrize("dtype", _DTYPES)
def test_source_and_offline_nvrtc(dtype: torch.dtype) -> None:
    keys = (
        scans.Kernel(dtype, 4, backward=False, cache_line=True),
        scans.Kernel(dtype, 1, backward=False, real_interleaved=True),
        scans.Kernel(dtype, 2, backward=True, compact=True),
        scans.Kernel(
            dtype, 1, backward=True, vertical=True, coarse=True, hidden8=True,
            cache_weights=True, compact=True, imag_interleaved=True,
        ),
        scans.Kernel(dtype, 4, backward=True, index32=False),
        scans.Kernel(
            dtype, 3, backward=False, vertical=True, hidden8=True, cache_weights=True,
            real_interleaved=True, imag_interleaved=True,
        ),
        scans.Kernel(
            dtype, 2, backward=True, vertical=True, shared_acc=True,
            index32=False,
        ),
        scans.Kernel(dtype, 2, backward=False, vertical=True, pure_vertical=True),
        scans.Kernel(
            dtype, 2, backward=True, vertical=True, pure_vertical=True,
            coarse=True, compact=True, cache_source=True, cache_line=True,
        ),
        scans.Kernel(
            dtype,
            4,
            backward=True,
            vertical=True,
            index32=False,
            shared_acc=True,
            global_checkpoints=True,
            cache_line=True,
            precise=False,
        ),
    )
    sources = [scans._kernel_source(key) for key in keys]
    assert all(
        s.startswith(scans._CUDA_SOURCE) and "int B, int H, int W, int M" in s for s in sources
    )
    assert len(set(sources)) == len(keys)
    assert "long long, false, 4, 8, 0, true, false, true, true, true, false" in sources[-1]
    assert scans._DTYPES[dtype] in sources[0]
    try:
        scans._nvrtc()
    except OSError:
        pytest.skip("NVRTC not installed for offline compilation")
    if not scans._include_dirs():
        pytest.skip("CUDA headers not installed for offline compilation")
    architectures = [(8, 0), (8, 9), (9, 0)]
    if dtype != torch.bfloat16:
        architectures.insert(0, (7, 0))
    for arch in architectures:
        for key in keys:
            ptx = scans._compile_ptx(key, arch)
            expected = (
                "legacy_vscan_bwd" if key.backward else "legacy_vscan_fwd"
            ) if key.vertical else "legacy_hscan"
            assert f".entry {expected}(".encode() in ptx.value


def _inputs(
    h: int,
    w: int,
    m: int,
    hidden: int,
    dtype: torch.dtype,
    device: str,
    batch: int = 1,
) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor, list[Tensor], list[Tensor]]:
    def sample(
        shape: tuple[int, ...], scale: float = 1.0, storage: torch.dtype = torch.float32
    ) -> Tensor:
        return (torch.randn(shape, device=device) * scale).to(storage).requires_grad_()

    # Keep poles contractive, complex, and nonzero in both components.
    px = [sample((1, m), 0.09) for _ in range(4)]
    py = [sample((m,), 0.12) for _ in range(4)]
    real = sample((batch, h, w, m), 0.4, dtype)
    imag = sample((batch, h, w, m), 0.4, dtype)
    auxiliary = list(static_variance_tables_reference(tuple(px), tuple(py), w, h))
    path = [sample(s, 0.17) for s in [(m, hidden, 8), (m, hidden), (m, 2, hidden), (m, 2)]]
    return px, py, real, imag, auxiliary, path


def _reference_vscan(
    py: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    coarse: bool,
) -> tuple[Tensor, Tensor]:
    pr, pi, nr, ni = states.unbind(0)
    products = _product_scans_from_tables_reference(tuple(py), (pr, pi), (nr, ni), auxiliary[0])
    select = _product_coarse4_reference if coarse else _product_full4_reference
    r, i = select(products, epsilon=1e-6, gain_normalization="local")
    packed = torch.cat((r, i), dim=-1).to(states.dtype)
    out = d4_grouped_path_collapse_packed_reference(packed, *path)
    return out.chunk(2, dim=-1)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(("h", "w", "m", "coarse"), [(7, 13, 1, False), (28, 56, 48, True)])
def test_fake_shapes_and_autograd(dtype: torch.dtype, h: int, w: int, m: int, coarse: bool) -> None:
    with FakeTensorMode():
        px, py, r, i, aux, path = _inputs(h, w, m, 5, dtype, "cpu", batch=2)
        states = scans.horizontal_forward(px, r, i)
        assert states.shape == (4, 2, h, w, m)
        v_real, v_imag = scans.vertical_forward(py, states, aux, path, coarse=coarse)
        step = 2 if coarse else 1
        assert v_real.shape == v_imag.shape == (2, h // step, w // step, m)
        v_grads = scans.vertical_backward(py, states, aux, path, v_real, v_imag, coarse)
        h_grads = scans.horizontal_backward(px, [r, i], states, v_grads[4])
        assert h_grads[-1].shape == (2, h, w, 2 * m)
        assert v_grads[4].shape == states.shape
        assert states.dtype == v_real.dtype == v_grads[4].dtype == h_grads[-1].dtype == dtype
        params = [r, i, *px, *py, *path]
        gradients = torch.autograd.grad(v_real.sum(), params)
        assert [(g.shape, g.dtype) for g in gradients] == [(p.shape, p.dtype) for p in params]


def test_reference_forward_mode_on_cpu() -> None:
    px, py, r, i, aux, path = _inputs(6, 10, 3, 5, torch.float32, "cpu")
    states = scans.horizontal_forward(px, r, i, mode="reference")
    torch.testing.assert_close(
        states,
        torch.stack(
            bidirectional_product_scan_reference(
                tuple(px),
                (r, i),
            )
        ),
    )
    actual = scans.vertical_forward(py, states, aux, path, coarse=True, mode="reference")
    expected = _reference_vscan(py, states, aux, path, coarse=True)
    torch.testing.assert_close(actual, expected)


def test_eligibility_and_memory_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    _, py, r, i, aux, path = _inputs(13, 19, 16, 4, torch.float32, "cpu")
    states = torch.empty((4, *r.shape))
    assert not scans.can_use_general_hscan(r, i)
    assert not scans.can_use_general_vscan(states, aux, path, pole=py)
    assert not scans._vertical_inputs_ok(states, aux, path, coarse=True)
    assert not scans._vertical_inputs_ok(states, aux[::-1], path, coarse=False)
    assert not scans._shape_supported((2, 0, 13, 16))
    assert not scans._shape_supported((2, 3, 4))
    fake_cuda = SimpleNamespace(
        is_cuda=True,
        dtype=torch.float32,
        device=SimpleNamespace(index=0),
        shape=r.shape,
        numel=r.numel,
    )
    monkeypatch.setattr(scans, "device_limits", lambda _: (100, 4096, 48))
    calls = []

    def resident(device: int, key: scans.Kernel, smem: int) -> int:
        calls.append((device, key, smem))
        return 0

    monkeypatch.setattr(scans, "_resident", resident)
    assert (
        scans._plan_launch(
            fake_cuda,
            tuple(r.shape),
            backward=True,
            vertical=True,
            coarse=False,
            hidden=4,
            precise=True,
        )
        is None
    )
    assert calls
    assert all(key.vertical and not key.cache_line for _, key, _ in calls)
    calls.clear()
    assert not scans._eligible(
        fake_cuda, tuple(r.shape), vertical=False, coarse=False, hidden=0, precise=True
    )
    assert len(calls) == 4
    assert all(call[2] == 0 for call in calls)
    scans._launch_for_shape.cache_clear()

    def unavailable(*_args: object) -> int:
        raise OSError("Driver unavailable")

    monkeypatch.setattr(scans, "_resident", unavailable)
    assert not scans._eligible(
        fake_cuda, tuple(r.shape), vertical=False, coarse=False, hidden=0, precise=True
    )


def test_index_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scans, "device_limits", lambda _: (128, 99 * 1024, 48))
    monkeypatch.setattr(scans, "_resident", lambda *_: 3)
    for shape in [(1, 7, 13, 48), (4, 200, 334, 2048)]:
        bound = 4 * torch.Size(shape).numel()
        launch = scans._launch_for_shape(
            0, torch.float32, shape, bound,
            backward=True, vertical=True, coarse=False, hidden=8, precise=True,
        )
        assert launch is not None
        assert launch.key.index32 == (bound < 2**31)
    scans._launch_for_shape.cache_clear()


@pytest.mark.parametrize(("batch", "index32"), [(1024, True), (1100, False)])
def test_packed_vertical_index_bound(
    monkeypatch: pytest.MonkeyPatch, batch: int, index32: bool
) -> None:
    monkeypatch.setenv("LNET_GENERAL_SCAN_OPT", "off")
    monkeypatch.setattr(scans, "device_limits", lambda _: (132, 232448, 64))
    monkeypatch.setattr(scans, "_resident", lambda *_: 3)
    monkeypatch.setattr(scans, "_scan_workspace_budget", lambda _: scans._DEFAULT_WORKSPACE_BYTES)
    scans._launch_for_shape.cache_clear()
    try:
        shape = (batch, 56, 56, 160)
        states = SimpleNamespace(
            is_cuda=True,
            dtype=torch.bfloat16,
            device=SimpleNamespace(index=0),
            numel=lambda: 4 * torch.Size(shape).numel(),
        )
        launch = scans._plan_launch(
            states, shape, backward=True, vertical=True, coarse=False, hidden=8, precise=False,
        )
        assert launch is not None
        assert launch.key.index32 == index32
    finally:
        scans._launch_for_shape.cache_clear()


# Cover every requested value with both eager and Inductor; include checkpoint
# boundaries, global checkpoints, a singleton shape, and lanes beyond M.
_CASES = [
    (1, 1, 1, False, torch.float32, 1),
    (7, 7, 16, False, torch.bfloat16, 4),
    (7, 9, 35, False, torch.float32, 17),
    (14, 14, 48, False, torch.float32, 8),
    (14, 14, 48, True, torch.float16, 8),
    (28, 28, 96, False, torch.float16, 16),
    (28, 28, 96, True, torch.bfloat16, 4),
    (56, 56, 128, False, torch.bfloat16, 8),
    (56, 56, 128, True, torch.float32, 16),
    (13, 19, 160, False, torch.float32, 4),
    (64, 96, 48, False, torch.float16, 16),
    (64, 96, 48, True, torch.float32, 8),
    (128, 192, 16, False, torch.float32, 4),
    (128, 192, 16, True, torch.float16, 4),
    (200, 334, 3, False, torch.float32, 5),
    (201, 9, 17, False, torch.float32, 5),
    (202, 10, 17, True, torch.float32, 5),
]


def _pipeline(
    px: list[Tensor],
    py: list[Tensor],
    real: Tensor,
    imag: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    coarse: bool,
    mode: str,
) -> tuple[Tensor, Tensor, Tensor]:
    states = scans.horizontal_forward(px, real, imag, mode)
    return states, *scans.vertical_forward(py, states, auxiliary, path, coarse, mode=mode)


_compiled_pipeline = torch.compile(_pipeline, fullgraph=True, dynamic=True)


def _compare_pipeline(
    h: int,
    w: int,
    m: int,
    coarse: bool,
    dtype: torch.dtype,
    hidden: int,
    compiled: bool,
    mode: str,
    batch: int | None = None,
) -> None:
    torch.manual_seed(791)
    args = _inputs(h, w, m, hidden, dtype, "cuda", batch=batch or (2 if h < 32 else 1))
    px, py, real, imag, auxiliary, path = args
    if coarse:
        packed = torch.cat((real, imag), dim=-1).detach().requires_grad_()
        real, imag = packed.chunk(2, dim=-1)
        args = px, py, real, imag, auxiliary, path
    run = _compiled_pipeline if compiled else _pipeline
    with torch._dynamo.config.patch(recompile_limit=64):
        actual = run(*args, coarse, mode)
    states = torch.stack(bidirectional_product_scan_reference(tuple(px), (real, imag)))
    expected = (states, *_reference_vscan(py, states, auxiliary, path, coarse))
    params = [real, imag, *px, *py, *path]
    adjoints = [torch.randn_like(output) for output in expected]
    actual_grads = torch.autograd.grad(actual, params, adjoints)
    expected_grads = torch.autograd.grad(expected, params, adjoints)
    forward_tol, backward_tol = {
        torch.float32: ((4e-5, 3e-6), (4e-4, 2e-4)),
        torch.float16: ((8e-3, 1e-3), (5e-2, 2e-2)),
        torch.bfloat16: ((4e-2, 5e-3), (1.2e-1, 8e-2)),
    }[dtype]
    for values, reference, (rtol, atol) in (
        (actual, expected, forward_tol),
        (actual_grads, expected_grads, backward_tol),
    ):
        for value, target in zip(values, reference, strict=True):
            torch.testing.assert_close(value, target, rtol=rtol, atol=atol)


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("h", "w", "coarse"), [(37, 53, False), (34, 46, True)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_batched_forward_path(
    monkeypatch: pytest.MonkeyPatch, h: int, w: int, coarse: bool, dtype: torch.dtype,
) -> None:
    torch.manual_seed(791)
    px, py, real, imag, auxiliary, path = _inputs(h, w, 35, 8, dtype, "cuda")
    states = scans.horizontal_forward(px, real, imag)
    monkeypatch.setenv("LNET_GENERAL_SCAN_OPT", "off")
    reference = scans._vscan(py, states, auxiliary, path, coarse, 1e-6, precise=True, packed=True)
    dispatch = scans._dispatch

    def with_batched_path(
        source: Tensor, launch: scans.Launch, args: tuple,
    ) -> None:
        dispatch(source, launch._replace(key=replace(launch.key, batch_path=True)), args)

    monkeypatch.setattr(scans, "_dispatch", with_batched_path)
    actual = scans._vscan(py, states, auxiliary, path, coarse, 1e-6, precise=True, packed=True)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("warps", [1, 2, 3, 4])
@pytest.mark.parametrize("layout", ["planar", "interleaved", "transposed", "compact"])
def test_cuda_horizontal_unroll_exact(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, warps: int, layout: str,
) -> None:
    torch.manual_seed(913)
    compact = layout == "compact"
    h, w, m = (56, 56, 160) if compact else (37, 53, 53)
    px, _, real, imag, _, _ = _inputs(h, w, m, 8, dtype, "cuda", batch=2)
    if layout == "interleaved":
        real, imag = torch.stack((real, imag), dim=-2).unbind(-2)
    if layout == "transposed":
        real, imag = (
            value.transpose(1, 2).contiguous().transpose(1, 2) for value in (real, imag)
        )
    plan_launch = scans._plan_launch
    unroll = 0

    def plan(source: Tensor, shape: tuple[int, int, int, int], **opts: object) -> scans.Launch:
        base = plan_launch(source, shape, **opts)
        assert base is not None
        backward = base.key.backward
        threads = warps * 32
        smem = threads * (8 if backward else 4) * (w // 2 if backward and compact else w)
        key = replace(
            base.key, warps=warps, cache_line=True,
            horizontal_unroll8=backward and unroll == 8,
        )
        return scans.Launch(
            scans.ScanPlan(warps, smem, cache_line=True), key, 2 * h, scans._ceil(2 * h, warps),
        )

    monkeypatch.setenv("LNET_GENERAL_SCAN_OPT", "off")
    monkeypatch.setattr(scans, "_plan_launch", plan)
    expected = scans.horizontal_forward(px, real, imag)
    grad = torch.randn(
        (4, 2, h, w // 2 if compact else w, m), dtype=dtype, device="cuda",
    )
    expected_grads = scans.horizontal_backward(px, [real, imag], expected, grad, compact)
    unroll = 8
    actual_grads = scans.horizontal_backward(px, [real, imag], expected, grad, compact)
    torch.testing.assert_close(actual_grads, expected_grads, atol=0, rtol=0)


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("h", "w", "coarse"), [(37, 53, False), (38, 54, True)])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_auto_odd_modes_against_fp32_reference(
    monkeypatch: pytest.MonkeyPatch, h: int, w: int, coarse: bool, dtype: torch.dtype,
) -> None:
    torch.manual_seed(501)
    px, py, real, imag, aux, path = _inputs(h, w, 53, 8, dtype, "cuda")
    params = [real, imag, *px, *py, *path]
    reference_states = torch.stack(
        bidirectional_product_scan_reference(tuple(px), (real.float(), imag.float())),
    )
    reference = (reference_states, *_reference_vscan(py, reference_states, aux, path, coarse))
    dy = [torch.randn_like(value) for value in reference]
    reference_grads = torch.autograd.grad(reference, params, dy, retain_graph=True)
    results = []
    for policy in ("off", "auto"):
        monkeypatch.setenv("LNET_GENERAL_SCAN_OPT", policy)
        actual = _pipeline(px, py, real, imag, aux, path, coarse, "cuda")
        grads = torch.autograd.grad(actual, params, dy, retain_graph=True)
        results.append((*actual, *grads))

    tolerance = {
        torch.float32: ((4e-5, 3e-6), (4e-4, 2e-4)),
        torch.float16: ((8e-3, 1e-3), (5e-2, 2e-2)),
        torch.bfloat16: ((4e-2, 5e-3), (1.2e-1, 8e-2)),
    }[dtype]
    for index, (off, auto, expected) in enumerate(
        zip(*results, (*reference, *reference_grads), strict=True),
    ):
        rtol, atol = tolerance[index >= len(reference)]
        torch.testing.assert_close(auto, off, rtol=rtol, atol=atol)
        if dtype == torch.float32 or index < len(reference):
            torch.testing.assert_close(auto.float(), expected.float(), rtol=rtol, atol=atol)
        else:
            # FP32 omits the storage casts, so cancellation can amplify relative error
            # at individual entries of the pole reductions.
            norm = torch.linalg.vector_norm(expected.float()).clamp_min(atol)
            error = torch.linalg.vector_norm(auto.float() - expected.float()) / norm
            assert error.item() <= (0.02 if dtype == torch.bfloat16 else 0.005), index


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("h", "w", "m", "coarse", "dtype", "hidden"), _CASES)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "inductor"])
def test_cuda_reference_equivalence(
    h: int,
    w: int,
    m: int,
    coarse: bool,
    dtype: torch.dtype,
    hidden: int,
    compiled: bool,
) -> None:
    _compare_pipeline(h, w, m, coarse, dtype, hidden, compiled, "cuda")


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("h", "w", "m", "batch", "coarse"),
    [
        (7, 9, 1, 1, False), (8, 6, 35, 3, True),
        (14, 14, 160, 1, False), (14, 4, 35, 128, True),
        (28, 4, 35, 128, False), (28, 28, 160, 3, True),
        (56, 8, 35, 17, False), (56, 16, 160, 1, True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_full_trajectory_reference(
    monkeypatch: pytest.MonkeyPatch, h: int, w: int, m: int, batch: int, coarse: bool,
    dtype: torch.dtype,
) -> None:
    original = scans._scan_plans

    def full_only(*args: object, **kwargs: bool | int) -> list[scans.ScanPlan]:
        kwargs["cache_trajectories"] = True
        plans = original(*args, **kwargs)
        if kwargs["vertical"] and kwargs["backward"]:
            return [p for p in plans if p.full_trajectory]
        return plans

    monkeypatch.setattr(scans, "_scan_plans", full_only)
    scans._launch_for_shape.cache_clear()
    try:
        _compare_pipeline(h, w, m, coarse, dtype, 8, compiled=False, mode="cuda", batch=batch)
    finally:
        scans._launch_for_shape.cache_clear()


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("h", "coarse"), [(57, False), (56, True)])
def test_cuda_trajectory_replay_preserves_gradients(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, h: int, coarse: bool
) -> None:
    monkeypatch.setenv("LNET_VSCAN_DECOUPLED", "0")
    torch.manual_seed(719)
    px, py, real, imag, auxiliary, path = _inputs(h, 8, 35, 8, dtype, "cuda", batch=3)
    states = scans.horizontal_forward(px, real, imag).detach()
    out = scans.vertical_forward(py, states, auxiliary, path, coarse=coarse)
    grad = tuple(torch.randn_like(part) for part in out)
    original = scans._scan_plans

    def force_trajectory(*args: object, **kwargs: bool | int) -> list[scans.ScanPlan]:
        kwargs["cache_trajectories"] = True
        plans = original(*args, **kwargs)
        return [
            p for p in plans
            if p.warps == 1 and p.cache_weights and p.cache_line and not p.shared_acc
            and (p.cache_source, p.full_trajectory) == _strategy
        ]

    # One warp per worker and fewer lines than workers fit in one wave. The
    # reduction order is the same across the checkpoint and cache variants.
    monkeypatch.setattr(scans, "_scan_plans", force_trajectory)
    result = []
    try:
        for _strategy in ((False, False), (True, False), (True, True)):
            scans._launch_for_shape.cache_clear()
            result.append(scans.vertical_backward(
                py, states, auxiliary, path, *grad, coarse=coarse, compact=coarse,
            ))
    finally:
        scans._launch_for_shape.cache_clear()

    for observed in result[1:]:
        for actual, expected in zip(observed, result[0], strict=True):
            if actual.dtype == torch.bfloat16:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                # CUDA can contract FP32 arithmetic differently across
                # specializations, even with the same reduction order.
                diff = actual - expected
                assert diff.abs().max().item() <= 3e-5
                assert diff.norm().item() <= 5e-7 * expected.norm().item()


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "inductor"])
@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cuda_reference_forward_cuda_backward(
    compiled: bool, coarse: bool, dtype: torch.dtype
) -> None:
    _compare_pipeline(14, 20, 48, coarse, dtype, 8, compiled, "reference")


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "inductor"])
@pytest.mark.parametrize(("h", "w", "m", "coarse"), [(128, 192, 7, True), (17, 35, 33, False)])
def test_cuda_dense_scratch_fallback(
    monkeypatch: pytest.MonkeyPatch, compiled: bool, h: int, w: int, m: int, coarse: bool,
) -> None:
    monkeypatch.setenv("LNET_VSCAN_DECOUPLED", "0")
    sms, _, max_warps = scans.device_limits(0)
    monkeypatch.setattr(scans, "device_limits", lambda _: (sms, 16 * 1024, max_warps))
    original_plans = scans._scan_plans
    monkeypatch.setattr(
        scans, "_scan_plans",
        lambda *args, **kwargs: [
            p for p in original_plans(*args, **kwargs)
            if not p.cache_line
        ],
    )
    scans._launch_for_shape.cache_clear()
    try:
        shape = (1, h, w, m)
        for vertical in (False, True):
            launch = scans._launch_for_shape(
                0, torch.float32, shape, 16 * torch.Size(shape).numel(),
                backward=True, vertical=vertical, coarse=coarse and vertical,
                hidden=8 if vertical else 0, precise=False,
            )
            assert launch is not None
            assert not launch.plan.cache_line
        _compare_pipeline(h, w, m, coarse=coarse, dtype=torch.float32, hidden=8,
                          compiled=compiled, mode="cuda")
    finally:
        scans._launch_for_shape.cache_clear()


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hidden", [8, 19])
def test_cuda_mixed_parameter_storage_and_packed_outputs(hidden: int) -> None:
    torch.manual_seed(719)
    px, py, real, imag, auxiliary, path = _inputs(8, 10, 35, hidden, torch.float32, "cuda")
    py = [p.to(dtype).detach().requires_grad_() for p, dtype in zip(
        py, (*_DTYPES, torch.float32), strict=True,
    )]
    path = [
        p.to(dtype).detach().requires_grad_()
        for p, dtype in zip(path, (*_DTYPES, torch.float16), strict=True)
    ]
    states = scans.horizontal_forward(px, real, imag).detach().requires_grad_()
    with torch.no_grad():
        standalone = scans.vertical_forward(py, states, auxiliary, path, coarse=True)
        for planar in (False, True):
            packed = scans.vertical_forward_packed(
                py, states, auxiliary, path, coarse=True, epsilon=1e-6, planar=planar,
            )
            combine = torch.stack if planar else torch.cat
            torch.testing.assert_close(packed, combine(standalone, dim=0 if planar else -1))

    gradients = torch.randn_like(torch.cat(standalone, dim=-1)).chunk(2, dim=-1)
    result = scans.vertical_backward(py, states, auxiliary, path, *gradients, coarse=True)
    mixed = scans.vertical_backward(
        py, states, auxiliary, path, gradients[0].contiguous(), gradients[1], coarse=True,
    )
    for actual, target in zip(mixed, result, strict=True):
        torch.testing.assert_close(actual, target)
    repeated = scans.vertical_backward(py, states, auxiliary, path, *gradients, coarse=True)
    for actual, again in zip(result, repeated, strict=True):
        torch.testing.assert_close(actual, again, rtol=0, atol=0)
    expected = torch.autograd.grad(
        _reference_vscan(py, states, auxiliary, path, coarse=True),
        [*py, states, *path],
        gradients,
    )
    for actual, target in zip(result, expected, strict=True):
        torch.testing.assert_close(actual, target, rtol=2e-2, atol=2e-3)
    assert torch.count_nonzero(result[4][:2, :, :, ::2]) == 0
    assert torch.count_nonzero(result[4][2:, :, :, 1::2]) == 0
    compact = scans.vertical_backward(
        py, states, auxiliary, path, *gradients, coarse=True, compact=True,
    )
    torch.testing.assert_close(compact[4][:2], result[4][:2, :, :, 1::2])
    torch.testing.assert_close(compact[4][2:], result[4][2:, :, :, ::2])
    full_h = scans.horizontal_backward(px, [real, imag], states, result[4])
    compact_h = scans.horizontal_backward(px, [real, imag], states, compact[4], compact=True)
    for actual, target in zip(compact_h, full_h, strict=True):
        # Compact and dense launches may reduce their partials in different orders.
        torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-6)


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compiled", [False, True])
@inductor_config.patch(emulate_precision_casts=False)
def test_cuda_terminal_dispatch(monkeypatch: pytest.MonkeyPatch, compiled: bool) -> None:
    monkeypatch.setenv("LNET_GENERAL_SCANS", "0")
    monkeypatch.delenv("LNET_GENERAL_SCAN_REF_FWD", raising=False)
    px, py, real, imag, _, _ = _inputs(7, 7, 35, 8, torch.float32, "cuda", batch=2)
    px, py = ([p.reshape(1, 1, 1, -1) for p in pole] for pole in (px, py))
    drive = torch.cat((real, imag), dim=-1)
    inputs = [*px, *py, drive]
    legacy = terminal.complex_correlation_descriptor(tuple(px), tuple(py), drive)
    dg = torch.randn_like(legacy)
    expected = torch.autograd.grad(legacy, inputs, dg)

    def unexpected_legacy(*_args: object) -> None:
        pytest.fail("Terminal scan fell back to Triton")

    monkeypatch.setattr(terminal, "triton_bidirectional_product_scan", unexpected_legacy)
    monkeypatch.delenv("LNET_GENERAL_SCANS")
    descriptor = terminal.complex_correlation_descriptor
    if compiled:
        descriptor = torch.compile(descriptor, fullgraph=True)
    actual = descriptor(tuple(px), tuple(py), drive)
    torch.testing.assert_close(actual, legacy, atol=1e-7, rtol=1e-4)
    actual_grad = torch.autograd.grad(actual, inputs, dg)
    for a, e in zip(actual_grad, expected, strict=True):
        torch.testing.assert_close(a, e, atol=1e-6, rtol=1e-4)


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@inductor_config.patch(emulate_precision_casts=False)
def test_cuda_coarse_drive_dtypes(monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype) -> None:
    monkeypatch.setenv("LNET_GENERAL_SCANS", "0")
    monkeypatch.delenv("LNET_GENERAL_SCAN_REF_FWD", raising=False)
    torch.manual_seed(218)
    px, py, real, imag, auxiliary, path = _inputs(8, 10, 35, 8, dtype, "cuda", batch=2)
    px, py = ([p.reshape(1, 1, 1, -1) for p in pole] for pole in (px, py))
    drive = torch.cat((real, imag), dim=-1)
    for run in (
        run_product_scan_coarse_path_collapse_packed,
        run_product_scan_coarse_path_collapse_planes_packed,
    ):
        with pytest.raises(TypeError, match="requires a BF16 CUDA drive"):
            run(tuple(px), tuple(py), drive, tuple(path), gain_normalization="pointwise")

    monkeypatch.delenv("LNET_GENERAL_SCANS")
    out = run_product_scan_coarse_path_collapse_packed(
        tuple(px), tuple(py), drive, tuple(path), gain_normalization="pointwise", epsilon=1e-6,
    )
    planes = run_product_scan_coarse_path_collapse_planes_packed(
        tuple(px), tuple(py), drive, tuple(path), gain_normalization="pointwise", epsilon=1e-6,
    )
    assert out.dtype == dtype
    torch.testing.assert_close(torch.cat(planes, dim=-1), out, atol=0, rtol=0)
    states = torch.stack(bidirectional_product_scan_reference(tuple(px), (real, imag)))
    expected = torch.cat(_reference_vscan(py, states, auxiliary, path, coarse=True), dim=-1)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=8e-3)
    dy = torch.randn_like(out)
    params = [drive, *px, *py, *path]
    packed_grads = torch.autograd.grad(out, params, dy, retain_graph=True)
    plane_grads = torch.autograd.grad(planes, params, dy.chunk(2, dim=-1))
    for actual, target in zip(plane_grads, packed_grads, strict=True):
        torch.testing.assert_close(actual, target)


def test_decoupled_chunk_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    class States:
        is_cuda = True
        dtype = torch.bfloat16
        device = torch.device("cuda:0")
        shape = (4, 80, 128, 192, 160)

        @staticmethod
        def element_size() -> int:
            return 2

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=24 * 1024**3)
    )
    states = States()
    monkeypatch.delenv("LNET_VSCAN_DECOUPLED", raising=False)
    assert scans._decoupled_chunk(states, coarse=False) == 0
    monkeypatch.setenv("LNET_VSCAN_DECOUPLED", "chunk")
    assert 0 < scans._decoupled_chunk(states, coarse=False) < states.shape[1]
    assert 0 < scans._decoupled_chunk(states, coarse=True) < states.shape[1]
    monkeypatch.setenv("LNET_VSCAN_DECOUPLED", "0")
    assert scans._decoupled_chunk(states, coarse=True) == 0
    monkeypatch.setattr(scans, "_l2_size", lambda _: 48 * 1024**2)
    assert scans._saved_backward_chunk(states, coarse=False) == scans._workspace_vscan_chunk(
        states, coarse=False,
    )
    states.shape = (4, 4, 128, 192, 160)
    assert scans._saved_backward_chunk(states, coarse=False) == 1
    states.shape = (4, 32, 128, 192, 160)
    assert scans._saved_backward_chunk(states, coarse=True) == 6


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    ("coarse", "compact", "hidden", "compiled"),
    [(False, False, 8, False), (True, False, 11, True), (True, True, 5, False)],
)
@pytest.mark.parametrize("policy", ["chunk", "test_saved"])
def test_cuda_decoupled_vscan(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    coarse: bool,
    compact: bool,
    hidden: int,
    compiled: bool,
    policy: str,
) -> None:
    torch.manual_seed(791)
    _, pole, real, _, auxiliary, path = _inputs(8, 10, 35, hidden, dtype, "cuda", batch=3)
    path[1] = path[1].to(torch.float16)
    path[3] = path[3].to(torch.bfloat16)
    states = torch.stack([real] * 4)
    monkeypatch.setenv("LNET_VSCAN_DECOUPLED", "0")
    expected = scans.vertical_forward(pole, states, auxiliary, path, coarse)
    dy = tuple(torch.randn_like(value) for value in expected)
    expected_grads = scans.vertical_backward(
        pole, states, auxiliary, path, *dy, coarse, compact=compact
    )

    monkeypatch.setenv("LNET_VSCAN_DECOUPLED", policy)
    plan_chunk = scans._decoupled_chunk
    monkeypatch.setattr(scans, "_decoupled_chunk", lambda *args: min(2, plan_chunk(*args)))
    monkeypatch.setattr(scans, "_saved_backward_chunk", lambda *_: 2)
    run = scans.vertical_forward
    if compiled:
        run = torch.compile(run, fullgraph=True)
    actual = run(pole, states, auxiliary, path, coarse)
    features = None
    if compact and policy == "test_saved":
        _, _, features = scans._vertical_forward_saved(
            pole, states, auxiliary, path, coarse, 1e-6,
            planar=True, precise=True, separate=False,
        )
    actual_grads = (
        scans.vertical_backward(
            pole, states, auxiliary, path, *dy, coarse, compact=True, saved_features=features,
        )
        if compact else torch.autograd.grad(actual, [*pole, states, *path], dy)
    )
    for planar in (False, True):
        packed = scans.vertical_forward_packed(pole, states, auxiliary, path, coarse, 1e-6, planar)
        target = torch.stack(actual) if planar else torch.cat(actual, dim=-1)
        torch.testing.assert_close(packed, target)
    if dtype == torch.float32:
        forward_tol, backward_tol = (4e-5, 3e-6), (4e-4, 2e-4)
    elif dtype == torch.float16:
        forward_tol, backward_tol = (8e-3, 1e-3), (5e-2, 2e-2)
    else:
        forward_tol, backward_tol = (4e-2, 5e-3), (1.2e-1, 8e-2)
    reference_out = _reference_vscan(pole, states, auxiliary, path, coarse)
    for value, target in zip(actual, reference_out, strict=True):
        torch.testing.assert_close(value, target, rtol=forward_tol[0], atol=forward_tol[1])
    if not compact:
        reference_grads = torch.autograd.grad(reference_out, [*pole, states, *path], dy)
        for value, target in zip(actual_grads, reference_grads, strict=True):
            torch.testing.assert_close(value, target, rtol=backward_tol[0], atol=backward_tol[1])
    for values, reference, (rtol, atol) in (
        (actual, expected, forward_tol),
        (actual_grads, expected_grads, backward_tol),
    ):
        for value, target in zip(values, reference, strict=True):
            torch.testing.assert_close(value, target, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("length", "width", "coarse", "span", "paired", "tail", "readonly"),
    [
        (13, 19, False, 13, True, True, True),
        (18, 22, True, 18, True, True, False),
        (10, 12, True, 10, False, False, False),
        (52, 34, False, 14, True, False, False),
        (52, 34, True, 20, True, False, True),
    ],
)
def test_scan_auto_backward(
    monkeypatch: pytest.MonkeyPatch, length: int, width: int, coarse: bool, span: int,
    paired: bool, tail: bool, readonly: bool,
) -> None:
    checkpointed = length > 32
    plan = scans.ScanPlan(
        2, 40_000, 8 if checkpointed else length, cache_weights=True,
        cache_source=True, full_trajectory=not checkpointed,
    )
    key = scans.Kernel(
        torch.bfloat16, plan.warps, backward=True, vertical=True, coarse=coarse,
        hidden8=True, chunk=8 if checkpointed else 0, cache_weights=True, cache_source=True,
    )
    base = scans.Launch(plan, key, 120, 60)
    monkeypatch.setattr(scans, "_launch_for_shape", lambda *_, **_opts: base)
    monkeypatch.setattr(scans, "device_limits", lambda _: (80, 99 << 10, 48))
    monkeypatch.setattr(scans, "_resident", lambda *_: 2)
    shape = (32, length, width, 63)
    source = SimpleNamespace(
        is_cuda=True, dtype=key.dtype, device=SimpleNamespace(index=0),
        numel=lambda: 4 * torch.Size(shape).numel(),
    )
    options = {"backward": True, "vertical": True, "coarse": coarse, "hidden": 8, "precise": False}
    monkeypatch.setenv("LNET_GENERAL_SCAN_OPT", "off")
    assert scans._plan_launch(source, shape, **options) == base
    monkeypatch.delenv("LNET_GENERAL_SCAN_OPT")
    actual = scans._plan_launch(source, shape, **options)
    assert actual is not None
    assert actual.plan.chunk == span
    assert actual.key.backward_batch2 is paired
    assert actual.key.backward_batch_tail is tail
    assert actual.key.backward_readonly is readonly
    assert actual.plan.smem <= 99 << 10


def test_scan_wave_alignment_respects_work_and_workspace() -> None:
    key = scans.Kernel(torch.float16, 2, backward=False, vertical=True)
    launch = scans.Launch(scans.ScanPlan(2, 2048), key, 105, 105)
    assert scans._align_workers(launch, 200, 3, 64).workers == 105
    assert scans._align_workers(launch, 200, 3, 64, max_workers=127).workers == 105
    assert scans._align_workers(launch, 200, 3, 64, max_workers=128).workers == 128
    assert scans._align_workers(launch, 240, 3, 64).workers == 84


@_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("variant", tuple(scans._W37A_KERNELS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize(("h", "w", "m"), [
    (56, 56, 160), (28, 28, 160), (14, 14, 160), (13, 19, 63), (7, 7, 128),
])
def test_cuda_wave37a_parity(
    monkeypatch: pytest.MonkeyPatch, variant: str, dtype: torch.dtype,
    coarse: bool, h: int, w: int, m: int,
) -> None:
    torch.manual_seed(137)
    _, py, real, _, aux, path = _inputs(h, w, m, 8, dtype, "cuda", batch=2)
    states = torch.randn((4, *real.shape), dtype=dtype, device="cuda") * 0.1
    step = 1 + int(coarse)
    grad = [torch.randn((2, h // step, w // step, m), dtype=dtype, device="cuda") * 0.05
            for _ in range(2)]
    recorded = []
    dispatch = scans._dispatch

    def record(source: Tensor, launch: scans.Launch | None, args: tuple) -> None:
        assert launch is not None
        if launch.key.backward and launch.key.vertical:
            recorded.append(launch.key.wave37a)
        dispatch(source, launch, args)

    monkeypatch.setattr(scans, "_dispatch", record)
    monkeypatch.setenv("LNET_SCAN_W37A", variant)
    actual = scans.vertical_backward(py, states, aux, path, *grad, coarse)
    assert recorded == [variant]
    if coarse and (h % 2 or w % 2):
        # The last row can affect the reverse recurrence. Use the full odd
        # source for the reference and only truncate at coarse selection.
        operands = [*py, states.requires_grad_(), *path]
        out = scans._reference_vertical(py, states, aux, path, coarse, 1e-6)
        expected = torch.autograd.grad(out, operands, grad)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, atol=5e-4, rtol=2e-3)
        return
    monkeypatch.setenv("LNET_SCAN_W37A", "off")
    expected = scans.vertical_backward(py, states, aux, path, *grad, coarse)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, atol=5e-4, rtol=2e-3)


@pytest.mark.parametrize("variant", tuple(scans._W37A_KERNELS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("h", "w", "m", "coarse"), [
    (13, 19, 63, False), (13, 19, 63, True), (9, 11, 34, True),
])
def test_wave37a_cpu_batched_accumulation(  # noqa: C901, PLR0915
    variant: str, dtype: torch.dtype, h: int, w: int, m: int, coarse: bool,
) -> None:
    rng = torch.Generator().manual_seed(132)

    def sample(*shape: int) -> Tensor:
        return (torch.randn(shape, generator=rng) * 0.15).to(dtype).float()

    def rounded(value: Tensor) -> Tensor:
        return value.to(dtype).float()

    def fma(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        return (a.double() * b.double() + c.double()).float()

    step = 1 + int(coarse)
    wi, bi, wo = sample(m, 8, 8), sample(m, 8), sample(m, 8, 2)
    source, gain = sample(w, h, m, 2, 4), sample(w, h, m, 2, 4).abs() + 0.5
    # Use both recurrences, retaining the trailing source in the reverse state.
    states = torch.zeros_like(source)
    for direction in (0, 1):
        carry = torch.zeros_like(source[:, 0, :, direction])
        rows = range(h) if direction == 0 else range(h - 1, -1, -1)
        for y in rows:
            carry = fma(torch.tensor(0.125), carry, source[:, y, :, direction])
            states[:, y, :, direction] = carry
    used = h // step * step
    pos = rounded(rounded(states[:, int(coarse):used:step, :, 0])
                  * gain[:, int(coarse):used:step, :, 0])
    neg = rounded(rounded(states[:, :used:step, :, 1]) * gain[:, :used:step, :, 1])
    p = torch.stack([pos[..., 0], pos[..., 2], neg[..., 0], neg[..., 2],
                     pos[..., 1], pos[..., 3], neg[..., 1], neg[..., 3]], dim=-1)
    dy = sample(w, h // step, m, 2)

    def adjoint(k: int, cta: bool) -> tuple[Tensor, Tensor]:
        # Assign each column to its own warp, grouping up to three warps per CTA.
        partial = torch.zeros(((w + 2) // 3 if cta else w, m, 90))
        dp = torch.zeros_like(p)
        for x in range(w):
            row = partial[x // 3 if cta else x]
            for end in range(h // step, 0, -k):
                cells = torch.arange(end - 1, max(-1, end - k - 1), -1)
                inp, grad = p[x, cells], dy[x, cells]
                delta = torch.zeros_like(row) if cta else row
                for jac in grad:
                    delta[:, 88:] += jac
                for j in range(8):
                    z = torch.zeros_like(inp[..., 0])
                    for i in range(8):
                        z = fma(wi[:, j, i], inp[..., i], z)
                    z = rounded(rounded(z) + bi[:, j])
                    sig = 1 / (1 + (-z).exp())
                    act = rounded(z * sig)
                    result = fma(grad[..., 0], wo[:, j, 0], grad[..., 1] * wo[:, j, 1])
                    dz = rounded(rounded(result) * (sig * fma(z, 1 - sig, torch.ones_like(z))))
                    dp[x, cells] = fma(dz[..., None], wi[:, j], dp[x, cells])
                    for t in range(len(cells)):
                        delta[:, 8 * j:8 * (j + 1)] = fma(
                            dz[t, :, None], inp[t], delta[:, 8 * j:8 * (j + 1)],
                        )
                        delta[:, 64 + j] += dz[t]
                        delta[:, 72 + j] = fma(grad[t, :, 0], act[t], delta[:, 72 + j])
                        delta[:, 80 + j] = fma(grad[t, :, 1], act[t], delta[:, 80 + j])
                if cta:
                    row += delta
        return dp, partial.sum(0)

    path_dp, reduced = adjoint(scans._W37A_KERNELS[variant][1], cta=True)
    ref_dp, ref_reduced = adjoint(1, cta=False)
    torch.testing.assert_close(path_dp, ref_dp, atol=0, rtol=0)
    torch.testing.assert_close(reduced, ref_reduced, atol=3e-6, rtol=0)

    def column_adjoint(dp: Tensor, chunk: int) -> tuple[Tensor, Tensor]:
        dstate = torch.zeros_like(source)
        dpoles = torch.zeros((m, 2, 2))
        decay = torch.tensor(0.125)
        for direction in (0, 1):
            dp_indices = [0, 4, 1, 5] if direction == 0 else [2, 6, 3, 7]
            inject = torch.zeros_like(source[..., direction, :])
            injections = dp[..., dp_indices]
            start = int(coarse and direction == 0)
            inject[:, start:used:step] = rounded(
                rounded(injections) * gain[:, start:used:step, :, direction],
            )
            order = list(range(h)) if direction == 0 else list(reversed(range(h)))
            prev = {}
            for offset in range(0, h, chunk):
                carry = (states[:, order[offset - 1], :, direction] if offset
                         else torch.zeros_like(source[:, 0, :, direction]))
                for row in order[offset:offset + chunk]:
                    prev[row] = carry
                    carry = fma(decay, carry, source[:, row, :, direction])
            adj = torch.zeros_like(source[:, 0, :, direction])
            for row in reversed(order):
                adj = fma(decay, adj, inject[:, row])
                dstate[:, row, :, direction] = rounded(adj)
                dpoles[:, direction, 0] += (adj * prev[row]).sum((0, 2))
                dpoles[:, direction, 1] += (adj * source[:, row, :, direction]).sum((0, 2))
        return dstate, dpoles

    for wave, ref in zip(column_adjoint(path_dp, 6), column_adjoint(ref_dp, h), strict=True):
        torch.testing.assert_close(wave, ref, atol=0, rtol=0)
