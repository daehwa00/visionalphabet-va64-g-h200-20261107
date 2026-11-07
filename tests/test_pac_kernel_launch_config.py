from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingParameterType=false
# pyright: reportPrivateUsage=false
# Triton kernel signatures are not ordinary Python signatures.
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch
import triton
import triton.language as tl

import lnet.pac_kernel_launch_config as launch_config
import lnet.pac_launch_tuning as launch_tuning
import lnet.pac_product_scan_normalization as product_scan_normalization
import lnet.pac_product_scan_ops as product_scan
import lnet.pac_triton_bidirectional_product_scan as bidirectional_scan
import lnet.pac_triton_complex_rmsnorm as complex_rmsnorm
import lnet.pac_triton_grouped_path_cffn as grouped_path_cffn
import lnet.pac_triton_phase_gate as phase_gate
import lnet.pac_triton_phase_gate_linear as phase_gate_linear
import lnet.pac_triton_phase_gate_linear_fused as fused_phase_gate_linear
import lnet.pac_triton_phase_gated_cffn_fused as fused_phase_gated_cffn
import lnet.pac_triton_rmsnorm_linear_fused as fused_rmsnorm_linear
from lnet.pac_kernel_launch_config import (
    LaunchGeometry,
    LaunchScope,
    UnknownKernelError,
    autotuned,
    capture_launch_scopes,
    device_key,
    make_launch_scope,
    register_default,
    registered_candidates,
    registered_default,
    registered_names,
    resolve,
    store_geometry,
    stored_geometry,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@triton.jit
def _probe_kernel(output, N: tl.constexpr, BLOCK_N: tl.constexpr) -> None:
    """Minimal real JITFunction; triton.autotune rejects plain callables."""
    offsets = tl.arange(0, BLOCK_N)
    tl.store(output + offsets, offsets, mask=offsets < N)


def _scope(
    *,
    shape: dict[str, int] | None = None,
    dtype: str = "bfloat16",
    execution_regime: str = "compiled",
) -> LaunchScope:
    return LaunchScope.build(
        execution_regime=execution_regime,
        dtype=dtype,
        shape={"batch": 64, "height": 56, "modes": 64, **(shape or {})},
        kernel_revision="kernel-revision-a",
        torch_version="torch-a",
        triton_version="triton-a",
        cuda_version="cuda-a",
        cuda_driver_version=12080,
    )


def test_geometry_is_hashable_so_it_can_key_the_autotuner_cache() -> None:
    geometry = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 32})
    assert {geometry: "cached"}[geometry] == "cached"
    assert geometry.blocks == {"BLOCK_N": 32}


def test_block_order_does_not_change_identity() -> None:
    first = LaunchGeometry.build(blocks={"BLOCK_A": 1, "BLOCK_B": 2})
    second = LaunchGeometry.build(blocks={"BLOCK_B": 2, "BLOCK_A": 1})
    assert first == second
    assert hash(first) == hash(second)


def test_registered_kernels_expose_their_measured_defaults() -> None:
    assert product_scan.FORWARD_LAUNCH_NAME in registered_names()
    geometry = registered_default(product_scan.FORWARD_LAUNCH_NAME)
    assert geometry.num_warps == 4
    assert geometry.blocks == {"BLOCK_LINES": 1, "BLOCK_MODES": 8}


@pytest.mark.parametrize(
    ("name", "symbol"),
    [
        (
            product_scan.DESCRIPTOR_FORWARD_LAUNCH_NAME,
            "_product_scan_coarse4_associative_forward_kernel",
        ),
        (
            product_scan.DESCRIPTOR_BACKWARD_LAUNCH_NAME,
            "_product_scan_coarse4_associative_backward_kernel",
        ),
        (
            product_scan_normalization.GLOBAL_GAIN_LAUNCH_NAME,
            "_static_global_inverse_gain_kernel",
        ),
    ],
)
def test_real_step_tuner_maps_descriptor_only_scan_kernels(
    name: str,
    symbol: str,
) -> None:
    assert launch_tuning._KERNEL_SYMBOLS[name] == symbol


@pytest.mark.parametrize(
    ("name", "symbol"),
    [
        (
            grouped_path_cffn.FORWARD_LAUNCH_NAME,
            "_d4_grouped_path_collapse_forward_kernel",
        ),
        (
            grouped_path_cffn.BACKWARD_LAUNCH_NAME,
            "_d4_grouped_path_collapse_backward_kernel",
        ),
    ],
)
def test_real_step_tuner_maps_grouped_d4_cffn_kernels(name: str, symbol: str) -> None:
    assert launch_tuning._KERNEL_SYMBOLS[name] == symbol


@pytest.mark.parametrize(
    ("name", "symbol"),
    [
        (
            complex_rmsnorm.FORWARD_LAUNCH_NAME,
            "_packed_complex_rmsnorm_forward_kernel",
        ),
        (
            complex_rmsnorm.BACKWARD_LAUNCH_NAME,
            "_packed_complex_rmsnorm_backward_kernel",
        ),
        (
            complex_rmsnorm.BACKWARD_REDUCE_LAUNCH_NAME,
            "_packed_complex_rmsnorm_backward_reduce_kernel",
        ),
        (phase_gate.FORWARD_LAUNCH_NAME, "_phase_gate_forward_kernel"),
        (
            phase_gate_linear.BACKWARD_LAUNCH_NAME,
            "_phase_gate_output_linear_backward_kernel",
        ),
        (fused_phase_gate_linear.FORWARD_LAUNCH_NAME, "_fused_forward_kernel"),
        (fused_phase_gate_linear.BACKWARD_LAUNCH_NAME, "_fused_backward_kernel"),
        (
            fused_phase_gated_cffn.FORWARD_LAUNCH_NAME,
            "_phase_gated_cffn_forward_kernel",
        ),
        (
            fused_phase_gated_cffn.BACKWARD_LAUNCH_NAME,
            "_phase_gated_cffn_backward_kernel",
        ),
        (fused_rmsnorm_linear.FORWARD_LAUNCH_NAME, "_fused_forward_kernel"),
        (fused_rmsnorm_linear.BACKWARD_LAUNCH_NAME, "_fused_backward_kernel"),
        (
            phase_gate.BACKWARD_REDUCE_LAUNCH_NAME,
            "_phase_gate_backward_reduce_kernel",
        ),
    ],
)
def test_real_step_tuner_maps_phase_gated_kernels(name: str, symbol: str) -> None:
    assert launch_tuning._KERNEL_SYMBOLS[name] == symbol


@pytest.mark.parametrize(
    "name",
    [
        bidirectional_scan.FORWARD_LAUNCH_NAME,
        bidirectional_scan.BACKWARD_LAUNCH_NAME,
    ],
)
def test_bidirectional_scan_autotunes_only_associative_geometries(name: str) -> None:
    candidates = registered_candidates(name)
    assert all(set(candidate.blocks) == {"BLOCK_MODES"} for candidate in candidates)
    assert {candidate.blocks["BLOCK_MODES"] for candidate in candidates} == {8, 16, 32, 64}


def test_unregistered_kernel_is_reported_rather_than_silently_defaulted() -> None:
    with pytest.raises(UnknownKernelError):
        resolve("no_such_kernel")


def test_resolution_order_is_argument_over_environment_over_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_default("probe_kernel", LaunchGeometry.build(num_warps=4, blocks={"BLOCK_N": 16}))

    assert resolve("probe_kernel").num_warps == 4
    assert resolve("probe_kernel").blocks == {"BLOCK_N": 16}

    monkeypatch.setenv("LNET_LAUNCH_PROBE_KERNEL_WARPS", "2")
    monkeypatch.setenv("LNET_LAUNCH_PROBE_KERNEL_BLOCK_N", "64")
    assert resolve("probe_kernel").num_warps == 2
    assert resolve("probe_kernel").blocks == {"BLOCK_N": 64}

    # An explicit argument outranks the environment.
    assert resolve("probe_kernel", num_warps=8).num_warps == 8
    assert resolve("probe_kernel", blocks={"BLOCK_N": 128}).blocks == {"BLOCK_N": 128}


def test_explicit_geometry_replaces_the_registered_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_default("probe_kernel2", LaunchGeometry.build(num_warps=4, blocks={"BLOCK_N": 16}))
    monkeypatch.delenv("LNET_LAUNCH_PROBE_KERNEL2_WARPS", raising=False)
    supplied = LaunchGeometry.build(num_warps=1, blocks={"BLOCK_N": 256})
    assert resolve("probe_kernel2", geometry=supplied) == supplied


def test_tuned_geometry_round_trips_through_the_per_device_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    register_default("store_probe", LaunchGeometry.build(num_warps=4, blocks={"BLOCK_N": 16}))
    assert stored_geometry("store_probe") is None
    assert resolve("store_probe").num_warps == 4

    tuned = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 128})
    store_geometry("store_probe", tuned)
    assert stored_geometry("store_probe") == tuned
    # A stored record outranks the registered default but not an override.
    assert resolve("store_probe") == tuned
    assert resolve("store_probe", num_warps=8).num_warps == 8


def test_corrupt_optional_store_falls_back_to_registered_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    (tmp_path / "launch_geometry.json").write_text("not-json")
    expected = LaunchGeometry.build(num_warps=4, blocks={"BLOCK_N": 16})
    register_default("corrupt_store_probe", expected)

    assert stored_geometry("corrupt_store_probe") is None
    assert resolve("corrupt_store_probe") == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_regime", "eager"),
        ("compile_profile", '{"mode":"reduce-overhead"}'),
        ("dtype", "float32"),
        ("shape_items", (("batch", 32), ("height", 56), ("modes", 64))),
        ("torch_version", "torch-b"),
        ("triton_version", "triton-b"),
        ("cuda_version", "cuda-b"),
        ("cuda_driver_version", 12090),
        ("kernel_revision", "kernel-revision-b"),
    ],
)
def test_scoped_geometry_requires_an_exact_workload_and_compiler_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    exact_scope = _scope()
    mismatch = replace(exact_scope, **{field: value})
    tuned = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 128})

    store_geometry("scoped_probe", tuned, scope=exact_scope)

    assert stored_geometry("scoped_probe", scope=exact_scope) == tuned
    assert stored_geometry("scoped_probe", scope=mismatch) is None
    assert stored_geometry("scoped_probe") is None


def test_launch_scope_uses_kernel_revision_dtype_shape_and_execution_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setenv(
        "LNET_COMPILE_PROFILE",
        '{"dynamic":false,"fullgraph":false,"mode":"default"}',
    )
    reference = torch.empty(2, dtype=torch.bfloat16)

    scope = make_launch_scope(_probe_kernel, reference, shape={"modes": 64, "batch": 32})

    assert scope.execution_regime == "compiled"
    assert scope.compile_profile == ('{"dynamic":false,"fullgraph":false,"mode":"default"}')
    assert scope.dtype == "bfloat16"
    assert scope.shape_items == (("batch", 32), ("modes", 64))
    assert scope.kernel_revision == _probe_kernel.cache_key


def test_scope_capture_only_collects_the_named_kernel() -> None:
    first = _scope(shape={"batch": 16})
    second = _scope(shape={"batch": 32})

    with capture_launch_scopes("capture_probe") as captured:
        resolve("capture_probe", scope=first, geometry=LaunchGeometry.build())
        resolve("another_probe", scope=second, geometry=LaunchGeometry.build())

    assert captured == {first}


def test_exact_scope_override_does_not_change_sibling_scopes() -> None:
    name = "exact_override_probe"
    default = LaunchGeometry.build(num_warps=4, blocks={"BLOCK_N": 16})
    candidate = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 32})
    register_default(name, default, candidates=(candidate,))
    first = _scope(shape={"batch": 16})
    second = _scope(shape={"batch": 32})

    with launch_config.override_launch_geometry(name, first, candidate):
        assert resolve(name, scope=first) == candidate
        assert resolve(name, scope=second) == default

    assert resolve(name, scope=first) == default


def test_real_step_tuner_persists_independent_winner_for_each_exact_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    name = "scoped_tuning_probe"
    default = LaunchGeometry.build(num_warps=4, blocks={"BLOCK_N": 16})
    small_candidate = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 32})
    large_candidate = LaunchGeometry.build(num_warps=8, blocks={"BLOCK_N": 64})
    register_default(name, default, candidates=(small_candidate, large_candidate))
    monkeypatch.setitem(launch_tuning._KERNEL_SYMBOLS, name, "probe_symbol")
    small_scope = _scope(shape={"batch": 16})
    large_scope = _scope(shape={"batch": 64})
    scopes = (small_scope, large_scope)
    costs = {
        small_scope: {default: 4.0, small_candidate: 1.0, large_candidate: 3.0},
        large_scope: {default: 4.0, small_candidate: 5.0, large_candidate: 1.0},
    }
    active: dict[LaunchScope, LaunchGeometry] = {}

    def step() -> None:
        active.clear()
        active.update((scope, resolve(name, scope=scope)) for scope in scopes)

    def fake_kernel_time_ms(
        measured_step: Callable[[], None],
        symbol: str,
        *,
        iterations: int,
    ) -> float:
        del symbol, iterations
        measured_step()
        return sum(costs[scope][geometry] for scope, geometry in active.items())

    monkeypatch.setattr(launch_tuning, "kernel_time_ms", fake_kernel_time_ms)

    results = launch_tuning.tune_kernel(name, step, iterations=1)

    assert [result.winner for result in results] == [small_candidate, large_candidate]
    assert [result.scopes for result in results] == [(small_scope,), (large_scope,)]
    assert stored_geometry(name, scope=small_scope) == small_candidate
    assert stored_geometry(name, scope=large_scope) == large_candidate
    assert stored_geometry(name) is None


def test_real_step_tuner_persists_both_bidirectional_compiled_winners(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launch_config, "device_key", lambda: "test-device")
    names = (
        bidirectional_scan.FORWARD_LAUNCH_NAME,
        bidirectional_scan.BACKWARD_LAUNCH_NAME,
    )
    scopes = {
        name: _scope(
            shape={"batch": 128, "height": 56, "width": 56, "modes": 64},
            execution_regime="compiled",
        )
        for name in names
    }
    candidates = {name: registered_candidates(name) for name in names}
    forward_winner = next(
        geometry
        for geometry in candidates[bidirectional_scan.FORWARD_LAUNCH_NAME]
        if geometry.blocks == {"BLOCK_MODES": 32} and geometry.num_warps == 4
    )
    backward_winner = next(
        geometry
        for geometry in candidates[bidirectional_scan.BACKWARD_LAUNCH_NAME]
        if geometry.blocks == {"BLOCK_MODES": 64} and geometry.num_warps == 8
    )
    winners = {
        bidirectional_scan.FORWARD_LAUNCH_NAME: forward_winner,
        bidirectional_scan.BACKWARD_LAUNCH_NAME: backward_winner,
    }
    names_by_symbol = {launch_tuning._KERNEL_SYMBOLS[name]: name for name in names}
    active: dict[str, LaunchGeometry] = {}

    def step() -> None:
        active.update((name, resolve(name, scope=scopes[name])) for name in names)

    def fake_kernel_time_ms(
        measured_step: Callable[[], None],
        symbol: str,
        *,
        iterations: int,
    ) -> float:
        del iterations
        measured_step()
        name = names_by_symbol[symbol]
        return 1.0 if active[name] == winners[name] else 4.0

    monkeypatch.setattr(launch_tuning, "kernel_time_ms", fake_kernel_time_ms)

    results = launch_tuning.tune_all(step, names=names, iterations=1)

    assert {result.name: result.winner for result in results} == winners
    assert all(result.scopes == (scopes[result.name],) for result in results)
    assert all(stored_geometry(name, scope=scopes[name]) == winners[name] for name in names)


def test_concurrent_scoped_writes_preserve_every_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    scopes = [_scope(shape={"batch": batch}) for batch in range(1, 17)]
    geometries = [
        LaunchGeometry.build(num_warps=1 << (index % 4), blocks={"BLOCK_N": 16 + index})
        for index in range(len(scopes))
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(store_geometry, "concurrent_probe", geometry, scope=scope)
            for scope, geometry in zip(scopes, geometries, strict=True)
        ]
        for future in futures:
            future.result()

    assert [stored_geometry("concurrent_probe", scope=scope) for scope in scopes] == geometries
    document = json.loads((tmp_path / "launch_geometry.json").read_text())
    assert document["schema_version"] == 2
    assert len(document["records"][device_key()]["concurrent_probe"]) == len(scopes)
    assert not tuple(tmp_path.glob(".launch_geometry.json.*.tmp"))


def test_legacy_store_is_readable_and_migrates_without_leaking_into_scopes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    legacy = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 64})
    path = tmp_path / "launch_geometry.json"
    path.write_text(
        json.dumps(
            {
                device_key(): {
                    "legacy_probe": {
                        "num_warps": legacy.num_warps,
                        "num_stages": legacy.num_stages,
                        "blocks": legacy.blocks,
                    }
                }
            }
        )
    )
    scope = _scope()
    scoped = LaunchGeometry.build(num_warps=4, blocks={"BLOCK_N": 128})

    assert stored_geometry("legacy_probe") == legacy
    assert stored_geometry("legacy_probe", scope=scope) is None
    store_geometry("legacy_probe", scoped, scope=scope)

    assert stored_geometry("legacy_probe") == legacy
    assert stored_geometry("legacy_probe", scope=scope) == scoped
    assert json.loads(path.read_text())["schema_version"] == 2


def test_unseen_eager_scope_searches_candidates_without_launch_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    monkeypatch.delenv("LNET_DISABLE_LAUNCH_AUTOTUNE", raising=False)
    default = LaunchGeometry.build(num_warps=1, blocks={"BLOCK_N": 16})
    candidate = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 64})
    register_default("auto_probe", default, candidates=(candidate,))

    tuner = autotuned(
        _probe_kernel,
        "auto_probe",
        key=("N",),
        scope=_scope(execution_regime="eager"),
    )
    configs = tuner.configs

    assert [(config.num_warps, config.kwargs) for config in configs] == [
        (1, {"BLOCK_N": 16}),
        (2, {"BLOCK_N": 64}),
    ]
    assert tuner.cache_results is True


def test_unseen_compiled_scope_searches_candidates_in_compiled_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    monkeypatch.delenv("LNET_DISABLE_LAUNCH_AUTOTUNE", raising=False)
    default = LaunchGeometry.build(num_warps=1, blocks={"BLOCK_N": 16})
    candidate = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 64})
    register_default("compiled_floor_probe", default, candidates=(candidate,))

    tuner = autotuned(
        _probe_kernel,
        "compiled_floor_probe",
        key=("N",),
        scope=_scope(execution_regime="compiled"),
    )

    assert [(config.num_warps, config.kwargs) for config in tuner.configs] == [
        (1, {"BLOCK_N": 16}),
        (2, {"BLOCK_N": 64}),
    ]
    assert tuner.cache_results is True


def test_autotuned_wrapper_pins_a_stored_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    monkeypatch.delenv("LNET_DISABLE_LAUNCH_AUTOTUNE", raising=False)
    default = LaunchGeometry.build(num_warps=1, blocks={"BLOCK_N": 16})
    stored = LaunchGeometry.build(num_warps=4, blocks={"BLOCK_N": 128})
    register_default("stored_auto_probe", default, candidates=(stored,))
    scope = _scope()
    store_geometry("stored_auto_probe", stored, scope=scope)

    tuner = autotuned(_probe_kernel, "stored_auto_probe", key=("N",), scope=scope)
    configs = tuner.configs

    assert len(configs) == 1
    assert configs[0].num_warps == 4
    assert configs[0].kwargs == {"BLOCK_N": 128}


def test_unscoped_legacy_geometry_does_not_pin_shape_autotuning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    monkeypatch.delenv("LNET_DISABLE_LAUNCH_AUTOTUNE", raising=False)
    default = LaunchGeometry.build(num_warps=1, blocks={"BLOCK_N": 16})
    legacy = LaunchGeometry.build(num_warps=4, blocks={"BLOCK_N": 128})
    register_default("legacy_auto_probe", default, candidates=(legacy,))
    store_geometry("legacy_auto_probe", legacy)

    tuner = autotuned(
        _probe_kernel,
        "legacy_auto_probe",
        key=("N",),
        scope=_scope(execution_regime="eager"),
    )
    configs = tuner.configs

    assert [(config.num_warps, config.kwargs) for config in configs] == [
        (1, {"BLOCK_N": 16}),
        (4, {"BLOCK_N": 128}),
    ]


def test_autotuned_wrapper_pins_an_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    monkeypatch.delenv("LNET_DISABLE_LAUNCH_AUTOTUNE", raising=False)
    monkeypatch.setenv("LNET_LAUNCH_ENV_AUTO_PROBE_WARPS", "2")
    monkeypatch.setenv("LNET_LAUNCH_ENV_AUTO_PROBE_BLOCK_N", "64")
    default = LaunchGeometry.build(num_warps=1, blocks={"BLOCK_N": 16})
    register_default("env_auto_probe", default)

    tuner = autotuned(_probe_kernel, "env_auto_probe", key=("N",))
    configs = tuner.configs

    assert len(configs) == 1
    assert configs[0].num_warps == 2
    assert configs[0].kwargs == {"BLOCK_N": 64}


def test_autotuned_wrapper_can_disable_first_use_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LNET_LAUNCH_CACHE", str(tmp_path))
    monkeypatch.setenv("LNET_DISABLE_LAUNCH_AUTOTUNE", "1")
    default = LaunchGeometry.build(num_warps=1, blocks={"BLOCK_N": 16})
    candidate = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 64})
    register_default("disabled_auto_probe", default, candidates=(candidate,))

    tuner = autotuned(_probe_kernel, "disabled_auto_probe", key=("N",))
    configs = tuner.configs

    assert len(configs) == 1
    assert configs[0].num_warps == 1
    assert configs[0].kwargs == {"BLOCK_N": 16}


def test_autotuned_wrapper_pins_an_explicit_geometry() -> None:
    geometry = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 64})
    tuner = autotuned(_probe_kernel, "wrapper_probe", key=("N",), geometry=geometry)
    configs = tuner.configs
    assert len(configs) == 1
    assert configs[0].num_warps == 2
    assert configs[0].kwargs == {"BLOCK_N": 64}


def test_autotuned_wrapper_forwards_and_keys_restore_value() -> None:
    geometry = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 64})
    restored = autotuned(
        _probe_kernel,
        "restore_probe",
        key=("N",),
        geometry=geometry,
        restore_value=("output",),
    )
    plain = autotuned(
        _probe_kernel,
        "restore_probe",
        key=("N",),
        geometry=geometry,
    )

    assert restored.restore_value == ["output"]
    assert plain.restore_value == []
    assert restored is not plain


def test_autotuned_wrapper_cache_is_device_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = LaunchGeometry.build(num_warps=2, blocks={"BLOCK_N": 64})
    active_device = "gpu-a"
    monkeypatch.setattr(
        "lnet.pac_kernel_launch_config.device_key",
        lambda: active_device,
    )

    first = autotuned(
        _probe_kernel,
        "device_scoped_wrapper_probe",
        key=("N",),
        geometry=geometry,
    )
    active_device = "gpu-b"
    second = autotuned(
        _probe_kernel,
        "device_scoped_wrapper_probe",
        key=("N",),
        geometry=geometry,
    )

    assert first is not second
