from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from scripts import r2k3_runner_registry as registry
from scripts import smoke_r2k3_campaign as smoke

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def test_unknown_runner_is_reported_with_its_canonical_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(module_name: str) -> ModuleType:
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(registry.importlib, "import_module", missing)

    with pytest.raises(ValueError, match="run_missing_imagenet100"):
        registry.resolve_runner("missing")


def test_runner_transitive_import_failure_is_not_mislabeled_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(_module_name: str) -> ModuleType:
        raise ModuleNotFoundError(name="required_dependency")

    monkeypatch.setattr(registry.importlib, "import_module", missing_dependency)

    with pytest.raises(ModuleNotFoundError) as raised:
        registry.resolve_runner("campaign")

    assert raised.value.name == "required_dependency"


def _run_smoke_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    active_runner: object,
) -> list[tuple[str, tuple[str, ...]]]:
    args = SimpleNamespace(
        runner=active_runner,
        root=tmp_path,
        variants=None,
    )
    calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(smoke, "_arguments", lambda _argv: args)
    monkeypatch.setattr(smoke, "_device", lambda _args: torch.device("cpu"))
    monkeypatch.setattr(smoke.shared, "_cell_reconstruction_check", lambda _device: 0.0)
    monkeypatch.setattr(smoke.shared, "_full_state_semantic_check", lambda _device: 0.0)
    monkeypatch.setattr(smoke.shared, "runner", smoke.shared.runner)
    monkeypatch.setattr(
        smoke,
        "_multi_variant_payload",
        lambda _args, _device, variants: calls.append(("multi", variants)) or {},
    )
    monkeypatch.setattr(
        smoke,
        "_single_variant_payload",
        lambda runner, _args, _device: calls.append(("single", (runner.VARIANT,))) or {},
    )

    smoke.main([])

    return calls


def test_smoke_main_accepts_a_variants_only_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    active_runner = SimpleNamespace(VARIANTS=("first", "second"))

    assert _run_smoke_main(monkeypatch, tmp_path, active_runner) == [
        ("multi", active_runner.VARIANTS)
    ]


def test_smoke_main_accepts_a_variant_only_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    active_runner = SimpleNamespace(VARIANT="only")

    assert _run_smoke_main(monkeypatch, tmp_path, active_runner) == [("single", ("only",))]
