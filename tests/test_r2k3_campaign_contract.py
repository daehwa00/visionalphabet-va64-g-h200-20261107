"""Pin the shared campaign contract assembly.

Campaign runners used to repeat the contract field sequence verbatim.  The
harness compares the whole payload for equality when resuming a run root, so a
drift in this shape silently invalidates every existing run.  These tests hold
the assembled shape and the seed-critical build order in place.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
from pathlib import Path

import pytest

# Import the same top-level module objects the runners themselves import.
# `scripts.X` and `X` are distinct entries in sys.modules, so patching the
# wrong one silently misses.
import r2k3_campaign as campaign
import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as clpost
import run_alphabet2d_imagenet100_nano as harness
import run_complex_scan_zero_init_imagenet100 as zero_init

CONTRACT_KEYS = {
    "schema",
    "evidence_status",
    "variants",
    "seeds",
    "variant_configs",
    "parameter_counts",
    "architecture",
    "source_sha256",
    "model",
    "recipe",
    "data",
}


@pytest.fixture
def contract_args(monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    """Supply the harness arguments a contract needs without touching the dataset."""
    monkeypatch.setattr(
        harness,
        "_dataset_digest",
        lambda _root: (zero_init.IMAGENET100_MANIFEST_SHA256, 130_000, 5_000),
    )
    return argparse.Namespace(
        root=Path("/nonexistent/run"),
        data_root=Path("/nonexistent/data"),
        epochs=100,
        batch_size=128,
        gradient_accumulation_steps=1,
        precision="bfloat16",
        workers=8,
        run_seeds=[501],
        variants=None,
        initialize_only=False,
    )


def test_campaign_contract_publishes_the_expected_field_set(
    contract_args: argparse.Namespace,
) -> None:
    payload = clpost._contract(contract_args)
    assert set(payload) >= CONTRACT_KEYS


def test_campaign_contract_scopes_every_variant_dict_to_the_variant(
    contract_args: argparse.Namespace,
) -> None:
    payload = clpost._contract(contract_args)
    for field in ("variant_configs", "parameter_counts", "architecture"):
        assert list(payload[field]) == [clpost.VARIANT], field
    assert payload["variants"] == list(clpost.VARIANTS)
    assert payload["seeds"] == list(clpost.SEEDS)


def test_campaign_contract_records_the_runner_and_its_extra_sources(
    contract_args: argparse.Namespace,
) -> None:
    payload = clpost._contract(contract_args)
    sources = payload["source_sha256"]
    assert "clpost_runner" in sources
    assert "gated_post_fusion" in sources
    assert all(len(value) == 64 for value in sources.values())


def test_build_derived_rejects_a_variant_the_campaign_does_not_own() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        clpost._build("some-other-variant", None)


def test_build_derived_runs_install_then_configure_then_assert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The order fixes seeded initialization, so it is part of the contract."""
    model = object()
    config = object()
    calls: list[tuple[object, ...]] = []

    def parent_build(variant: str, active_config: object) -> object:
        calls.append(("parent_build", variant, active_config))
        return model

    def configure(variants: tuple[str, ...], seeds: tuple[int, ...]) -> None:
        calls.append(("configure", variants, seeds))

    monkeypatch.setattr(campaign.runtime, "configure", configure)

    campaign.build_derived(
        requested="V",
        variant="V",
        base_variant="B",
        config=config,
        parent_build=parent_build,
        install=lambda active_model: calls.append(("install", active_model)),
        assert_model=lambda active_model: calls.append(("assert", active_model)),
        variants=("V",),
        seeds=(501,),
    )
    assert calls == [
        ("parent_build", "B", config),
        ("install", model),
        ("configure", ("V",), (501,)),
        ("assert", model),
    ]


def test_campaign_contract_rejects_variant_metadata_that_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(campaign.runtime, "base_contract", lambda _args: {"source_sha256": {}})

    with pytest.raises(ValueError, match="cover exactly"):
        campaign.campaign_contract(
            argparse.Namespace(),
            runner_file=tmp_path / "runner.py",
            runner_source_key="runner",
            variants=("first", "second"),
            seeds=(501,),
            schema="test",
            evidence_status="test",
            variant_configs={"first": {}},
            architectures={"first": "first architecture", "second": "second architecture"},
            parameter_counts={"first": 1, "second": 2},
        )
