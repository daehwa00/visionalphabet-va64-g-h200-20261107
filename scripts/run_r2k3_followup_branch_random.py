#!/usr/bin/env python3
"""Run P0-5 branch scaling and P0-6 shared matched-random pole controls.

This is a frozen-checkpoint follow-up runner.  It intentionally owns separate
families and caches so that it can resume safely after a GPU/WSL interruption
without changing the original checkpoint-study artifacts.
"""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportMissingImports=false, reportPrivateUsage=false
import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import types
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import torch
from r2k3_frozen_study_common import (
    PredictionResult,
    atomic_json,
    bh_fdr,
    bootstrap_mean_ci,
    coefficient_intervention,
    collect_predictions,
    imagefolder,
    load_frozen_model,
    load_predictions,
    make_loader,
    model_state_sha256,
    paired_effect,
    pole_mask_transform,
    prediction_metrics,
    random_matched_indices,
    same_resolution_blocks,
    save_predictions,
    scan_modules,
    sha256_file,
)
from scipy import stats
from torch import Tensor, nn

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock


SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
BRANCH_FAMILY = "P05_memory_carry_scale_sweep"
RANDOM_FAMILY = "P06_pole_matched_random_expansion"
SCHEMA = "lnet.r2k3.followup.family.v1"
BASE_RANDOM_SEED = 91_000


class CheckpointYieldError(RuntimeError):
    """Signal a deliberate restart after a bounded number of new GPU passes."""


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source-study-root",
        type=Path,
        help="Original study root used only for validated cache reuse (defaults to --root).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--hierarchical-repeats", type=int, default=10_000)
    parser.add_argument(
        "--random-repeats-per-contract",
        type=int,
        default=100,
        help="Unique random subsets for each distinct module/count contract.",
    )
    parser.add_argument(
        "--max-new-conditions",
        type=int,
        default=5,
        help="Yield cleanly after this many uncached GPU evaluations; 0 disables yielding.",
    )
    parser.add_argument("--expected-val-samples", type=int, default=5_000)
    parser.add_argument(
        "--group",
        choices=("branches", "random", "all"),
        default="all",
    )
    return parser.parse_args()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_seed(value: str, *, offset: int = 0) -> int:
    digest = hashlib.sha256(value.encode()).digest()
    return offset + int.from_bytes(digest[:4], "big")


def _scale_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _condition_id(memory_scale: float, carry_scale: float) -> str:
    return f"memory_{_scale_token(memory_scale)}__carry_{_scale_token(carry_scale)}"


@contextlib.contextmanager
def scaled_memory_carry(
    model: nn.Module,
    *,
    memory_scale: float,
    carry_scale: float,
) -> Generator[None]:
    """Scale both branches at every coarsening and same-resolution merge.

    Main transitions receive collapsed pole memory in arguments 0/1 and the
    pre-pooled S2D carry in arguments 2/3.  Same-resolution blocks are wrapped
    at their explicit ``carry + collapsed_memory`` merge.  Hooks/methods are
    removed or restored even when evaluation raises.
    """

    if not (math.isfinite(memory_scale) and math.isfinite(carry_scale)):
        raise ValueError("branch scales must be finite")
    blocks = same_resolution_blocks(model)
    backups: dict[SameResolutionPoleScanBlock, Any] = {}
    with contextlib.ExitStack() as stack:
        for stage_name in ("stage1", "stage2", "stage3"):
            transition = getattr(model, stage_name).augmented

            def prehook(
                _module: nn.Module,
                inputs: tuple[Tensor, ...],
                *,
                active_memory_scale: float = memory_scale,
                active_carry_scale: float = carry_scale,
            ) -> tuple[Tensor, ...]:
                if len(inputs) != 4:
                    raise RuntimeError("main branch-scale hook expected four complex coordinates")
                return (
                    inputs[0] * active_memory_scale,
                    inputs[1] * active_memory_scale,
                    inputs[2] * active_carry_scale,
                    inputs[3] * active_carry_scale,
                )

            stack.callback(transition.register_forward_pre_hook(prehook).remove)

        for block in blocks.values():
            original = block.forward
            backups[block] = original

            def replacement(
                active: SameResolutionPoleScanBlock,
                real: Tensor,
                imag: Tensor,
                *,
                active_memory_scale: float = memory_scale,
                active_carry_scale: float = carry_scale,
            ) -> tuple[Tensor, Tensor]:
                memory_real, memory_imag = active.collapsed_memory(real, imag)
                return active.post_fusion(
                    active_carry_scale * real + active_memory_scale * memory_real,
                    active_carry_scale * imag + active_memory_scale * memory_imag,
                )

            block.forward = types.MethodType(replacement, block)  # type: ignore[method-assign]
        try:
            yield
        finally:
            for block, original in backups.items():
                block.forward = original  # type: ignore[method-assign]


def _p_value(delta: Tensor) -> float:
    if delta.numel() < 2 or float(delta.std(unbiased=False)) == 0.0:
        return 1.0 if float(delta.abs().max()) == 0.0 else 0.0
    result = stats.ttest_1samp(delta.double().numpy(), popmean=0.0)
    return float(result.pvalue)


def _materialize_cache(source: Path | None, destination: Path) -> dict[str, Any] | None:
    if destination.is_file():
        return {"kind": "existing_destination", "path": str(destination)}
    if source is None or not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".reuse-tmp")
    try:
        os.link(source, temporary)
        reuse_kind = "hardlink"
    except OSError:
        shutil.copy2(source, temporary)
        reuse_kind = "copy"
    temporary.replace(destination)
    return {
        "kind": reuse_kind,
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
    }


class ConditionRunner:
    """Full-validation paired evaluator with deterministic record-level resume."""

    def __init__(
        self,
        *,
        root: Path,
        model: nn.Module,
        loader: Any,
        device: torch.device,
        baseline: PredictionResult,
        baseline_sha256: str,
        checkpoint_sha256: str,
        initial_state_sha256: str,
        bootstrap_repeats: int,
        max_new_conditions: int,
    ) -> None:
        self.root = root
        self.model = model
        self.loader = loader
        self.device = device
        self.baseline = baseline
        self.baseline_sha256 = baseline_sha256
        self.checkpoint_sha256 = checkpoint_sha256
        self.initial_state_sha256 = initial_state_sha256
        self.bootstrap_repeats = bootstrap_repeats
        self.max_new_conditions = max_new_conditions
        self.new_conditions = 0

    def _paths(self, family_id: str, condition_id: str) -> tuple[Path, Path]:
        prediction = self.root / "predictions" / family_id / f"{condition_id}.pt"
        record = self.root / "condition_records" / family_id / f"{condition_id}.json"
        return prediction, record

    def execute(
        self,
        family_id: str,
        condition_id: str,
        context: contextlib.AbstractContextManager[Any],
        *,
        factors: Mapping[str, Any],
        reuse_source: Path | None = None,
        matched_control: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        prediction_path, record_path = self._paths(family_id, condition_id)
        factors_dict = dict(factors)
        factors_sha256 = _json_sha256(factors_dict)
        if prediction_path.is_file() and record_path.is_file():
            record = json.loads(record_path.read_text())
            valid = (
                record.get("checkpoint_sha256") == self.checkpoint_sha256
                and record.get("baseline_sha256") == self.baseline_sha256
                and record.get("factors_sha256") == factors_sha256
                and record.get("per_example_sha256") == sha256_file(prediction_path)
            )
            if valid:
                record["prediction_source"] = "record_cache"
                return cast("dict[str, Any]", record)
            raise RuntimeError(f"stale or incompatible condition cache: {record_path}")

        reuse = _materialize_cache(reuse_source, prediction_path)
        if prediction_path.is_file():
            candidate = load_predictions(prediction_path)
            source = "validated_reuse" if reuse and reuse.get("source_path") else "prediction_cache"
        else:
            if self.max_new_conditions and self.new_conditions >= self.max_new_conditions:
                raise CheckpointYieldError(
                    f"yield after {self.new_conditions} new full-validation conditions"
                )
            before = model_state_sha256(self.model)
            if before != self.initial_state_sha256:
                raise RuntimeError(f"model state is dirty before {family_id}/{condition_id}")
            with context:
                candidate = collect_predictions(self.model, self.loader, self.device)
            after = model_state_sha256(self.model)
            if after != self.initial_state_sha256:
                raise RuntimeError(
                    f"intervention failed exact state restoration: {family_id}/{condition_id}"
                )
            save_predictions(prediction_path, candidate)
            source = "evaluation"
            self.new_conditions += 1

        if not torch.equal(candidate.indices, self.baseline.indices) or not torch.equal(
            candidate.labels, self.baseline.labels
        ):
            raise RuntimeError(f"unpaired prediction cache: {prediction_path}")
        effect_seed = _stable_seed(f"{family_id}/{condition_id}", offset=20_260_904)
        effect = paired_effect(
            self.baseline,
            candidate,
            bootstrap_repeats=self.bootstrap_repeats,
            seed=effect_seed,
        )
        effect["p_value_ce_delta_two_sided"] = _p_value(
            candidate.losses - self.baseline.losses
        )
        record = {
            "condition_id": condition_id,
            "status": "complete",
            "factors": factors_dict,
            "factors_sha256": factors_sha256,
            "matched_control": dict(matched_control or {}),
            "metrics": prediction_metrics(candidate),
            "paired_effect": effect,
            "per_example_path": str(prediction_path.relative_to(self.root)),
            "per_example_sha256": sha256_file(prediction_path),
            "per_example_bytes": prediction_path.stat().st_size,
            "checkpoint_sha256": self.checkpoint_sha256,
            "baseline_sha256": self.baseline_sha256,
            "prediction_source": source,
            "reuse": reuse or {},
        }
        atomic_json(record_path, record)
        print(
            json.dumps(
                {
                    "event": "condition_complete",
                    "family": family_id,
                    "condition": condition_id,
                    "accuracy": cast("Mapping[str, Any]", record["metrics"])["accuracy"],
                    "accuracy_delta_pp": effect["accuracy_delta_pp"],
                    "source": source,
                }
            ),
            flush=True,
        )
        return record


def _write_family(root: Path, family: Mapping[str, Any]) -> Path:
    path = root / "families" / f"{family['family_id']}.json"
    atomic_json(path, dict(family))
    return path


def _family(
    family_id: str,
    conditions: Sequence[Mapping[str, Any]],
    *,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    p_values = {
        str(condition["condition_id"]): float(
            cast("Mapping[str, Any]", condition["paired_effect"])[
                "p_value_ce_delta_two_sided"
            ]
        )
        for condition in conditions
        if condition.get("status") == "complete"
    }
    adjusted = bh_fdr(p_values)
    output_conditions: list[dict[str, Any]] = []
    for source in conditions:
        condition = dict(source)
        condition_id = str(condition["condition_id"])
        if condition_id in adjusted:
            effect = dict(cast("Mapping[str, Any]", condition["paired_effect"]))
            effect["q_value_ce_delta_bh"] = adjusted[condition_id]
            condition["paired_effect"] = effect
        output_conditions.append(condition)
    return {
        "family_id": family_id,
        "schema": SCHEMA,
        "status": "complete",
        "required_condition_ids": sorted(str(item["condition_id"]) for item in conditions),
        "observed_condition_ids": sorted(str(item["condition_id"]) for item in conditions),
        "conditions": output_conditions,
        "analysis": dict(analysis),
        "statistics": {
            "per_condition": (
                "full-val paired metrics; class-stratified paired bootstrap; two-sided "
                "one-sample t-test on per-image CE delta; BH within family"
            ),
        },
    }


def _branch_reuse_source(
    source_root: Path,
    memory_scale: float,
    carry_scale: float,
) -> Path | None:
    if memory_scale == 1.0 and carry_scale == 1.0:
        return source_root / "predictions" / "baseline_full_val.pt"
    aliases = {
        (0.0, 1.0): "memory_zero",
        (1.0, 0.0): "carry_zero",
        (0.0, 0.0): "memory_and_carry_zero",
    }
    alias = aliases.get((memory_scale, carry_scale))
    if alias is None:
        return None
    return source_root / "predictions" / "F06_transition_decomposition" / f"{alias}.pt"


def _render_branch_heatmap(root: Path, family: Mapping[str, Any]) -> list[Path]:
    conditions = cast("Sequence[Mapping[str, Any]]", family["conditions"])
    by_pair = {
        (
            float(cast("Mapping[str, Any]", item["factors"])["memory_scale"]),
            float(cast("Mapping[str, Any]", item["factors"])["carry_scale"]),
        ): float(cast("Mapping[str, Any]", item["metrics"])["accuracy"])
        for item in conditions
    }
    matrix = torch.tensor(
        [[100.0 * by_pair[(memory, carry)] for carry in SCALES] for memory in SCALES]
    )
    output_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "P05_memory_carry_scale_heatmap.png"
    pdf = output_dir / "P05_memory_carry_scale_heatmap.pdf"
    sidecar = output_dir / "P05_memory_carry_scale_heatmap.json"

    figure: Figure
    figure, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    image = axis.imshow(matrix.numpy(), origin="lower", cmap="viridis", aspect="equal")
    axis.set_xticks(range(len(SCALES)), [f"{value:g}" for value in SCALES])
    axis.set_yticks(range(len(SCALES)), [f"{value:g}" for value in SCALES])
    axis.set_xlabel("Carry scale")
    axis.set_ylabel("Memory scale")
    axis.set_title("Frozen K128/D2262: memory x carry sweep")
    midpoint = 0.5 * (float(matrix.min()) + float(matrix.max()))
    for row in range(len(SCALES)):
        for column in range(len(SCALES)):
            value = float(matrix[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < midpoint else "black",
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.84)
    colorbar.set_label("Top-1 accuracy (%)")
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    atomic_json(
        sidecar,
        {
            "family_id": BRANCH_FAMILY,
            "memory_scales": list(SCALES),
            "carry_scales": list(SCALES),
            "accuracy_percent": matrix.tolist(),
            "orientation": "rows=memory scale ascending, columns=carry scale ascending",
        },
    )
    return [png, pdf, sidecar]


def _run_branches(
    runner: ConditionRunner,
    model: nn.Module,
    source_root: Path,
) -> tuple[dict[str, Any], list[Path]]:
    conditions: list[dict[str, Any]] = []
    for memory_scale in SCALES:
        for carry_scale in SCALES:
            condition_id = _condition_id(memory_scale, carry_scale)
            conditions.append(
                runner.execute(
                    BRANCH_FAMILY,
                    condition_id,
                    scaled_memory_carry(
                        model,
                        memory_scale=memory_scale,
                        carry_scale=carry_scale,
                    ),
                    factors={
                        "memory_scale": memory_scale,
                        "carry_scale": carry_scale,
                        "scope": "all three coarsening and all eight same-resolution merges",
                    },
                    reuse_source=_branch_reuse_source(
                        source_root,
                        memory_scale,
                        carry_scale,
                    ),
                )
            )

    by_pair = {
        (
            float(cast("Mapping[str, Any]", item["factors"])["memory_scale"]),
            float(cast("Mapping[str, Any]", item["factors"])["carry_scale"]),
        ): 100.0 * float(cast("Mapping[str, Any]", item["metrics"])["accuracy"])
        for item in conditions
    }
    baseline = by_pair[(1.0, 1.0)]
    interaction = {
        _condition_id(memory, carry): (
            by_pair[(memory, carry)]
            - by_pair[(memory, 1.0)]
            - by_pair[(1.0, carry)]
            + baseline
        )
        for memory in SCALES
        for carry in SCALES
    }
    family = _family(
        BRANCH_FAMILY,
        conditions,
        analysis={
            "design": "complete 5x5 factorial grid on frozen checkpoint",
            "sample_contract": "same ordered ImageNet-100 val5000 for every cell",
            "accuracy_percent_by_condition": {
                _condition_id(memory, carry): by_pair[(memory, carry)]
                for memory in SCALES
                for carry in SCALES
            },
            "difference_in_differences_pp": interaction,
            "difference_in_differences_definition": (
                "A(m,c)-A(m,1)-A(1,c)+A(1,1); nonzero values indicate non-additive "
                "frozen-checkpoint branch interaction, not retrained architectural necessity"
            ),
        },
    )
    path = _write_family(runner.root, family)
    return family, [path, *_render_branch_heatmap(runner.root, family)]


def _band_indices(model: nn.Module, kind: str) -> dict[str, dict[str, list[int]]]:
    output = {"low": {}, "mid": {}, "high": {}}
    for name, module in scan_modules(model).items():
        damping_x, damping_y = (
            value.reshape(-1).detach().float().cpu() for value in module._damping_fields()
        )
        if kind == "memory":
            values = -torch.sqrt(damping_x * damping_y)
        elif kind == "frequency":
            values = torch.hypot(
                module.phase_x.detach().float().cpu(),
                module.phase_y.detach().float().cpu(),
            )
        else:
            raise ValueError(kind)
        chunks = torch.tensor_split(torch.argsort(values), 3)
        for label, chunk in zip(("low", "mid", "high"), chunks, strict=True):
            output[label][name] = sorted(map(int, chunk.tolist()))
    return output


def _count_contract(selection: Mapping[str, Sequence[int]]) -> dict[str, int]:
    return {name: len(indices) for name, indices in sorted(selection.items())}


def _contract_id(counts: Mapping[str, int]) -> str:
    uniform_counts = sorted(set(counts.values()))
    prefix = "x".join(map(str, uniform_counts))
    return f"removed_{prefix}_{_json_sha256(dict(sorted(counts.items())))[:12]}"


def _random_designs(
    model: nn.Module,
    target_selections: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    repeats_per_contract: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if repeats_per_contract <= 0:
        raise ValueError("random repeats per contract must be positive")
    scans = scan_modules(model)
    target_to_contract: dict[str, str] = {}
    contracts: dict[str, dict[str, Any]] = {}
    for target_id, selection in sorted(target_selections.items()):
        counts = _count_contract(selection)
        contract_id = _contract_id(counts)
        target_to_contract[target_id] = contract_id
        contract = contracts.setdefault(
            contract_id,
            {
                "contract_id": contract_id,
                "removed_count_by_module": counts,
                "target_condition_ids": [],
                "controls": [],
            },
        )
        if contract["removed_count_by_module"] != counts:
            raise RuntimeError("count-contract hash collision")
        cast("list[str]", contract["target_condition_ids"]).append(target_id)

    for contract_id, contract in sorted(contracts.items()):
        counts = cast("dict[str, int]", contract["removed_count_by_module"])
        seen: set[str] = set()
        controls: list[dict[str, Any]] = []
        for replicate in range(repeats_per_contract):
            seed = BASE_RANDOM_SEED + replicate
            while True:
                selection = random_matched_indices(scans, counts, seed=seed)
                selection_sha256 = _json_sha256(selection)
                if selection_sha256 not in seen:
                    break
                seed += 1_000_003
            seen.add(selection_sha256)
            controls.append(
                {
                    "condition_id": f"{contract_id}__random_{replicate:03d}",
                    "replicate": replicate,
                    "seed": seed,
                    "selection": selection,
                    "selection_sha256": selection_sha256,
                }
            )
        contract["controls"] = controls
        contract["unique_subset_count"] = len(seen)
        contract["design_sha256"] = _json_sha256(controls)
    return contracts, target_to_contract


def _old_random_source(
    source_root: Path,
    contract: Mapping[str, Any],
    replicate: int,
) -> Path | None:
    # The original F04 runner used this exact seed schedule and evaluated the
    # same subset repeatedly for every band sharing a count contract.  Reuse
    # one of those outputs, never all redundant aliases.
    if replicate >= 20:
        return None
    predictions = source_root / "predictions" / "F04_pole_targeted"
    for target_id in cast("Sequence[str]", contract["target_condition_ids"]):
        candidate = predictions / f"{target_id}_random_{replicate:02d}.pt"
        if candidate.is_file():
            return candidate
    return None


def _class_means(values: Tensor, labels: Tensor) -> Tensor:
    return torch.stack(
        [values[labels == label].float().mean() for label in labels.unique(sorted=True)]
    )


def _hierarchical_control_class_ci(
    target_values: Tensor,
    control_values: Tensor,
    labels: Tensor,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap control subsets and classes around the shared-pool contrast."""

    if control_values.ndim != 2 or control_values.shape[1] != target_values.numel():
        raise ValueError("hierarchical controls must have shape [control, example]")
    target_class = _class_means(target_values, labels)
    control_class = torch.stack([_class_means(row, labels) for row in control_values])
    contrast = target_class[None, :] - control_class
    generator = torch.Generator().manual_seed(seed)
    draws: list[Tensor] = []
    control_count, class_count = contrast.shape
    for start in range(0, repeats, 512):
        active = min(512, repeats - start)
        control_indices = torch.randint(
            control_count,
            (active, control_count),
            generator=generator,
        )
        mean_by_class = contrast[control_indices].mean(1)
        class_indices = torch.randint(
            class_count,
            (active, class_count),
            generator=generator,
        )
        draws.append(mean_by_class.gather(1, class_indices).mean(1))
    distribution = torch.cat(draws)
    return float(distribution.quantile(0.025)), float(distribution.quantile(0.975))


def _target_contrast(
    target: PredictionResult,
    controls: Sequence[PredictionResult],
    baseline: PredictionResult,
    *,
    bootstrap_repeats: int,
    hierarchical_repeats: int,
    seed: int,
) -> dict[str, Any]:
    baseline_correct = baseline.predictions.eq(baseline.labels).float()
    target_accuracy_change = target.predictions.eq(target.labels).float() - baseline_correct
    control_accuracy_change = torch.stack(
        [control.predictions.eq(control.labels).float() - baseline_correct for control in controls]
    )
    target_loss_change = target.losses - baseline.losses
    control_loss_change = torch.stack([control.losses - baseline.losses for control in controls])
    paired_accuracy = target_accuracy_change - control_accuracy_change.mean(0)
    paired_loss = target_loss_change - control_loss_change.mean(0)
    paired_accuracy_ci = bootstrap_mean_ci(
        paired_accuracy,
        baseline.labels,
        repeats=bootstrap_repeats,
        seed=seed,
    )
    paired_loss_ci = bootstrap_mean_ci(
        paired_loss,
        baseline.labels,
        repeats=bootstrap_repeats,
        seed=seed + 1,
    )
    hierarchical_accuracy_ci = _hierarchical_control_class_ci(
        target_accuracy_change,
        control_accuracy_change,
        baseline.labels,
        repeats=hierarchical_repeats,
        seed=seed + 2,
    )
    hierarchical_loss_ci = _hierarchical_control_class_ci(
        target_loss_change,
        control_loss_change,
        baseline.labels,
        repeats=hierarchical_repeats,
        seed=seed + 3,
    )
    target_accuracy_delta = 100.0 * float(target_accuracy_change.mean())
    control_accuracy_deltas = 100.0 * control_accuracy_change.mean(1)
    target_loss_delta = float(target_loss_change.mean())
    control_loss_deltas = control_loss_change.mean(1)
    return {
        "target_accuracy_delta_pp": target_accuracy_delta,
        "random_accuracy_delta_pp_mean": float(control_accuracy_deltas.mean()),
        "random_accuracy_delta_pp_std": float(control_accuracy_deltas.std(unbiased=True)),
        "random_accuracy_delta_pp_quantiles": {
            "q025": float(control_accuracy_deltas.quantile(0.025)),
            "q500": float(control_accuracy_deltas.quantile(0.5)),
            "q975": float(control_accuracy_deltas.quantile(0.975)),
        },
        "accuracy_delta_pp_vs_random_mean": 100.0 * float(paired_accuracy.mean()),
        "accuracy_delta_pp_vs_random_mean_paired_stratified_ci95": [
            100.0 * paired_accuracy_ci[0],
            100.0 * paired_accuracy_ci[1],
        ],
        "accuracy_delta_pp_vs_random_mean_hierarchical_control_class_ci95": [
            100.0 * hierarchical_accuracy_ci[0],
            100.0 * hierarchical_accuracy_ci[1],
        ],
        "target_cross_entropy_delta": target_loss_delta,
        "random_cross_entropy_delta_mean": float(control_loss_deltas.mean()),
        "cross_entropy_delta_vs_random_mean": float(paired_loss.mean()),
        "cross_entropy_delta_vs_random_mean_paired_stratified_ci95": list(paired_loss_ci),
        "cross_entropy_delta_vs_random_mean_hierarchical_control_class_ci95": list(
            hierarchical_loss_ci
        ),
        "empirical_p_accuracy_more_harmful_one_sided": (
            1 + int((control_accuracy_deltas <= target_accuracy_delta).sum())
        )
        / (1 + len(controls)),
        "empirical_p_cross_entropy_more_harmful_one_sided": (
            1 + int((control_loss_deltas >= target_loss_delta).sum())
        )
        / (1 + len(controls)),
        "random_control_count": len(controls),
        "paired_bootstrap_repeats": bootstrap_repeats,
        "hierarchical_control_class_bootstrap_repeats": hierarchical_repeats,
        "hierarchical_units": ["matched random subset", "ImageNet-100 class"],
    }


def _run_random(
    args: argparse.Namespace,
    runner: ConditionRunner,
    model: nn.Module,
    source_root: Path,
) -> tuple[dict[str, Any], list[Path]]:
    scans = scan_modules(model)
    bands = {kind: _band_indices(model, kind) for kind in ("memory", "frequency")}
    target_selections = {
        f"{kind}_{band}_off": selection
        for kind, band_map in bands.items()
        for band, selection in band_map.items()
    }
    contracts, target_to_contract = _random_designs(
        model,
        target_selections,
        repeats_per_contract=args.random_repeats_per_contract,
    )
    design_path = args.root / "designs" / "P06_matched_random_subsets.json"
    design_payload = {
        "schema": "lnet.r2k3.followup.random-design.v1",
        "base_seed": BASE_RANDOM_SEED,
        "generator": "python random.Random via random_matched_indices",
        "target_to_contract": target_to_contract,
        "contracts": contracts,
    }
    if design_path.is_file():
        existing = json.loads(design_path.read_text())
        if _json_sha256(existing) != _json_sha256(design_payload):
            raise RuntimeError(
                "existing random-subset design disagrees with current arguments/checkpoint"
            )
    else:
        atomic_json(design_path, design_payload)

    conditions: list[dict[str, Any]] = []
    target_records: dict[str, dict[str, Any]] = {}
    control_records: dict[str, list[dict[str, Any]]] = {}
    for target_id, selection in sorted(target_selections.items()):
        contract_id = target_to_contract[target_id]
        contract = contracts[contract_id]
        counts = _count_contract(selection)
        if counts != contract["removed_count_by_module"]:
            raise RuntimeError(f"target {target_id} does not satisfy {contract_id}")
        control_ids = [
            str(control["condition_id"])
            for control in cast("Sequence[Mapping[str, Any]]", contract["controls"])
        ]
        record = runner.execute(
            RANDOM_FAMILY,
            target_id,
            coefficient_intervention(scans, pole_mask_transform(selection)),
            factors={
                "kind": target_id.split("_", 1)[0],
                "band": target_id.split("_")[1],
                "selection_sha256": _json_sha256(selection),
                "removed_count_by_module": counts,
                "matched_contract_id": contract_id,
            },
            matched_control={
                "shared_pool_id": contract_id,
                "condition_ids": control_ids,
                "matching_keys": ["module", "removed_count"],
                "repeats": len(control_ids),
            },
            reuse_source=(
                source_root / "predictions" / "F04_pole_targeted" / f"{target_id}.pt"
            ),
        )
        conditions.append(record)
        target_records[target_id] = record

    for contract_id, contract in sorted(contracts.items()):
        records: list[dict[str, Any]] = []
        counts = cast("Mapping[str, int]", contract["removed_count_by_module"])
        for control in cast("Sequence[Mapping[str, Any]]", contract["controls"]):
            replicate = int(control["replicate"])
            selection = cast("Mapping[str, Sequence[int]]", control["selection"])
            if _count_contract(selection) != dict(counts):
                raise RuntimeError(f"random subset violates contract: {control['condition_id']}")
            record = runner.execute(
                RANDOM_FAMILY,
                str(control["condition_id"]),
                coefficient_intervention(scans, pole_mask_transform(selection)),
                factors={
                    "control_role": "shared_matched_random_pool",
                    "matched_contract_id": contract_id,
                    "replicate": replicate,
                    "seed": int(control["seed"]),
                    "selection_sha256": str(control["selection_sha256"]),
                    "removed_count_by_module": dict(counts),
                    "design_path": str(design_path.relative_to(args.root)),
                },
                reuse_source=_old_random_source(source_root, contract, replicate),
            )
            records.append(record)
            conditions.append(record)
        control_records[contract_id] = records

    contrasts: list[dict[str, Any]] = []
    for target_id in sorted(target_records):
        contract_id = target_to_contract[target_id]
        target_prediction = load_predictions(
            args.root / str(target_records[target_id]["per_example_path"])
        )
        controls = [
            load_predictions(args.root / str(record["per_example_path"]))
            for record in control_records[contract_id]
        ]
        contrast = _target_contrast(
            target_prediction,
            controls,
            runner.baseline,
            bootstrap_repeats=args.bootstrap_repeats,
            hierarchical_repeats=args.hierarchical_repeats,
            seed=_stable_seed(target_id, offset=20_261_104),
        )
        contrasts.append(
            {
                "target_condition_id": target_id,
                "shared_pool_id": contract_id,
                "count_contract_verified": True,
                **contrast,
            }
        )

    for key, output_name in (
        (
            "empirical_p_accuracy_more_harmful_one_sided",
            "empirical_q_accuracy_more_harmful_bh",
        ),
        (
            "empirical_p_cross_entropy_more_harmful_one_sided",
            "empirical_q_cross_entropy_more_harmful_bh",
        ),
    ):
        adjusted = bh_fdr(
            {
                str(contrast["target_condition_id"]): float(contrast[key])
                for contrast in contrasts
            }
        )
        for contrast in contrasts:
            contrast[output_name] = adjusted[str(contrast["target_condition_id"])]

    family = _family(
        RANDOM_FAMILY,
        conditions,
        analysis={
            "design_path": str(design_path.relative_to(args.root)),
            "design_sha256": sha256_file(design_path),
            "target_to_contract": target_to_contract,
            "distinct_count_contracts": len(contracts),
            "unique_random_subsets_total": sum(
                int(contract["unique_subset_count"]) for contract in contracts.values()
            ),
            "controls_are_shared_only_for_exact_count_contracts": True,
            "target_contrasts": contrasts,
            "empirical_test": (
                "plus-one one-sided randomization p; six target hypotheses BH-corrected "
                "together for accuracy and cross-entropy separately"
            ),
            "minimum_attainable_empirical_p": 1.0
            / (1 + args.random_repeats_per_contract),
            "uncertainty": {
                "paired_stratified": "resamples images within each of 100 classes",
                "hierarchical_control_class": (
                    "resamples matched subsets and classes; complements rather than replaces "
                    "the paired image bootstrap"
                ),
            },
        },
    )
    family_path = _write_family(args.root, family)
    return family, [design_path, family_path]


def _find_baseline(source_root: Path, root: Path) -> Path | None:
    candidates = (
        root / "predictions" / "baseline_full_val.pt",
        source_root / "predictions" / "baseline_full_val.pt",
    )
    return next((path for path in candidates if path.is_file()), None)


def _load_or_evaluate_baseline(
    root: Path,
    source_root: Path,
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> tuple[PredictionResult, Path]:
    destination = root / "predictions" / "baseline_full_val.pt"
    source = _find_baseline(source_root, root)
    _materialize_cache(source, destination)
    if destination.is_file():
        return load_predictions(destination), destination
    result = collect_predictions(model, loader, device)
    save_predictions(destination, result)
    return result, destination


def _manifest(
    args: argparse.Namespace,
    *,
    baseline_path: Path,
    baseline: PredictionResult,
    checkpoint_sha256: str,
    initial_state_sha256: str,
    final_state_sha256: str,
    outputs: Sequence[Path],
) -> dict[str, Any]:
    known_outputs = (
        root_path
        for root_path in (
            args.root / "families" / f"{BRANCH_FAMILY}.json",
            args.root / "figures" / "P05_memory_carry_scale_heatmap.png",
            args.root / "figures" / "P05_memory_carry_scale_heatmap.pdf",
            args.root / "figures" / "P05_memory_carry_scale_heatmap.json",
            args.root / "designs" / "P06_matched_random_subsets.json",
            args.root / "families" / f"{RANDOM_FAMILY}.json",
            *outputs,
        )
        if root_path.is_file()
    )
    unique_outputs = {str(path.resolve()): path for path in known_outputs}
    return {
        "schema": "lnet.r2k3.followup.branch-random.v1",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256_before": checkpoint_sha256,
            "sha256_after": sha256_file(args.checkpoint),
        },
        "baseline": {
            "path": str(baseline_path.relative_to(args.root)),
            "sha256": sha256_file(baseline_path),
            "metrics": prediction_metrics(baseline),
        },
        "evaluation": {
            "split": "val",
            "samples": baseline.count,
            "group": args.group,
            "batch_size": args.batch_size,
            "random_repeats_per_contract": args.random_repeats_per_contract,
            "bootstrap_repeats": args.bootstrap_repeats,
            "hierarchical_repeats": args.hierarchical_repeats,
        },
        "no_training_guard": {
            "optimizer_steps": 0,
            "backward_calls": 0,
            "model_state_sha256_before": initial_state_sha256,
            "model_state_sha256_after": final_state_sha256,
            "state_restored_exactly": final_state_sha256 == initial_state_sha256,
        },
        "outputs": [
            {
                "path": str(path.relative_to(args.root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(unique_outputs.values())
        ],
    }


def main() -> None:
    args = _args()
    args.root.mkdir(parents=True, exist_ok=True)
    source_root = args.source_study_root or args.root
    if args.random_repeats_per_contract <= 0:
        raise ValueError("--random-repeats-per-contract must be positive")
    if min(args.bootstrap_repeats, args.hierarchical_repeats) <= 0:
        raise ValueError("bootstrap repeat counts must be positive")
    device = torch.device(args.device)
    frozen = load_frozen_model(args.checkpoint, device)
    frozen.model.requires_grad_(requires_grad=False)
    dataset = imagefolder(args.data_root, "val")
    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    baseline, baseline_path = _load_or_evaluate_baseline(
        args.root,
        source_root,
        frozen.model,
        loader,
        device,
    )
    if baseline.count != args.expected_val_samples or baseline.count != len(dataset):
        message = f"full-validation contract violated: predictions={baseline.count}, dataset={len(dataset)}, expected={args.expected_val_samples}"
        raise RuntimeError(message)
    baseline_sha256 = sha256_file(baseline_path)
    runner = ConditionRunner(
        root=args.root,
        model=frozen.model,
        loader=loader,
        device=device,
        baseline=baseline,
        baseline_sha256=baseline_sha256,
        checkpoint_sha256=frozen.checkpoint_sha256,
        initial_state_sha256=frozen.initial_state_sha256,
        bootstrap_repeats=args.bootstrap_repeats,
        max_new_conditions=args.max_new_conditions,
    )
    outputs: list[Path] = []
    try:
        if args.group in {"branches", "all"}:
            _branch_family, branch_outputs = _run_branches(
                runner,
                frozen.model,
                source_root,
            )
            outputs.extend(branch_outputs)
        if args.group in {"random", "all"}:
            _random_family, random_outputs = _run_random(
                args,
                runner,
                frozen.model,
                source_root,
            )
            outputs.extend(random_outputs)
    except CheckpointYieldError as error:
        state_after_yield = model_state_sha256(frozen.model)
        if state_after_yield != frozen.initial_state_sha256:
            raise RuntimeError("model state changed before checkpoint yield") from error
        if sha256_file(args.checkpoint) != frozen.checkpoint_sha256:
            raise RuntimeError("checkpoint changed before checkpoint yield") from error
        print(
            json.dumps(
                {
                    "event": "checkpoint_yield",
                    "group": args.group,
                    "new_conditions": runner.new_conditions,
                    "reason": str(error),
                    "state_restored": True,
                }
            ),
            flush=True,
        )
        return

    final_state_sha256 = model_state_sha256(frozen.model)
    if final_state_sha256 != frozen.initial_state_sha256:
        raise RuntimeError("model state was not restored at end of follow-up study")
    if sha256_file(args.checkpoint) != frozen.checkpoint_sha256:
        raise RuntimeError("checkpoint changed during follow-up study")
    manifest_path = args.root / "followup_branch_random_study.json"
    manifest = _manifest(
        args,
        baseline_path=baseline_path,
        baseline=baseline,
        checkpoint_sha256=frozen.checkpoint_sha256,
        initial_state_sha256=frozen.initial_state_sha256,
        final_state_sha256=final_state_sha256,
        outputs=outputs,
    )
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "event": "followup_complete",
                "group": args.group,
                "new_conditions": runner.new_conditions,
                "baseline_accuracy": prediction_metrics(baseline)["accuracy"],
                "manifest": str(manifest_path),
                "state_restored": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
