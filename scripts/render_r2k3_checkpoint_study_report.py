#!/usr/bin/env python3
"""Render figures and a Korean evidence report from completed study artifacts."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportMissingImports=false
# pyright: reportPrivateUsage=false
# ruff: noqa: RUF001
import argparse
import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
from r2k3_frozen_study_common import artifact_record, atomic_json, sha256_file
from run_r2k3_frozen_study_observational import _figure_omega

mpl.use("Agg")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _families(root: Path, study: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        family_id: _load(root / path)
        for family_id, path in cast("dict[str, str]", study["families"]).items()
    }


def _upsert_artifact(study: dict[str, Any], record: dict[str, Any]) -> None:
    artifacts = cast("list[dict[str, Any]]", study.setdefault("artifacts", []))
    artifacts[:] = [
        item
        for item in artifacts
        if item.get("artifact_id") != record["artifact_id"]
        and item.get("path") != record["path"]
    ]
    artifacts.append(record)


def _upsert_figure(study: dict[str, Any], record: dict[str, Any]) -> None:
    figures = cast("list[dict[str, Any]]", study.setdefault("figures", []))
    figures[:] = [item for item in figures if item.get("figure_id") != record["figure_id"]]
    figures.append(record)


def _normalize_f02_metric_labels(
    root: Path,
    study: dict[str, Any],
    families: dict[str, dict[str, Any]],
) -> None:
    family = families["F02_synthetic_identification"]
    changed = False
    for module in cast("dict[str, dict[str, Any]]", family["modules"]).values():
        for axis in ("x", "y"):
            old_key = f"alpha_empirical_half_life_spearman_{axis}"
            new_key = f"theoretical_empirical_half_life_spearman_{axis}"
            if old_key in module and new_key not in module:
                module[new_key] = module.pop(old_key)
                changed = True
    if not changed:
        return
    family_path = root / cast("dict[str, str]", study["families"])[
        "F02_synthetic_identification"
    ]
    atomic_json(family_path, family)
    _upsert_artifact(
        study,
        artifact_record(
            root,
            "F02_synthetic_identification",
            family_path,
            "application/json",
        ),
    )


def _refresh_study_source_closure(study: dict[str, Any]) -> None:
    repo = Path(__file__).resolve().parents[1]
    study_sources = (
        "scripts/a2d_r2k3_diagnostics.py",
        "scripts/analyze_a2d_r2k3_capacity_diagnostics.py",
        "scripts/r2k3_frozen_study_common.py",
        "scripts/run_r2k3_frozen_study_observational.py",
        "scripts/run_r2k3_frozen_study_evaluation.py",
        "scripts/run_r2k3_frozen_study_representations.py",
        "scripts/run_r2k3_frozen_study_erf.py",
        "scripts/run_r2k3_frozen_study_robustness.py",
        "scripts/render_r2k3_checkpoint_study_report.py",
        "scripts/render_r2k3_checkpoint_study_latex.py",
        "scripts/validate_r2k3_checkpoint_study.py",
    )
    source = cast("dict[str, Any]", study["provenance"])["source"]
    records = {
        str(record["path"]): record
        for record in cast("list[dict[str, Any]]", source["files"])
        if (repo / str(record["path"])).is_file()
    }
    for relative in study_sources:
        path = repo / relative
        records[relative] = {
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    source["files"] = [records[key] for key in sorted(records)]
    source["dependency_tree_sha256"] = hashlib.sha256(
        json.dumps(source["files"], sort_keys=True).encode()
    ).hexdigest()


def _figure_record(root: Path, figure_id: str, path: Path, sidecar: Path) -> dict[str, Any]:
    image = artifact_record(root, figure_id, path, "image/png")
    metadata = artifact_record(root, figure_id + "_sidecar", sidecar, "application/json")
    return {
        "figure_id": figure_id,
        **{key: image[key] for key in ("path", "sha256", "bytes")},
        "sidecar_path": metadata["path"],
        "sidecar_sha256": metadata["sha256"],
        "sidecar_bytes": metadata["bytes"],
    }


def _path_figure(root: Path, families: dict[str, dict[str, Any]]) -> tuple[Path, Path]:
    synthetic = families["F02_synthetic_identification"]
    paths = families["F05_d4_paths"]
    heatmap = torch.tensor(synthetic["path_orientation_mean_response"])
    path_conditions = [
        condition
        for condition in paths["conditions"]
        if condition.get("status") == "complete"
        and str(condition["condition_id"]).startswith("path_")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    image = axes[0].imshow(heatmap, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set(xlabel="orientation index", ylabel="D4 path", title="Intrinsic response")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    labels = [condition["condition_id"] for condition in path_conditions]
    values = [condition["paired_effect"]["accuracy_delta_pp"] for condition in path_conditions]
    axes[1].barh(range(len(labels)), values)
    axes[1].set_yticks(range(len(labels)), labels, fontsize=7)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="accuracy delta (percentage points)", title="Frozen path interventions")
    figure.tight_layout()
    path = root / "figures" / "FIG03_path_orientation.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    sidecar = path.with_suffix(".json")
    atomic_json(
        sidecar,
        {
            "figure_id": "FIG03_path_orientation",
            "panels": [
                {"panel_id": "intrinsic", "source": "families/F02_synthetic_identification.json#/path_orientation_mean_response"},
                {"panel_id": "causal", "source": "families/F05_d4_paths.json#/conditions"},
            ],
            "caption": "Intrinsic D4 orientation response and frozen path-intervention effects.",
        },
    )
    return path, sidecar


def _effect_conditions(families: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family_id in ("F03_recurrence_causality", "F04_pole_targeted", "F05_d4_paths", "F06_transition_decomposition", "F07_rank_capacity"):
        for condition in families[family_id]["conditions"]:
            condition_id = str(condition["condition_id"])
            if condition.get("status") != "complete" or "paired_effect" not in condition:
                continue
            if "_random_" in condition_id:
                continue
            records.append({"family_id": family_id, **condition})
    return records


def _forest_figure(root: Path, families: dict[str, dict[str, Any]]) -> tuple[Path, Path]:
    records = sorted(
        _effect_conditions(families),
        key=lambda record: abs(float(record["paired_effect"]["accuracy_delta_pp"])),
        reverse=True,
    )[:24]
    records.reverse()
    labels = [f"{record['family_id']}:{record['condition_id']}" for record in records]
    values = [float(record["paired_effect"]["accuracy_delta_pp"]) for record in records]
    lower = [
        value - float(record["paired_effect"]["accuracy_delta_pp_ci95"][0])
        for value, record in zip(values, records, strict=True)
    ]
    upper = [
        float(record["paired_effect"]["accuracy_delta_pp_ci95"][1]) - value
        for value, record in zip(values, records, strict=True)
    ]
    figure, axis = plt.subplots(figsize=(11, 9))
    axis.errorbar(values, range(len(records)), xerr=[lower, upper], fmt="o", capsize=2)
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("accuracy delta (percentage points), paired bootstrap 95% CI")
    axis.set_title("Largest frozen-checkpoint intervention effects")
    figure.tight_layout()
    path = root / "figures" / "FIG04_intervention_forest.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    sidecar = path.with_suffix(".json")
    atomic_json(
        sidecar,
        {
            "figure_id": "FIG04_intervention_forest",
            "panels": [
                {
                    "panel_id": "forest",
                    "sources": [f"families/{record['family_id']}.json#/conditions" for record in records],
                }
            ],
            "caption": "Largest paired accuracy effects with class-stratified bootstrap intervals.",
        },
    )
    return path, sidecar


def _condition(family: dict[str, Any], condition_id: str) -> dict[str, Any]:
    return next(
        condition for condition in family["conditions"] if condition["condition_id"] == condition_id
    )


def _top_effects(families: dict[str, dict[str, Any]], count: int = 12) -> list[dict[str, Any]]:
    return sorted(
        _effect_conditions(families),
        key=lambda record: abs(float(record["paired_effect"]["accuracy_delta_pp"])),
        reverse=True,
    )[:count]


def _report(study: dict[str, Any], families: dict[str, dict[str, Any]]) -> str:
    baseline = study["baseline"]
    f01 = families["F01_pole_atlas"]
    f02 = families["F02_synthetic_identification"]
    f03 = families["F03_recurrence_causality"]
    f04 = families["F04_pole_targeted"]
    f05 = families["F05_d4_paths"]
    f06 = families["F06_transition_decomposition"]
    f07 = families["F07_rank_capacity"]
    f08 = families["F08_terminal_q"]
    f09 = families["F09_representation_geometry"]
    f10 = families["F10_erf_attribution"]
    f11 = families["F11_robustness_invariance"]
    f12 = families["F12_calibration_error_statistics"]
    omega_rho = statistics.median(
        module["omega_preferred_frequency_spearman"]["rho"]
        for module in f02["modules"].values()
    )
    half_life_rho = statistics.median(
        0.5
        * (
            module[
                "theoretical_empirical_half_life_spearman_x"
                if "theoretical_empirical_half_life_spearman_x" in module
                else "alpha_empirical_half_life_spearman_x"
            ]["rho"]
            + module[
                "theoretical_empirical_half_life_spearman_y"
                if "theoretical_empirical_half_life_spearman_y" in module
                else "alpha_empirical_half_life_spearman_y"
            ]["rho"]
        )
        for module in f02["modules"].values()
    )
    recurrence = _condition(f03, "recurrence_off")["paired_effect"]
    memory_zero = _condition(f06, "memory_zero")["paired_effect"]
    carry_zero = _condition(f06, "carry_zero")["paired_effect"]
    q_metrics = _condition(f08, "terminal_q_contribution")["metrics"]
    alpha_shuffle_effect = _condition(f04, "alpha_shuffle")["paired_effect"]
    omega_shuffle_effect = _condition(f04, "omega_shuffle")["paired_effect"]
    rank75_effect = _condition(f07, "postfusion_all_activation_rank_75")["paired_effect"]
    rank50_effect = _condition(f07, "postfusion_all_activation_rank_50")["paired_effect"]
    rank25_effect = _condition(f07, "postfusion_all_activation_rank_25")["paired_effect"]
    representation_rows = sorted(
        (
            condition["condition_id"],
            condition["metrics"]["knn_accuracy_k5"],
            condition["metrics"]["spectrum"]["participation_ratio"],
        )
        for condition in f09["conditions"]
    )
    erf_rows = {
        condition["condition_id"]: condition["metrics"]["mean"] for condition in f10["conditions"]
    }
    robust = sorted(
        f11["conditions"],
        key=lambda condition: condition["paired_effect"]["accuracy_delta_pp"],
    )
    matched_contrasts = sorted(
        f04["matched_control_contrasts"],
        key=lambda contrast: abs(float(contrast["accuracy_delta_pp_vs_random_mean"])),
        reverse=True,
    )
    robustness_crosses = [
        condition
        for condition in f11["conditions"]
        if str(condition["condition_id"]).startswith("half_life_cap_2_x_")
    ]
    recurrence_rows = [
        condition
        for condition in f03["conditions"]
        if condition.get("status") == "complete"
    ]
    path_rows = [
        condition
        for condition in f05["conditions"]
        if condition.get("status") == "complete"
    ]
    branch_scale_rows = [
        condition
        for condition in f06["conditions"]
        if str(condition["condition_id"]).startswith("postfusion_scale_")
    ]
    postfusion_off_rows = [
        condition
        for condition in f06["conditions"]
        if str(condition["condition_id"]).startswith("postfusion_off_")
    ]
    rank_rows = [
        condition
        for condition in f07["conditions"]
        if condition.get("status") == "complete" and "paired_effect" in condition
    ]
    q_curve = q_metrics["removal_curve"]
    robustness_calibration = _condition(f12, "robustness_calibration")["metrics"]
    worst_calibration = sorted(
        (
            (condition_id, metrics)
            for condition_id, metrics in robustness_calibration.items()
            if not str(condition_id).startswith("half_life_cap_2_x_")
        ),
        key=lambda item: float(item[1]["ece_15"]),
        reverse=True,
    )[:10]
    top_effects = _top_effects(families)
    lines = [
        "# K128-P160-160-160-128-D2262 동결 체크포인트 메커니즘·표현 연구",
        "",
        "## 요약",
        "",
        f"- 분석 checkpoint는 seed 501, epoch 100이며 독립 재평가 Top-1은 **{100*baseline['accuracy']:.2f}%**였다.",
        "- 학습 기록의 **88.34%는 epoch 99의 historical best**이고, 지정 epoch-100 체크포인트의 기록은 **88.30%**다. 본문은 둘을 구분한다.",
        f"- pole-only 전달함수에서 theoretical–empirical half-life 중앙 Spearman ρ는 **{half_life_rho:.3f}**, |ω|–preferred-frequency 중앙 ρ는 **{omega_rho:.3f}**였다.",
        f"- recurrence-off 정확도 변화는 **{recurrence['accuracy_delta_pp']:.2f}pp**, memory-zero는 **{memory_zero['accuracy_delta_pp']:.2f}pp**, carry-zero는 **{carry_zero['accuracy_delta_pp']:.2f}pp**였다.",
        f"- terminal-Q의 effective classifier poles는 **{q_metrics['effective_classifier_poles']:.1f}**개였다.",
        "",
        "## 한눈에 보는 해석과 판정",
        "",
        "| 영역 | 무엇을 말하는가 | 판정 |",
        "|---|---|---|",
        f"| Baseline | 독립 재평가 Top-1 {100*baseline['accuracy']:.2f}%; 학습 기록과 일치 | **좋음** — 체크포인트 재현성은 높음 |",
        f"| Pole 식별 | half-life ρ={half_life_rho:.3f}, frequency ρ={omega_rho:.3f} | **좋음** — 파라미터가 의도한 시간·주파수 의미를 가짐 |",
        f"| Recurrence/branch | recurrence {recurrence['accuracy_delta_pp']:.2f}pp, memory {memory_zero['accuracy_delta_pp']:.2f}pp, carry {carry_zero['accuracy_delta_pp']:.2f}pp | **강한 양성 결과** — 세 경로가 실제 추론에 사용됨 |",
        f"| Pole 위상 | omega shuffle {omega_shuffle_effect['accuracy_delta_pp']:.2f}pp, alpha shuffle {alpha_shuffle_effect['accuracy_delta_pp']:.2f}pp | **혼합** — 위상 배치는 핵심, damping의 정확한 배치는 상대적으로 덜 민감 |",
        f"| PostFusion rank | 75% {rank75_effect['accuracy_delta_pp']:.2f}pp, 50% {rank50_effect['accuracy_delta_pp']:.2f}pp, 25% {rank25_effect['accuracy_delta_pp']:.2f}pp | **매우 좋음** — 50–75% 범위의 압축 여지가 큼 |",
        "| Terminal-Q | effective pole 123/128 | **분산 표현** — 소수 pole에 의존하지 않지만 큰 폭 pruning은 어려움 |",
        "| Extra14 depth | kNN-5 20.84% → 64.10% | **좋음** — 추가 block이 class geometry를 단계적으로 정리 |",
        f"| Calibration | accuracy {100*baseline['accuracy']:.2f}% 대비 mean confidence {100*baseline['mean_confidence']:.2f}%, ECE {baseline['ece_15']:.3f} | **개선 필요** — 정확도보다 confidence가 낮은 under-confidence |",
        "| Robustness | scale·수평 flip·작은 translation에는 안정, blur·저주파 제거·회전·grayscale에는 취약 | **혼합** — 일부 invariance는 좋고 spectral/directional robustness는 약함 |",
        "",
        "여기서 ‘좋음’은 이 체크포인트가 해당 구조를 실제로 사용하거나 압축 가능성이 있다는 뜻이다. 제거 시 성능이 크게 떨어지는 결과는 모델 정확도에는 나쁜 현상이 아니라, 메커니즘 의존성을 확인하는 데는 좋은 양성 결과다.",
        "",
        "## 연구 대상과 provenance — 직접 증거",
        "",
        f"- Variant: `{study['study']['variant']}`",
        f"- Checkpoint SHA256: `{study['provenance']['checkpoint']['sha256']}`",
        f"- Payload: `{study['provenance']['checkpoint']['payload']}`",
        "- Parameters: 4,621,476; K=[128,128,128,128], P=[160,160,160,128], D=[2,2,6,2].",
        f"- Validation: {baseline['correct']}/{baseline['samples']} correct, CE={baseline['cross_entropy_mean']:.4f}, ECE={baseline['ece_15']:.4f}, Brier={baseline['brier']:.4f}.",
        "- optimizer step, parameter update, linear probe는 실행하지 않았다. 모든 개입은 메모리에서 복원됐고 checkpoint 전후 SHA가 같다.",
        "",
        "## F01–F02. Pole atlas와 synthetic identification — 직접 증거",
        "",
        f"12개 scan module, 총 {len(f01['rows'])}개 module-pole 좌표를 기록했다.",
        f"Intrinsic transfer의 module 중앙 theoretical/empirical half-life ρ={half_life_rho:.3f}, |ω|/frequency ρ={omega_rho:.3f}.",
        "이 결과는 구현된 pole 전달함수의 식별 가능성을 확인하지만 자연 이미지 분류에서 실제로 사용한다는 인과 증거는 아니다.",
        "**직관·판정:** alpha는 기억이 얼마나 빨리 사라지는지, omega는 어떤 반복 주파수에 반응하는지를 정한다. 높은 상관은 학습 후에도 이 물리적 의미가 보존됐다는 뜻이므로 좋은 결과지만, 실제 자연 이미지 사용 여부는 뒤의 개입 실험으로 판단해야 한다.",
        "",
        "![alpha half-life](figures/FIG01_alpha_half_life.png)",
        "",
        "![omega frequency](figures/FIG02_omega_preferred_frequency.png)",
        "",
        "## F03–F08. 동결 체크포인트 개입 — 직접 증거",
        "",
        "| family | condition | Δ accuracy (pp) | 95% CI | Δ CE | flips |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in top_effects:
        effect = record["paired_effect"]
        lines.append(
            "".join(
                (
                    f"| {record['family_id']} | {record['condition_id']} | {effect['accuracy_delta_pp']:.2f} | ",
                    f"[{effect['accuracy_delta_pp_ci95'][0]:.2f}, {effect['accuracy_delta_pp_ci95'][1]:.2f}] | ",
                    f"{effect['cross_entropy_delta']:.4f} | {effect['prediction_flips']} |",
                )
            )
        )
    lines.extend(
        [
            "",
            "### 표적 pole 제거 대 matched-random 제거",
            "",
            "| target | target−random mean Δacc (pp) | paired 95% CI | empirical p | BH q |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for contrast in matched_contrasts:
        ci = contrast["accuracy_delta_pp_vs_random_mean_ci95"]
        lines.append(
            "".join(
                (
                    f"| {contrast['target_condition_id']} | ",
                    f"{contrast['accuracy_delta_pp_vs_random_mean']:.2f} | ",
                    f"[{ci[0]:.2f}, {ci[1]:.2f}] | ",
                    f"{contrast['empirical_p_more_harmful']:.4f} | ",
                    f"{contrast['empirical_q_more_harmful_bh']:.4f} |",
                )
            )
        )
    lines.extend(
        [
            "",
            "**직관·판정:** band별 target-minus-random의 image-paired CI는 일부 차이를 보이지만 20회 intervention-repeat empirical p는 BH 보정 후 유의하지 않았다. 따라서 ‘pole 1/3 제거는 위험하다’는 결론은 강하지만 특정 memory/frequency band만 특별히 필수라는 결론은 아직 약한 null 결과다.",
            "",
            "### F03 recurrence 범위",
            "",
            "| condition | accuracy | Δ accuracy (pp) | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in recurrence_rows:
        effect = condition["paired_effect"]
        ci = effect["accuracy_delta_pp_ci95"]
        lines.append(
            "".join(
                (
                    f"| {condition['condition_id']} | ",
                    f"{100*condition['metrics']['accuracy']:.2f}% | ",
                    f"{effect['accuracy_delta_pp']:.2f} | ",
                    f"[{ci[0]:.2f}, {ci[1]:.2f}] |",
                )
            )
        )
    lines.extend(
        [
            "",
            "`effective_half_life_cap_16`의 비단조적 붕괴는 그대로 보고한다. 이 sweep은 position reset이 아닌 static coefficient 개입이므로 horizon 효과로 해석하지 않으며, 비단조성의 정확한 원인은 미확정이다.",
            "**직관·판정:** recurrence를 거의 끄면 정확도가 10.90%까지 내려가므로 scan이 단순 장식이 아니라 핵심 계산 경로라는 강한 양성 결과다. cap 2/4/8의 작은 손실은 유효 정보의 상당 부분이 비교적 짧은 기억에서도 유지됨을 시사하지만, cap 16 이상 현상은 단순 기억 길이로 해석하면 안 된다.",
            "",
            "### F05 D4 path 개입",
            "",
            "| condition | accuracy | Δ accuracy (pp) |",
            "|---|---:|---:|",
        ]
    )
    for condition in path_rows:
        lines.append(
            "".join(
                (
                    f"| {condition['condition_id']} | ",
                    f"{100*condition['metrics']['accuracy']:.2f}% | ",
                    f"{condition['paired_effect']['accuracy_delta_pp']:.2f} |",
                )
            )
        )
    lines.extend(
        [
            "",
            "**직관·판정:** 한 path만 제거하면 다른 세 path가 일부 보완하지만 두 path를 동시에 제거하면 정확도가 크게 무너지고, path 순서를 바꾸면 거의 붕괴한다. 즉 네 path는 부분적으로 중복되면서도 서로 교환 가능한 복사본은 아니며, 학습된 순서와 조합이 중요하다. 구조 사용을 확인하는 좋은 결과다.",
            "",
            "### F06 PostFusion scale과 블록별 제거",
            "",
            "| condition | accuracy | Δ accuracy (pp) |",
            "|---|---:|---:|",
        ]
    )
    for condition in (*branch_scale_rows, *postfusion_off_rows):
        lines.append(
            "".join(
                (
                    f"| {condition['condition_id']} | ",
                    f"{100*condition['metrics']['accuracy']:.2f}% | ",
                    f"{condition['paired_effect']['accuracy_delta_pp']:.2f} |",
                )
            )
        )
    lines.extend(
        [
            "",
            "**직관·판정:** memory와 carry 전체를 끄는 것은 치명적이지만 개별 PostFusion block 제거는 대체로 수 pp 손실이다. 전역적으로 두 branch가 모두 필요하고, local refinement 기여는 여러 block에 분산되어 있다는 뜻이다. scale 0.75–1.25의 작은 손실은 약간의 gain 변화에는 내성이 있음을 보여준다.",
            "",
            "### F07 reader·activation-aware rank",
            "",
            "저랭크 basis는 validation이 아니라 train split의 deterministic class-balanced calibration bank에서 추정했다.",
            "",
            "| condition | accuracy | Δ accuracy (pp) |",
            "|---|---:|---:|",
        ]
    )
    for condition in rank_rows:
        lines.append(
            "".join(
                (
                    f"| {condition['condition_id']} | ",
                    f"{100*condition['metrics']['accuracy']:.2f}% | ",
                    f"{condition['paired_effect']['accuracy_delta_pp']:.2f} |",
                )
            )
        )
    lines.extend(
        [
            "",
            "**직관·판정:** 출력 activation이 실제로 차지하는 부분공간을 기준으로 75% rank를 남겨도 성능이 거의 같고 50%에서도 손실이 0.70pp뿐이다. 가장 직접적인 최적화 후보이며 매우 좋은 결과다. 다만 25%에서 14.68pp가 떨어지므로 무리한 압축은 피해야 한다.",
            "",
            "### F08 terminal-Q necessity·sufficiency",
            "",
            f"Effective classifier poles는 {q_metrics['effective_classifier_poles']:.1f}/{len(q_metrics['pole_importance_rms'])}, 상위 10% importance share는 {100*q_metrics['top_10_percent_importance_share']:.2f}%였다.",
            "",
            "| fraction | most-important removed acc | random removed acc | most-important only acc | random only acc |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in q_curve:
        lines.append(
            "".join(
                (
                    f"| {row['removed_fraction']:.2f} | ",
                    f"{100*row['most_contributing_first_accuracy']:.2f}% | ",
                    f"{100*row['random_accuracy_mean']:.2f}% | ",
                    f"{100*row['most_contributing_only_accuracy']:.2f}% | ",
                    f"{100*row['random_only_accuracy_mean']:.2f}% |",
                )
            )
        )
    lines.extend(
        [
            "",
            "**직관·판정:** classifier 정보가 128개 중 약 123개 pole에 넓게 퍼져 있고 상위 10%도 전체 importance의 14.1%만 차지한다. 특정 소수 pole 고장에는 비교적 안전한 분산 표현이지만, terminal-Q를 몇 개 pole만 남기는 방식의 큰 pruning에는 불리하다.",
            "",
            "![path orientation](figures/FIG03_path_orientation.png)",
            "",
            "![intervention forest](figures/FIG04_intervention_forest.png)",
            "",
            "## F09. 표현 geometry — 직접 증거",
            "",
            "train bank는 class당 10장의 deterministic bicubic evaluation view이며, CKA는 val class당 10장(총 1,000장), retrieval은 class당 1개 query를 사용했다. 학습된 probe는 없다.",
            "수집 범위는 11개 depth 지점의 `E_out`이며, 내부 reader/path/collapse 지점과 clean–corruption CKA는 이번 실행 범위에 포함하지 않았다.",
            "",
            "| representation | kNN-5 | participation rank |",
            "|---|---:|---:|",
        ]
    )
    for name, knn, rank in representation_rows:
        lines.append(f"| {name} | {100*knn:.2f}% | {rank:.1f} |")
    lines.extend(
        [
            "",
            "**직관·판정:** 14×14 base에서 extra4로 갈수록 kNN 정확도와 effective rank가 함께 상승한다. 단순히 feature를 압축하는 것이 아니라 class-separable 방향을 점진적으로 만들어 내는 모습이며, extra14 깊이의 유용성을 지지하는 좋은 관찰 결과다. 다만 E_out 관찰만으로 정확도 원인까지 확정할 수는 없다.",
            "",
            "![depth rank CKA kNN](figures/FIG06_depth_rank_cka_knn.png)",
            "",
            "## F10. ERF와 attribution — 직접 증거",
            "",
            "Gradient ERF는 class-balanced 100장, Integrated Gradients와 occlusion은 각각 20장으로 계산했다.",
            "",
            "| condition | area90 | radius90 px | entropy |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, metrics in erf_rows.items():
        lines.append(
            f"| {name} | {metrics['area90_fraction']:.4f} | {metrics['radius90_pixels']:.1f} | {metrics['entropy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "**직관·판정:** 모든 조건에서 radius90이 약 113–126 px로 넓어 모델이 국소 패치만이 아니라 이미지 전역을 사용한다. memory/carry 제거가 ERF 모양을 바꾸지만 범위를 완전히 국소화하지는 않아, 두 branch가 서로 다른 전역 정보 흐름을 담당한다는 보조 증거다. 정확도처럼 단순히 ‘클수록 좋다’는 지표는 아니다.",
            "",
            "![ERF](figures/FIG05_full_memory_carry_erf.png)",
            "",
            "## F11–F12. Robustness·invariance·calibration — 직접 증거",
            "",
            "| condition | reference | accuracy | Δ accuracy (pp) | agreement |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for condition in robust:
        reference = condition.get("factors", {}).get("comparison_baseline", "clean")
        lines.append(
            "".join(
                (
                    f"| {condition['condition_id']} | {reference} | ",
                    f"{100*condition['metrics']['accuracy']:.2f}% | ",
                    f"{condition['paired_effect']['accuracy_delta_pp']:.2f} | ",
                    f"{100*condition['prediction_agreement']:.2f}% |",
                )
            )
        )
    lines.extend(
        [
            "",
            "### Half-life cap 2 × 대표 교란 상호작용",
            "",
            "아래 interaction은 difference-in-differences 점추정치이며 별도 interaction CI/검정은 계산하지 않았다. `intervention effect`의 paired 통계는 cross 대 해당 corruption 비교다.",
            "",
            "| cross | intervention effect within corruption (pp) | interaction Δacc (pp) |",
            "|---|---:|---:|",
        ]
    )
    for condition in robustness_crosses:
        lines.append(
            "".join(
                (
                    f"| {condition['condition_id']} | ",
                    f"{condition['paired_effect']['accuracy_delta_pp']:.2f} | ",
                    f"{condition['interaction_accuracy_delta_pp']:.2f} |",
                )
            )
        )
    lines.extend(
        [
            "",
            "### Calibration이 가장 나쁜 대표 교란 10개",
            "",
            "| condition | NLL | ECE-15 | Brier | mean confidence |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition_id, metrics in worst_calibration:
        lines.append(
            "".join(
                (
                    f"| {condition_id} | {metrics['nll']:.4f} | ",
                    f"{metrics['ece_15']:.4f} | {metrics['brier']:.4f} | ",
                    f"{metrics['mean_confidence']:.4f} |",
                )
            )
        )
    lines.extend(
        [
            "",
            "**직관·판정:** 크기 변화·수평 flip·작은 translation은 잘 견디므로 scale/일부 위치 변화 일반화는 좋다. 반면 blur·low-pass·band-pass·grayscale·90° 회전과 수직 flip 손실은 크다. 즉 이 모델은 세밀한 주파수와 자연 영상의 방향·색 통계에 강하게 의존하며, 배포 관점에서는 가장 명확한 약점이다. Baseline ECE 0.153과 mean confidence 72.98%는 88.28% 정확도보다 낮아 under-confident calibration도 개선 여지가 있다.",
            "",
            "## 통계 계약",
            "",
            "- 모든 정확도·CE·margin 효과는 동일한 validation 5,000장의 paired comparison이다. 일반 조건의 reference는 clean baseline이고, F11 cross의 reference는 표에 적힌 해당 corruption이다.",
            "- CI는 class-stratified paired image bootstrap 10,000회이며 family 안의 p-value는 Benjamini–Hochberg로 보정했다.",
            "- F04 표적 제거는 module별 제거 개수를 맞춘 random control 20회와 별도 비교했다. image-paired CI와 intervention-repeat empirical p는 서로 다른 불확실성을 나타내므로 함께 보고한다.",
            "- 모든 완료 개입은 per-example logits·label·index artifact를 갖고, 조건 종료 후 model state hash가 초기값과 일치한다.",
        ]
    )
    lines.extend(
        [
            "",
            "## 해석 — 추론",
            "",
            "- α/ω의 intrinsic identification과 표적 개입이 같은 방향으로 정렬될 때만 memory/frequency specialization으로 해석한다.",
            "- memory/carry 제거 효과와 branch ERF 차이는 두 경로의 상보성을 지지할 수 있지만, frozen-model OOD surgery에 대한 민감성도 일부 포함한다.",
            "- 14×14 extra block의 rank·kNN·CKA 변화는 추가 깊이가 표현을 확장하는지 압축하는지 보여주지만, 그 변화 자체가 정확도 원인임을 증명하지 않는다.",
            "",
            "## Null result와 제한",
            "",
            "- 정확한 periodic horizon reset은 production scan이 mode-static compact pole만 받기 때문에 실행하지 못했고, static half-life cap으로 대체했다.",
            "- distant-region state reset도 같은 position-dependent state-control 부재 때문에 실행하지 못했다.",
            "- F05의 내부 D4 path-permutation rotation equivariance는 실행하지 않았으며, F11의 rotate90 결과는 classifier-level invariance만 측정한다.",
            "- F06의 same/different-class branch activation swap은 실행하지 않았다. F09 표현 geometry는 이를 대신하는 인과 실험이 아니다.",
            "- linear probe와 retrained ablation은 ‘추가 학습 없음’ 계약에 따라 제외했다.",
            "- checkpoint intervention은 현재 학습된 모델의 의존성을 보여주지만 해당 구조가 재학습 후에도 필수임을 증명하지 않는다.",
            "- seed 501 한 개의 결과이므로 seed 일반화는 확인되지 않았다.",
            "- F02 transfer response는 reader·nonlinearity를 제외한 pole-only 분석이다.",
            "- F09는 `E_out` depth geometry에 한정되며 reader/directional/collapse 내부 지점과 clean–corruption CKA는 포함하지 않는다.",
            "- prediction cache는 output SHA로 무결성을 검증하지만 checkpoint·transform·CLI 전체를 하나의 생성 cache key로 묶지는 않았다. 모든 cache는 이 연구 전용 root에서 같은 checkpoint와 data manifest로 생성했다.",
            "- 192/256/288 robustness 조건은 원본에서 직접 bicubic resize와 center crop을 적용한 실제 가변 입력 해상도이며, 나머지 조건은 224×224 기준 view다.",
            "",
            "## 결론",
            "",
            "이 보고서의 강한 결론은 관찰값이 아니라 paired intervention, synthetic identification, ERF, representation geometry가 함께 지지하는 범위로 제한한다. 88.34% historical best와 epoch-100 체크포인트 88.30%는 혼동하지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _args()
    study_path = args.root / "study.json"
    study = _load(study_path)
    families = _families(args.root, study)
    _normalize_f02_metric_labels(args.root, study, families)
    _refresh_study_source_closure(study)
    expected = {
        "F01_pole_atlas",
        "F02_synthetic_identification",
        "F03_recurrence_causality",
        "F04_pole_targeted",
        "F05_d4_paths",
        "F06_transition_decomposition",
        "F07_rank_capacity",
        "F08_terminal_q",
        "F09_representation_geometry",
        "F10_erf_attribution",
        "F11_robustness_invariance",
        "F12_calibration_error_statistics",
    }
    missing = expected - set(families)
    if missing:
        raise RuntimeError(f"cannot render report; missing families: {sorted(missing)}")
    figure2, sidecar2 = _figure_omega(args.root, families["F02_synthetic_identification"])
    figure3, sidecar3 = _path_figure(args.root, families)
    figure4, sidecar4 = _forest_figure(args.root, families)
    for figure_id, path, sidecar in (
        ("FIG02_omega_preferred_frequency", figure2, sidecar2),
        ("FIG03_path_orientation", figure3, sidecar3),
        ("FIG04_intervention_forest", figure4, sidecar4),
    ):
        _upsert_figure(study, _figure_record(args.root, figure_id, path, sidecar))
    report_path = args.root / "report.md"
    report_path.write_text(_report(study, families))
    report_record = artifact_record(args.root, "REPORT_KO", report_path, "text/markdown")
    _upsert_artifact(study, report_record)
    study["report"] = {
        "path": report_record["path"],
        "sha256": report_record["sha256"],
        "bytes": report_record["bytes"],
        "language": "ko",
    }
    study["study"]["status"] = "complete_pending_validation"
    study["study"]["completed_utc"] = dt.datetime.now(dt.UTC).isoformat()
    atomic_json(study_path, study)
    print(json.dumps({"event": "report_rendered", "path": str(report_path)}))


if __name__ == "__main__":
    main()
