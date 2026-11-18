#!/usr/bin/env python3
"""Build an integrated XeLaTeX paper for the original and P0 follow-up studies."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
# ruff: noqa: RUF001
import argparse
import json
from pathlib import Path
from typing import Any, cast

from render_r2k3_checkpoint_study_latex import (
    _compile,
    _longtable,
    _mono,
    _preamble,
    _sha256,
    _table_star,
    _tex,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _condition(family: dict[str, Any], condition_id: str) -> dict[str, Any]:
    return next(
        condition
        for condition in cast("list[dict[str, Any]]", family["conditions"])
        if condition["condition_id"] == condition_id
    )


def _original_family(root: Path, study: dict[str, Any], family_id: str) -> dict[str, Any]:
    relative = cast("dict[str, str]", study["families"])[family_id]
    return _load(root / "original" / relative)


def _graphic_preamble() -> str:
    return _preamble().replace(
        r"\graphicspath{{figures/}}",
        r"\graphicspath{{original/figures/}{followup/figures/}{followup/state_reset/figures/}}",
    )


def _build(root: Path) -> tuple[str, dict[str, Any]]:
    original_study = _load(root / "original/study.json")
    original = {
        family_id: _original_family(root, original_study, family_id)
        for family_id in cast("dict[str, str]", original_study["families"])
    }
    grating = _load(root / "followup/followups/P0_3_actual_input_grating.json")
    rotation = _load(root / "followup/families/P0-4_input_rotation_d4_path_consistency.json")
    branch = _load(root / "followup/families/P05_memory_carry_scale_sweep.json")
    random_family = _load(root / "followup/families/P06_pole_matched_random_expansion.json")
    reset = _load(root / "followup/state_reset/state_reset.json")
    baseline = cast("dict[str, Any]", original_study["baseline"])
    f07 = original["F07_rank_capacity"]
    rank75 = _condition(f07, "postfusion_all_activation_rank_75")["paired_effect"]
    rank50 = _condition(f07, "postfusion_all_activation_rank_50")["paired_effect"]

    reset_by_id = {
        str(row["condition_id"]): row
        for row in cast("list[dict[str, Any]]", reset["conditions"])
    }
    state_rows = [
        (
            _mono(condition_id),
            f"{100*row['metrics']['accuracy']:.2f}",
            f"{row['vs_segmented_full_reference']['accuracy_delta_pp']:.2f}",
            _tex(
                f"[{row['vs_segmented_full_reference']['accuracy_delta_pp_ci95'][0]:.2f}, {row['vs_segmented_full_reference']['accuracy_delta_pp_ci95'][1]:.2f}]"
            ),
        )
        for condition_id, row in reset_by_id.items()
    ]
    state_table = _table_star(
        "실제 direction-aware state-reset horizon curve",
        "tab:state-reset",
        r"Yrrr",
        ("조건", r"정확도 (\%)", r"$\Delta$Acc (pp)", r"95\% CI"),
        state_rows,
        note="Reset 지점에서 position-dependent decay를 0으로 만들어 horizontal/vertical state와 variance를 실제로 초기화했다. reset-full은 production prediction을 재현했다.",
    )

    grating_aggregate = cast("list[dict[str, Any]]", grating["aggregate_descriptive"])
    grating_rows = [
        (
            _tex(row["response_kind"]),
            f"{row['descriptive_frequency_spearman_rho']:.3f}",
            f"{row['median_frequency_error_octaves']:.3f}",
            f"{row['median_orientation_error_degrees']:.2f}",
            f"{100*row['within_22_5_degrees_fraction']:.1f}",
        )
        for row in grating_aggregate
    ]
    grating_modules = cast("list[dict[str, Any]]", grating["module_summaries"])
    frequency_significant = {
        kind: sum(
            row["frequency_permutation_q_bh"] < 0.05
            for row in grating_modules
            if row["response_kind"] == kind
        )
        for kind in ("reader_drive", "scan_state")
    }
    orientation_significant = {
        kind: sum(
            row["orientation_permutation_q_bh"] < 0.05
            for row in grating_modules
            if row["response_kind"] == kind
        )
        for kind in ("reader_drive", "scan_state")
    }
    grating_table = _table_star(
        "실제 입력 grating response 요약",
        "tab:grating",
        r"Yrrrr",
        ("응답", "Frequency $\rho$", "Freq. error (oct.)", "Orientation error", r"Within 22.5$^\circ$ (\%)"),
        grating_rows,
        note=(
            f"BH-significant frequency modules: reader {frequency_significant['reader_drive']}/12, scan {frequency_significant['scan_state']}/12; "
            f"orientation modules: reader {orientation_significant['reader_drive']}/12, scan {orientation_significant['scan_state']}/12."
        ),
    )

    rotation_summary = cast("dict[str, Any]", rotation["summary"])
    rotation_rows = [
        (
            _mono(row["module"]),
            f"{row['expected_direct_complex_cosine']:.3f}",
            str(row["expected_direct_rank_of_24"]),
            f"{row['best_direct_complex_cosine']:.3f}",
            _mono(row["best_direct_permutation"]),
            f"{row['expected_magnitude_cosine']:.3f}",
        )
        for row in cast("list[dict[str, Any]]", rotation["modules"])
    ]
    rotation_table = _table_star(
        r"90$^\circ$ input rotation과 D4 path permutation",
        "tab:rotation",
        r"Yrrrrr",
        ("모듈", "Expected cosine", "Rank/24", "Best cosine", "Best permutation", "Magnitude cosine"),
        rotation_rows,
        note="Geometry-derived expected permutation은 [1,3,0,2]. Best-of-24는 ceiling이며 primary test가 아니다.",
    )

    branch_accuracy = cast("dict[str, float]", branch["analysis"]["accuracy_percent_by_condition"])
    scales = (0.0, 0.25, 0.5, 0.75, 1.0)
    branch_matrix_rows = []
    for memory in scales:
        values = []
        for carry in scales:
            condition_id = f"memory_{memory:.2f}__carry_{carry:.2f}".replace(".", "p")
            values.append(f"{branch_accuracy[condition_id]:.2f}")
        branch_matrix_rows.append((f"{memory:.2f}", *values))
    branch_table = _table_star(
        "Memory$\times$carry 5$\times$5 scale sweep accuracy (\\%)",
        "tab:branch-grid",
        r"rrrrrr",
        ("Memory / Carry", "0", "0.25", "0.50", "0.75", "1.00"),
        branch_matrix_rows,
        note="각 cell은 같은 validation 5,000장의 frozen-checkpoint intervention이다. Scale은 branch merge 직전 적용된다.",
    )

    target_contrasts = cast("list[dict[str, Any]]", random_family["analysis"]["target_contrasts"])
    random_rows = [
        (
            _mono(row["target_condition_id"]),
            f"{row['target_accuracy_delta_pp']:.2f}",
            f"{row['random_accuracy_delta_pp_mean']:.2f}",
            f"{row['accuracy_delta_pp_vs_random_mean']:.2f}",
            f"{row['empirical_p_accuracy_more_harmful_one_sided']:.4f}",
            f"{row['empirical_q_accuracy_more_harmful_bh']:.4f}",
        )
        for row in target_contrasts
    ]
    random_table = _table_star(
        "100회 matched-random pole contrast",
        "tab:random-expanded",
        r"Yrrrrr",
        ("Target", r"Target $\Delta$pp", "Random mean", "Target--random", "$p$", "BH $q$"),
        random_rows,
        note="세 exact count contract마다 100개 unique subset을 사용했다. 최소 attainable p=1/101=0.0099; 여섯 target의 accuracy hypothesis를 함께 BH 보정했다.",
    )

    claim_rows = [
        (
            _tex("실제 memory 거리"),
            _tex("recurrence-off 의존성만 확인"),
            _tex("32-step −0.02pp, 16-step −0.54pp, 8-step −2.22pp, 4-step −6.88pp"),
            r"\good{강화: Level 4}",
        ),
        (
            _tex("ω↔visual frequency"),
            _tex("intrinsic pole-only ρ=0.983"),
            _tex("actual scan pooled ρ=0.103; 2/12 modules significant"),
            r"\warn{약화/null}",
        ),
        (
            _tex("D4 semantic direction"),
            _tex("path identity/order는 중요"),
            _tex("expected rotation permutation best 2/11; median rank 9/24"),
            r"\warn{부분/null}",
        ),
        (
            _tex("Memory/carry 역할"),
            _tex("둘 다 제거 시 붕괴"),
            _tex("5×5 grid에서 carry scale에 강한 비대칭·비선형 민감도"),
            r"\good{상호의존 강화}",
        ),
        (
            _tex("Pole band specialization"),
            _tex("20회 random control에서 null"),
            _tex("100회/contract에서도 모든 accuracy BH q≥0.743"),
            r"\good{Null 강화}",
        ),
        (
            _tex("PostFusion low-rank"),
            _tex(f"75% {rank75['accuracy_delta_pp']:.2f}pp, 50% {rank50['accuracy_delta_pp']:.2f}pp"),
            _tex("이번 P0에서 변경 없음; 실제 factorization은 P1"),
            _tex("강한 checkpoint 증거 유지"),
        ),
    ]
    claim_table = _table_star(
        "후속 실험에 따른 claim--evidence 업데이트",
        "tab:claim-update",
        r"p{0.17\textwidth}p{0.23\textwidth}Yp{0.16\textwidth}",
        ("주장", "기존 증거", "새 증거", "판정"),
        claim_rows,
    )

    preamble = _graphic_preamble()
    main = preamble + rf"""
\title{{\vspace{{-2.0em}}\sffamily\bfseries Vision ALPHABET\\[2pt]
\large Frozen Checkpoint Study와 P0 Mechanism Follow-up 통합 보고서}}
\author{{LNet Mechanistic Analysis}}
\date{{2026-09-04}}
\maketitle
\begin{{abstract}}
기존 K128-P160-160-160-128-D2262 seed501 frozen-checkpoint 분석을 기반으로, 실제 state reset, actual-input grating, input rotation--D4 path consistency, memory$\times$carry 5$\times$5 sweep, 100회 matched-random pole control을 추가 수행했다. Exact reset은 16--32 spatial step에서 거의 모든 정확도를 유지하고 4 step 이하에서 단조롭게 하락해 memory distance를 직접 확인했다. 반면 intrinsic $|\omega|$--frequency 상관은 실제 full-network grating response에서 약해졌고, 예상 D4 rotation permutation도 2/11 module에서만 최적이었다. Pole band specialization은 100회 random control에서도 지지되지 않아 distributed pole usage 해석이 강화됐다. 모든 P0 실험은 동일 epoch-100 checkpoint에서 추가 학습 없이 수행됐다.
\end{{abstract}}
\noindent\textbf{{Keywords:}} exact state reset, grating tuning, D4 path, branch interaction, randomization test, frozen checkpoint

\section{{통합 결론}}
이번 후속 실험은 기존 결과를 단순히 추가하는 것이 아니라 claim의 강도를 재분류한다. 가장 강해진 주장은 실제 memory distance와 distributed pole usage이며, 가장 약해진 주장은 ``각 learned $\omega$가 실제 visual frequency에 직접 대응한다''와 ``D4 path가 정확한 rotation permutation을 따른다''이다. Table~\ref{{tab:claim-update}}에 변화가 요약돼 있다.
{claim_table}
\FloatBarrier

\section{{대상과 실험 계약}}
대상 checkpoint는 seed 501, epoch 100, SHA256 \nolinkurl{{{original_study['provenance']['checkpoint']['sha256']}}}이다. 독립 baseline은 {baseline['correct']}/{baseline['samples']}={100*baseline['accuracy']:.2f}\%다. Original study는 12 families와 6 figures를 포함하며, 이번 follow-up은 P0-2부터 P0-6까지 수행했다. Optimizer step, backward, parameter update는 모두 0이고 각 runner는 checkpoint/model-state hash를 전후 비교했다.

동일 seed의 509/521 checkpoint는 존재하지 않아 P0-1 cross-seed replication은 실행하지 않았다. P1 retrained ablation과 factorized PostFusion, P2 external baseline은 별도의 학습/모델 실행이 필요한 다음 단계다.

\section{{P0-2: 실제 state-reset memory horizon}}
Production-compatible dynamic recurrence에서 scan 방향별 reset boundary의 decay를 0으로 만들어 state와 variance를 실제 초기화했다. Reset-full은 production baseline 88.28\%를 정확히 재현했다. 32-step은 88.26\%, 16-step 87.74\%, 8-step 86.06\%, 4-step 81.40\%, 2-step 68.06\%, 1-step 10.94\%다.

\good{{직관.}} 최소 16 spatial step 정도의 memory가 있으면 대부분의 성능이 유지되지만 4 step 이하에서는 유용한 장거리 정보가 급격히 사라진다. Static half-life cap의 비단조 문제를 해소하고 실제 memory distance를 직접 보여주는 가장 강한 후속 결과다.
{state_table}
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.70\textwidth]{{state_reset_horizon.png}}
\caption{{Exact periodic state-reset horizon curve on the full validation set.}}
\label{{fig:reset}}
\end{{figure*}}
\FloatBarrier

\section{{P0-3: actual-input grating}}
576개 sinusoidal grating(12 frequencies$\times$12 orientations$\times$4 phases)과 5개 control을 ImageNet normalization 후 전체 reader--nonlinearity--scan 경로에 통과시켰다. Pole-only intrinsic $|\omega|$--preferred frequency $\rho=0.983$과 달리 실제 pooled frequency $\rho$는 reader 0.108, scan 0.103이다. BH-significant frequency module은 reader 3/12, scan 2/12뿐이다.

반면 median orientation error는 reader 43.32$^\circ$에서 scan 26.82$^\circ$로 감소하고, significant orientation module은 2/12에서 7/12로 증가했다. 따라서 \warn{{직접 visual-frequency specialization은 약화}}되지만 \good{{scan이 방향 구조를 정리한다는 증거}}는 강화된다.
{grating_table}
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.96\textwidth]{{P0_3_actual_input_grating.png}}
\caption{{Actual-input grating frequency/orientation response after reader and scan.}}
\label{{fig:grating}}
\end{{figure*}}
\FloatBarrier

\section{{P0-4: rotation과 D4 path}}
100개 class-balanced validation image를 90$^\circ$ 회전하고 activation을 공간적으로 원위치 정렬한 뒤 24개 path permutation을 비교했다. Expected permutation [1,3,0,2]의 mean direct cosine은 {rotation_summary['mean_expected_direct_complex_cosine']:.3f}, median rank는 {rotation_summary['median_expected_direct_rank_of_24']}/24이며 best module은 {rotation_summary['modules_expected_is_best']}/11이다. Magnitude cosine은 높지만 identity permutation이 여러 module에서 더 좋다.

\warn{{직관.}} Path index와 downstream weights의 co-adaptation은 강하지만, 네 path가 전 층에서 정확한 rotation-equivariant semantic direction을 담당한다고 말할 수는 없다. Stage3와 extra14.2처럼 late module 일부에서만 예상 mapping이 최적이다.
{rotation_table}
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.88\textwidth]{{FIG_P0-4_rotation_path_consistency.png}}
\caption{{Expected and best-of-24 D4 path permutations under 90$^\circ$ input rotation.}}
\label{{fig:rotation}}
\end{{figure*}}
\FloatBarrier

\section{{P0-5: memory$\times$carry interaction}}
두 branch scale을 각각 0, 0.25, 0.5, 0.75, 1.0으로 조절했다. Carry=1일 때 memory 0.50/0.75는 83.10/87.44\%까지 회복하지만, memory=1이어도 carry=0.75는 8.26\%다. Scale 0.75 이하에서 대부분의 조합이 급격히 붕괴한다.

\warn{{직관.}} 두 branch는 단순 가산 성분이 아니며 특히 carry amplitude에 강하게 co-adapted돼 있다. 이는 상호의존성의 직접 증거지만 ``memory=global, carry=local'' 역할 분담을 증명하지는 않는다. Frozen internal scale intervention의 OOD sensitivity도 포함한다.
{branch_table}
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.66\textwidth]{{P05_memory_carry_scale_heatmap.png}}
\caption{{Memory$\times$carry scale sweep accuracy heatmap.}}
\label{{fig:branch-grid}}
\end{{figure*}}
\FloatBarrier

\section{{P0-6: matched-random 확대}}
실제 제거-count contract 세 종류마다 100개 unique random subset, 총 300개를 평가했다. 여섯 target의 one-sided accuracy randomization test를 함께 BH 보정했으며 모든 $q\geq0.743$이다. Frequency low/mid target은 random mean보다 약 5.92/6.20pp 더 해롭지만 subset-level 분산 안에서 유의하지 않다.

\good{{직관.}} ``특정 pre-defined band가 특별히 중요하다''는 가설은 더 약해졌고, ``많은 pole이 co-adapted representation을 분산해 담당한다''는 null-based story가 더 강해졌다. Frequency-high 제거는 random보다 33.49pp 덜 해로워 오히려 높은-frequency band의 상대적 중복을 시사한다.
{random_table}
\FloatBarrier

\section{{기존 결과와 함께 본 최종 스토리}}
\begin{{enumerate}}
\item \textbf{{Stable complex recurrence는 실제로 사용된다.}} Frozen off intervention과 exact reset이 함께 이를 지지한다. 유효 spatial memory는 16--32 step에서 포화되고 4 step 이하에서 급격히 부족해진다.
\item \textbf{{Complex phase organization은 중요하지만 단순 visual-frequency dictionary는 아니다.}} Channel alignment를 깨는 $\omega$ intervention은 치명적이나 actual grating frequency alignment는 약하다. 반면 orientation alignment는 scan 이후 개선된다.
\item \textbf{{D4 path는 기능적으로 non-exchangeable하지만 완전한 rotation equivariance는 아니다.}} Path order/identity와 learned downstream mapping의 공동 적응이 더 안전한 설명이다.
\item \textbf{{Memory와 carry는 비선형적으로 상호의존한다.}} 특히 carry scale calibration이 중요하다. Local/global 이분법보다 co-adapted branch merge가 현재 증거와 맞는다.
\item \textbf{{Pole usage는 sparse band specialization보다 distributed organization에 가깝다.}} Terminal-Q 123/128 effective poles와 100-control null이 일치한다.
\item \textbf{{가장 실용적인 최적화 후보는 여전히 PostFusion rank다.}} 75\% rank {rank75['accuracy_delta_pp']:.2f}pp, 50\% rank {rank50['accuracy_delta_pp']:.2f}pp이지만 실제 factorized layer와 latency는 P1에서 검증해야 한다.
\end{{enumerate}}

\section{{좋은 결과, 약점, 다음 단계}}
\subsection{{좋은 결과}}
Exact reset은 mechanism-specific Level 4 증거를 제공했다. Actual grating은 frequency null과 orientation positive를 분리했고, 100-control 확대는 distributed pole usage를 강화했다. 이러한 null result는 기존 주장을 약하게 만드는 대신 논문 스토리를 더 안전하게 만든다.

\subsection{{약점}}
Cross-seed replication이 없고, D4 exact equivariance와 direct frequency semantics는 지지되지 않았다. Branch grid는 frozen internal scaling이라 retrained role separation을 대신하지 못한다. 기존 robustness에서도 blur/low-pass/grayscale/vertical transform과 under-confidence가 남는다.

\subsection{{우선순위}}
가장 싼 다음 단계는 동일 architecture seed509/521 checkpoint를 생성해 핵심 7개 intervention만 반복하는 것이다. 그 다음은 no-recurrence/no-memory/no-carry 중 2--3개 retrained ablation, 실제 rank50/75 factorization과 latency 측정이다. External baseline은 ConvNeXt V2 Atto와 TinyViM-S 정도의 최소 비교가 적절하다.

\section{{결론}}
통합 결과는 Vision ALPHABET의 recurrence 사용과 16--32 step memory horizon을 강하게 지지한다. 반면 learned pole을 단순한 visual-frequency channel로 해석하거나 D4 path를 완전한 rotation-equivariant basis로 보는 주장은 지지되지 않는다. Memory/carry는 gain-sensitive하게 상호의존하고 pole usage는 넓게 분산된다. 따라서 현재 가장 강한 논문 기여는 \emph{{exact memory-distance evidence, distributed pole usage, scan-emergent orientation organization, and low-rank PostFusion opportunity}}의 조합이다.

\clearpage
\onecolumn
\appendix
\section{{후속 실험 전체 수치}}
"""

    grating_module_rows = [
        (
            _mono(row["module"]),
            _tex(row["response_kind"]),
            f"{row['frequency_spearman_rho']:.3f}",
            f"{row['frequency_permutation_q_bh']:.4f}",
            f"{row['orientation_pairing_alignment_mean_cos_2delta']:.3f}",
            f"{row['orientation_permutation_q_bh']:.4f}",
            f"{row['median_orientation_error_degrees']:.2f}",
        )
        for row in grating_modules
    ]
    branch_rows = [
        (
            _mono(condition["condition_id"]),
            f"{100*condition['metrics']['accuracy']:.2f}",
            f"{condition['paired_effect']['accuracy_delta_pp']:.2f}",
            f"{branch['analysis']['difference_in_differences_pp'][condition['condition_id']]:.2f}",
        )
        for condition in cast("list[dict[str, Any]]", branch["conditions"])
    ]
    appendices = "".join(
        (
            _longtable(
                "P0-2 exact state-reset conditions",
                "tab:app-reset",
                r"p{0.42\textwidth}rrr",
                ("조건", "정확도", r"$\Delta$Acc", r"95\% CI"),
                state_rows,
            ),
            _longtable(
                "P0-3 module-level actual grating statistics",
                "tab:app-grating",
                r"p{0.25\textwidth}p{0.15\textwidth}rrrrr",
                ("모듈", "응답", "Freq. $\rho$", "Freq. $q$", "Orient. cos", "Orient. $q$", "Error deg."),
                grating_module_rows,
            ),
            _longtable(
                "P0-4 rotation--path module statistics",
                "tab:app-rotation",
                r"p{0.28\textwidth}rrrrp{0.20\textwidth}",
                ("모듈", "Expected cos", "Rank", "Best cos", "Magnitude cos", "Best permutation"),
                [
                    (row[0], row[1], row[2], row[3], row[5], row[4])
                    for row in rotation_rows
                ],
            ),
            _longtable(
                "P0-5 memory--carry full factorial grid",
                "tab:app-branch",
                r"p{0.56\textwidth}rrr",
                ("조건", "정확도", r"$\Delta$Acc", "Difference-in-differences"),
                branch_rows,
            ),
            _longtable(
                "P0-6 target contrasts against 100 matched controls",
                "tab:app-random",
                r"p{0.36\textwidth}rrrrr",
                ("Target", r"Target $\Delta$", "Random mean", "Difference", "$p$", "$q$"),
                random_rows,
            ),
        )
    )
    document = main + appendices + "\n\\end{document}\n"
    inputs = {
        "original_study": root / "original/study.json",
        "grating": root / "followup/followups/P0_3_actual_input_grating.json",
        "rotation": root / "followup/families/P0-4_input_rotation_d4_path_consistency.json",
        "branch": root / "followup/families/P05_memory_carry_scale_sweep.json",
        "state_reset": root / "followup/state_reset/state_reset.json",
        "random": root / "followup/families/P06_pole_matched_random_expansion.json",
    }
    manifest = {
        "schema": "lnet.r2k3.integrated_followup_report.v1",
        "status": "pending_render",
        "checkpoint_sha256": original_study["provenance"]["checkpoint"]["sha256"],
        "baseline_accuracy": baseline["accuracy"],
        "families": {
            "original": 12,
            "followup_complete": ["P0-2", "P0-3", "P0-4", "P0-5", "P0-6"],
            "not_executed": {
                "P0-1": "seed509/521 checkpoints absent",
                "P1": "requires retraining/factorization",
                "P2": "requires external baseline models",
            },
        },
        "inputs": {
            key: {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in inputs.items()
        },
    }
    return document, manifest


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = _args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("output must be inside integrated report root")
    document, manifest = _build(root)
    tex_path = output.with_suffix(".tex")
    tex_path.write_text(document, encoding="utf-8")
    _compile(root, tex_path, output)
    manifest.update(
        {
            "status": "complete",
            "outputs": {
                "tex": {
                    "path": str(tex_path.relative_to(root)),
                    "sha256": _sha256(tex_path),
                    "bytes": tex_path.stat().st_size,
                },
                "pdf": {
                    "path": str(output.relative_to(root)),
                    "sha256": _sha256(output),
                    "bytes": output.stat().st_size,
                },
            },
        }
    )
    manifest_path = root / "integrated_report_manifest.json"
    _atomic_json(manifest_path, manifest)
    print(f"compiled {tex_path} -> {output}")
    print(f"manifest {manifest_path}")


if __name__ == "__main__":
    main()
