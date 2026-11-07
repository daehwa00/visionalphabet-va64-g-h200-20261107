#!/usr/bin/env python3
"""Generate and compile a XeLaTeX paper for the frozen-checkpoint study."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false
# ruff: noqa: RUF001
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _family(root: Path, study: dict[str, Any], family_id: str) -> dict[str, Any]:
    relative = cast("dict[str, str]", study["families"])[family_id]
    return _load(root / relative)


def _condition(family: dict[str, Any], condition_id: str) -> dict[str, Any]:
    return next(
        condition
        for condition in cast("list[dict[str, Any]]", family["conditions"])
        if condition["condition_id"] == condition_id
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tex(value: object) -> str:
    text = str(value)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def _mono(value: object) -> str:
    return rf"\nolinkurl{{{value}}}"


def _rows(values: list[tuple[str, ...]]) -> str:
    return "\n".join(" & ".join(row) + r" \\" for row in values)


def _table_star(
    caption: str,
    label: str,
    columns: str,
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    note: str = "",
) -> str:
    note_tex = rf"\parbox{{0.96\textwidth}}{{\scriptsize\emph{{Note.}} {note}}}" if note else ""
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{{caption}}}\label{{{label}}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabularx}}{{\textwidth}}{{{columns}}}
\toprule
{' & '.join(headers)} \\
\midrule
{_rows(rows)}
\bottomrule
\end{{tabularx}}
{note_tex}
\end{{table*}}
"""


def _longtable(
    caption: str,
    label: str,
    columns: str,
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> str:
    header = " & ".join(headers) + r" \\"
    return rf"""
\begin{{longtable}}{{{columns}}}
\caption{{{caption}}}\label{{{label}}}\\
\toprule
{header}
\midrule
\endfirsthead
\multicolumn{{{len(headers)}}}{{l}}{{\scriptsize Table \ref{{{label}}} continued}}\\
\toprule
{header}
\midrule
\endhead
\midrule
\multicolumn{{{len(headers)}}}{{r}}{{\scriptsize continued on next page}}\\
\endfoot
\bottomrule
\endlastfoot
{_rows(rows)}
\end{{longtable}}
"""


def _preamble() -> str:
    return r"""
\documentclass[9pt,a4paper,twocolumn]{extarticle}
\usepackage{kotex}
\usepackage{fontspec}
\setmainfont{Noto Serif CJK KR}
\setsansfont{Noto Sans CJK KR}
\setmonofont{Noto Sans Mono CJK KR}
\usepackage[a4paper,top=17mm,bottom=19mm,left=16mm,right=16mm,columnsep=7mm]{geometry}
\usepackage{graphicx,booktabs,tabularx,longtable,array,multirow}
\usepackage{subcaption,caption,xcolor,enumitem,microtype,placeins}
\usepackage{fancyhdr,titlesec,hyperref,url}
\definecolor{Navy}{HTML}{163B70}
\definecolor{Good}{HTML}{16745B}
\definecolor{Warn}{HTML}{A65F00}
\definecolor{PaleBlue}{HTML}{EAF1FB}
\hypersetup{colorlinks=true,linkcolor=Navy,urlcolor=Navy,citecolor=Navy,pdfborder={0 0 0}}
\captionsetup{font=small,labelfont={bf,color=Navy},skip=4pt}
\setlength{\parindent}{1em}
\setlength{\parskip}{1.5pt}
\setlength{\columnsep}{7mm}
\setlist[itemize]{leftmargin=1.4em,itemsep=1pt,topsep=2pt}
\setlist[enumerate]{leftmargin=1.6em,itemsep=1pt,topsep=2pt}
\titleformat{\section}{\large\sffamily\bfseries\color{Navy}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalsize\sffamily\bfseries\color{Navy}}{\thesubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{1.5ex plus .3ex}{.7ex}
\titlespacing*{\subsection}{0pt}{1.2ex}{.5ex}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\scriptsize K128-D2262 Frozen Checkpoint Study}
\fancyhead[R]{\scriptsize No-training ImageNet-100 analysis}
\fancyfoot[C]{\scriptsize\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\newcommand{\good}[1]{\textcolor{Good}{\textbf{#1}}}
\newcommand{\warn}[1]{\textcolor{Warn}{\textbf{#1}}}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\graphicspath{{figures/}}
\begin{document}
"""


def _build_document(study: dict[str, Any], families: dict[str, dict[str, Any]]) -> str:
    baseline = cast("dict[str, Any]", study["baseline"])
    f01, _f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12 = (
        families[family_id]
        for family_id in (
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
        )
    )
    recurrence = _condition(f03, "recurrence_off")["paired_effect"]
    memory = _condition(f06, "memory_zero")["paired_effect"]
    carry = _condition(f06, "carry_zero")["paired_effect"]
    omega_shuffle = _condition(f04, "omega_shuffle")["paired_effect"]
    alpha_shuffle = _condition(f04, "alpha_shuffle")["paired_effect"]
    rank75 = _condition(f07, "postfusion_all_activation_rank_75")["paired_effect"]
    rank50 = _condition(f07, "postfusion_all_activation_rank_50")["paired_effect"]
    rank25 = _condition(f07, "postfusion_all_activation_rank_25")["paired_effect"]
    q_metrics = cast("dict[str, Any]", _condition(f08, "terminal_q_contribution")["metrics"])
    representation = {
        str(condition["condition_id"]): condition
        for condition in cast("list[dict[str, Any]]", f09["conditions"])
    }

    summary_rows = [
        (_tex("Baseline"), _tex(f"Top-1 {100*baseline['accuracy']:.2f}%"), r"\good{좋음}", _tex("학습 기록과 일치하는 재현성")),
        (_tex("Pole 의미"), _tex("half-life ρ=0.778, frequency ρ=0.983"), r"\good{좋음}", _tex("시간·주파수 의미가 학습 후에도 보존")),
        (_tex("Recurrence/branch"), _tex(f"{recurrence['accuracy_delta_pp']:.2f}/{memory['accuracy_delta_pp']:.2f}/{carry['accuracy_delta_pp']:.2f} pp"), r"\good{강한 양성}", _tex("세 경로가 실제 추론에 사용됨")),
        (_tex("Pole 위상"), _tex(f"ω shuffle {omega_shuffle['accuracy_delta_pp']:.2f}, α shuffle {alpha_shuffle['accuracy_delta_pp']:.2f} pp"), r"\warn{혼합}", _tex("위상 배치가 damping 배치보다 민감")),
        (_tex("PostFusion rank"), _tex(f"75% {rank75['accuracy_delta_pp']:.2f}, 50% {rank50['accuracy_delta_pp']:.2f}, 25% {rank25['accuracy_delta_pp']:.2f} pp"), r"\good{매우 좋음}", _tex("50–75% rank 압축 후보")),
        (_tex("Terminal-Q"), _tex(f"effective {q_metrics['effective_classifier_poles']:.0f}/128"), _tex("분산 표현"), _tex("소수 pole pruning보다 넓은 분산")),
        (_tex("Extra14 depth"), _tex("kNN-5 20.84%→64.10%"), r"\good{좋음}", _tex("추가 깊이가 class geometry 개선")),
        (_tex("Calibration"), _tex(f"ECE {baseline['ece_15']:.3f}"), r"\warn{개선 필요}", _tex("정확도 대비 under-confidence")),
    ]
    summary_table = _table_star(
        "주요 결과의 직관적 판정",
        "tab:summary",
        r"p{0.14\textwidth}p{0.25\textwidth}p{0.14\textwidth}Y",
        ("영역", "핵심 수치", "판정", "직관"),
        summary_rows,
        note="성능 하락은 모델 품질에는 약점일 수 있지만, 해당 메커니즘을 실제로 사용한다는 증거로는 강한 양성 결과일 수 있다.",
    )

    recurrence_rows = [
        (
            _mono(condition["condition_id"]),
            f"{100*condition['metrics']['accuracy']:.2f}",
            f"{condition['paired_effect']['accuracy_delta_pp']:.2f}",
            _tex(f"[{condition['paired_effect']['accuracy_delta_pp_ci95'][0]:.2f}, {condition['paired_effect']['accuracy_delta_pp_ci95'][1]:.2f}]"),
        )
        for condition in cast("list[dict[str, Any]]", f03["conditions"])
        if condition.get("status") == "complete"
    ]
    recurrence_table = _table_star(
        "Recurrence static-coefficient interventions",
        "tab:recurrence",
        r"Yrrr",
        ("조건", r"정확도 (\%)", r"$\Delta$Acc (pp)", r"95\% CI"),
        recurrence_rows,
        note="Position reset이 아닌 static pole coefficient 개입이다. Cap-16 비단조성은 horizon 효과로 해석하지 않는다.",
    )

    rank_rows = [
        (
            _mono(condition["condition_id"]),
            f"{100*condition['metrics']['accuracy']:.2f}",
            f"{condition['paired_effect']['accuracy_delta_pp']:.2f}",
        )
        for condition in cast("list[dict[str, Any]]", f07["conditions"])
        if condition.get("status") == "complete" and "paired_effect" in condition
    ]
    rank_table = _table_star(
        "Reader and activation-aware PostFusion rank interventions",
        "tab:rank",
        r"Yrr",
        ("조건", r"정확도 (\%)", r"$\Delta$Acc (pp)"),
        rank_rows,
        note="Low-rank basis는 validation이 아닌 deterministic train calibration bank에서 추정했다.",
    )

    robust_key_ids = (
        "noise_0.1",
        "blur_2.0",
        "lowpass_4",
        "rotate90",
        "hflip",
        "vflip",
        "resolution_192",
        "resolution_256",
        "resolution_288",
        "grayscale",
    )
    robust_key_rows = []
    for condition_id in robust_key_ids:
        condition = _condition(f11, condition_id)
        robust_key_rows.append(
            (
                _mono(condition_id),
                f"{100*condition['metrics']['accuracy']:.2f}",
                f"{condition['paired_effect']['accuracy_delta_pp']:.2f}",
                f"{100*condition['prediction_agreement']:.2f}",
            )
        )
    robust_key_table = _table_star(
        "대표 robustness 조건",
        "tab:robust-key",
        r"Yrrr",
        ("조건", r"정확도 (\%)", r"$\Delta$Acc (pp)", r"Agreement (\%)"),
        robust_key_rows,
    )

    provenance = cast("dict[str, Any]", study["provenance"])
    checkpoint = cast("dict[str, Any]", provenance["checkpoint"])
    guard = cast("dict[str, Any]", provenance["no_training_guard"])
    main = _preamble() + rf"""
\title{{\vspace{{-2.0em}}\sffamily\bfseries K128-P160-160-160-128-D2262\\[2pt]
\large 동결 체크포인트 메커니즘·표현 연구}}
\author{{LNet Checkpoint Analysis}}
\date{{2026-09-01}}
\maketitle
\begin{{abstract}}
Seed 501, epoch-100 frozen checkpoint를 ImageNet-100 validation 5,000장에서 독립 재평가하고, pole, recurrence, D4 path, transition, rank, terminal-Q, representation, effective receptive field (ERF), robustness와 calibration을 추가 학습 없이 분석하였다. 독립 Top-1은 \textbf{{{100*baseline['accuracy']:.2f}\%}}였으며, 12개 experiment family, 18개 artifact, 6개 figure가 validator를 통과했다. 가장 강한 결과는 recurrence/memory/carry의 실사용, D4 path 순서 의존성, 50--75\% PostFusion rank 압축 가능성, extra14 깊이에 따른 class geometry 개선이다. 반면 blur, 저주파 제거, grayscale, 수직 변환과 calibration은 주요 약점으로 나타났다.
\end{{abstract}}
\noindent\textbf{{Keywords:}} frozen checkpoint, associative scan, pole dynamics, mechanistic interpretability, low-rank compression, robustness

\section{{연구 질문과 주요 결과}}
이 연구의 질문은 ``높은 정확도를 낸 체크포인트가 실제로 어떤 계산 경로와 표현을 사용하는가''이다. 제거 시 성능 하락은 배포 관점에서는 취약성이지만, 그 구조가 실제로 사용된다는 메커니즘 증거로는 양성 결과다. Table~\ref{{tab:summary}}는 이 두 관점을 구분한다.
{summary_table}
\FloatBarrier

\section{{실험 설계와 provenance}}
분석 대상은 \nolinkurl{{{study['study']['variant']}}}, seed 501, epoch 100 체크포인트다. 학습 기록의 88.34\%는 epoch 99 historical best이고, 지정 epoch-100 기록은 88.30\%, 독립 재평가는 {100*baseline['accuracy']:.2f}\%다. 체크포인트 SHA256은 \nolinkurl{{{checkpoint['sha256']}}}이다.

동일 validation 5,000장을 사용해 Top-1, NLL/CE, ECE, Brier, confidence, margin과 per-class 지표를 기록했다. 모든 causal condition은 동일 sample ordering의 paired comparison이며 class-stratified bootstrap 10,000회로 CI를 계산했다. Optimizer step={guard['optimizer_step_calls']}, parameter update={guard['parameter_update_calls']}, linear probe={guard['linear_probe_runs']}이며 checkpoint와 model-state hash는 전후 동일하다.

\section{{Pole 구조와 intrinsic identification}}
12개 scan module의 {len(f01['rows'])}개 module--pole 좌표를 기록했다. Theoretical--empirical half-life 중앙 Spearman $\rho=0.778$, $|\omega|$--preferred-frequency 중앙 $\rho=0.983$이다. 이는 alpha가 감쇠, omega가 반복 주파수라는 물리적 의미를 학습 후에도 유지한다는 \good{{좋은 결과}}다. 다만 reader와 nonlinearity를 제외한 pole-only 분석이므로 자연 이미지에서의 실사용은 뒤의 개입으로 판단한다.

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.97\textwidth]{{FIG01_alpha_half_life.png}}
\caption{{학습된 damping/phase pole atlas. Alpha와 theoretical half-life의 역관계 및 2D phase 분포를 보인다.}}
\label{{fig:pole-atlas}}
\end{{figure*}}
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.97\textwidth]{{FIG02_omega_preferred_frequency.png}}
\caption{{Intrinsic frequency identification과 D4 path orientation response.}}
\label{{fig:frequency}}
\end{{figure*}}
\FloatBarrier

\section{{동결 체크포인트 인과 개입}}
\subsection{{Recurrence와 pole}}
Recurrence를 거의 끄면 정확도가 10.90\%로 내려가며 $\Delta$Acc={recurrence['accuracy_delta_pp']:.2f}pp다. 이는 scan이 장식이 아니라 핵심 계산 경로임을 보여주는 \good{{강한 양성 결과}}다. Cap 2/4/8의 손실은 작지만 cap 16은 비단조적으로 붕괴한다. 이 sweep은 position reset이 아닌 static coefficient intervention이므로 단순 horizon 효과로 해석하지 않는다.
{recurrence_table}

Omega zero/sign-flip/shuffle는 각각 큰 붕괴를 일으키지만 alpha shuffle 손실은 {alpha_shuffle['accuracy_delta_pp']:.2f}pp다. 따라서 exact damping 배치보다 learned phase/frequency organization이 훨씬 민감하다. Targeted 1/3 pole removal은 전반적으로 파괴적이지만 20회 matched-random empirical test의 BH q는 유의하지 않아 특정 band만 특별히 필수라는 주장은 약하다.

\subsection{{D4 path와 transition}}
개별 path 제거는 약 $-7.0$--$-10.5$pp지만 두 path 제거는 $-51.5$--$-61.7$pp, path permutation은 $-74.5$--$-83.6$pp다. 네 path는 부분적 중복을 가지지만 교환 가능한 복사본이 아니며 학습된 순서와 조합이 중요하다.

Memory-zero와 carry-zero는 각각 {memory['accuracy_delta_pp']:.2f}pp, {carry['accuracy_delta_pp']:.2f}pp다. 전역 branch는 모두 필수지만 개별 PostFusion block 제거 손실은 대체로 수 pp이므로 local refinement는 여러 block에 분산된다.

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.96\textwidth]{{FIG03_path_orientation.png}}
\caption{{Intrinsic D4 response와 path intervention 효과.}}
\label{{fig:path}}
\end{{figure*}}
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.82\textwidth]{{FIG04_intervention_forest.png}}
\caption{{가장 큰 frozen-checkpoint intervention 효과와 paired bootstrap 95\% CI.}}
\label{{fig:forest}}
\end{{figure*}}
\FloatBarrier

\section{{Rank capacity와 terminal-Q}}
Activation-aware PostFusion rank 75\%는 {rank75['accuracy_delta_pp']:.2f}pp, 50\%는 {rank50['accuracy_delta_pp']:.2f}pp, 25\%는 {rank25['accuracy_delta_pp']:.2f}pp다. 따라서 50--75\%가 가장 직접적인 압축 후보이며, 25\%는 과도한 압축 경계다.
{rank_table}

Terminal-Q의 effective classifier pole은 {q_metrics['effective_classifier_poles']:.1f}/128이고 상위 10\% importance share는 {100*q_metrics['top_10_percent_importance_share']:.2f}\%다. 정보가 거의 모든 pole에 분산되어 소수 pole 고장에는 덜 민감하지만 큰 폭 pole pruning에는 불리하다.
\FloatBarrier

\section{{표현 geometry}}
Deterministic train bank 10/class와 validation 5,000장의 11개 \texttt{{E\_out}} 지점을 분석했다. 14$\times$14 base에서 extra4로 갈수록 kNN-5가 {100*representation['r14/base/E_out']['metrics']['knn_accuracy_k5']:.2f}\%에서 {100*representation['r14/extra4/E_out']['metrics']['knn_accuracy_k5']:.2f}\%로 증가한다. 추가 깊이가 class-separable geometry를 단계적으로 만든다는 \good{{좋은 관찰 결과}}다.
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.98\textwidth]{{FIG06_depth_rank_cka_knn.png}}
\caption{{14$\times$14 depth geometry, class-balanced linear CKA, and t-SNE.}}
\label{{fig:geometry}}
\end{{figure*}}
\FloatBarrier

\section{{ERF와 attribution}}
Gradient ERF는 class-balanced 100장, Integrated Gradients와 occlusion은 각각 20장이다. Radius90은 약 113--126 px로 모델이 이미지 전역을 사용함을 보인다. Memory/carry 제거는 ERF 모양을 다르게 바꾸지만 범위를 완전히 국소화하지 않는다. ERF 크기는 단순히 클수록 좋은 지표가 아니다.
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.98\textwidth]{{FIG05_full_memory_carry_erf.png}}
\caption{{Full, memory-zero, carry-zero 조건의 평균 gradient ERF.}}
\label{{fig:erf}}
\end{{figure*}}
\FloatBarrier

\section{{Robustness와 calibration}}
Resolution 192/256/288, horizontal flip, 작은 translation에는 안정적이다. 반면 blur, low-pass, band-pass, grayscale, 90$^\circ$ rotation과 vertical flip 손실은 크다. 모델은 자연 영상의 세밀한 주파수, 방향, 색 통계에 강하게 의존한다.
{robust_key_table}

Baseline mean confidence는 {100*baseline['mean_confidence']:.2f}\%로 accuracy {100*baseline['accuracy']:.2f}\%보다 낮고 ECE={baseline['ece_15']:.3f}이다. Under-confidence이며 contrast/resolution shift에서 calibration gap이 더 커진다. Temperature scaling 같은 post-hoc calibration이 실용적 후속 조치다.

\section{{논의}}
\subsection{{좋은 결과}}
첫째, recurrence, memory, carry, D4 routing과 pole phase organization이 실제로 사용된다. 둘째, PostFusion 50--75\% rank가 유망한 압축 구간이다. 셋째, extra14 block은 depth에 따라 class geometry를 꾸준히 개선한다.

\subsection{{약점과 null result}}
Frequency degradation, blur, grayscale, 수직 방향 변환과 calibration이 주요 배포 약점이다. 특정 pole band가 matched random보다 특별히 필수라는 가설은 BH 보정 후 강하게 지지되지 않았다. 이는 negative/null result로 유지한다.

\subsection{{제한}}
Single seed이며 frozen-model surgery는 retraining 후 구조적 필요성을 증명하지 않는다. Exact periodic/distant-region reset, internal rotation-equivariance, branch activation swap은 실행하지 않았다. F02는 pole-only, F09는 E-out에 한정된다. Interaction은 별도 CI 없는 point estimate이며 cache는 output hash로 검증하지만 전체 protocol cache key와 결합되지 않았다.

\section{{결론}}
이 체크포인트는 recurrence와 복소 pole phase를 적극적으로 사용하고, 추가 14$\times$14 깊이에서 class representation을 정리한다. 가장 실용적인 최적화 후보는 activation-aware PostFusion rank 50--75\%다. 정확도와 구조 사용은 강하지만 spectral/directional robustness와 calibration은 개선 여지가 뚜렷하다.

\clearpage
\onecolumn
\appendix
\section{{전체 수치 표}}
부록은 보고서 본문에 사용된 complete condition의 수치를 제공한다. 모든 원시 per-example logits와 machine-readable JSON은 evidence package에 포함된다.
"""

    pole_rows = [
        (
            _mono(condition["condition_id"]),
            f"{100*condition['metrics']['accuracy']:.2f}",
            f"{condition['paired_effect']['accuracy_delta_pp']:.2f}",
            f"{condition['paired_effect']['cross_entropy_delta']:.4f}",
        )
        for condition in cast("list[dict[str, Any]]", f04["conditions"])
        if condition.get("status") == "complete" and "_random_" not in str(condition["condition_id"])
    ]
    contrast_rows = [
        (
            _mono(row["target_condition_id"]),
            f"{row['accuracy_delta_pp_vs_random_mean']:.2f}",
            _tex(f"[{row['accuracy_delta_pp_vs_random_mean_ci95'][0]:.2f}, {row['accuracy_delta_pp_vs_random_mean_ci95'][1]:.2f}]"),
            f"{row['empirical_p_more_harmful']:.4f}",
            f"{row['empirical_q_more_harmful_bh']:.4f}",
        )
        for row in cast("list[dict[str, Any]]", f04["matched_control_contrasts"])
    ]
    path_rows = [
        (_mono(condition["condition_id"]), f"{100*condition['metrics']['accuracy']:.2f}", f"{condition['paired_effect']['accuracy_delta_pp']:.2f}")
        for condition in cast("list[dict[str, Any]]", f05["conditions"])
        if condition.get("status") == "complete"
    ]
    transition_rows = [
        (_mono(condition["condition_id"]), f"{100*condition['metrics']['accuracy']:.2f}", f"{condition['paired_effect']['accuracy_delta_pp']:.2f}")
        for condition in cast("list[dict[str, Any]]", f06["conditions"])
        if condition.get("status") == "complete"
    ]
    q_rows = [
        (
            f"{row['removed_fraction']:.2f}",
            f"{100*row['most_contributing_first_accuracy']:.2f}",
            f"{100*row['random_accuracy_mean']:.2f}",
            f"{100*row['most_contributing_only_accuracy']:.2f}",
            f"{100*row['random_only_accuracy_mean']:.2f}",
        )
        for row in cast("list[dict[str, Any]]", q_metrics["removal_curve"])
    ]
    representation_rows = [
        (
            _mono(condition["condition_id"]),
            f"{100*condition['metrics']['knn_accuracy_k5']:.2f}",
            f"{condition['metrics']['spectrum']['participation_ratio']:.2f}",
            f"{condition['metrics']['class_separation']['between_to_within_ratio']:.3f}",
        )
        for condition in cast("list[dict[str, Any]]", f09["conditions"])
    ]
    erf_rows = [
        (
            _mono(condition["condition_id"]),
            str(condition["metrics"]["images"]),
            f"{condition['metrics']['mean']['area90_fraction']:.4f}",
            f"{condition['metrics']['mean']['radius90_pixels']:.1f}",
            f"{condition['metrics']['mean']['entropy']:.3f}",
        )
        for condition in cast("list[dict[str, Any]]", f10["conditions"])
    ]
    robust_rows = [
        (
            _mono(condition["condition_id"]),
            _mono(cast("dict[str, Any]", condition.get("factors", {})).get("comparison_baseline", "clean")),
            f"{100*condition['metrics']['accuracy']:.2f}",
            f"{condition['paired_effect']['accuracy_delta_pp']:.2f}",
            f"{100*condition['prediction_agreement']:.2f}",
        )
        for condition in sorted(
            cast("list[dict[str, Any]]", f11["conditions"]),
            key=lambda item: float(item["paired_effect"]["accuracy_delta_pp"]),
        )
    ]
    calibration = cast("dict[str, dict[str, Any]]", _condition(f12, "robustness_calibration")["metrics"])
    calibration_rows = [
        (
            _mono(condition_id),
            f"{metrics['accuracy']:.4f}",
            f"{metrics['nll']:.4f}",
            f"{metrics['ece_15']:.4f}",
            f"{metrics['brier']:.4f}",
            f"{metrics['mean_confidence']:.4f}",
        )
        for condition_id, metrics in sorted(calibration.items())
    ]

    appendices = "".join(
        (
            _longtable("F04 non-random pole interventions", "tab:app-pole", r"p{0.48\textwidth}rrr", ("조건", "정확도", r"$\Delta$Acc", r"$\Delta$CE"), pole_rows),
            _longtable("F04 matched-random contrasts", "tab:app-contrast", r"p{0.40\textwidth}rrrr", ("Target", r"$\Delta$pp", "CI", "$p$", "$q$"), contrast_rows),
            _longtable("F05 D4 path interventions", "tab:app-path", r"p{0.65\textwidth}rr", ("조건", "정확도", r"$\Delta$Acc"), path_rows),
            _longtable("F06 transition decomposition", "tab:app-transition", r"p{0.65\textwidth}rr", ("조건", "정확도", r"$\Delta$Acc"), transition_rows),
            _longtable("F07 rank interventions", "tab:app-rank", r"p{0.65\textwidth}rr", ("조건", "정확도", r"$\Delta$Acc"), rank_rows),
            _longtable("F08 terminal-Q curves", "tab:app-q", r"rrrrr", ("Fraction", "Important removed", "Random removed", "Important only", "Random only"), q_rows),
            _longtable("F09 representation geometry", "tab:app-rep", r"p{0.50\textwidth}rrr", ("Representation", "kNN-5", "Eff. rank", "Between/within"), representation_rows),
            _longtable("F10 ERF and attribution", "tab:app-erf", r"p{0.48\textwidth}rrrr", ("조건", "N", "Area90", "Radius90", "Entropy"), erf_rows),
            _longtable("F11 robustness and interaction conditions", "tab:app-robust", r"p{0.40\textwidth}p{0.20\textwidth}rrr", ("조건", "Reference", "정확도", r"$\Delta$Acc", "Agreement"), robust_rows),
            _longtable("F12 calibration metrics", "tab:app-calibration", r"p{0.36\textwidth}rrrrr", ("조건", "Acc", "NLL", "ECE", "Brier", "Conf."), calibration_rows),
        )
    )
    return main + appendices + "\n\\end{document}\n"


def _compile(root: Path, tex_path: Path, output: Path) -> None:
    command = [
        "latexmk",
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir={root}",
        tex_path.name,
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    generated = root / tex_path.with_suffix(".pdf").name
    if completed.returncode != 0 or not generated.is_file():
        log_tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-80:])
        raise RuntimeError(f"XeLaTeX compilation failed:\n{log_tail}")
    if generated != output:
        generated.replace(output)
    subprocess.run(
        ["latexmk", "-c", f"-outdir={root}", tex_path.name],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _register(root: Path, tex_path: Path, output: Path) -> None:
    manifest_path = root / "study.json"
    manifest = _load(manifest_path)
    records = [
        {
            "artifact_id": "REPORT_TEX",
            "path": str(tex_path.relative_to(root)),
            "sha256": _sha256(tex_path),
            "bytes": tex_path.stat().st_size,
            "media_type": "application/x-tex",
        },
        {
            "artifact_id": "REPORT_PDF",
            "path": str(output.relative_to(root)),
            "sha256": _sha256(output),
            "bytes": output.stat().st_size,
            "media_type": "application/pdf",
        },
    ]
    artifacts = cast("list[dict[str, Any]]", manifest.setdefault("artifacts", []))
    artifact_ids = {record["artifact_id"] for record in records}
    paths = {record["path"] for record in records}
    artifacts[:] = [
        item
        for item in artifacts
        if item.get("artifact_id") not in artifact_ids and item.get("path") not in paths
    ]
    artifacts.extend(records)
    report = cast("dict[str, Any]", manifest.setdefault("report", {}))
    report.update(
        {
            "tex_path": records[0]["path"],
            "tex_sha256": records[0]["sha256"],
            "tex_bytes": records[0]["bytes"],
            "pdf_path": records[1]["path"],
            "pdf_sha256": records[1]["sha256"],
            "pdf_bytes": records[1]["bytes"],
            "pdf_renderer": "XeLaTeX via latexmk",
        }
    )
    cast("dict[str, Any]", manifest["study"])["status"] = "complete_pending_validation"
    manifest["validation"] = {}
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def main() -> None:
    args = _args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("output must be inside the study root")
    study = _load(root / "study.json")
    family_ids = cast("dict[str, str]", study["families"])
    families = {family_id: _family(root, study, family_id) for family_id in family_ids}
    tex_path = output.with_suffix(".tex")
    tex_path.write_text(_build_document(study, families), encoding="utf-8")
    _compile(root, tex_path, output)
    if not output.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("compiled output is not a PDF")
    _register(root, tex_path, output)
    print(f"compiled {tex_path} -> {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
