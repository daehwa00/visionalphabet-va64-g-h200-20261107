#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
suite=h200/liverms_v2_opt
budget=55

if [[ ${1:-} != --within-budget ]]; then
    export RUN_ROOT
    RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/va-h200-liverms-v2.XXXXXX")
    export PERSIST_ROOT
    PERSIST_ROOT=$(mktemp -d /app/output/va-h200-liverms-v2.XXXXXX) || true
    trap 'python3 h200/liverms_v2_opt/export_result.py "$?"' EXIT
    timeout --kill-after=5s "${budget}m" bash "$suite/run_h200_verify.sh" --within-budget "$@"
    exit
fi

export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS
unset LNET_DISABLE_LAUNCH_AUTOTUNE LNET_NR4_FUSED

verify_environment() {
    "$1" -c 'import sys; raise SystemExit(sys.version_info < (3, 13))' &&
        "$1" -c 'import numpy, torch, triton, torchvision; raise SystemExit(not torch.cuda.is_available())'
}

ENV_PYTHON=
for candidate in \
    /app/output/va-h200-effective1024-r3-combined-20261119/environment/bin/python \
    /app/output/va-h200-r4-marker-fix-20261119/environment/bin/python \
    "$(command -v python3)"; do
    if [[ -x $candidate ]] && verify_environment "$candidate"; then
        ENV_PYTHON=$candidate
        break
    fi
done
if [[ -z $ENV_PYTHON ]]; then
    printf '%s\n' '{"status":"started","method":"uv Python 3.13 bootstrap"}' > "$RUN_ROOT/setup.json"
    export UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" UV_CACHE_DIR="$RUN_ROOT/uv-cache"
    python3 -m pip install -q --disable-pip-version-check --target "$RUN_ROOT/bootstrap" uv==0.11.26
    uv_run() { PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv "$@"; }
    uv_run python install 3.13.11
    uv_run venv --python 3.13.11 "$RUN_ROOT/environment"
    ENV_PYTHON="$RUN_ROOT/environment/bin/python"
    uv_run pip install --python "$ENV_PYTHON" \
        'torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=63381a109a569b280ed3319da89d3afe5cf9ab5c879936382a212affb5c90552' \
        'torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=5f7c5e0fa08d2cbee93b6e04bbedd59b5e11462cff6cefd07949217265df2370' \
        triton==3.5.1 numpy==2.4.6 pytest==9.0.3 timm==1.0.26
    verify_environment "$ENV_PYTHON"
    printf '%s\n' '{"status":"ok","method":"uv Python 3.13 bootstrap"}' > "$RUN_ROOT/setup.json"
fi

EXTRA_PYPATH=
if ! "$ENV_PYTHON" -c 'import pytest' 2>/dev/null; then
    "$ENV_PYTHON" -m pip install --disable-pip-version-check --target "$RUN_ROOT/deps" pytest
    EXTRA_PYPATH=$RUN_ROOT/deps
fi
if ! "$ENV_PYTHON" -c 'import timm' 2>/dev/null; then
    # Reuse the selected CUDA torch/torchvision install.
    "$ENV_PYTHON" -m pip install --disable-pip-version-check --target "$RUN_ROOT/deps" --no-deps timm
    EXTRA_PYPATH=$RUN_ROOT/deps
fi
export PYTHONPATH="$PWD/src:$PWD/scripts${EXTRA_PYPATH:+:$EXTRA_PYPATH}"
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/inductor"

run_bench() {
    local phase=$1 out=$2 rc=0
    shift 2
    mkdir -p "$(dirname -- "$out")"
    if ! test -f "$out"; then
        printf '%s\n' '{"status":"started"}' > "$out"
    fi
    "$ENV_PYTHON" "$suite/bench_h200.py" "$phase" --out "$out" "$@" || rc=$?
    if ((rc)); then
        printf '{"status":"process_error","exit_code":%d}\n' "$rc" > "$out"
    fi
}

run_pytest() {
    local name=$1 limit=$2 rc=0 started=$SECONDS
    shift 2
    printf '%s\n' '{"status":"started"}' > "$RUN_ROOT/$name.json"
    timeout --foreground --kill-after=5s "$limit" "$ENV_PYTHON" -m pytest "$@" \
        > "$RUN_ROOT/$name.log" 2>&1 || rc=$?
    run_bench wrap-pytest "$RUN_ROOT/$name.json" --log "$RUN_ROOT/$name.log" \
        --exit-code "$rc" --elapsed "$((SECONDS - started))"
}

variants=(17-p1-nr4-1024d-v3 17-live-complex-rms-s03 17-e2-multiscale-s3s4-live-rms-v2)
names=(p1_nr4 liverms multiscale)
run_bench probe "$RUN_ROOT/probe.json"
if [[ ${2:-} == --probe-only ]]; then
    run_bench kernel-probe "$RUN_ROOT/kernel_probe.json"
elif [[ ${2:-} != --infer-only ]]; then
    run_pytest pytest_full 12m -v
    run_bench dynamo "$RUN_ROOT/dynamo.json"

    for i in "${!variants[@]}"; do
        LNET_NR4_FUSED=0 LNET_GENERAL_SCANS=0 LNET_FUSED_FACTORIZED_READER=0 run_bench train-arm \
            "$RUN_ROOT/arms/va17_${names[i]}_opt_off_b1024.json" --variant "${variants[i]}" --batch 1024
    done
    LNET_NR4_FUSED=0 run_bench train-arm \
        "$RUN_ROOT/arms/va17_${names[0]}_nr4_off_b1024.json" --variant "${variants[0]}" --batch 1024
    for i in "${!variants[@]}"; do
        run_bench train-arm "$RUN_ROOT/arms/va17_${names[i]}_opt_on_b1024.json" \
            --variant "${variants[i]}" --batch 1024 --profile
    done
    for batch in 128 256 512; do
        for i in "${!variants[@]}"; do
            run_bench train-arm "$RUN_ROOT/arms/va17_${names[i]}_opt_on_b$batch.json" \
                --variant "${variants[i]}" --batch "$batch"
        done
    done
fi
if [[ ${2:-} != --probe-only ]]; then
    for batch in 128 1024; do
        for i in "${!variants[@]}"; do
            run_bench infer-fold "$RUN_ROOT/infer/va17_${names[i]}_b$batch.json" \
                --variant "${variants[i]}" --batch "$batch"
        done
    done
fi

# Complete all stages and export their diagnostics even when one stage fails.
"$ENV_PYTHON" - "$RUN_ROOT" <<'PYTHON'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
failures = []
paths = [*root.glob("*.json"), *root.glob("arms/*.json"), *root.glob("infer/*.json")]
for path in sorted(paths):
    row = json.loads(path.read_text())
    passed = row.get("status") == "ok"
    if path.stem == "dynamo" and passed:
        graphs = [pair for modes in row["graphs"].values() for pair in modes.values()]
        passed = graphs == [[1, 0]] * 18
    if path.parent == root / "arms" and passed:
        passed = all(row[key] for key in ("finite_losses", "finite_gradients", "running_ms_finite"))
    if path.parent == root / "infer" and passed:
        passed = (
            row["fp32_head_parity_by_precision"]["highest"]["passed"]
            and row["strict_reload_passed"] and row["folded_train_guard"]
        )
    if not passed:
        failures.append(path.stem)
if failures:
    print("Verification failures:", ", ".join(failures), file=sys.stderr)
    sys.exit(1)
PYTHON
