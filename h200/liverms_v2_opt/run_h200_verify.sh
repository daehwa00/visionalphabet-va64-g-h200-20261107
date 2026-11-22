#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
suite=h200/liverms_v2_opt
budget=35

if [[ ${1:-} != --within-budget ]]; then
    export RUN_ROOT
    RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/va-h200-liverms-v2.XXXXXX")
    export PERSIST_ROOT
    PERSIST_ROOT=$(mktemp -d /app/output/va-h200-liverms-v2.XXXXXX) || true
    trap 'python3 h200/liverms_v2_opt/export_result.py "$?"' EXIT
    timeout --kill-after=5s "${budget}m" bash "$suite/run_h200_verify.sh" --within-budget
    exit
fi

export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS
unset LNET_DISABLE_LAUNCH_AUTOTUNE

ENV_PYTHON=
for candidate in \
    /app/output/va-h200-effective1024-r3-combined-20261119/environment/bin/python \
    /app/output/va-h200-r4-marker-fix-20261119/environment/bin/python \
    "$(command -v python3)"; do
    if [[ -x $candidate ]] && "$candidate" -c \
        'import torch, triton, torchvision, numpy; raise SystemExit(not torch.cuda.is_available())'; then
        ENV_PYTHON=$candidate
        break
    fi
done
if [[ -z $ENV_PYTHON ]]; then
    printf '%s\n' '{"status":"error","reason":"No Python environment with CUDA torch, triton, torchvision and numpy"}' > "$RUN_ROOT/setup.json"
    exit 1
fi

EXTRA_PYPATH=
if ! "$ENV_PYTHON" -c 'import pytest'; then
    "$ENV_PYTHON" -m pip install --disable-pip-version-check --target "$RUN_ROOT/deps" pytest
    EXTRA_PYPATH=$RUN_ROOT/deps
fi
if ! "$ENV_PYTHON" -c 'import timm'; then
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

run_bench probe "$RUN_ROOT/probe.json"
run_pytest pytest_targeted 5m -v tests/test_live_complex_rms.py \
    tests/test_calibrated_bn_wrapper.py tests/test_eight_ablation_model.py
run_pytest pytest_full 8m -q
run_bench dynamo "$RUN_ROOT/dynamo.json"

variants=(17-live-complex-rms-s03 17-e2-multiscale-s3s4-live-rms-v2)
names=(liverms multiscale)
for i in 0 1; do
    LNET_GENERAL_SCANS=0 LNET_FUSED_FACTORIZED_READER=0 run_bench train-arm \
        "$RUN_ROOT/arms/va17_${names[i]}_opt_off_b1024.json" --variant "${variants[i]}" --batch 1024
done
for i in 0 1; do
    run_bench train-arm "$RUN_ROOT/arms/va17_${names[i]}_opt_on_b1024.json" \
        --variant "${variants[i]}" --batch 1024 --profile
done
for batch in 128 256 512; do
    for i in 0 1; do
        run_bench train-arm "$RUN_ROOT/arms/va17_${names[i]}_opt_on_b$batch.json" \
            --variant "${variants[i]}" --batch "$batch"
    done
done
for batch in 128 1024; do
    for i in 0 1; do
        run_bench infer-fold "$RUN_ROOT/infer/va17_${names[i]}_b$batch.json" \
            --variant "${variants[i]}" --batch "$batch"
    done
done
