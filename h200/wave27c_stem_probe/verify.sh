#!/usr/bin/env bash
# Run from the repository tree on gunsoo.
set -uo pipefail
export PYTHONPATH=src:scripts:.
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor"
export LNET_FUSED_STEM_CONV=on
python_path=${1:-python}
results=${2:-results}
mkdir -p "$results"
failed=0

record() {
    local label=$1
    shift
    local status=0
    "$python_path" -u "$@" >"$results/$label.log" 2>&1 || status=$?
    printf '%s\n' "$status" >"$results/$label.status"
    if ((status)); then
        failed=1
    fi
}

verify() {
    record isolated h200/wave27c_stem_probe/isolated.py
    for mode in off stage0_opt stage0_v2 stage01 auto; do
        export LNET_STEM_W27C=$mode
        record "pytest-$mode" -m pytest -q -s tests/test_fused_stem_ops.py \
            tests/test_fused_conv_stem_ln_gelu.py
        if [[ $mode == off || $mode == auto ]]; then
            continue
        fi
        record "dynamo-$mode" h200/wave25c_stem_probe/training.py dynamo \
            --out "$results/dynamo-$mode.json"
        for model in 17-p1-nr4-1024d-v3 17-live-complex-rms-s03 17-e2-multiscale-s3s4-live-rms-v2; do
            record "fold-$mode-$model" h200/wave25c_stem_probe/training.py infer-fold \
                --variant "$model" --batch 2 --out "$results/fold-$mode-$model.json"
        done
    done
}

(
    flock -w 2400 9 || exit
    verify
    exit "$failed"
) 9>/tmp/va-gpu.lock
