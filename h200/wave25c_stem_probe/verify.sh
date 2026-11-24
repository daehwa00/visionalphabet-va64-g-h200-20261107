#!/usr/bin/env bash
# Run inside the gunsoo staging tree.
set -uo pipefail
export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor"
mkdir -p results
python_path=${1:-python}
variant=17-p1-nr4-1024d-v3

with_gpu() (
    flock -w 2400 9 || exit
    "$@"
) 9>/tmp/va-gpu.lock

record() {
    local label=$1
    shift
    if [[ ${WAVE25C_RESUME:-0} == 1 && -f results/$label.status ]] &&
        [[ $(<"results/$label.status") == 0 ]]; then
        return
    fi
    (
        nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv
        "$python_path" -u "$@"
        result=$?
        nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv
        exit "$result"
    ) >"results/$label.log" 2>&1
    printf '%s\n' "$?" >"results/$label.status"
}

train() {
    for pair in "$@"; do
        modes=(off on)
        [[ $pair != reverse ]] || modes=(on off)
        export WAVE25C_CUDNN_BENCHMARK=1
        [[ $pair != controlled* ]] || export WAVE25C_CUDNN_BENCHMARK=0
        for mode in "${modes[@]}"; do
            export LNET_FUSED_STEM_CONV=$mode
            name="train-$pair-$mode"
            record "$name" h200/wave25c_stem_probe/training.py train-arm --profile \
                --variant "$variant" --batch 128 --out "results/$name.json"
        done
    done
}

regression() {
    export LNET_FUSED_STEM_CONV=on
    record pytest -m pytest -q tests/test_p1_nr4.py tests/test_eight_ablation_model.py \
        tests/test_live_complex_rms.py tests/test_fused_conv_stem_ln_gelu.py
    record dynamo h200/wave25c_stem_probe/training.py dynamo --out results/dynamo.json
    for model in "$variant" 17-live-complex-rms-s03 17-e2-multiscale-s3s4-live-rms-v2; do
        record "fold-$model" h200/wave25c_stem_probe/training.py infer-fold \
            --variant "$model" --batch 2 --out "results/fold-$model.json"
    done
}

if [[ ${2:-} != regression ]]; then
    with_gpu train forward reverse
    with_gpu train controlled1 controlled2
fi
if [[ ${2:-} != training ]]; then
    with_gpu regression
fi
