#!/usr/bin/env bash
# Run from the staging repository. Each process releases the shared GPU lock.
set -uo pipefail
export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor"
mkdir -p results
python_path=${1:-python}

record() {
    local label=$1
    shift
    (flock -w 2400 9 && "$python_path" -u "$@") 9>/tmp/va-gpu.lock \
        >"results/$label.log" 2>&1
    printf '%s\n' "$?" >"results/$label.status"
}

for pair in forward reverse controlled; do
    modes=(off auto)
    [[ $pair != reverse ]] || modes=(auto off)
    export WAVE24C_CUDNN_BENCHMARK=1
    [[ $pair != controlled ]] || export WAVE24C_CUDNN_BENCHMARK=0
    for mode in "${modes[@]}"; do
        export LNET_STEM_CONV_OPT=$mode LNET_FUSED_SWIGLU=$mode LNET_NR4_OPT=$mode
        name="train-$pair-$mode"
        record "$name" h200/wave24c_backbone_probe/training.py train-arm --profile \
            --variant 17-p1-nr4-1024d-v3 --batch 128 --out "results/$name.json"
    done
done
