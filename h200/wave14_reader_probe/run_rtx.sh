#!/usr/bin/env bash
set -euo pipefail
stage=$1
lock=$2
mode=$3
py=$4
case "$stage" in /tmp/va-t58-*) ;; *) exit 64 ;; esac
cd "$stage"
mkdir -p code results
tar -xf - -C code
cd code
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts:$stage/deps"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR="$stage/inductor" TRITON_CACHE_DIR="$stage/triton"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export WANDB_MODE=disabled
results="$stage/results"
(
    flock -w 1800 9
    nvidia-smi >"$results/$mode-device.log"
    if [[ $mode == screen ]]; then
        "$py" h200/wave14_reader_probe/probe.py --out "$results/screen.json" \
            >"$results/screen.log" 2>&1
    else
        "$py" -m pytest -q tests/test_factorized_reader_kernels.py tests/test_p1_nr4.py \
            >"$results/pytest.log" 2>&1
        "$py" h200/liverms_v2_opt/bench_h200.py dynamo --out "$results/dynamo.json" \
            >"$results/dynamo.log" 2>&1
        "$py" h200/wave14_reader_probe/probe.py --reference --out "$results/graphs.json" \
            >"$results/graphs.log" 2>&1
    fi
) 9>"/tmp/$lock"
