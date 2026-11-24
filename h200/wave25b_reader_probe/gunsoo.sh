#!/usr/bin/env bash
set -eu

cd /tmp/va-t90-gunsoo
py="$HOME/miniconda3/envs/alphabet/bin/python"
probe=h200/wave25b_reader_probe
export PYTHONPATH=src:scripts:h200/wave17b_vec_reader_probe
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor"
export LNET_READER_VEC=auto

gpu() {
    (flock -w 2400 9 && "$@") 9>/tmp/va-gpu.lock
}

train() {
    local arm=$1 order=$2
    shift 2
    local telemetry="$order-$arm-telemetry.csv"
    nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw \
        --format=csv -l 1 > "$telemetry" &
    local monitor=$!
    local result=0
    env LNET_READER_W25B="$arm" "$py" "$probe/train.py" train-arm \
        --variant 17-p1-nr4-1024d-v3 --batch 128 --profile \
        --out "$order-$arm.json" "$@" > "$order-$arm.log" 2>&1 || result=$?
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    return "$result"

}

case "${1:?ablate|bench|verify|train-pair}" in
    ablate)
        gpu "$py" "$probe/screen.py" bench --ablate --out isolated-ablation.jsonl
        ;;
    bench)
        gpu "$py" "$probe/screen.py" bench --out isolated.jsonl
        ;;
    verify)
        export LNET_READER_W25B=on
        result=0
        gpu "$py" "$probe/screen.py" parity --out parity.jsonl > parity.log 2>&1 || result=1
        gpu "$py" -m pytest -q tests/test_factorized_reader_kernels.py tests/test_p1_nr4.py \
            tests/test_eight_ablation_model.py tests/test_live_complex_rms.py \
            > cuda-pytest.log 2>&1 || result=1
        gpu "$py" "$probe/train.py" dynamo --out dynamo.json > dynamo.log 2>&1 || result=1
        gpu "$py" "$probe/train.py" infer-fold --variant 17-p1-nr4-1024d-v3 --batch 2 \
            --out fold.json > fold.log 2>&1 || result=1
        exit "$result"
        ;;
    train-pair)
        order=${2:?pair name required}
        first=${3:?first arm required}
        second=${4:?second arm required}
        shift 4
        pair() {
            sleep 25
            train "$first" "$order" "$@"
            sleep 25
            train "$second" "$order" "$@"
        }
        gpu pair "$@"
        ;;
    *) exit 2 ;;
esac
