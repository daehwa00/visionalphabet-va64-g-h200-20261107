#!/usr/bin/env bash
set -eu

cd /tmp/va-t87-gunsoo
py="$HOME/miniconda3/envs/alphabet/bin/python"
probe=h200/wave24b_reader_probe
export PYTHONPATH=src:scripts:h200/wave17b_vec_reader_probe
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor"
export LNET_READER_VEC=auto

gpu() {
    (flock -w 2400 9 && "$@") 9>/tmp/va-gpu.lock
}

train_arm() {
    local arm=$1 order=$2
    shift 2
    env LNET_READER_W24B="$arm" "$py" "$probe/train.py" train-arm \
        --variant 17-p1-nr4-1024d-v3 --batch 128 --profile \
        --out "$order-$arm.json" "$@" > "$order-$arm.log" 2>&1
}

train() {
    gpu train_arm "$@"
}

train_paired() {
    train_arm off paired-forward
    train_arm on paired-forward
    train_arm on paired-reversed
    train_arm off paired-reversed
    train_arm off paired-controlled --no-cudnn-benchmark
    train_arm on paired-controlled --no-cudnn-benchmark
}

case "${1:?ablate|bench|verify|train|train-base|train-paired}" in
    ablate)
        gpu "$py" "$probe/screen.py" bench --ablate --out isolated-ablation.jsonl
        ;;
    bench)
        gpu "$py" "$probe/screen.py" bench --out isolated.jsonl
        ;;
    verify)
        export LNET_READER_W24B=on
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
    train-paired)
        gpu train_paired
        ;;
    train-base)
        train off forward
        ;;
    train)
        test -s forward-off.json || train off forward
        train on forward
        train on reversed
        train off reversed
        train off controlled --no-cudnn-benchmark
        train on controlled --no-cudnn-benchmark
        ;;
    *) exit 2 ;;
esac
