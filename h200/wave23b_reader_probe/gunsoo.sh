#!/usr/bin/env bash
set -euo pipefail

cd /tmp/va-t83-gunsoo
py="$HOME/miniconda3/envs/alphabet/bin/python"
probe=h200/wave23b_reader_probe
train_probe=$probe/train.py
variant=17-p1-nr4-1024d-v3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor"
unset LNET_READER_VEC_MODE
export LNET_READER_VEC=auto
export PYTHONPATH=src:scripts:h200/wave17b_vec_reader_probe

gpu() {
    (flock -w 2400 9 && "$@") 9>/tmp/va-gpu.lock
}
train() {
    local name=$1 source=$2
    shift 2
    gpu env PYTHONPATH="$source/src:$source/scripts" "$py" "$train_probe"         train-arm --variant "$variant" --batch 128 --profile --out "$name.json" "$@"
}

case "${1:?usage: gunsoo.sh ablate|release|verify|train}" in
    ablate|release)
        gpu "$py" "$probe/screen.py" bench --refs refs --out "$1.jsonl" --arms "$1"
        ;;
    verify)
        result=0
        gpu "$py" "$probe/screen.py" parity --refs refs --out parity.jsonl || result=1
        gpu "$py" -m pytest -q tests/test_factorized_reader_kernels.py tests/test_p1_nr4.py             tests/test_eight_ablation_model.py > cuda-pytest.log 2>&1 || result=1
        gpu "$py" "$train_probe" dynamo --out dynamo.json || result=1
        gpu "$py" "$train_probe" infer-fold --variant "$variant" --batch 2 --out fold.json ||
            result=1
        exit "$result"
        ;;
    train)
        train wave20-forward "$PWD/wave20"
        train wave23-forward "$PWD"
        train wave23-reverse "$PWD"
        train wave20-reverse "$PWD/wave20"
        train wave17-control "$PWD/wave17" --no-cudnn-benchmark
        train wave23-control "$PWD" --no-cudnn-benchmark
        ;;
    *) exit 2 ;;
esac
