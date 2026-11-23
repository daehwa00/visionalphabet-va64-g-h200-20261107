#!/usr/bin/env bash
set -euo pipefail

cd /tmp/va-t74-gunsoo
py="$HOME/miniconda3/envs/alphabet/bin/python"
probe=h200/wave20b_reader_probe
baseline="$PWD/wave17"
variant=17-p1-nr4-1024d-v3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor"
export LNET_READER_VEC=auto

gpu() {
    (flock -w 2400 9 && "$@") 9>/tmp/va-gpu.lock
}
train() {
    local name=$1 source=$2
    shift 2
    gpu env PYTHONPATH="$source/src:$source/scripts" "$py" "$probe/train.py" \
        train-arm --variant "$variant" --batch 128 --out "$name.json" "$@"
}
case "${1:?usage: gunsoo.sh train|verify|general}" in
    train)
        train wave17-forward "$baseline"
        train wave20-forward "$PWD"
        train wave20-reverse "$PWD"
        train wave17-reverse "$baseline"
        train wave17-control "$baseline" --no-cudnn-benchmark
        train wave20-control "$PWD" --no-cudnn-benchmark
        ;;
    verify)
        export PYTHONPATH=src:scripts:h200/wave17b_vec_reader_probe
        result=0
        gpu "$py" "$probe/screen.py" parity --refs refs --out parity.jsonl || result=1
        gpu "$py" -m pytest -q tests/test_factorized_reader_kernels.py tests/test_p1_nr4.py \
            > cuda-pytest.log 2>&1 || result=1
        gpu "$py" "$probe/train.py" dynamo --out dynamo.json || result=1
        gpu "$py" "$probe/train.py" infer-fold --variant "$variant" --batch 2 --out fold.json ||
            result=1
        exit "$result"
        ;;
    general)
        export PYTHONPATH=src:scripts:h200/wave17b_vec_reader_probe
        gpu "$py" "$probe/screen.py" bench --refs refs --out general.jsonl --general-only
        ;;
    *) exit 2 ;;
esac
