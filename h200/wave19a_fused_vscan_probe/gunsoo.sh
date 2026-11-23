#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src:$PWD/scripts"
export TORCHINDUCTOR_CACHE_DIR="$PWD/results/inductor"
export LNET_VSCAN_DECOUPLED=off
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
py="$HOME/miniconda3/envs/alphabet/bin/python"
mkdir -p results
gpu() (
    flock -w 2400 9
    "$py" "$@"
) 9>/tmp/va-gpu.lock

case "$1" in
isolated) shift; gpu h200/wave19a_fused_vscan_probe/isolated.py results/isolated.jsonl "$@" ;;
pytest) LNET_VSCAN_FUSED_OPT=on gpu -m pytest -q tests/test_general_cuda_scans.py tests/test_p1_nr4.py ;;
train-arm|dynamo|infer-fold)
    export LNET_VSCAN_FUSED_OPT="$2"
    args=(--variant 17-p1-nr4-1024d-v3 --batch 128)
    if [[ $1 == dynamo ]]; then args=(); fi
    gpu h200/liverms_v2_opt/bench_h200.py "$1" --out "results/$1-$2.json" "${args[@]}"
    ;;
*) exit 2 ;;
esac
