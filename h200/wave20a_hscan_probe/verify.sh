#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
py=${PYTHON:-"$HOME/miniconda3/envs/alphabet/bin/python"}
export PYTHONPATH="$PWD/src:$PWD/scripts"
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PWD/inductor-cache}"
probe=h200/wave20a_hscan_probe
base=h200/liverms_v2_opt/bench_h200.py

locked() (
    flock -w 2400 9
    "$py" "$@"
) 9>/tmp/va-gpu.lock

case "${1:?select bench, parity, pytest, train, dynamo, or fold}" in
    bench) locked "$probe/bench.py" ;;
    parity) locked "$probe/bench.py" --parity ;;
    pytest) locked -m pytest -q tests/test_general_cuda_scans.py tests/test_p1_nr4.py ;;
    train) shift; locked "$probe/train.py" "$@" ;;
    dynamo)
        locked "$base" dynamo --out "$PWD/dynamo.json"
        cat "$PWD/dynamo.json"
        ;;
    fold)
        locked "$base" infer-fold --variant 17-p1-nr4-1024d-v3 --batch 2 --out "$PWD/fold.json"
        cat "$PWD/fold.json"
        ;;
    *) exit 2 ;;
esac
