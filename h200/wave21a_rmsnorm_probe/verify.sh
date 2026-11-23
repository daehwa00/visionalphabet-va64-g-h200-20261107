#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
py=${PYTHON_BIN:-"$HOME/miniconda3/envs/alphabet/bin/python"}
out=${OUT_DIR:-results}
export PYTHONPATH="$PWD/src:$PWD/scripts"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
mkdir -p -- "$out"
probe=h200/wave21a_rmsnorm_probe/bench.py
baseline=h200/liverms_v2_opt/bench_h200.py
variant=17-p1-nr4-1024d-v3

gpu() {
    (flock -w 2400 9 && "$py" "$@") 9>/tmp/va-gpu.lock
}

case "${1:?expected train, isolated, unit, model, dynamo or fold}" in
    train)
        export LNET_RMSNORM_OPT="${2:?expected arm}"
        gpu "$probe" train --cudnn "${3:?expected cudnn mode}" --profile \
            --out "$out/${4:?expected output stem}.json"
        ;;
    isolated)
        for shard in 0 1 2; do
            gpu "$probe" isolated --shard "$shard" --out "$out/isolated-$shard.jsonl"
        done
        ;;
    unit)
        LNET_RMSNORM_OPT=on gpu -m pytest -q tests/test_fused_rmsnorm_ops.py
        ;;
    model)
        for test in test_p1_nr4 test_live_complex_rms test_terminal_input_rmsnorm test_memory_backbone; do
            LNET_RMSNORM_OPT=on gpu -m pytest -q "tests/$test.py"
        done
        ;;
    dynamo)
        LNET_RMSNORM_OPT=on gpu "$baseline" dynamo --out "$out/dynamo.json"
        ;;
    fold)
        variant=${2:-$variant}
        LNET_RMSNORM_OPT=on gpu "$baseline" infer-fold --batch 2 --variant "$variant" \
            --out "$out/fold-$variant.json"
        ;;
    *)
        exit 2
        ;;
esac
