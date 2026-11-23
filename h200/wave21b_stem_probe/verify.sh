#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
py=${1:?pass the remote Python executable}
out=${2:?pass an output directory}
mkdir -p -- "$out"
export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor-cache"

locked() (
    flock -w 2400 9
    "$py" "$@"
) 9>/tmp/va-gpu.lock

variant=17-p1-nr4-1024d-v3
for settings in controlled default; do
    for order in forward reverse; do
        arms=(off auto)
        if [[ $order == reverse ]]; then
            arms=(auto off)
        fi
        for arm in "${arms[@]}"; do
            label="$settings-$order-$arm"
            LNET_STEM_OPT=$arm locked h200/wave21b_stem_probe/train.py "$settings" train-arm \
                --variant "$variant" --batch 128 --profile --out "$out/$label.json" \
                > "$out/$label.log" 2>&1 || touch "$out/$label.failed"
        done
    done
done

locked -m pytest -q tests/test_fused_stem_ops.py tests/test_p1_nr4.py tests/test_eight_ablation_model.py \
    > "$out/cuda-tests.log" 2>&1 || touch "$out/cuda-tests.failed"
locked h200/liverms_v2_opt/bench_h200.py dynamo --out "$out/dynamo.json" \
    > "$out/dynamo.log" 2>&1 || touch "$out/dynamo.failed"
locked h200/liverms_v2_opt/bench_h200.py infer-fold --variant "$variant" --batch 2 \
    --out "$out/fold.json" > "$out/fold.log" 2>&1 || touch "$out/fold.failed"
