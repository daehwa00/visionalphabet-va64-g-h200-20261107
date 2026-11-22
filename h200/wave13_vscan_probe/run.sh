#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src:$PWD/scripts"
export WANDB_MODE=disabled OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
out=$(mktemp -d /app/output/va-t51-wave13.XXXXXX)
echo "output=$out"
run_python=python3
for candidate in /app/output/va-h200-{effective1024-r3-combined,r4-marker-fix}-20261119/environment/bin/python; do
    if [[ -x $candidate ]] && "$candidate" -c 'import torch, torchvision, timm'; then
        run_python=$candidate
        break
    fi
done
"$run_python" h200/wave13_vscan_probe/bench.py --out "$out/isolated.jsonl" | tee "$out/isolated.log"
bench=h200/liverms_v2_opt/bench_h200.py
for batch in 1024 128 256 512; do
    for policy in 0 1; do
        LNET_VSCAN_DECOUPLED=$policy "$run_python" "$bench" train-arm \
            --out "$out/p1_b${batch}_${policy}.json" --variant 17-p1-nr4-1024d-v3 \
            --batch "$batch" --profile
    done
done
