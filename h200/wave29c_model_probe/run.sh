#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root in the coordinator's existing H200 allocation.
out=${1:?provide an output directory}
phase=${2:-train}
python_bin=${PYTHON:-python}
export PYTHONPATH="src:scripts${PYTHONPATH:+:$PYTHONPATH}"
export LNET_CAPTURE_SAFE_ORTHOGONAL=on
bench=h200/liverms_v2_opt/bench_h200.py
variant=17-p1-nr4-1024d-v3
mkdir -p "$out"

if [[ $phase == verify ]]; then
    "$python_bin" -m pytest -q tests/test_model_w29c.py
    for preset in combined_bwd combined_full; do
        export LNET_MODEL_W29C=$preset
        "$python_bin" "$bench" dynamo --out "$out/$preset-dynamo.json"
        "$python_bin" "$bench" infer-fold --variant "$variant" --batch 2             --out "$out/$preset-fold.json"
    done
else
    export LNET_MODEL_W28C=combined
    for preset in off swiglu_vec2d rmsnorm_bwd rmsnorm_full combined_bwd combined_full; do
        LNET_MODEL_W29C=$preset "$python_bin" "$bench" train-arm             --variant "$variant" --batch 1024 --profile --out "$out/$preset.json"
    done
fi
