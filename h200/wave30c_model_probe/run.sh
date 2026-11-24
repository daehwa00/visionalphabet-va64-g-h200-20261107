#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root inside an existing Whole H200 allocation.
out=${1:?provide an output directory}
phase=${2:-train}
python_bin=${PYTHON:-python}
export PYTHONPATH="src:scripts${PYTHONPATH:+:$PYTHONPATH}"
export LNET_CAPTURE_SAFE_ORTHOGONAL=on
bench=h200/liverms_v2_opt/bench_h200.py
variant=17-p1-nr4-1024d-v3
mkdir -p "$out"

case "$phase" in
    verify)
        LNET_MODEL_W28C=off LNET_MODEL_W29C=off LNET_MODEL_W30C=combined_v2 \
            "$python_bin" -m pytest -q tests/test_model_w30c.py
        export LNET_MODEL_W28C=off LNET_MODEL_W29C=off LNET_MODEL_W30C=combined_v2
        "$python_bin" "$bench" dynamo --out "$out/combined_v2-dynamo.json"
        "$python_bin" "$bench" infer-fold --variant "$variant" --batch 2 \
            --out "$out/combined_v2-fold.json"
        ;;
    train)
        LNET_MODEL_W28C=off LNET_MODEL_W29C=off LNET_MODEL_W30C=off \
            "$python_bin" "$bench" train-arm --variant "$variant" --batch 1024 \
            --profile --out "$out/wave26.json"
        for preset in off swiglu_v2 rmsnorm_bwd_v2 input_norm_v2 combined_v2; do
            LNET_MODEL_W28C=combined LNET_MODEL_W29C=combined_bwd LNET_MODEL_W30C=$preset \
                "$python_bin" "$bench" train-arm --variant "$variant" --batch 1024 \
                --profile --out "$out/$preset.json"
        done
        "$python_bin" - "$out" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
presets = ("wave26", "off", "swiglu_v2", "rmsnorm_bwd_v2", "input_norm_v2", "combined_v2")
results = {preset: json.loads((output / f"{preset}.json").read_text()) for preset in presets}
budget = 20
passed = True
for preset in presets[2:]:
    row = results[preset]
    allocated = row["peak_allocated_gib"]
    deltas = [(allocated - results[base]["peak_allocated_gib"]) * 1024 for base in presets[:2]]
    print(f"{preset} peak allocated MiB delta vs Wave 26 / 29C: {deltas[0]:+.3f} / {deltas[1]:+.3f}")
    passed &= max(deltas) <= budget and row["finite_losses"] and row["finite_gradients"]
sys.exit(0 if passed else 1)
PY
        ;;
    *)
        echo "Expected train or verify" >&2
        exit 2
        ;;
esac
