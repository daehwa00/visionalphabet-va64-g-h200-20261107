#!/usr/bin/env bash
set -euo pipefail

# Start from the candidate checkout under the device allocation's lock.
out=${1:?output directory required}
python_bin=${PYTHON_BIN:-python}
batch=${SCAN_BATCH:-128}
variant=${SPATIAL_VARIANT:-auto}
mkdir -p "$out"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export LNET_VSCAN_DECOUPLED=0
unset LNET_VSCAN_SPATIAL
suite=h200/liverms_v2_opt/bench_h200.py
model=17-p1-nr4-1024d-v3
read -r -a graph_variants <<< "${GRAPH_VARIANTS:-4 8 4-cached 8-cached 4x8 4x16 8x8 8x16 auto}"

nvidia-smi > "$out/nvidia-smi.txt"
"$python_bin" "$suite" probe --out "$out/probe.json"
"$python_bin" -m pytest -q tests/test_general_cuda_scans.py -k test_cuda_spatial_vscan |
    tee "$out/edge-pytest.log"
"$python_bin" h200/wave16_tc_vscan_probe/bench.py \
    --batch "$batch" --out "$out/isolated.jsonl" --variants "${graph_variants[@]}" |
    tee "$out/isolated.log"
if [[ ${COMBINED_SCAN:-0} == 1 ]]; then
    "$python_bin" h200/wave16_tc_vscan_probe/bench.py --combined --batch "$batch" \
        --variants "$variant" --out "$out/combined.jsonl" | tee "$out/combined.log"
fi
if [[ ${REFERENCE_CHECK:-1} == 1 ]]; then
    "$python_bin" h200/wave16_tc_vscan_probe/validate.py \
        --batch "${REFERENCE_BATCH:-128}" --variant "$variant" --out "$out/parity.jsonl" |
        tee "$out/parity.log"
fi

for model_batch in ${MODEL_BATCHES:-128}; do
    for arm in baseline spatial; do
        setting=
        if [[ $arm == spatial ]]; then setting=$variant; fi
        LNET_VSCAN_SPATIAL=$setting "$python_bin" "$suite" train-arm --variant "$model" \
            --batch "$model_batch" --profile --out "$out/b${model_batch}_${arm}.json"
    done
done

export LNET_VSCAN_SPATIAL=$variant
"$python_bin" "$suite" dynamo --out "$out/dynamo.json"
"$python_bin" "$suite" infer-fold --variant "$model" --batch "$batch" --out "$out/folded.json"
status=0
started=$SECONDS
timeout --foreground --kill-after=5s 13m "$python_bin" -m pytest -q \
    tests/test_general_cuda_scans.py tests/test_product_scan_pipeline.py tests/test_p1_nr4.py \
    > "$out/targeted-pytest.log" 2>&1 || status=$?
"$python_bin" "$suite" wrap-pytest --out "$out/targeted-pytest.json" \
    --log "$out/targeted-pytest.log" --exit-code "$status" --elapsed "$((SECONDS - started))"
test "$status" -eq 0

"$python_bin" - "$out" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
for path in folder.glob("*.json"):
    result = json.loads(path.read_text())
    assert result["status"] == "ok", (path.name, result)
    if path.name.startswith("b") and "_" in path.name:
        assert result["finite_losses"] and result["finite_gradients"], path
dynamo = json.loads((folder / "dynamo.json").read_text())
assert [pair for row in dynamo["graphs"].values() for pair in row.values()] == [[1, 0]] * 18
folded = json.loads((folder / "folded.json").read_text())
assert folded["fp32_head_parity_by_precision"]["highest"]["passed"]
assert folded["strict_reload_passed"] and folded["folded_train_guard"]
PY
