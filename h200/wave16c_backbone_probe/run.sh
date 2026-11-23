#!/usr/bin/env bash
# Run from the Wave 16C candidate checkout with a pre-provisioned CUDA Python.
set -euo pipefail
run_python=${1:?provide the CUDA Python executable}
out=${2:?provide an output directory}
suite=h200/wave16c_backbone_probe
bench=h200/liverms_v2_opt/bench_h200.py
baseline=42ff1ee5170892b94d44a4a17106af28b71400ab
mkdir -p "$out"
out=$(realpath "$out")
if [[ $# -ge 3 ]]; then
    cp "$3" "$out/baseline_backbone.py"
else
    git show "$baseline:src/lnet/memory_backbone.py" > "$out/baseline_backbone.py"
fi
export PYTHONPATH="$PWD/src:$PWD/scripts"
variants=(17-p1-nr4-1024d-v3 17-live-complex-rms-s03 17-e2-multiscale-s3s4-live-rms-v2)
exec 9>/tmp/va-gpu.lock
flock -w 1800 9
"$run_python" "$bench" probe --out "$out/probe.json"
"$run_python" "$bench" dynamo --out "$out/dynamo.json"
"$run_python" "$suite/micro.py" > "$out/micro.jsonl"
for variant in "${variants[@]}"; do
    "$run_python" "$suite/compare.py" --baseline "$out/baseline_backbone.py"         --variant "$variant" --profile --out "$out/ab-$variant.json"
    "$run_python" "$bench" infer-fold --variant "$variant" --batch 128         --out "$out/fold-$variant.json"
done
"$run_python" "$suite/compare.py" --baseline "$out/baseline_backbone.py"     --dense --profile --out "$out/ab-dense.json"
"$run_python" -m pytest -q tests/test_memory_backbone.py tests/test_fused_stem_ops.py     tests/test_p1_nr4.py tests/test_live_complex_rms.py > "$out/pytest.log" 2>&1
