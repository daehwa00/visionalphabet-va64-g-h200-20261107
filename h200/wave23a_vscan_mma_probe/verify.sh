#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
out=${1:?output directory required}
python=${PYTHON:-python3}
mkdir -p "$out"
export PYTHONPATH="$PWD/src:$PWD/scripts"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export WANDB_MODE=disabled
unset LNET_VSCAN_DECOUPLED LNET_GENERAL_SCAN_OPT
suite=h200/wave23a_vscan_mma_probe
variant=17-p1-nr4-1024d-v3
train() {
    local name=$1 policy=$2 benchmark=$3
    LNET_VSCAN_MMA="$policy" VA_T82_CUDNN_BENCHMARK="$benchmark" "$python" "$suite/model.py" \
        train-arm --out "$out/$name.json" --variant "$variant" --batch 128 --profile \
        >"$out/$name.log" 2>&1
}
parity_status=0
PYTHONPATH="$PYTHONPATH:$PWD/tests" "$python" "$suite/parity.py" \
    >"$out/fp32-parity.jsonl" 2>"$out/fp32-parity.log" || parity_status=$?
printf '%s\n' "$parity_status" >"$out/fp32-parity.exit"
train p1-first-off off 1
train p1-first-auto auto 1
train p1-second-auto auto 1
train p1-second-off off 1
train p1-controlled-off off 0
train p1-controlled-auto auto 0
LNET_VSCAN_MMA=on "$python" "$suite/model.py" dynamo --out "$out/dynamo.json"
LNET_VSCAN_MMA=on "$python" "$suite/model.py" infer-fold --out "$out/folded.json" \
    --variant "$variant" --batch 128
LNET_VSCAN_MMA=on "$python" -m pytest -q >"$out/cuda-pytest.log" 2>&1
