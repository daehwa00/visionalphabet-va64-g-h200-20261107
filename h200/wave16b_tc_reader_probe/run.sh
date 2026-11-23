#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
py=${PYTHON:-python3}
out=${1:?Provide an output directory}
shift
mkdir -p "$out"
nvidia-smi >"$out/device.log"
probe=h200/wave16b_tc_reader_probe/probe.py
bench=h200/liverms_v2_opt/bench_h200.py
variant=17-p1-nr4-1024d-v3
status=0
run_check() {
    local label=$1
    shift
    local rc=0
    "$@" >"$out/$label.log" 2>&1 || rc=$?
    printf '%s %d\n' "$label" "$rc" >>"$out/status.log"
    if (( rc != 0 )); then status=1; fi
}
: >"$out/status.log"
run_check isolated "$py" "$probe" --reference --shift --out "$out/isolated.json" "$@"
for policy in off force; do
    export VA_READER_MMA="$policy"
    run_check "pytest-$policy" "$py" -m pytest -q tests/test_factorized_reader_kernels.py
    run_check "dynamo-$policy" "$py" "$bench" dynamo --out "$out/dynamo-$policy.json"
    run_check "fold-$policy" "$py" "$bench" infer-fold --variant "$variant" --batch 128 \
        --out "$out/fold-$policy.json"
    run_check "train-$policy" "$py" "$bench" train-arm --variant "$variant" \
        --batch "${TRAIN_BATCH:-128}" --profile --out "$out/train-$policy.json"
done
run_check validate "$py" h200/wave16b_tc_reader_probe/validate.py "$out"
exit "$status"
