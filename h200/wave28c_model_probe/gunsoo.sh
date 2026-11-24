#!/usr/bin/env bash
set -euo pipefail

cd /tmp/va-t100-gunsoo
py="$HOME/miniconda3/envs/alphabet/bin/python"
bench=h200/liverms_v2_opt/bench_h200.py
out="$PWD/results"
mkdir -p "$out"
export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$PWD/inductor" TRITON_CACHE_DIR="$PWD/triton"
export LNET_MODEL_W28C=combined
export LNET_SCAN_W25A=auto LNET_READER_W25B=auto LNET_FUSED_STEM_CONV=auto

check() {
    local name=$1 rc=0
    shift
    "$@" > "$out/$name.log" 2>&1 || rc=$?
    printf '%s\t%s\n' "$name" "$rc" >> "$out/checks.tsv"
}

(
    flock -w 2400 9
    check parity "$py" -m pytest -q tests/test_model_w28c.py
    export LNET_CAPTURE_SAFE_ORTHOGONAL=on
    check dynamo "$py" "$bench" dynamo --out "$out/dynamo.json"
    unset LNET_CAPTURE_SAFE_ORTHOGONAL
    check cuda-pytest "$py" -m pytest -q tests/test_p1_nr4.py tests/test_memory_backbone.py
    export LNET_CAPTURE_SAFE_ORTHOGONAL=on
    check fold "$py" "$bench" infer-fold --variant 17-p1-nr4-1024d-v3 --batch 2 \
        --out "$out/fold.json"
) 9>/tmp/va-gpu.lock
awk -F '\t' '$2 != 0 {failed=1} END {exit failed}' "$out/checks.tsv"
