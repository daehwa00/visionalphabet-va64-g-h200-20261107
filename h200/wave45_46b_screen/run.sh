#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
OUT=$(mktemp -d /app/output/va-t162-h200.XXXXXX)
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset LNET_GENERAL_SCAN_OPT LNET_VSCAN_MMA LNET_GENERAL_SCANS

stage() {
    local name=$1 limit=$2 rc=0 start=$SECONDS
    shift 2
    timeout --kill-after=5s "$limit" "$@" > "$OUT/$name.log" 2>&1 || rc=$?
    printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - start))" >> "$OUT/stages.tsv"
    if (( rc != 0 )); then
        printf '[stage] %s exited %s\n' "$name" "$rc"
        tail -n 40 "$OUT/$name.log"
    fi
}

stage scan 300s python3 scripts/bench_scan_micro.py --ops vscan_bwd \
    --modes off,strict_ro,tc_mode_shared --flush-l2 --parity --out "$OUT/scan_vscan_bwd.json"
stage readproj 150s python3 scripts/bench_readproj_micro.py \
    --modes off,dual_fwd_only,dual_mma_triton,dual_all,cublas_cat_ablation \
    --parity --json-out "$OUT/readproj_wave46b.json"
python3 h200/wave45_46b_screen/emit.py "$OUT"
