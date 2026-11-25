#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OUT="${OUT:-/app/output/va-t159}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
SECONDS=0
run_stage() {
    local stage="$1" limit="$2" file="$3" started="$SECONDS" rc
    shift 3
    rm -f "$OUT/$file"
    set +e
    timeout --kill-after=5s "$limit" python3 "$@" >"$OUT/$stage.log" 2>&1
    rc=$?
    set -e
    printf '%s %s %s\n' "$stage" "$rc" "$((SECONDS - started))" >>"$OUT/status"
    if ((rc != 0)); then
        printf '%s exited %s\n' "$stage" "$rc"
        tail -n 8 "$OUT/$stage.log"
    fi
}

: >"$OUT/status"
export VA_NVIDIA_SMI="$(nvidia-smi 2>&1 || true)"
export VA_SHA="$(git rev-parse HEAD)"
export LNET_FUSED_STEM_CONV=on
run_stage scan 180s scan_vscan_bwd.json scripts/bench_scan_micro.py \
    --ops vscan_bwd --modes off,strict_ro,tc_mode_shared --flush-l2 --parity \
    --out "$OUT/scan_vscan_bwd.json"
run_stage reader 210s reader_wave45b.json scripts/bench_reader_micro.py \
    --modes off dw_fullrank dw_fullrank_cvt fwd_vec1_r8 fwd_vec1_r4 dinput_wide all_opt \
    --flush-l2 --parity --out "$OUT/reader_wave45b.json"
run_stage stem 120s stem_stage0.json scripts/bench_stem_micro.py \
    --modes off,cudnn_split,cudnn_pad4 --flush-l2 --parity --json-out "$OUT/stem_stage0.json"

python3 h200/wave45_mega_screen/results.py emit
