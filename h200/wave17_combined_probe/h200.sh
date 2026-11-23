#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

out=$(mktemp -d /app/output/va-t65-wave17-h200.XXXXXX)
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS
unset LNET_DISABLE_LAUNCH_AUTOTUNE LNET_NR4_FUSED LNET_VSCAN_DECOUPLED LNET_READER_VEC
export TORCHINDUCTOR_CACHE_DIR="$out/inductor"

baseline=42ff1ee5170892b94d44a4a17106af28b71400ab
variant=17-p1-nr4-1024d-v3
bench=h200/liverms_v2_opt/bench_h200.py
mkdir "$out/wave14a"
git archive "$baseline" src scripts | tar -x -C "$out/wave14a"
BASE_PP="$out/wave14a/src:$out/wave14a/scripts"
CAND_PP="$PWD/src:$PWD/scripts"

finish() {
    local rc=$?
    trap - EXIT
    printf 'Wave 17 combined results %s, runner exit %d\n' "$out" "$rc"
    printf '%d\n' "$rc" > "$out/exit-code.txt"
    if [[ -n ${PYTHON_BIN:-} && -x ${PYTHON_BIN:-} ]]; then
        "$PYTHON_BIN" - "$out" <<'PY' || true
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
print("=== SUMMARY: B=1024 4-WAY A/B ===")
for name in ("b1024_baseline_w14a", "b1024_wave17a_only", "b1024_wave17b_only", "b1024_wave17ab_combined"):
    p = out / f"{name}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    buckets = {k: round(v, 2) for k, v in (d.get("buckets_ms") or {}).items()}
    print(f"{name:26s} step={d.get('step_ms_median', d.get('median_ms'))} ms | "
          f"ips={d.get('throughput_img_s')} | "
          f"alloc={d.get('peak_allocated_gib', d.get('max_allocated_gib'))} GiB | "
          f"res={d.get('peak_reserved_gib', d.get('max_reserved_gib'))} GiB | "
          f"buckets={buckets}")
print("=== SUMMARY: BATCH SWEEP ===")
for b in (128, 256, 512):
    for arm in ("baseline_w14a", "wave17ab_combined"):
        p = out / f"b{b}_{arm}.json"
        if p.exists():
            d = json.loads(p.read_text())
            print(f"B={b:4d} {arm:20s} step={d.get('step_ms_median', d.get('median_ms'))} ms | "
                  f"alloc={d.get('peak_allocated_gib', d.get('max_allocated_gib'))} GiB")
for name in ("dynamo_combined", "fold_combined"):
    p = out / f"{name}.json"
    if p.exists():
        print(f"=== {name} ===")
        print(p.read_text()[:600])
PY
    fi
    (cd -- "$out" && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 tar -cf -) |
        xz -T2 | base64 -w 120 |
        sed 's/^/VA_T65_XZ /'
    exit "$rc"
}
trap finish EXIT

ready() {
    test -x "$1" && "$1" -c '
import sys
import numpy, pytest, timm, torch, torchvision
raise SystemExit(sys.version_info < (3, 13) or not torch.cuda.is_available())
' 2>/dev/null
}

for candidate in \
    /app/output/va-h200-effective1024-r3-combined-20261119/environment/bin/python \
    /app/output/va-h200-r4-marker-fix-20261119/environment/bin/python \
    "$(command -v python3)"; do
    if ready "$candidate"; then
        export PYTHON_BIN=$candidate
        break
    fi
done
if [[ -z ${PYTHON_BIN:-} ]]; then
    setup="$out/environment"
    mkdir -p "$setup"
    export UV_PYTHON_INSTALL_DIR="$setup/python" UV_CACHE_DIR="$setup/uv-cache"
    python3 -m pip install -q --disable-pip-version-check --target "$setup/bootstrap" uv==0.11.26
    uv_run() { PYTHONPATH="$setup/bootstrap" python3 -m uv "$@"; }
    uv_run python install 3.13.11
    uv_run venv --python 3.13.11 "$setup/venv"
    export PYTHON_BIN="$setup/venv/bin/python"
    uv_run pip install --python "$PYTHON_BIN" \
        'torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=63381a109a569b280ed3319da89d3afe5cf9ab5c879936382a212affb5c90552' \
        'torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=5f7c5e0fa08d2cbee93b6e04bbedd59b5e11462cff6cefd07949217265df2370' \
        triton==3.5.1 numpy==2.4.6 pytest==9.0.3 timm==1.0.26
    ready "$PYTHON_BIN"
fi

echo "[1] Device probe"
PYTHONPATH="$CAND_PP" "$PYTHON_BIN" "$bench" probe --out "$out/device.json"

echo "[2] B=1024 4-way A/B (with CUDA profiler bucket breakdown)"
PYTHONPATH="$BASE_PP" "$PYTHON_BIN" "$bench" train-arm \
    --variant "$variant" --batch 1024 --profile --out "$out/b1024_baseline_w14a.json"

LNET_VSCAN_DECOUPLED=auto LNET_READER_VEC=1 PYTHONPATH="$CAND_PP" "$PYTHON_BIN" "$bench" train-arm \
    --variant "$variant" --batch 1024 --profile --out "$out/b1024_wave17a_only.json"

LNET_VSCAN_DECOUPLED=off LNET_READER_VEC=auto PYTHONPATH="$CAND_PP" "$PYTHON_BIN" "$bench" train-arm \
    --variant "$variant" --batch 1024 --profile --out "$out/b1024_wave17b_only.json"

LNET_VSCAN_DECOUPLED=auto LNET_READER_VEC=auto PYTHONPATH="$CAND_PP" "$PYTHON_BIN" "$bench" train-arm \
    --variant "$variant" --batch 1024 --profile --out "$out/b1024_wave17ab_combined.json"

echo "[3] Batch sweep (B=128, 256, 512): baseline vs combined"
for b in 128 256 512; do
    PYTHONPATH="$BASE_PP" "$PYTHON_BIN" "$bench" train-arm \
        --variant "$variant" --batch "$b" --out "$out/b${b}_baseline_w14a.json"
    LNET_VSCAN_DECOUPLED=auto LNET_READER_VEC=auto PYTHONPATH="$CAND_PP" "$PYTHON_BIN" "$bench" train-arm \
        --variant "$variant" --batch "$b" --out "$out/b${b}_wave17ab_combined.json"
done

echo "[4] Isolated H200 CUDA Graph microbenchmarks (B=1024 BF16)"
PYTHONPATH="$CAND_PP" "$PYTHON_BIN" h200/wave17a_decoupled_tc_vscan_probe/bench.py micro \
    --dtype bfloat16 --batch 1024 --repeats 10 --case 0 1 2 3 4 5 6 \
    --out "$out/vscan_micro_b1024_bf16.jsonl"

for shape in "56 56 160 4" "28 28 320 4" "14 14 640 4" "7 7 640 4"; do
    # shellcheck disable=SC2086
    PYTHONPATH="$CAND_PP" "$PYTHON_BIN" h200/wave17b_vec_reader_probe/bench.py bench \
        --batch 1024 --samples 10 --shape $shape \
        --baseline "$out/wave14a/src/lnet/factorized_reader_kernels.py" \
        --out "$out/reader_micro_${shape// /x}.jsonl"
done

echo "[5] Dynamo 18/18 & Folded inference FP32 highest parity (combined auto)"
LNET_VSCAN_DECOUPLED=auto LNET_READER_VEC=auto PYTHONPATH="$CAND_PP" "$PYTHON_BIN" "$bench" dynamo \
    --out "$out/dynamo_combined.json"
LNET_VSCAN_DECOUPLED=auto LNET_READER_VEC=auto PYTHONPATH="$CAND_PP" "$PYTHON_BIN" "$bench" infer-fold \
    --variant "$variant" --batch 2 --out "$out/fold_combined.json"

echo "[6] Targeted CUDA pytest (combined auto, 12m cap)"
set +e
LNET_VSCAN_DECOUPLED=auto LNET_READER_VEC=auto PYTHONPATH="$CAND_PP" timeout --kill-after=10s 720s \
    "$PYTHON_BIN" -m pytest -q \
    tests/test_general_cuda_scans.py \
    tests/test_product_scan_pipeline.py \
    tests/test_factorized_reader_kernels.py \
    tests/test_p1_nr4.py > "$out/pytest_combined.log" 2>&1
pytest_rc=$?
set -e
printf '%d\n' "$pytest_rc" > "$out/pytest_rc.txt"
tail -n 25 "$out/pytest_combined.log"
test "$pytest_rc" -eq 0
