#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

out=$(mktemp -d /app/output/va-t60-h200.XXXXXX)
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS
unset LNET_DISABLE_LAUNCH_AUTOTUNE LNET_NR4_FUSED LNET_VSCAN_DECOUPLED
export TORCHINDUCTOR_CACHE_DIR="$out/inductor"
export SCAN_BATCH=1024 MODEL_BATCHES="128 256 512 1024" COMBINED_SCAN=1 REFERENCE_CHECK=0
export SPATIAL_VARIANT=16x8-batched GRAPH_VARIANTS="auto-batched 16x8-batched"
baseline=42ff1ee5170892b94d44a4a17106af28b71400ab
mkdir "$out/wave14a"
git archive "$baseline" src scripts | tar -x -C "$out/wave14a"
export BASELINE_PYTHONPATH="$out/wave14a/src:$out/wave14a/scripts"

finish() {
    local rc=$?
    trap - EXIT
    printf 'Wave 16A results %s, runner exit %d\n' "$out" "$rc"
    printf '%d\n' "$rc" > "$out/exit-code.txt"
    (cd -- "$out" && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 tar -cf -) |
        xz -T2 | base64 -w 120 |
        sed 's/^/VA_T60_XZ /'
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
bash h200/wave16_tc_vscan_probe/run.sh "$out"
