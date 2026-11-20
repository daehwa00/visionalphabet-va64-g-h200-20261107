#!/usr/bin/env bash
set -euo pipefail
umask 077

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.."
CAND12=$PWD
export HOPPER_R12_START
HOPPER_R12_START=$(date +%s)
export HOPPER_REQUESTED_POOL="${1:?Expected pool is required}"
export HOPPER_R12_MODE="${2:-full}"
case "$HOPPER_R12_MODE" in
    full) ;;
    *) exit 2 ;;
esac
case "$HOPPER_REQUESTED_POOL" in
    whole_h200) budget=38 ;;
    *) exit 2 ;;
esac
export RUN_ROOT
RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/va-h200-hopper-r12-20261120.XXXXXX")
export PERSIST_ROOT=
PERSIST_ROOT=$(mktemp -d /app/output/va-h200-hopper-r12-results.XXXXXX) || true
df -Pk "$RUN_ROOT" /app/output
export XDG_CACHE_HOME="$RUN_ROOT/xdg" UV_CACHE_DIR="$RUN_ROOT/uv-cache"
export HOPPER_R12_COMMIT
HOPPER_R12_COMMIT=$(git rev-parse HEAD)
export HOPPER_R12_DEADLINE=$(( HOPPER_R12_START + budget * 60 ))
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS LNET_DISABLE_LAUNCH_AUTOTUNE
unset LNET_R6_HBACK LNET_R6_CABWD LNET_R7_AFWD LNET_R7_DIRECT
unset LNET_R8_HFWD LNET_R8_HBACK LNET_R9_RMS_DW
unset LNET_R10_RMS_DW LNET_R10_VFWD LNET_R10_HFWD56_SEG4 LNET_R10_RMS_DW56 LNET_R10_TUNED_LAUNCH LNET_R10_SEG4PACK2
unset LNET_R11_CUDA_VFWD LNET_R11_CUDA_VBACK LNET_R11_CUDA_HFWD LNET_R11_CUDA_HBACK
unset LNET_R12_PACKED LNET_R12_INDEX32 LNET_R12_PRECISE
export CONTRACT="$CAND12/contract.json"
finish() {
    result=$?
    if ! python3 "$HERE/export_result.py" "$result"; then
        result=1
    fi
    exit "$result"
}
trap finish EXIT

python3 "$HERE/probe_topology.py"
persistent=/app/output/va-h200-effective1024-r3-combined-20261119
markerfix=/app/output/va-h200-r4-marker-fix-20261119
verify_environment() {
    "$1" -c 'import torch, triton, torchvision, numpy, timm, requests, cryptography; assert torch.__version__ == "2.9.1+cu128"; assert triton.__version__ == "3.5.1"'
}
env_python=
for candidate in "$persistent/environment/bin/python" "$markerfix/environment/bin/python"; do
    if [[ -x "$candidate" ]] && verify_environment "$candidate"; then
        env_python="$candidate"
        break
    fi
done
if [[ -z "$env_python" ]]; then
    env_python="$RUN_ROOT/environment/bin/python"
    export UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python"
    timeout --kill-after=10s 300s bash -euc '
        python3 -m pip install --disable-pip-version-check --target "$RUN_ROOT/bootstrap" uv==0.11.26
        export PYTHONPATH="$RUN_ROOT/bootstrap"
        python3 -m uv python install 3.13.11
        python3 -m uv venv --python 3.13.11 "$RUN_ROOT/environment"
        python3 -m uv pip install --python "$1" -r "$2" cryptography
    ' bash "$env_python" "$CAND12/../terminal_rawg/requirements.txt"
    verify_environment "$env_python"
    rm -rf -- "$UV_CACHE_DIR"
fi
echo "ENVIRONMENT_PYTHON=$env_python"
printf '%s\n' "$env_python" > "$RUN_ROOT/environment-source"

if ! COLD_INPUTS=$("$env_python" "$HERE/stage_r12.py" "$RUN_ROOT/private-input"); then
    printf '%s\n' 'Seven verified inputs could not be staged' > "$RUN_ROOT/setup-error"
    exit 1
fi
export COLD_INPUTS
printf '%s\n' "$HERE/inputs_r12.sha256" > "$RUN_ROOT/fixture-manifest"
printf '%s\n' sealed-relay > "$RUN_ROOT/fixture-source"

# Assemble the benchmark sources with the pinned launch records.
"$env_python" "$HERE/stage_r12.py" "$RUN_ROOT" --sources "$COLD_INPUTS/packaged-4090-effective.json"
remaining=$(( HOPPER_R12_DEADLINE - $(date +%s) ))
(( remaining > 0 ))
timeout --kill-after=3s "${remaining}s" "$env_python" -u "$HERE/run_all.py"
