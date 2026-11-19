#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.."
CAND12=$PWD
export HOPPER_R8_START
HOPPER_R8_START=$(date +%s)
export HOPPER_REQUESTED_POOL="${1:?Expected pool is required}"
case "$HOPPER_REQUESTED_POOL" in
    mig_1g18gb) budget=24 ;;
    whole_h200) budget=38 ;;
    *) exit 2 ;;
esac
export RUN_ROOT
RUN_ROOT=$(mktemp -d /app/output/va-h200-hopper-r8-20261119.XXXXXX)
export HOPPER_R8_COMMIT
HOPPER_R8_COMMIT=$(git rev-parse HEAD)
export HOPPER_R8_DEADLINE=$(( HOPPER_R8_START + budget * 60 ))
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS LNET_DISABLE_LAUNCH_AUTOTUNE
unset LNET_R6_HBACK LNET_R6_CABWD LNET_R7_AFWD LNET_R7_DIRECT
export CONTRACT="$CAND12/contract.json"
trap 'result=$?; python3 "$HERE/export_result.py" "$result"; exit "$result"' EXIT

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
fi
echo "ENVIRONMENT_PYTHON=$env_python"
printf '%s\n' "$env_python" > "$RUN_ROOT/environment-source"

if ! COLD_INPUTS=$("$env_python" "$HERE/stage_r8.py" "$RUN_ROOT/private-input"); then
    printf '%s\n' 'Seven verified inputs could not be staged' > "$RUN_ROOT/setup-error"
    exit 1
fi
export COLD_INPUTS
printf '%s\n' "$HERE/inputs_r8.sha256" > "$RUN_ROOT/fixture-manifest"
if [[ "$COLD_INPUTS" == "$RUN_ROOT/private-input" ]]; then
    printf '%s\n' sealed-relay > "$RUN_ROOT/fixture-source"
else
    printf '%s\n' verified-persistent > "$RUN_ROOT/fixture-source"
fi

mkdir -p "$RUN_ROOT/r4" "$RUN_ROOT/current"
for layer in code-cand12 common replay/known-overlay segment/overlay official/overlay effective/overlay effective_cli/overlay; do
    cp -a "$layer/." "$RUN_ROOT/r4/"
done
cp "$COLD_INPUTS/packaged-4090-effective.json" "$RUN_ROOT/r4/src/lnet/validated_effective_updates.json"
(cd "$RUN_ROOT/r4" && sha256sum -c "$CAND12/effective_cli/common-release.sha256")
(cd "$HERE/overlay" && sha256sum --quiet -c "$HERE/overlay.sha256")
cp -a "$RUN_ROOT/r4/." "$RUN_ROOT/current/"
cp -a "$HERE/overlay/." "$RUN_ROOT/current/"
remaining=$(( HOPPER_R8_DEADLINE - $(date +%s) ))
(( remaining > 0 ))
timeout --kill-after=3s "${remaining}s" "$env_python" -u "$HERE/run_all.py"
