#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/.."
CAND12=$PWD
export RUN_ROOT
RUN_ROOT=$(mktemp -d /app/output/va-h200-hopper-r7-20261119.XXXXXX)
export HOPPER_R7_COMMIT
HOPPER_R7_COMMIT=$(git rev-parse HEAD)
export HOPPER_R7_DEADLINE=$(( $(date +%s) + 31 * 60 ))
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS LNET_DISABLE_LAUNCH_AUTOTUNE
unset LNET_R6_HBACK LNET_R6_CABWD LNET_R7_R5_ASSOC
export CONTRACT="$CAND12/contract.json"
trap 'result=$?; python3 "$HERE/export_result.py" "$result"; exit "$result"' EXIT

persistent=/app/output/va-h200-effective1024-r3-combined-20261119
markerfix=/app/output/va-h200-r4-marker-fix-20261119
unset COLD_INPUTS
for candidate in "$persistent/private-input" "$markerfix/private-input"; do
    if [[ -d "$candidate" ]] && (cd "$candidate" && sha256sum -c "$CAND12/effective_cli/inputs.sha256"); then
        export COLD_INPUTS="$candidate"
        break
    fi
done
: "${COLD_INPUTS:?No verified persistent fixture}"
echo "FIXTURE_SOURCE=$COLD_INPUTS"
printf '%s\n' "$COLD_INPUTS" > "$RUN_ROOT/fixture-source"

verify_environment() {
    "$1" -c 'import torch, triton, torchvision, numpy, timm; assert torch.__version__ == "2.9.1+cu128"; assert triton.__version__ == "3.5.1"'
}
env_python="$persistent/environment/bin/python"
if ! [[ -x "$env_python" ]] || ! verify_environment "$env_python"; then
    env_python="$RUN_ROOT/environment/bin/python"
    export UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python"
    timeout --kill-after=10s 300s bash -euc '
        python3 -m pip install --disable-pip-version-check --target "$RUN_ROOT/bootstrap" uv==0.11.26
        export PYTHONPATH="$RUN_ROOT/bootstrap"
        python3 -m uv python install 3.13.11
        python3 -m uv venv --python 3.13.11 "$RUN_ROOT/environment"
        python3 -m uv pip install --python "$1" -r "$2"
    ' bash "$env_python" "$CAND12/../terminal_rawg/requirements.txt"
    verify_environment "$env_python"
fi
echo "ENVIRONMENT_PYTHON=$env_python"
printf '%s\n' "$env_python" > "$RUN_ROOT/environment-source"

mkdir -p "$RUN_ROOT/r4" "$RUN_ROOT/current"
for layer in code-cand12 common replay/known-overlay segment/overlay official/overlay effective/overlay effective_cli/overlay; do
    cp -a "$layer/." "$RUN_ROOT/r4/"
done
cp "$COLD_INPUTS/packaged-4090-effective.json" "$RUN_ROOT/r4/src/lnet/validated_effective_updates.json"
(cd "$RUN_ROOT/r4" && sha256sum -c "$CAND12/effective_cli/common-release.sha256")
(cd "$HERE/overlay" && sha256sum --quiet -c "$HERE/overlay.sha256")
cp -a "$RUN_ROOT/r4/." "$RUN_ROOT/current/"
cp -a "$HERE/overlay/." "$RUN_ROOT/current/"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv
remaining=$(( HOPPER_R7_DEADLINE - $(date +%s) ))
(( remaining > 0 ))
timeout --kill-after=15s "${remaining}s" "$env_python" -u "$HERE/run_all.py"
