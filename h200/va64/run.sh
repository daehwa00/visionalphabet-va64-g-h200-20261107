#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
MODE="${1:-preflight}"
[[ "$MODE" == preflight || "$MODE" == train ]] || { echo 'Expected preflight or train'; exit 2; }
RUN_ID="${VA64_RUN_ID:-va64-g-seed0-preflight-20261107}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run ID'; exit 2; }
RUN_ROOT="/app/output/$RUN_ID"
[[ ! -e "$RUN_ROOT" ]] || { echo 'Output already exists; inspect it before retry'; exit 1; }
mkdir -p "$RUN_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export LNET_DISABLE_LAUNCH_AUTOTUNE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WANDB_MODE=disabled
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/inductor"
trap 'rc=$?; echo "VA64_JOB_EXIT=$rc"; exit "$rc"' EXIT
python3 -m pip install --disable-pip-version-check --target "$RUN_ROOT/bootstrap" uv==0.11.26
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv python install 3.13.11
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv venv --python 3.13.11 "$RUN_ROOT/environment"
PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv pip install --python "$RUN_ROOT/environment/bin/python" -r h200/va64/requirements.txt
PY="$RUN_ROOT/environment/bin/python"
DATA_ROOT="${VA64_DATA_ROOT:-/app/data/ImageNet-2012}"
"$PY" h200/va64/preflight.py --source-root "$PROJECT_ROOT" --data-root "$DATA_ROOT" --output-root "$RUN_ROOT/environment-check"
# Tests use tiny fields. The smoke below checks the full 224px path separately.
"$PY" -m pytest -q tests/test_va64_g.py
 timeout --signal=INT --kill-after=60 20m "$PY" h200/va64/smoke_g.py --output "$RUN_ROOT/smoke" --steps 20 --image-size 224 --device cuda --compile
if [[ "$MODE" == train ]]; then
  # Full training is a separate issue after retrieval and stop access are verified.
  [[ "${VA64_OPERATIONS_VERIFIED:-}" == yes ]] || { echo 'Verify checkpoint retrieval and remote stopping before full training'; exit 2; }
  exec "$PY" scripts/run_va64_g_h200.py --variant G --seed 0 --data-root "$DATA_ROOT" --data-manifest "$RUN_ROOT/environment-check/dataset-manifest.json" --output-root "$RUN_ROOT/runs" --batch-size 128 --workers 8 --gpu-mixup --stop-file "$RUN_ROOT/STOP"
fi
printf 'VA64_PREFLIGHT_COMPLETE output=%s training_started=false\n' "$RUN_ROOT"
