#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
MODE="${1:-preflight}"
[[ "$MODE" == preflight || "$MODE" == train ]] || { echo 'Expected preflight or train'; exit 2; }
RUN_ID="${VA64_RUN_ID:-va64-g-seed0-preflight-20261107}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run ID'; exit 2; }
RUN_ROOT="/app/output/$RUN_ID"
[[ "${2:-}" == --inside-guard || ! -e "$RUN_ROOT" ]] || { echo 'Output already exists; inspect it before retry'; exit 1; }
if [[ "${2:-}" != --inside-guard ]]; then
  mkdir -p "$RUN_ROOT"
  export VA64_G_CODE_SHA="$(git rev-parse HEAD)"
  export VA64_G_MANAGED_ROOT="$RUN_ROOT"
  export VA64_G_JOB=va64-g-preflight
  [[ "$MODE" == train ]] && export VA64_G_JOB=va64-g-seed0
  export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts"
  exec python3 -u scripts/va64_h200_guard.py --root "$RUN_ROOT" --code-sha "$VA64_G_CODE_SHA" \
    --relay-url https://visionalphabet-va64-g-20261107.gpupulse-monitor.workers.dev/va64g \
    --share-token-with-child --enroll-wait-seconds 900 --compact-results \
    --grace-seconds 1200 --stall-seconds 1800 -- bash h200/va64/run.sh "$MODE" --inside-guard
fi
mkdir -p "$RUN_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export LNET_DISABLE_LAUNCH_AUTOTUNE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WANDB_MODE=disabled
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/inductor"
trap 'touch "$RUN_ROOT/STOP"' TERM INT
JOB=va64-g-preflight
[[ "$MODE" == train ]] && JOB=va64-g-seed0
check_stop() { [[ ! -e "$RUN_ROOT/STOP" ]] || { echo "VA64_STOP_REQUESTED"; exit 130; }; }
python3 scripts/va64_runtime_operations.py ready --root "$RUN_ROOT" --job "$JOB"
check_stop
trap 'rc=$?; echo "VA64_JOB_EXIT=$rc"; exit "$rc"' EXIT
python3 -m pip install --disable-pip-version-check --target "$RUN_ROOT/bootstrap" uv==0.11.26
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv python install 3.13.11
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv venv --python 3.13.11 "$RUN_ROOT/environment"
PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv pip install --python "$RUN_ROOT/environment/bin/python" -r h200/va64/requirements.txt
PY="$RUN_ROOT/environment/bin/python"
DATA_ROOT="${VA64_DATA_ROOT:-/app/data/ImageNet-2012}"
check_stop
"$PY" h200/va64/preflight.py --source-root "$PROJECT_ROOT" --data-root "$DATA_ROOT" --output-root "$RUN_ROOT/environment-check"
check_stop
# Tests use tiny fields. The smoke below checks the full 224px path separately.
"$PY" -m pytest -q tests/test_va64_g.py
if [[ "$MODE" == train ]]; then
  # Full training is a separate issue after retrieval and stop access are verified.
  [[ "${VA64_OPERATIONS_VERIFIED:-}" == yes ]] || { echo 'Verify checkpoint retrieval and remote stopping before full training'; exit 2; }
  exec "$PY" scripts/run_va64_g_h200.py --variant G --seed 0 --data-root "$DATA_ROOT" --data-manifest "$RUN_ROOT/environment-check/dataset-manifest.json" --output-root "$RUN_ROOT/runs" --batch-size 128 --workers 8 --gpu-mixup --stop-file "$RUN_ROOT/STOP"
fi
check_stop
"$PY" - "$RUN_ROOT" "$JOB" <<'PYCODE'
import sys
from pathlib import Path
from va64_runtime_operations import current
current(Path(sys.argv[1]),sys.argv[2],'preflight',Path(sys.argv[1])/'smoke')
PYCODE
 timeout --signal=INT --kill-after=60 20m "$PY" h200/va64/smoke_g.py --output "$RUN_ROOT/smoke" --steps 20 --image-size 224 --device cuda --compile
"$PY" scripts/va64_runtime_operations.py backup-smoke --root "$RUN_ROOT" --job "$JOB" --output "$RUN_ROOT/smoke"
printf 'VA64_PREFLIGHT_COMPLETE output=%s training_started=false\n' "$RUN_ROOT"
