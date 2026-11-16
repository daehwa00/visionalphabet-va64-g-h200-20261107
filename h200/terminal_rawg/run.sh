#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$PWD"
RUN_ID="${RAWG_RUN_ID:-terminal-rawg-v3-in1k-seed0-eb1024-20261116}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
RUN_ROOT="/app/output/$RUN_ID"
export VA64_G_RELAY_URL=https://visionalphabet-terminal-rawg-v3-20261116.gpupulse-monitor.workers.dev/va64g
export VA64_G_CHECKPOINT_POLICY=local_only VA64_G_JOB=va64-g-seed0 VA64_G_MANAGED_ROOT="$RUN_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts" PYTHONUNBUFFERED=1
export VA64_G_CODE_SHA="$(git rev-parse HEAD)"
if [[ "${1:-}" != --inside-guard ]]; then
 [[ ! -e "$RUN_ROOT" ]] || { echo 'Output exists; inspect before restarting'; exit 1; }
 mkdir -p "$RUN_ROOT"
 exec python3 -u scripts/va64_h200_guard.py --root "$RUN_ROOT" --code-sha "$VA64_G_CODE_SHA" \
  --relay-url "$VA64_G_RELAY_URL" --share-token-with-child --enroll-wait-seconds 900 \
  --compact-results --metrics-optional --grace-seconds 900 --stall-seconds 2400 \
  -- bash h200/terminal_rawg/run.sh --inside-guard
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export VA_REAL_MEMORY_SCAN_BATCH_PARTIALS=1
export LNET_DISABLE_LAUNCH_AUTOTUNE=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WANDB_MODE=disabled
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/inductor"
trap 'touch "$RUN_ROOT/STOP"' TERM INT
python3 -m pip install --disable-pip-version-check --target "$RUN_ROOT/bootstrap" uv==0.11.26
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv python install 3.13.11
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv venv --python 3.13.11 "$RUN_ROOT/environment"
PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv pip install --python "$RUN_ROOT/environment/bin/python" -r h200/terminal_rawg/requirements.txt
PY="$RUN_ROOT/environment/bin/python"
DATA_ROOT="${RAWG_DATA_ROOT:-/app/data/ImageNet-2012}"
[[ ! -e "$RUN_ROOT/STOP" ]] || exit 130
"$PY" h200/terminal_rawg/preflight.py --source-root "$PROJECT_ROOT" --data-root "$DATA_ROOT" --output-root "$RUN_ROOT/environment-check"
# Reuse only a readable cache produced by the same H200/software environment.
"$PY" - "$RUN_ROOT" <<'CACHE'
import json, shutil, sys
from pathlib import Path
root=Path(sys.argv[1]);old=Path('/app/output/terminal-rawg-in1k-seed0-eb1024-20261116')
old_report=old/'environment-check/preflight.json';new_report=root/'environment-check/preflight.json'
reused=False
if old_report.is_file() and (old/'cache').is_dir():
    a=json.loads(old_report.read_text());b=json.loads(new_report.read_text())
    if all(a.get(k)==b.get(k) for k in ('python','versions','gpu','cuda_runtime')):
        shutil.copytree(old/'cache',root/'cache',dirs_exist_ok=True);reused=True
print('RAWG_H200_CACHE_REUSED='+str(reused),flush=True)
(root/'cache-reuse.json').write_text(json.dumps({'reused':reused,'source':str(old/'cache'),'new_target':str(root/'cache'),'checkpoint_reused':False},indent=2)+'\n')
CACHE
"$PY" -m pytest -q tests/test_terminal_imcorr.py tests/test_terminal_rawg_in1k.py tests/test_coarse_backward_autotune.py tests/test_pac_triton_bidirectional_product_scan.py tests/test_real_memory_training.py
BATCH=256
set +e
"$PY" h200/terminal_rawg/smoke.py --batch-size "$BATCH" --output "$RUN_ROOT/smoke256"
rc=$?
set -e
if [[ "$rc" == 42 ]]; then
 BATCH=128
 "$PY" h200/terminal_rawg/smoke.py --batch-size "$BATCH" --output "$RUN_ROOT/smoke128"
elif [[ "$rc" != 0 ]]; then exit "$rc"; fi
"$PY" - "$RUN_ROOT" "$BATCH" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);b=int(sys.argv[2]);(r/'batch-selection.json').write_text(json.dumps({'physical_batch':b,'gradient_accumulation_steps':1024//b,'effective_batch':1024,'learning_rate':0.003,'seed':0,'epochs':100},indent=2)+'\n')
PY
[[ ! -e "$RUN_ROOT/STOP" ]] || exit 130
exec "$PY" -u scripts/run_terminal_rawg_imagenet1k.py --data-root "$DATA_ROOT" \
 --data-manifest "$RUN_ROOT/environment-check/dataset-manifest.json" --output-root "$RUN_ROOT/runs" \
 --batch-size "$BATCH" --workers 8 --stop-file "$RUN_ROOT/STOP"
