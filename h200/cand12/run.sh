#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export RUN_ROOT=/app/output/cand12-bench-20261118
mkdir -p "$RUN_ROOT"
cd ../..
python3 -m pip install --disable-pip-version-check --target "$RUN_ROOT/bootstrap" uv==0.11.26
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv python install 3.13.11
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv venv --python 3.13.11 "$RUN_ROOT/environment"
PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv pip install --python "$RUN_ROOT/environment/bin/python" -r h200/terminal_rawg/requirements.txt
export PATH="$RUN_ROOT/environment/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
cd h200/cand12
PYTHONPATH=code-control/src:code-control/scripts:code-control python init.py
python -c "import torch; assert 'H200' in torch.cuda.get_device_name(), torch.cuda.get_device_name()"
export VA_REAL_MEMORY_SCAN_BATCH_PARTIALS=1 LNET_DISABLE_LAUNCH_AUTOTUNE=0 PYTHONHASHSEED=0
mkdir -p results
for spec in control:1 cand12:1 cand12:2 control:2 control:3 cand12:3; do
  side=${spec%:*}
  round=${spec#*:}
  PYTHONPATH="code-$side/scripts:code-$side/src:code-$side" TORCHINDUCTOR_CACHE_DIR="$PWD/cache-$side" python benchmark_dw_terminal_rmsnorm_training.py --checkpoint checkpoint.pt --contract contract.json --images images.pt --batch 128 --output "results/$side-$round.json" --warmup 20 --steps 10 --repeats 5 --spatial --vector --reset-each-step
done
python - <<'RESULT'
import json, pathlib, statistics
values = {}
for side in ('control', 'cand12'):
    values[side] = [json.loads(p.read_text())['median_ms'] for p in sorted(pathlib.Path('results').glob(side+'-*.json'))]
print('H200_RESULT',values,'speedup',statistics.median(values['control'])/statistics.median(values['cand12']))
RESULT
