#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts"
export RUN_ROOT
RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/va-t52-h200.XXXXXX")
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
trap 'python3 h200/liverms_v2_opt/export_result.py "$?"' EXIT

py=$(command -v python3)
for candidate in /app/output/va-h200-effective1024-r3-combined-20261119/environment/bin/python /app/output/va-h200-r4-marker-fix-20261119/environment/bin/python; do
    if [[ -x $candidate ]]; then
        py=$candidate
        break
    fi
done
if ! "$py" -c 'import torch, timm, triton; from typing import override' 2>/dev/null; then
    python3 -m pip install -q --target "$RUN_ROOT/bootstrap" uv==0.11.26
    export UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python"
    uv_bin="$RUN_ROOT/bootstrap/bin/uv"
    "$uv_bin" venv --python 3.13 "$RUN_ROOT/environment"
    py="$RUN_ROOT/environment/bin/python"
    "$uv_bin" pip install --python "$py" --extra-index-url https://download.pytorch.org/whl/cu128 \
        'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128' triton numpy pytest timm
fi
"$py" -m h200.wave13_reader_probe.bench --out "$RUN_ROOT/reader.json"
variant=17-p1-nr4-1024d-v3
for vec in 1 2 4; do
    VA_READER_VECTOR_PROBE=$vec "$py" h200/liverms_v2_opt/bench_h200.py train-arm \
        --variant "$variant" --batch 1024 --profile --out "$RUN_ROOT/train-$vec.json" || true
done
