#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts"
export RUN_ROOT
RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/va-t52-h200.XXXXXX")
export PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/inductor"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export PERSIST_ROOT
PERSIST_ROOT=$(mktemp -d /app/output/va-t52-reader.XXXXXX) || true
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
variant=17-p1-nr4-1024d-v3
bench=h200/liverms_v2_opt/bench_h200.py
for batch in 1024 128 256 512; do
    "$py" "$bench" train-arm --variant "$variant" --batch "$batch" --profile \
        --out "$RUN_ROOT/train-b$batch.json"
done
for batch in 128 1024; do
    "$py" "$bench" infer-fold --variant "$variant" --batch "$batch" \
        --out "$RUN_ROOT/fold-b$batch.json"
done
"$py" "$bench" dynamo --out "$RUN_ROOT/dynamo.json"
