#!/usr/bin/env bash
set -euo pipefail

runtime=${1:?runtime path is required}
evidence=${2:?evidence path is required}
python_bin=${3:?python path is required}

cd "$runtime"
mkdir -p "$evidence"
export PYTHONPATH="$runtime/src:$runtime/scripts"
export CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

"$python_bin" scripts/evaluate_r2k3_scan_unification.py \
  --provenance-only \
  --output "$evidence/source-provenance.json"

"$python_bin" -m pytest -q \
  tests/test_evaluate_r2k3_scan_unification.py \
  tests/test_pac_product_scan_module_boundaries.py \
  tests/test_pac_product_scan_pipeline.py \
  tests/test_pac_product_scan_ops.py \
  tests/test_product_scan_path_collapse.py \
  tests/test_pac_same_resolution_depth.py \
  tests/test_pac_kernel_launch_config.py \
  tests/test_complex_scan_module_boundaries.py \
  tests/test_complex_scan.py

"$python_bin" scripts/evaluate_r2k3_scan_unification.py \
  --output "$evidence/scan-unification-matrix-performance.json" \
  --require-parity \
  --require-no-regression \
  --require-matrix
