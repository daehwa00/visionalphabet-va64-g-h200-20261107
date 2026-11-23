#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
src=h200/wave14_reader_probe
root=$(mktemp -d /app/output/va-t58-wave14-reader.XXXXXX)
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$root/inductor" TRITON_CACHE_DIR="$root/triton"
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts:$root/deps"
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS
unset LNET_DISABLE_LAUNCH_AUTOTUNE LNET_NR4_FUSED LNET_VSCAN_DECOUPLED
trap 'rc=$?; python3 "$src/export.py" "$root" "$rc"; exit "$rc"' EXIT
echo "Wave 14B raw files: $root"

py=$(command -v python3)
for candidate in /app/output/va-h200-effective1024-r3-combined-20261119/environment/bin/python /app/output/va-h200-r4-marker-fix-20261119/environment/bin/python; do
    if [[ -x $candidate ]]; then
        py=$candidate
        break
    fi
done
if ! "$py" -c 'import torch, triton; from typing import override' 2>/dev/null; then
    python3 -m pip install -q --target "$root/bootstrap" uv==0.11.26
    export UV_PYTHON_INSTALL_DIR="$root/python"
    uv_bin="$root/bootstrap/bin/uv"
    "$uv_bin" venv --python 3.13 "$root/environment"
    py="$root/environment/bin/python"
    "$uv_bin" pip install --python "$py" --extra-index-url https://download.pytorch.org/whl/cu128 \
        'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128' triton numpy timm
fi
if ! "$py" -c 'import timm' 2>/dev/null; then
    "$py" -m pip install --target "$root/deps" --no-deps timm
fi

names=ring4x7,ring4x14,ring8x14,smem8x14_scalar,smem8x14_pair,smem8x16_pair
names+=,smem8x28_pair,smem8x14_ldg128,smem8x14_block_pair,smem8x14_block_scalar
names+=,smem8x28_block_pair,weight8x14_pair,weight8x14_reuse,weight8x28_reuse
names+=,weight8x28_pair,weight_t1r8_vec2,weight_t1r14_vec2,weight_t2r8_scalar
names+=,weight_t1r2_packed,weight_t1r4_packed
"$py" "$src/probe.py" --batch 1024 --samples 7 --reference --names "$names" \
    --out "$root/isolated.json" 2>&1 | tee "$root/isolated.log"
if [[ ${1:-} == --isolated-only ]]; then
    # Restore a full set of raw samples if the earlier pod's output is unavailable.
    "$py" "$src/export.py" "$root" --samples
    exit
fi
"$py" "$src/full.py" select --policy "$root/isolated.json" --out "$root/policy.json"
for batch in 128 256 512 1024; do
    for arm in wave13 wave14; do
        policy=()
        if [[ $arm == wave14 ]]; then policy=(--policy "$root/policy.json"); fi
        "$py" "$src/full.py" train --batch "$batch" --out "$root/${arm}-train-b$batch.json" \
            "${policy[@]}" 2>&1 | tee "$root/${arm}-train-b$batch.log"
    done
done
"$py" "$src/full.py" dynamo --policy "$root/policy.json" --out "$root/dynamo.json"
for arm in wave14 wave13; do
    policy=()
    if [[ $arm == wave14 ]]; then policy=(--policy "$root/policy.json"); fi
    "$py" "$src/full.py" fold --batch 128 --out "$root/${arm}-fold-b128.json" "${policy[@]}"
done
