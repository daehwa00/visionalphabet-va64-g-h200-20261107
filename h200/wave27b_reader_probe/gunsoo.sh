#!/usr/bin/env bash
set -eu

cd /tmp/va-t95-gunsoo
py="$HOME/miniconda3/envs/alphabet/bin/python"
probe=h200/wave27b_reader_probe
export PYTHONPATH=src:scripts:h200/wave17b_vec_reader_probe:h200/wave25b_reader_probe
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LNET_READER_W25B=auto LNET_READER_VEC=auto
status=0
(
    flock -w 2400 9 || exit 1
    "$py" "$probe/gpu_probe.py" parity --out parity.jsonl >parity.log 2>&1 || status=1
    for arm in auto dweight_2cta scalar_hiocc full_hopper; do
        LNET_READER_W27B="$arm" "$py" -m pytest -q tests/test_factorized_reader_kernels.py \
            >"pytest-$arm.log" 2>&1 || status=1
    done
    "$py" "$probe/gpu_probe.py" bench --out timing.jsonl >timing.log 2>&1 || status=1
    exit "$status"
) 9>/tmp/va-gpu.lock
