#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=/app/output/in1k-va-pico-v11-600-opt-20261127
mkdir -p "$root/logs"
exec > >(tee -a "$root/logs/launch-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
exec 9>"$root/train.lock"
flock -n 9
envdir=$(mktemp -d)
python3 -m pip install --disable-pip-version-check --target "$envdir/bootstrap" uv==0.11.26
export UV_PYTHON_INSTALL_DIR="$envdir/python" PYTHONPATH="$envdir/bootstrap"
python3 -m uv python install 3.13.11
python3 -m uv venv --python 3.13.11 "$envdir/venv"
python3 -m uv pip install --python "$envdir/venv/bin/python" -r requirements.txt
export PATH="$envdir/venv/bin:$PATH" PYTHONPATH="$PWD:$PWD/../../scripts:$PWD/../../src"
export ALPHABET_ALLOW_FP8=0 ALPHABET_SCAN_SPLIT_BF16=1 LNET_CUDAGRAPHS_ACTIVE=1
unset ALPHABET_SCAN_SPLIT_FP8 ALPHABET_SCAN_SMALL_SPATIAL_FP8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
c++ -O3 -std=c++17 -fPIC -shared imagenet_uint8_bicubic.cpp -o imagenet_uint8_bicubic.so
workers=$(python -c 'import os; print(max(1,min(24,len(os.sched_getaffinity(0))-2)))')
python -c 'import torch; assert torch.cuda.get_device_capability()==(9,0) and "H200" in torch.cuda.get_device_name()'
exec python -u train.py --root "$root" --data-root /app/data/ImageNet-2012 \
  --epochs 600 --workers "$workers" --prefetch-factor 2 --batch-size 512 \
  --gradient-accumulation-steps 2 --compact-manifest --clean-loader-workers \
  --fast-gpu-augment --tensor-crop --gpu-randaugment --compile --precision bfloat16 \
  --eval-every 10 --eval-dense-from 580 --checkpoint-every 20 --progress-every 250 \
  --no-val-persistent-workers --dual-eval --resume
