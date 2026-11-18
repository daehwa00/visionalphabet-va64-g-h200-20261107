#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export RUN_ROOT=/app/output/cand12-kernel-profile-20261118
mkdir -p "$RUN_ROOT"
cd ../..
python3 -m pip install --disable-pip-version-check --target "$RUN_ROOT/bootstrap" uv==0.11.26
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv python install 3.13.11
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv venv --python 3.13.11 "$RUN_ROOT/environment"
PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv pip install --python "$RUN_ROOT/environment/bin/python" -r h200/terminal_rawg/requirements.txt
export PATH="$RUN_ROOT/environment/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
cd h200/cand12
export VA_REAL_MEMORY_SCAN_BATCH_PARTIALS=1 LNET_DISABLE_LAUNCH_AUTOTUNE=0 PYTHONHASHSEED=0
python -c "import torch; assert 'H200' in torch.cuda.get_device_name(), torch.cuda.get_device_name()"
df -h /dev/shm
python -c 'import os; print("CPU_AFFINITY_COUNT", len(os.sched_getaffinity(0)))'
export PYTHONPATH="code-cand12/scripts:code-cand12/src:code-cand12"
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache-cand12"
python -u benchmark_imagenet1k.py --data /app/data/ImageNet-2012 --batch 1024 --loader-batch 512 --workers 16 --kernel-profile trace
ncu_cmd="$(command -v ncu || find /usr/local /opt -name ncu -type f -print -quit 2>/dev/null || true)"
if [[ -z "$ncu_cmd" ]]; then
  echo "NCU_UNAVAILABLE"
else
  metrics=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active
  if "$ncu_cmd" --profile-from-start off --target-processes all --kernel-name-base function --kernel-name 'regex:(_product_scan_coarse4_associative_backward_kernel|_bidirectional_backward_kernel)' --launch-count 30 --metrics "$metrics" --csv --log-file "$RUN_ROOT/ncu.csv" "$RUN_ROOT/environment/bin/python" -u benchmark_imagenet1k.py --data /app/data/ImageNet-2012 --batch 1024 --loader-batch 512 --workers 16 --kernel-profile counters; then
    cat "$RUN_ROOT/ncu.csv"
  else
    echo "NCU_FAILED"
    tail -30 "$RUN_ROOT/ncu.csv" || true
  fi
fi
