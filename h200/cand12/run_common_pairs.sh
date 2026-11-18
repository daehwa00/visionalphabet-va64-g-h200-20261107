#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export RUN_ROOT=/app/output/cand12-common-layout-20261118
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
mkdir -p "$RUN_ROOT/common"
cp -a code-cand12/. "$RUN_ROOT/common/"
cp -a common/. "$RUN_ROOT/common/"
export TRITON_CACHE_DIR="$RUN_ROOT/triton"
bench() {
    local arm=$1 tag=$2
    shift 2
    local tree="$RUN_ROOT/common" entry=common_trial.py
    if [[ $arm == control ]]; then tree="$PWD/code-cand12"; entry=benchmark_imagenet1k.py; fi
    echo "BEGIN $tag"
    ARM="$arm" LNET_LAUNCH_CACHE="$RUN_ROOT/trials/$arm" PYTHONPATH="$tree/scripts:$tree/src:$tree" TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache-$arm" \
        python -u "$entry" --data /app/data/ImageNet-2012 --batch 1024 --loader-batch 512 --workers 16 "$@" > "$RUN_ROOT/$tag.log" 2>&1 || { tail -8 "$RUN_ROOT/$tag.log"; return 1; }
    grep -E 'LAYOUT_GATE|IMAGENET_PROFILE=|IMAGENET_BENCH=|KERNEL_ROW=.*_product_scan_coarse4_associative_backward_kernel' "$RUN_ROOT/$tag.log" | head -8 || true
}
bench control reference --layout-check seed
# Save a reference real-data measurement even if all candidates fail the gate.
bench control baseline --profile-updates 96
bench control baseline-trace --kernel-profile trace
# Preserve other Inductor kernel choices from cand12, excluding the traced graph.
for arm in discovery common8 common2w; do
    python - "$RUN_ROOT/cache-control" "$RUN_ROOT/cache-$arm" <<'COPY'
import shutil,sys
shutil.copytree(sys.argv[1],sys.argv[2],ignore=shutil.ignore_patterns("aotautograd","fxgraph","locks"))
COPY
done
bench discovery discovery --layout-check compare
passed=()
for arm in common8 common2w; do
    if bench "$arm" "gate-$arm" --layout-check compare; then
        passed+=("$arm")
        bench "$arm" "trace-$arm" --kernel-profile trace
    fi
done
for arm in "${passed[@]}"; do
    for pair in 1 2 3; do
        order=(control "$arm")
        if [[ $pair == 2 ]]; then order=("$arm" control); fi
        for side in "${order[@]}"; do
            bench "$side" "$arm-p$pair-$side" --profile-updates 96
        done
    done
done
python - <<'PY'
import json, os
from pathlib import Path
root=Path(os.environ["RUN_ROOT"])
for p in sorted(root.glob("*.log")):
    for line in p.read_text().splitlines():
        if line.startswith("IMAGENET_BENCH="):
            d=json.loads(line.split("=",1)[1])
            print("PAIR_RESULT",p.stem,round(1024000/d["images_s"],4),round(d["images_s"],2),round(d["peak_mib"],2))
        if line.startswith("KERNEL_ROW="):
            d=json.loads(line.split("=",1)[1])
            if d["kernel"]=="_product_scan_coarse4_associative_backward_kernel" and d["ms_per_update"]>55:
                print("HOT_LAUNCH",p.stem,d)
        if line.startswith(("VERIFIED_LAUNCH","LAYOUT_GATE","OBSERVED_SCOPE")):
            print(line)
PY

for arm in "${passed[@]}"; do
    echo "TRIAL_RECORD $arm $(cat "$RUN_ROOT/trials/$arm/launch_geometry.json" | tr -d '\n')"
done
