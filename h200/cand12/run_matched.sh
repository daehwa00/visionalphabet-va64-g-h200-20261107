#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export RUN_ROOT=/app/output/cand12-in100-matched-20261118
mkdir -p "$RUN_ROOT"
cd ../..
python3 -m pip install --disable-pip-version-check --target "$RUN_ROOT/bootstrap" uv==0.11.26
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv python install 3.13.11
UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv venv --python 3.13.11 "$RUN_ROOT/environment"
PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv pip install --python "$RUN_ROOT/environment/bin/python" -r h200/terminal_rawg/requirements.txt cryptography==46.0.3
export PATH="$RUN_ROOT/environment/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
cd h200/cand12
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS LNET_DISABLE_LAUNCH_AUTOTUNE
export PYTHONHASHSEED=0 TRITON_CACHE_DIR="$RUN_ROOT/triton"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
sha256sum -c common-fixed.sha256
mkdir -p "$RUN_ROOT/common" "$RUN_ROOT/results"
cp -a code-cand12/. "$RUN_ROOT/common/"
cp -a common/. "$RUN_ROOT/common/"
fixture=${VA_MATCHED_FIXTURE_DIR:-"$RUN_ROOT/private-input"}
if [[ ! -d "$fixture" ]]; then python stage_private_inputs.py "$fixture"; fi
(cd "$fixture" && sha256sum -c "$OLDPWD/matched-inputs.sha256")
affinity=$(python - <<'PY'
import os
from pathlib import Path
def key(cpu):
    p=Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return tuple((p/f).read_text().strip() for f in ["physical_package_id","core_id"])
seen={}
for n in sorted(os.sched_getaffinity(0)):seen.setdefault(key(n), n)
print(",".join(map(str,list(seen.values())[:8])))
PY
)
echo "MATCHED_AFFINITY=$affinity"
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,clocks.max.sm,memory.total --format=csv
run() {
    local arm=$1 tag=$2
    shift 2
    local tree="$RUN_ROOT/common" entry=run_matched_trial.py
    if [[ $arm == cand12 ]]; then tree="$PWD/code-cand12"; entry="${MATCHED_ENTRY:-benchmark_matched.py}"; fi
    echo "BEGIN $tag"
    ARM="$arm" LNET_LAUNCH_CACHE="$RUN_ROOT/store-$arm" PYTHONPATH="$tree/scripts:$tree/src:$tree" TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache-$arm" \
      taskset -c "$affinity" python "$entry" --checkpoint "$fixture/checkpoint.pt" --contract contract.json --images "$fixture/images.pt" \
      --output "$RUN_ROOT/results/$tag.json" --batch 128 --cpu-threads 4 --spatial --vector --reset-each-step --warmup 5 --steps 10 --repeats 5 "$@" \
      > "$RUN_ROOT/results/$tag.log" 2>&1 || { tail -12 "$RUN_ROOT/results/$tag.log"; return 1; }
}
r="$RUN_ROOT/results"
run cand12 reference --profile
for side in discover matched h8w4 h4w1; do
    python - "$RUN_ROOT/cache-cand12" "$RUN_ROOT/cache-$side" <<'COPY'
import shutil,sys
shutil.copytree(sys.argv[1],sys.argv[2],ignore=shutil.ignore_patterns("aotautograd","fxgraph","locks"))
COPY
done
run discover discovery
MATCHED_ENTRY=check_matched_forward.py VA_FORWARD_CONTROL=actual-ref run cand12 actual-ref
for side in matched h8w4 h4w1; do
    run "$side" "screen-$side" --profile
    python summarize_matched.py "$r/screen-$side.json"
    # Report cross-device agreement independently of same-device validation.
    python summarize_matched.py compare "$r/screen-$side.validation.pt" "$fixture/4090.validation.pt" "cross-$side"
    ok=1
    python summarize_matched.py compare "$r/screen-$side.validation.pt" "$r/reference.validation.pt" "local-$side" --gate || ok=0
    MATCHED_ENTRY=check_matched_forward.py VA_FORWARD_CONTROL=actual-ref run "$side" "actual-$side" --reference "$r/actual-ref.json" || ok=0
    if [[ $ok == 1 ]]; then
        echo "GATED $side yes"
    else
        echo "GATED $side no"
        # Measure the required fixed-workload cross-device comparison even if
        # native numerics differ; such a trial must never be registered.
        if [[ $side != matched ]]; then continue; fi
    fi
    comparator=matched
    if [[ $side == matched ]]; then comparator=cand12; fi
    for i in 1 2 3; do
        order=("$comparator" "$side")
        if [[ $i == 2 ]]; then order=("$side" "$comparator"); fi
        for which in "${order[@]}"; do
            profile=()
            if [[ $i != 2 ]]; then profile+=(--profile); fi
            run "$which" "$side-p$i-$which" "${profile[@]}"
            python summarize_matched.py "$r/$side-p$i-$which.json"
            python summarize_matched.py compare "$r/$side-p$i-$which.validation.pt" "$r/reference.validation.pt" "pair-$side-$i-$which"
        done
    done
done
python - "$r" <<'PY'
from pathlib import Path
import json,sys
for path in Path(sys.argv[1]).glob("actual-*.json"):
    d=json.loads(path.read_text())
    print("ACTUAL_LR",path.stem,json.dumps({g:{"count":len(rows),"failed":sum(not v["passed"] for v in rows.values())} for g,rows in d.get("gradient_checks",{}).items()}),json.dumps({k:v.get("passed") for k,v in d.get("outputs",{}).items()}))
PY
