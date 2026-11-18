#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_ROOT=/app/output/cand12-four-row-20261118
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
if [[ ! -d "$fixture" ]]; then python segment/stage.py "$fixture"; fi
(cd "$fixture" && sha256sum -c "$OLDPWD/segment/inputs.sha256")
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
cp -a replay/known-overlay/. "$RUN_ROOT/common/"
cp -a segment/overlay/. "$RUN_ROOT/common/"
python - "$RUN_ROOT/common" <<'PY'
from pathlib import Path
import json,sys,hashlib
m=json.loads(Path("segment/source.json").read_text());r=Path(sys.argv[1])
for n,h in m.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
print("SOURCE_VERIFIED",len(m))
PY
for arm in control direct coarse both; do cp -a "$RUN_ROOT/common" "$RUN_ROOT/tree-$arm"; done
run() {
    local arm=$1 label=$2 phase=$3
    shift 3
    local tree="$RUN_ROOT/tree-$arm" entry="segment/trial.py"
    if [[ $arm == seed ]]; then tree="$PWD/code-cand12"; entry=benchmark_matched.py; fi
    local out="$RUN_ROOT/results/$label"
    ARM=$arm PHASE=$phase PYTHONPATH="$tree/scripts:$tree/src:$tree:$PWD" TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache-$arm" LNET_LAUNCH_CACHE="$RUN_ROOT/empty-$arm" \
        taskset -c "$affinity" python "$entry" --checkpoint "$fixture/checkpoint.pt" --contract contract.json --images "$fixture/images.pt" \
        --output "$out.json" --batch 128 --cpu-threads 4 --spatial --vector --reset-each-step --warmup 5 --steps 10 --repeats 5 "$@" > "$out.log" 2>&1 || { tail -18 "$out.log"; return 1; }
    grep -E '^CANDIDATE_RECORD|^TRAIN_LAUNCHES|^OBSERVED' "$out.log" || true
}
r="$RUN_ROOT/results"
trap 'python replay/evidence.py train' EXIT
run seed preseed run
for arm in control direct coarse both; do
    python - "$RUN_ROOT/cache-seed" "$RUN_ROOT/cache-$arm" <<'PY'
import shutil,sys
shutil.copytree(sys.argv[1],sys.argv[2],ignore=shutil.ignore_patterns("aotautograd","fxgraph","locks"))
PY
done
run control observed observe
run control control-record package
run control reference run --profile
MATCHED_ENTRY=check_matched_forward.py VA_FORWARD_CONTROL=actual-ref run control actual-ref run
paired() {
    local arm=$1 comparator=$2 series=$3
    for i in 1 2 3; do
        order=("$comparator" "$arm")
        if [[ $i == 2 ]]; then order=("$arm" "$comparator"); fi
        for which in "${order[@]}"; do
            prof=()
            if [[ $i != 2 ]]; then prof+=(--profile); fi
            run "$which" "$series-p$i-$which" run "${prof[@]}" || return 1
        done
        python summarize_matched.py compare "$r/$series-p$i-$arm.validation.pt" "$r/$series-p$i-$comparator.validation.pt" "pair-$series-$i" --gate || echo "PAIR_NUMERICAL_FAIL $series $i"
    done
}
for arm in direct coarse both; do
    if ! run "$arm" "$arm-record" package || ! run "$arm" "screen-$arm" run --profile; then echo "UNAVAILABLE $arm"; continue; fi
    python summarize_matched.py compare "$r/screen-$arm.validation.pt" "$r/reference.validation.pt" "screen-$arm" --gate || echo "SCREEN_NUMERICAL_FAIL $arm"
    MATCHED_ENTRY=check_matched_forward.py VA_FORWARD_CONTROL=actual-ref run "$arm" "actual-$arm" run --reference "$r/actual-ref.json" || echo "ACTUAL_NUMERICAL_FAIL $arm"
    paired "$arm" control "$arm" || echo "PAIR_RUN_FAIL $arm"
done
best=$(python segment/runoff.py "$r")
if [[ -n "$best" ]]; then
    echo "RUNOFF $best both"
    paired both "$best" "runoff-$best" || echo "RUNOFF_FAIL"
fi
