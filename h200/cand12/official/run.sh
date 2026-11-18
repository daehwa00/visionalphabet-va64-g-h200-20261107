#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_ROOT=/app/output/cand12-shared-dot-coarse8-20261118
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
if [[ ! -d "$fixture" ]]; then python official/stage.py "$fixture"; fi
(cd "$fixture" && sha256sum -c "$OLDPWD/official/inputs.sha256")
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
BASE_VARIANT=$(cat official/baseline.txt); export BASE_VARIANT
cp -a replay/known-overlay/. "$RUN_ROOT/common/"
cp -a segment/overlay/. "$RUN_ROOT/common/"
cp post714/overlay/src/lnet/validated_segmented_scans.json "$RUN_ROOT/common/src/lnet/"
for arm in control coarse direct both donor; do
    tree="$RUN_ROOT/tree-$arm"
    cp -a "$RUN_ROOT/common" "$tree"
    cp -a official/overlay/. "$tree/"
    (cd "$tree" && sha256sum --quiet -c "$OLDPWD/official/fixed.sha256")
done
run() {
    local arm=$1 label=$2 phase=$3
    shift 3
    local tree="$RUN_ROOT/tree-$arm" entry=official/trial.py
    if [[ $arm == seed ]]; then tree="$PWD/code-cand12"; entry=benchmark_matched.py; fi
    out="$RUN_ROOT/results/$label"
    ARM="$arm" PHASE="$phase" PYTHONPATH="$tree/scripts:$tree/src:$tree:$PWD" TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache-$arm" LNET_LAUNCH_CACHE="$RUN_ROOT/empty-$arm" \
      taskset -c "$affinity" python "$entry" --checkpoint "$fixture/checkpoint.pt" --contract contract.json --images "$fixture/images.pt" \
      --output "$out.json" --batch 128 --cpu-threads 4 --spatial --vector --reset-each-step --warmup 5 --steps 10 --repeats 5 "$@" > "$out.log" 2>&1 || { tail -15 "$out.log" | cut -c1-1100; return 1; }
    grep -E '^RECORD |^TRAIN_LAUNCHES|^OBSERVED_DOT|^DONOR_LOWER_VERIFIED' "$out.log" >> "$RUN_ROOT/measurements.txt" || true
}
r="$RUN_ROOT/results"
emit() { python replay/evidence.py train > "$RUN_ROOT/training-summary.txt" || true; python post714/emit.py "$RUN_ROOT"; }
trap emit EXIT
run seed leaf-seed run
for arm in control direct coarse both donor; do
    python - "$RUN_ROOT/cache-seed" "$RUN_ROOT/cache-$arm" <<'COPY'
import shutil,sys
shutil.copytree(sys.argv[1],sys.argv[2],ignore=shutil.ignore_patterns('fxgraph','aotautograd','locks'))
COPY
done
run direct observation observe --warmup 1 --steps 1 --repeats 1
run control reference run --profile
MATCHED_ENTRY=check_matched_forward.py VA_FORWARD_CONTROL=actual-ref run control actual-ref run
pair() {
    local arm=$1 cmp=$2 series=$3 passed=1
    for i in 1 2 3; do
        order=("$cmp" "$arm")
        if [[ $i == 2 ]]; then order=("$arm" "$cmp"); fi
        for which in "${order[@]}"; do
            profile=();if [[ $i != 2 ]];then profile+=(--profile);fi
            run "$which" "$series-p$i-$which" run "${profile[@]}" || return 1
        done
        python summarize_matched.py compare "$r/$series-p$i-$arm.validation.pt" "$r/$series-p$i-$cmp.validation.pt" "pair-$series-$i" --gate >> "$RUN_ROOT/measurements.txt" || passed=0
    done
    ((passed))
}
qualify() {
    local arm=$1
    run "$arm" "package-$arm" package || return 1
    run "$arm" "screen-$arm" run --profile || return 1
    python summarize_matched.py compare "$r/screen-$arm.validation.pt" "$r/reference.validation.pt" "screen-$arm" --gate >> "$RUN_ROOT/measurements.txt" || return 1
    MATCHED_ENTRY=check_matched_forward.py VA_FORWARD_CONTROL=actual-ref run "$arm" "actual-$arm" run --reference "$r/actual-ref.json" || return 1
    pair "$arm" control "$arm" || return 1
    touch "$RUN_ROOT/checked-$arm"
}
for arm in direct coarse both donor; do
    echo "QUALIFY $arm"
    if ! qualify "$arm";then echo "UNQUALIFIED $arm";fi
done
best=$(python official/choose.py "$RUN_ROOT")
if [[ -n "$best" ]]; then
    echo "RUNOFF $best"
    pair both "$best" "runoff-$best" || echo RUNOFF_NUMERICAL_FAIL
fi

if [[ -f "$r/actual-donor.validation.pt" ]]; then
    python summarize_matched.py compare "$r/actual-donor.validation.pt" "$fixture/official4090.validation.pt" cross-4090-checkpoint-lr >> "$RUN_ROOT/measurements.txt"
fi
