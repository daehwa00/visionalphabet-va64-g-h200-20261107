#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_ROOT=/app/output/cand12-post714-direct-20261118
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
if [[ ! -d "$fixture" ]]; then python post714/stage.py "$fixture"; fi
(cd "$fixture" && sha256sum -c "$OLDPWD/post714/inputs.sha256")
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
# Correct the device's base to the direct-only deployment qualified by #714.
cp post714/overlay/src/lnet/validated_segmented_scans.json "$RUN_ROOT/common/src/lnet/"
python - "$RUN_ROOT/common" <<'PY'
import json,sys
from pathlib import Path
r=json.loads((Path(sys.argv[1])/"src/lnet/validated_segmented_scans.json").read_text())['records']['NVIDIA_H200_NVL-sm90']['retained_four_row_prefix']
assert len(r)==1 and json.loads(next(iter(r)))['shape']['coarse']==0
PY
for arm in tree native outdot; do
    args=(--summary "$arm")
    if [[ $arm == outdot ]]; then args=(--mlp outdot --role direct); fi
    python post714/materialize_common.py "$RUN_ROOT/tree-$arm" --base "$RUN_ROOT/common" "${args[@]}"
done
for arm in control rows2 rows8 buckets deploy; do
    cp -a "$RUN_ROOT/common" "$RUN_ROOT/tree-$arm"
    cp -a post714/overlay/. "$RUN_ROOT/tree-$arm/"
done
python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import json,hashlib,sys
r=Path(sys.argv[1]);d=json.loads(Path("post714/generated-source.json").read_text())
for name,h in d.items():
    p=Path("post714/variants.py") if name=="generator" else r/("tree-"+name)/"src/lnet"/("direct_path_scan_kernels.py" if name=="outdot" else "segmented_path_scan.py")
    assert hashlib.sha256(p.read_bytes()).hexdigest()==h,(name,p)
    print("PINNED_SOURCE",name,h)
PY
run() {
    local arm=$1 label=$2 phase=$3
    shift 3
    local id=$arm portable=0
    if [[ "${DEPLOY:-0}" == 1 ]]; then id=deploy; portable=1; fi
    local tree="$RUN_ROOT/tree-$id" entry=post714/trial.py
    if [[ $arm == seed ]]; then tree="$PWD/code-cand12"; entry=benchmark_matched.py; fi
    local out="$RUN_ROOT/results/$label"
    ARM="$arm" PORTABLE="$portable" PHASE="$phase" PYTHONPATH="$tree/scripts:$tree/src:$tree:$PWD" TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache-$id" LNET_LAUNCH_CACHE="$RUN_ROOT/empty-$id" \
        taskset -c "$affinity" python "$entry" --checkpoint "$fixture/checkpoint.pt" --contract contract.json --images "$fixture/images.pt" \
        --output "$out.json" --batch 128 --cpu-threads 4 --spatial --vector --reset-each-step --warmup 5 --steps 10 --repeats 5 "$@" > "$out.log" 2>&1 || { tail -14 "$out.log" | cut -c1-1000; return 1; }
    grep -E '^CANDIDATE_RECORD|^SOURCE_FINGERPRINT|^TRAIN_LAUNCHES|^TOP_KERNELS' "$out.log" >> "$RUN_ROOT/measurements.txt" || true
}
r="$RUN_ROOT/results"
emit() {
    python replay/evidence.py train > "$RUN_ROOT/training-summary.txt" || true
    python post714/emit.py "$RUN_ROOT"
}
trap emit EXIT
run seed seed run
for arm in control tree native outdot rows2 rows8 buckets deploy; do
    python - "$RUN_ROOT/cache-seed" "$RUN_ROOT/cache-$arm" <<'PY'
import shutil,sys
shutil.copytree(sys.argv[1],sys.argv[2],ignore=shutil.ignore_patterns('aotautograd','fxgraph','locks'))
PY
done
run control observation-control observe
run control reference run --profile
MATCHED_ENTRY=check_matched_forward.py VA_FORWARD_CONTROL=actual-ref run control actual-ref run
qualify() {
    local arm=$1 name=$2
    if [[ "${DEPLOY:-0}" == 1 ]]; then
        cp "$RUN_ROOT/control-observed.json" "$RUN_ROOT/$arm-portable-observed.json" || return 1
    elif [[ $arm == rows2 || $arm == rows8 || $arm == buckets ]]; then
        cp "$RUN_ROOT/control-observed.json" "$RUN_ROOT/$arm-observed.json" || return 1
    else
        run "$arm" "obs-$name" observe --warmup 1 --steps 1 --repeats 1 || return 1
    fi
    run "$arm" "package-$name" package || return 1
    run "$arm" "screen-$name" run --profile || return 1
    python summarize_matched.py compare "$r/screen-$name.validation.pt" "$r/reference.validation.pt" "screen-$name" --gate >> "$RUN_ROOT/measurements.txt" || return 1
    MATCHED_ENTRY=check_matched_forward.py VA_FORWARD_CONTROL=actual-ref run "$arm" "actual-$name" run --reference "$r/actual-ref.json" || return 1
    local gates=1
    for i in 1 2 3; do
        order=(control "$arm")
        if [[ $i == 2 ]]; then order=("$arm" control); fi
        for which in "${order[@]}"; do
            prof=()
            if [[ $i != 2 ]]; then prof+=(--profile); fi
            if [[ $which == control ]]; then
                DEPLOY=0 run control "$name-p$i-control" run "${prof[@]}" || return 1
            else
                run "$arm" "$name-p$i-$arm" run "${prof[@]}" || return 1
            fi
        done
        python summarize_matched.py compare "$r/$name-p$i-$arm.validation.pt" "$r/$name-p$i-control.validation.pt" "pair-$name-$i" --gate >> "$RUN_ROOT/measurements.txt" || gates=0
    done
    ((gates)) || return 1
    touch "$RUN_ROOT/eligible-$name"
}
for arm in outdot tree native rows2 rows8 buckets; do
    echo "QUALIFY $arm"
    if ! qualify "$arm" "$arm"; then echo "GATE_OR_RUN_FAILED $arm"; fi
done
winner=$(python post714/choose.py "$RUN_ROOT")
if [[ -n "$winner" ]]; then
    echo "DEPLOY_TRIAL $winner"
    if ! DEPLOY=1 qualify "$winner" deployed; then echo "DEPLOY_FAILED $winner"; fi
fi
