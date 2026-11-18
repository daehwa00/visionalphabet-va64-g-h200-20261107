#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_ROOT=/app/output/cand12-b56-replay-20261118
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
if [[ ! -d "$fixture" ]]; then python replay/stage.py "$fixture"; fi
(cd "$fixture" && sha256sum -c "$OLDPWD/replay/inputs.sha256")
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
export FIXTURE="$fixture"
if taskset -c "$affinity" python -u replay/replay.py "$fixture" "$RUN_ROOT/micro" > "$RUN_ROOT/replay.log" 2>&1; then
    python replay/evidence.py micro
else
    tail -12 "$RUN_ROOT/replay.log"
    echo "ISOLATED_REPLAY_FAILED"
fi
bash replay/probe_ncu.sh replay/replay.py "$fixture" "$RUN_ROOT/ncu-test"
cp -a "$RUN_ROOT/common" "$RUN_ROOT/source-combo"
cp -a replay/known-overlay/. "$RUN_ROOT/source-combo/"
python - "$RUN_ROOT/source-combo" <<'PY'
from pathlib import Path
import json,hashlib,sys
root=Path(sys.argv[1]);m=json.loads(Path('replay/known-files.json').read_text())
for name,h in m.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
print('KNOWN_SNAPSHOT_VERIFIED',len(m))
PY
cp -a "$RUN_ROOT/source-combo" "$RUN_ROOT/source-mixed"
if [[ -f "$RUN_ROOT/micro/result.json" ]]; then python replay/make_mixed.py "$RUN_ROOT"; fi
# The exact archived benchmark and all numerical checks run on fresh processes.
run() {
    local arm=$1 label=$2 phase=$3
    shift 3
    local tree="$RUN_ROOT/source-$arm" entry="replay/train_trial.py" store="empty-$arm"
    if [[ $arm == control ]]; then tree="$RUN_ROOT/common"; fi
    if [[ $phase == package ]]; then store="staging-$arm"; fi
    if [[ $arm == seed ]]; then tree="$PWD/code-cand12"; entry=benchmark_matched.py; fi
    local out="$RUN_ROOT/results/$label"
    ARM=$arm TRIAL_PHASE=$phase PYTHONPATH="$tree/scripts:$tree/src:$tree:$PWD" TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache-$arm" LNET_LAUNCH_CACHE="$RUN_ROOT/$store" \
      taskset -c "$affinity" python "$entry" --checkpoint "$fixture/checkpoint.pt" --contract contract.json --images "$fixture/images.pt" \
      --output "$out.json" --batch 128 --cpu-threads 4 --spatial --vector --reset-each-step --warmup 5 --steps 10 --repeats 5 "$@" > "$out.log" 2>&1 || { tail -16 "$out.log"; return 1; }
    grep -E '^H200_RECORD|^TRAIN_LAUNCHES' "$out.log" || true
}
seed_cache() {
    python - "$RUN_ROOT/cache-seed" "$RUN_ROOT/cache-$1" <<'COPY'
import shutil,sys
shutil.copytree(sys.argv[1],sys.argv[2],ignore=shutil.ignore_patterns('aotautograd','fxgraph','locks'))
COPY
}
r="$RUN_ROOT/results"
trap 'python replay/evidence.py train' EXIT
run seed leaf-cache run
seed_cache control
run control discover-control observe
run control package-control package
run control reference run --profile
VA_FORWARD_CONTROL=actual-ref MATCHED_ENTRY=check_matched_forward.py run control actual-ref run
for arm in combo mixed; do
    if [[ $arm == mixed && ( ! -f "$RUN_ROOT/mixed-mode" || ! -f "$RUN_ROOT/gate-combo" ) ]]; then break; fi
    # Inherit only the real H200 final14 fallback scopes. Keep separate leaf and graph caches.
    cp "$RUN_ROOT/common/src/lnet/validated_path_layouts.json" "$RUN_ROOT/source-$arm/src/lnet/"
    seed_cache "$arm"
    run "$arm" "discover-$arm" observe
    run "$arm" "package-$arm" package
    run "$arm" "screen-$arm" run --profile
    if ! python summarize_matched.py compare "$r/screen-$arm.validation.pt" "$r/reference.validation.pt" "screen-$arm" --gate; then echo "REJECT_INITIAL_LR $arm"; continue; fi
    if ! VA_FORWARD_CONTROL=actual-ref MATCHED_ENTRY=check_matched_forward.py run "$arm" "actual-$arm" run --reference "$r/actual-ref.json"; then echo "REJECT_CHECKPOINT_LR $arm"; continue; fi
    touch "$RUN_ROOT/gate-$arm"
    comparator=control
    if [[ $arm == mixed ]]; then comparator=combo; fi
    for i in 1 2 3; do
        order=("$comparator" "$arm")
        if [[ $i == 2 ]]; then order=("$arm" "$comparator"); fi
        for which in "${order[@]}"; do
            profile=()
            if [[ $i != 2 ]]; then profile+=(--profile); fi
            run "$which" "$arm-p$i-$which" run "${profile[@]}"
        done
        python summarize_matched.py compare "$r/$arm-p$i-$arm.validation.pt" "$r/$arm-p$i-$comparator.validation.pt" "pair-$arm-$i" || exit 1
    done
done
