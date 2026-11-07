#!/usr/bin/env bash
set -euo pipefail

base=${1:-/home/qai/experiments/r2k3-k128-d2262-checkpoint-study-20260901}
output=${2:-/home/qai/experiments/r2k3-k128-d2262-followup-20260904}
python_bin=${3:-/home/qai/miniconda3/envs/alphabet/bin/python}
runtime=$base/runtime
results=$output/results
checkpoint=$base/inputs/K128-P160-160-160-128-D2262__seed501.pt
data_root=$base/data/ImageNet100
log=$output/followup.log

mkdir -p "$output" "$results"
exec >>"$log" 2>&1
exec 9>"$output/gpu.lock"
flock 9

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "[$(date -Is)] GPU busy; refusing to interfere"
  exit 75
fi

cd "$runtime"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=8
export LNET_DISABLE_LAUNCH_AUTOTUNE=1

json_complete() {
  [ -f "$1" ] && "$python_bin" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)' \
    "$1"
}

echo "[$(date -Is)] P0-3 actual-input grating"
attempt=0
grating_family=$results/followups/P0_3_actual_input_grating.json
while [ ! -f "$grating_family" ]; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 30 ]; then
    echo "[$(date -Is)] grating retry budget exceeded"
    exit 1
  fi
  "$python_bin" scripts/run_r2k3_followup_grating.py \
    --checkpoint "$checkpoint" --root "$results" --device cuda \
    --batch-size 16 --permutations 2000 --max-new-chunks 4 || true
done

echo "[$(date -Is)] P0-4 rotation-path consistency"
rotation_family=$results/families/P0-4_input_rotation_d4_path_consistency.json
if [ ! -f "$rotation_family" ]; then
  "$python_bin" scripts/run_r2k3_followup_rotation_paths.py \
    --checkpoint "$checkpoint" --data-root "$data_root" \
    --output-root "$results" --device cuda --per-class 1 \
    --batch-size 4 --workers 4
fi

echo "[$(date -Is)] P0-5 memory-carry scale grid"
branch_family=$results/families/P05_memory_carry_scale_sweep.json
attempt=0
while ! json_complete "$branch_family"; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 12 ]; then
    echo "[$(date -Is)] branch-grid retry budget exceeded"
    exit 1
  fi
  "$python_bin" scripts/run_r2k3_followup_branch_random.py \
    --checkpoint "$checkpoint" --data-root "$data_root" \
    --source-study-root "$base/results" --root "$results" \
    --device cuda --batch-size 64 --workers 8 \
    --bootstrap-repeats 10000 --hierarchical-repeats 10000 \
    --max-new-conditions 5 --group branches
done

echo "[$(date -Is)] P0-2 exact periodic state reset"
state_root=$results/state_reset
state_family=$state_root/state_reset.json
attempt=0
while ! json_complete "$state_family"; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 12 ]; then
    echo "[$(date -Is)] state-reset retry budget exceeded"
    exit 1
  fi
  "$python_bin" scripts/run_r2k3_followup_state_reset.py \
    --checkpoint "$checkpoint" --data-root "$data_root" \
    --base-study-root "$base/results" --root "$state_root" \
    --device cuda --batch-size 32 --workers 8 --per-class 50 \
    --bootstrap-repeats 10000 --max-new-conditions 1
done

echo "[$(date -Is)] P0-6 matched-random pole expansion"
random_family=$results/families/P06_pole_matched_random_expansion.json
attempt=0
while ! json_complete "$random_family"; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 70 ]; then
    echo "[$(date -Is)] random-control retry budget exceeded"
    exit 1
  fi
  "$python_bin" scripts/run_r2k3_followup_branch_random.py \
    --checkpoint "$checkpoint" --data-root "$data_root" \
    --source-study-root "$base/results" --root "$results" \
    --device cuda --batch-size 64 --workers 8 \
    --bootstrap-repeats 10000 --hierarchical-repeats 10000 \
    --random-repeats-per-contract 100 --max-new-conditions 5 --group random
done

echo "[$(date -Is)] all no-training P0 follow-ups complete"
