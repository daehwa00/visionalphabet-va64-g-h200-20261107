#!/usr/bin/env bash
set -euo pipefail

study_root=${1:?study root is required}
data_root=${2:?data root is required}
python_bin=${3:?python executable is required}
batch_size=${STUDY_BATCH_SIZE:-64}
runtime=$study_root/runtime
results=$study_root/results
checkpoint=$study_root/inputs/K128-P160-160-160-128-D2262__seed501.pt
log=$study_root/finish.log

exec >>"$log" 2>&1
exec 9>"$study_root/gpu.lock"
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

family_complete() {
  [ -f "$1" ] && "$python_bin" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)' \
    "$1"
}

for specification in \
  poles:F04_pole_targeted \
  paths:F05_d4_paths \
  branches:F06_transition_decomposition \
  rank:F07_rank_capacity \
  q:F08_terminal_q; do
  group=${specification%%:*}
  family=${specification#*:}
  attempt=0
  while ! family_complete "$results/families/$family.json"; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt 40 ]; then
      echo "[$(date -Is)] exceeded retry budget for $group"
      exit 1
    fi
    echo "[$(date -Is)] evaluation group=$group attempt=$attempt"
    "$python_bin" scripts/run_r2k3_frozen_study_evaluation.py \
      --checkpoint "$checkpoint" --data-root "$data_root" --root "$results" \
      --device cuda --batch-size "$batch_size" --workers 8 \
      --intervention-per-class 50 --random-repeats 20 \
      --bootstrap-repeats 10000 --max-new-conditions 5 --group "$group"
  done
done

echo "[$(date -Is)] representation geometry"
"$python_bin" scripts/run_r2k3_frozen_study_representations.py \
  --checkpoint "$checkpoint" --data-root "$data_root" --root "$results" \
  --device cuda --batch-size "$batch_size" --workers 8 --train-bank-per-class 10

echo "[$(date -Is)] ERF and attribution"
attempt=0
while ! family_complete "$results/families/F10_erf_attribution.json"; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 12 ]; then
    echo "[$(date -Is)] exceeded ERF retry budget"
    exit 1
  fi
  "$python_bin" scripts/run_r2k3_frozen_study_erf.py \
    --checkpoint "$checkpoint" --data-root "$data_root" --root "$results" \
    --device cuda --per-class 1 --ig-images 20 --ig-steps 16 \
    --occlusion-images 20 --occlusion-grid 7 --max-new-conditions 1
done

echo "[$(date -Is)] robustness and calibration"
attempt=0
while [ ! -f "$results/families/F11_robustness_invariance.json" ]; do
  attempt=$((attempt + 1))
  if [ "$attempt" -gt 20 ]; then
    echo "[$(date -Is)] exceeded robustness retry budget"
    exit 1
  fi
  "$python_bin" scripts/run_r2k3_frozen_study_robustness.py \
    --checkpoint "$checkpoint" --data-root "$data_root" --root "$results" \
    --device cuda --batch-size "$batch_size" --workers 8 \
    --bootstrap-repeats 10000 --max-new-conditions 5
done

echo "[$(date -Is)] report and validation"
"$python_bin" scripts/render_r2k3_checkpoint_study_report.py --root "$results"
"$python_bin" scripts/validate_r2k3_checkpoint_study.py \
  --root "$results" \
  --checkpoint-sha256 0f5d65e667ddd49f9c68c16db5a16d873631549ab773ef7acf8d6db5dab94f46 \
  --finalize \
  >"$study_root/validation.json"

echo "[$(date -Is)] finish pipeline complete"
