#!/usr/bin/env bash
set -euo pipefail

study_root=${1:?study root is required}
runtime=$study_root/runtime
results=$study_root/results
checkpoint=$study_root/inputs/K128-P160-160-160-128-D2262__seed501.pt
data_root=${2:-/data/ImageNet100}
python_bin=${3:-/home/daehwa/anaconda3/envs/alphabet/bin/python}
log=$study_root/study.log

mkdir -p "$results" "$study_root/inputs"
exec >>"$log" 2>&1

echo "[$(date -Is)] watcher started"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; do
  echo "[$(date -Is)] GPU busy; waiting"
  sleep 30
done

exec 9>"$study_root/gpu.lock"
flock 9
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "[$(date -Is)] GPU became busy before lock; exiting for safe retry"
  exit 75
fi

cd "$runtime"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=8
export LNET_DISABLE_LAUNCH_AUTOTUNE=1

echo "[$(date -Is)] baseline and frozen interventions"
"$python_bin" scripts/run_r2k3_frozen_study_evaluation.py \
  --checkpoint "$checkpoint" --data-root "$data_root" --root "$results" \
  --device cuda --batch-size 128 --workers 8 --intervention-per-class 50 \
  --random-repeats 20 --bootstrap-repeats 10000 --group all

echo "[$(date -Is)] representation geometry"
"$python_bin" scripts/run_r2k3_frozen_study_representations.py \
  --checkpoint "$checkpoint" --data-root "$data_root" --root "$results" \
  --device cuda --batch-size 128 --workers 8 --train-bank-per-class 10

echo "[$(date -Is)] ERF and attribution"
"$python_bin" scripts/run_r2k3_frozen_study_erf.py \
  --checkpoint "$checkpoint" --data-root "$data_root" --root "$results" \
  --device cuda --per-class 1 --ig-images 20 --ig-steps 16 \
  --occlusion-images 20 --occlusion-grid 7

echo "[$(date -Is)] robustness and calibration"
"$python_bin" scripts/run_r2k3_frozen_study_robustness.py \
  --checkpoint "$checkpoint" --data-root "$data_root" --root "$results" \
  --device cuda --batch-size 128 --workers 8 --bootstrap-repeats 10000

echo "[$(date -Is)] report and validation"
"$python_bin" scripts/render_r2k3_checkpoint_study_report.py --root "$results"
"$python_bin" scripts/validate_r2k3_checkpoint_study.py \
  --root "$results" \
  --checkpoint-sha256 0f5d65e667ddd49f9c68c16db5a16d873631549ab773ef7acf8d6db5dab94f46 \
  >"$study_root/validation.json"

echo "[$(date -Is)] study complete"
