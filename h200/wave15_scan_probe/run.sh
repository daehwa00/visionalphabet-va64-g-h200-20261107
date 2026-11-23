#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
suite=h200/wave15_scan_probe
repo=$PWD
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS
unset LNET_DISABLE_LAUNCH_AUTOTUNE LNET_NR4_FUSED LNET_VSCAN_DECOUPLED
unset VA_WAVE15_LAYOUT VA_WAVE15_SCRATCH VA_WAVE15_SINGLE_STORE
unset VA_WAVE15_DROP_SOURCE VA_WAVE15_SHARED_HIDDEN
model_arms=(wave13 wave14 row column)
if [[ ${VA_WAVE15_H200_REFINED:-0} == 1 ]]; then
    model_arms=(wave13 wave14 column-shared2)
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/va-t59-h200.XXXXXX")
export RUN_ROOT PERSIST_ROOT
RUN_ROOT=$work/results
PERSIST_ROOT=$(mktemp -d /app/output/va-t59-h200.XXXXXX)
mkdir -p "$RUN_ROOT"/{arms,infer}
echo "Wave 15A output $PERSIST_ROOT"
finish() {
    local status=$?
    trap - EXIT
    cp -a "$RUN_ROOT/." "$PERSIST_ROOT/"
    python3 h200/liverms_v2_opt/export_result.py "$status"
    rm -rf -- "$work"
    exit "$status"
}
trap finish EXIT
ready() {
    test -x "$1" && "$1" -c 'import sys, torch, torchvision, timm, pytest; raise SystemExit(sys.version_info < (3, 13) or not torch.cuda.is_available())'
}

run_python=
for candidate in /app/output/va-h200-{effective1024-r3-combined,r4-marker-fix}-20261119/environment/bin/python "$(command -v python3)"; do
    if ready "$candidate"; then
        run_python=$candidate
        break
    fi
done
if [[ -z $run_python ]]; then
    export UV_PYTHON_INSTALL_DIR="$work/python" UV_CACHE_DIR="$work/uv-cache"
    python3 -m pip install -q --target "$work/bootstrap" uv==0.11.26
    uv_run() { PYTHONPATH="$work/bootstrap" python3 -m uv "$@"; }
    uv_run python install 3.13.11
    uv_run venv --python 3.13.11 "$work/environment"
    run_python="$work/environment/bin/python"
    uv_run pip install --python "$run_python" --index https://download.pytorch.org/whl/cu128         torch==2.9.1 torchvision==0.24.1 triton==3.5.1 numpy==2.4.6 pytest==9.0.3 timm==1.0.26
    ready "$run_python"
fi
export PYTHONPATH="$repo/src:$repo/scripts"
export TORCHINDUCTOR_CACHE_DIR="$work/inductor"
wave13=fb4a33c76335c369c5f0dfbc0d8fc57dcafacea5
wave14=42ff1ee5170892b94d44a4a17106af28b71400ab
for revision in "$wave13" "$wave14"; do
    mkdir "$work/$revision"
    git archive "$revision" src scripts | tar -x -C "$work/$revision"
done
"$run_python" "$suite/bench.py" --batch 1024 --repeats 5 --out "$RUN_ROOT/ablation.jsonl"     | tee "$RUN_ROOT/ablation.log"
if [[ ${VA_WAVE15_H200_REFINED:-0} == 1 ]]; then
    "$run_python" "$suite/plans.py" > "$RUN_ROOT/occupancy.jsonl"
    "$run_python" "$suite/bench.py" --batch 1024 --reference-only         --arms baseline row column column-shared2 --out "$RUN_ROOT/torch-reference.jsonl"
fi
for revision in "$wave13" "$wave14"; do
    PYTHONPATH="$work/$revision/src:$work/$revision/scripts"         "$run_python" "$suite/bench.py" --batch 1024 --arms baseline         --out "$RUN_ROOT/$revision.jsonl" | tee "$RUN_ROOT/$revision.log"
done
"$run_python" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path
for path in Path(sys.argv[1]).glob("*.jsonl"):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        for arm in row.get("results", {}).values():
            arm.pop("sampled_parity", None)
            for plan in arm["plan"].values():
                plan.pop("kernel", None)
    path.with_suffix(".json").write_text(json.dumps(rows))
PY

bench=h200/liverms_v2_opt/bench_h200.py
variant=17-p1-nr4-1024d-v3
status=0
for batch in 128 256 512 1024; do
    for arm in "${model_arms[@]}"; do
        shared=0
        case $arm in
            wave13) checkout="$work/$wave13"; layout=0 ;;
            wave14) checkout="$work/$wave14"; layout=0 ;;
            row) checkout="$repo"; layout=1 ;;
            column) checkout="$repo"; layout=2 ;;
            column-shared2) checkout="$repo"; layout=2; shared=2 ;;
        esac
        PYTHONPATH="$checkout/src:$checkout/scripts" VA_WAVE15_LAYOUT=$layout VA_WAVE15_SHARED_HIDDEN=$shared             "$run_python" "$bench" train-arm --variant "$variant" --batch "$batch" --profile             --out "$RUN_ROOT/arms/b${batch}_${arm}.json" || status=1
    done
done
export VA_WAVE15_LAYOUT=2
if [[ ${VA_WAVE15_H200_REFINED:-0} == 1 ]]; then
    export VA_WAVE15_SHARED_HIDDEN=2
fi
"$run_python" "$bench" dynamo --out "$RUN_ROOT/infer/dynamo.json" || status=1
"$run_python" "$bench" infer-fold --variant "$variant" --batch 1024     --out "$RUN_ROOT/infer/fold.json" || status=1
pytest_status=0
started=$SECONDS
timeout --foreground --kill-after=5s 13m "$run_python" -m pytest -q     tests/test_general_cuda_scans.py tests/test_product_scan_pipeline.py tests/test_p1_nr4.py     > "$RUN_ROOT/pytest_targeted.log" 2>&1 || pytest_status=$?
"$run_python" "$bench" wrap-pytest --out "$RUN_ROOT/pytest_targeted.json"     --log "$RUN_ROOT/pytest_targeted.log" --exit-code "$pytest_status" --elapsed "$((SECONDS-started))"
if [[ $pytest_status != 0 ]]; then
    status=1
fi
"$run_python" - "$RUN_ROOT" "${#model_arms[@]}" <<'PY_CHECK' || status=1
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = [json.loads(path.read_text()) for path in (root / "arms").glob("*.json")]
assert len(arms) == 4 * int(sys.argv[2])
assert all(row["status"] == "ok" and row["finite_losses"] and row["finite_gradients"] for row in arms)
dynamo = json.loads((root / "infer/dynamo.json").read_text())
assert [pair for rows in dynamo["graphs"].values() for pair in rows.values()] == [[1, 0]] * 18
fold = json.loads((root / "infer/fold.json").read_text())
assert fold["fp32_head_parity_by_precision"]["highest"]["passed"]
assert fold["strict_reload_passed"] and fold["folded_train_guard"]
pytest = json.loads((root / "pytest_targeted.json").read_text())
assert not pytest["counts"]["skipped"]
PY_CHECK
exit "$status"
