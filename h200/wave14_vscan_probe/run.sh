#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

suite=h200/wave14_vscan_probe
baseline=fb4a33c76335c369c5f0dfbc0d8fc57dcafacea5
refined=79042ae47c67f68f8b7d6e1fb18cf350927fefc2
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset VA_REAL_MEMORY_SCAN_BATCH_PARTIALS VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS
unset LNET_DISABLE_LAUNCH_AUTOTUNE LNET_NR4_FUSED LNET_VSCAN_DECOUPLED

export RUN_ROOT PERSIST_ROOT
RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/va-t57-h200.XXXXXX")
PERSIST_ROOT=$(mktemp -d /app/output/va-t57-h200.XXXXXX)
echo "Wave 14A output $PERSIST_ROOT"
finish() {
    local rc=$?
    trap - EXIT
    cp -a "$RUN_ROOT/results/." "$PERSIST_ROOT/"
    RUN_ROOT="$RUN_ROOT/results" python3 h200/liverms_v2_opt/export_result.py "$rc"
    rm -rf -- "$RUN_ROOT"
    exit "$rc"
}
mkdir -p "$RUN_ROOT/results"
trap finish EXIT
out=$RUN_ROOT/results

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
    export UV_PYTHON_INSTALL_DIR="$RUN_ROOT/python" UV_CACHE_DIR="$RUN_ROOT/uv-cache"
    python3 -m pip install -q --target "$RUN_ROOT/bootstrap" uv==0.11.26
    uv_run() { PYTHONPATH="$RUN_ROOT/bootstrap" python3 -m uv "$@"; }
    uv_run python install 3.13.11
    uv_run venv --python 3.13.11 "$RUN_ROOT/environment"
    run_python="$RUN_ROOT/environment/bin/python"
    uv_run pip install --python "$run_python"         'torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=63381a109a569b280ed3319da89d3afe5cf9ab5c879936382a212affb5c90552'         'torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=5f7c5e0fa08d2cbee93b6e04bbedd59b5e11462cff6cefd07949217265df2370'         triton==3.5.1 numpy==2.4.6 pytest==9.0.3 timm==1.0.26
    ready "$run_python"
fi
export PYTHONPATH="$PWD/src:$PWD/scripts"
export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/inductor"
baseline_dir=$RUN_ROOT/wave13
mkdir -p "$baseline_dir"
git archive "$baseline" src scripts | tar -x -C "$baseline_dir"
base_path="$baseline_dir/src:$baseline_dir/scripts"
experimental=$RUN_ROOT/refined
mkdir -p "$experimental"
git archive "$refined" src scripts | tar -x -C "$experimental"
"$run_python" "$suite/bench.py" --out "$out/isolated.jsonl" --repeats 5 --forward --backward |
    tee "$out/isolated.log"
PYTHONPATH="$base_path" "$run_python" "$suite/bench.py" --out "$out/isolated_wave13.jsonl"     --repeats 5 --forward --backward | tee "$out/isolated_wave13.log"
PYTHONPATH="$experimental/src:$experimental/scripts" "$run_python" "$suite/bench.py"     --out "$out/isolated_refined.jsonl" --repeats 5     --forward fast-packed traffic async async-line     --backward async-line async-full async-shared shared-packed |
    tee "$out/isolated_refined.log"
"$run_python" "$suite/hscan.py" --batch 1024 --out "$out/hscan.json"
"$run_python" - "$out" <<'PYTHON'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for source in sorted(root.glob("isolated*.jsonl")):
    rows = []
    for line in source.read_text().splitlines():
        row = json.loads(line)
        for phase in ("forward", "backward"):
            for arm in row[phase].values():
                if "parity" in arm:
                    arm["parity"] = {
                        "max_abs": max(v["max_abs"] for v in arm["parity"]),
                        "finite": all(v["finite"] for v in arm["parity"]),
                    }
        rows.append(row)
    source.with_suffix(".json").write_text(json.dumps(rows) + "\n")
PYTHON

PYTHONPATH="$base_path" "$run_python" "$suite/model.py" --out "$out" --arm baseline --fold-batch 1024 |
    tee "$out/model_wave13.log"
"$run_python" "$suite/model.py" --out "$out" --arm selected --fold-batch 1024 |
    tee "$out/model_wave14.log"

status=0
started=$SECONDS
timeout --foreground --kill-after=5s 13m "$run_python" -m pytest -q     tests/test_general_cuda_scans.py tests/test_product_scan_pipeline.py tests/test_p1_nr4.py     > "$out/pytest_targeted.log" 2>&1 || status=$?
"$run_python" h200/liverms_v2_opt/bench_h200.py wrap-pytest     --out "$out/pytest_targeted.json" --log "$out/pytest_targeted.log"     --exit-code "$status" --elapsed "$((SECONDS - started))"
exit "$status"
