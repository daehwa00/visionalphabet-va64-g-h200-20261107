#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_ROOT=/app/output/va-h200-effective1024-r3-combined-20261119
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
if [[ ! -d "$fixture" ]]; then python effective_cli/stage.py "$fixture"; fi
(cd "$fixture" && sha256sum -c "$OLDPWD/effective_cli/inputs.sha256")
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
cp -a official/overlay/. "$RUN_ROOT/common/"
cp -a effective/overlay/. "$RUN_ROOT/common/"
mv "$RUN_ROOT/common" "$RUN_ROOT/source"
python - "$RUN_ROOT/source" "$fixture" <<'PY'
import hashlib,json,sys
from pathlib import Path
s,root=map(Path,sys.argv[1:]);m=json.loads(Path('effective/source.json').read_text())
for f,h in m.items():assert hashlib.sha256((s/f).read_bytes()).hexdigest()==h,f
manifest=json.loads((root/'manifest.json').read_text());meta=json.loads((root/'fixture.json').read_text())
assert len(set(v['sha256'] for v in manifest['train']))==len(manifest['train'])==1024
print('SOURCE_AND_FIXTURE_OK',len(m),meta['input_content_sha256'],meta['mixup']['lambda'])
PY
cp -a effective_cli/overlay/. "$RUN_ROOT/source/"
cp "$fixture/packaged-4090-effective.json" "$RUN_ROOT/source/src/lnet/validated_effective_updates.json"
python - "$RUN_ROOT/source" <<'PY'
from pathlib import Path
import hashlib,json,sys
r=Path(sys.argv[1]);m=json.loads(Path('effective_cli/source.json').read_text())
for n,h in m.items():
    assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
(r.parent/'source-manifest.json').write_text(json.dumps(m))
print('ORDINARY_FROZEN',len(m))
PY
(cd "$RUN_ROOT/source" && sha256sum -c "$OLDPWD/effective_cli/common-release.sha256")
export CONTRACT="$PWD/contract.json"
export COLD_INPUTS="$fixture"
export PYTHONPATH="$RUN_ROOT/source/scripts:$RUN_ROOT/source/src:$PWD/effective_cli" WANDB_MODE=disabled
export TORCHINDUCTOR_COMPILE_THREADS=2
trap 'python effective_cli/frontier_export.py' EXIT
taskset -c "$affinity" python -u effective_cli/frontier.py > "$RUN_ROOT/frontier.log" 2>&1 || {
    tail -10 "$RUN_ROOT/frontier.log" | cut -c1-1200
    exit 1
}
taskset -c "$affinity" python -u effective_cli/matched_table.py > "$RUN_ROOT/matched.log" 2>&1 || {
    tail -8 "$RUN_ROOT/matched.log" | cut -c1-1200
    exit 1
}
export R3_JOB_COMMIT
R3_JOB_COMMIT=$(git rev-parse HEAD)
python effective_cli/r3_catalog.py
python effective_cli/data.py
if taskset -c "$affinity" python -u effective_cli/cli_e2e.py > "$RUN_ROOT/cli-e2e.log" 2>&1; then
    echo ORDINARY_R3_VERIFIED
else
    echo ORDINARY_R3_NEEDS_REVIEW
    tail -12 "$RUN_ROOT/cli-e2e.log" | cut -c1-1100
    exit 1
fi
