#!/usr/bin/env bash
set -euo pipefail
SECONDS=0
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
out=$(mktemp -d /app/output/va-t151-h200.XXXXXX)
export LC_ALL=C PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD" VA_BUNDLE_COMMIT
VA_BUNDLE_COMMIT=$(git rev-parse HEAD)
for variable in ${!LNET_SCAN_@} ${!LNET_READER_@} ${!LNET_MODEL_@} ${!LNET_STEM_@}; do
    unset "$variable"
done
unset LNET_GENERAL_SCANS LNET_FUSED_FACTORIZED_READER LNET_GENERAL_SCAN_REF_FWD
unset LNET_SCAN_SGS LNET_VSCAN_DECOUPLED LNET_VSCAN_MMA LNET_READER_VEC LNET_READER_OPT LNET_GENERAL_SCAN_OPT
unset LNET_MODEL_OPT LNET_FUSED_STEM_CONV LNET_CAPTURE_SAFE_ORTHOGONAL
unset TORCHINDUCTOR_FORCE_DISABLE_CACHES
export TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=0 TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0
bootstrap_failure() {
    local rc=$?
    trap - ERR
    mkdir -p "$out"
    python3 -c '
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
log = root / "environment.log"
(root / "summary.json").write_text(json.dumps({
    "verdict": "FAIL", "status": "environment_error", "returncode": int(sys.argv[2]),
    "error": log.read_text(errors="replace")[-4096:] if log.exists() else "Setup failed",
    "commit": sys.argv[3],
}))
' "$out" "$rc" "$VA_BUNDLE_COMMIT"
    python3 -c 'from pathlib import Path; from h200.wave43_h200_lockin.run_wave43_h200 import transport; import sys; transport(Path(sys.argv[1]))' "$out"
    exit "$rc"
}
trap bootstrap_failure ERR
ready() {
    test -x "$1" && "$1" -c '
import sys
import numpy, pytest, timm, torch, torchvision, triton
raise SystemExit(sys.version_info[:3] != (3, 13, 11) or torch.__version__ != "2.9.1+cu128"
                 or triton.__version__ != "3.5.1" or not torch.cuda.is_available())
' 2>/dev/null
}

unset PYTHON_BIN
for candidate in \
    /app/output/va-t145-h200.*/environment/venv/bin/python \
    /app/output/va-t142-h200.*/environment/venv/bin/python \
    /app/output/va-t140-h200.*/environment/venv/bin/python \
    /app/output/va-t134-h200.*/environment/venv/bin/python \
    /app/output/va-t131-h200.*/environment/venv/bin/python \
    /app/output/va-t128-h200.*/environment/venv/bin/python \
    /app/output/va-h200-effective1024-r3-combined-20261119/environment/bin/python \
    /app/output/va-h200-r4-marker-fix-20261119/environment/bin/python \
    /app/output/va-t92-h200.aOb6kB/environment/venv/bin/python \
    /app/output/va-t93-h200.TrMSFC/environment/venv/bin/python \
    /app/output/va-t93-h200.XMqpcl/environment/venv/bin/python \
    /app/output/va-t101-h200.*/environment/venv/bin/python \
    /app/output/va-t105-h200.*/environment/venv/bin/python \
    /app/output/va-t97-h200.*/environment/venv/bin/python \
    /app/output/va-t85-h200.*/environment/venv/bin/python \
    "$(command -v python3)"; do
    if ready "$candidate"; then
        export PYTHON_BIN=$candidate
        break
    fi
done
if [[ -z ${PYTHON_BIN:-} ]]; then
    setup="$out/environment"
    mkdir -p "$setup"
    export UV_PYTHON_INSTALL_DIR="$setup/python" UV_CACHE_DIR="$setup/uv-cache"
    timeout --kill-after=5s 45s python3 -m pip install -q --disable-pip-version-check --target "$setup/bootstrap" uv==0.11.26
    timeout --kill-after=5s 60s env PYTHONPATH="$setup/bootstrap" python3 -m uv python install 3.13.11
    install_remaining=$((900 - SECONDS))
    (( install_remaining > 5 ))
    timeout --kill-after=2s 15s env PYTHONPATH="$setup/bootstrap" python3 -m uv venv \
        --python 3.13.11 "$setup/venv"
    install_remaining=$((895 - SECONDS))
    (( install_remaining > 0 ))
    export PYTHON_BIN="$setup/venv/bin/python"
    # Cold downloads are several GiB and have exceeded the former 240s cap.
    export UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=4 UV_CONCURRENT_DOWNLOADS=6
    timeout --kill-after=5s "${install_remaining}s" env PYTHONPATH="$setup/bootstrap" python3 -m uv pip install --python "$PYTHON_BIN" \
        'torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=63381a109a569b280ed3319da89d3afe5cf9ab5c879936382a212affb5c90552' \
        'torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=5f7c5e0fa08d2cbee93b6e04bbedd59b5e11462cff6cefd07949217265df2370' \
        triton==3.5.1 numpy==2.4.6 pytest==9.0.3 timm==1.0.26
    ready "$PYTHON_BIN"
fi > "$out/environment.log" 2>&1


"$PYTHON_BIN" -c 'import sys, torch, triton, timm, va_model_builder, lnet.general_cuda_scans; assert sys.version_info >= (3, 13) and torch.cuda.is_available()' >> "$out/environment.log" 2>&1
export TORCHINDUCTOR_CACHE_DIR="$out/cache/inductor"
export TRITON_CACHE_DIR="$out/cache/triton"
export CUDA_CACHE_PATH="$out/cache/cuda"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
export VA_BOOTSTRAP_SECONDS="$SECONDS"
trap - ERR
printf 'Python environment ready after %ss\n' "$SECONDS" >&2
exec "$PYTHON_BIN" h200/wave43_h200_lockin/run_wave43_h200.py "$out"
