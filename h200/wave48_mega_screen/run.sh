#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT="$(mktemp -d /tmp/va-t167-results.XXXXXXXX)"
trap 'rm -rf "$OUT"' EXIT
export PYTHONUNBUFFERED=1 VA_SCREEN_OUT="$OUT"

stage() {
    local name="$1" started="$SECONDS" rc
    shift
    set +e
    "$@" >"$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s %s %s\n' "$name" "$rc" "$((SECONDS-started))" >> "$OUT/stages"
    if (( rc != 0 )); then
        printf 'Stage %s exited %s\n' "$name" "$rc"
        tail -n 30 "$OUT/$name.log"
    fi
}

stage scan timeout 300s bash -c '
    out="$1"
    failed=0
    scan() {
        python3 scripts/bench_scan_micro.py --ops "$1" --op-modes "$2" \
            --state-layouts planar,quad_hwm,quad_whm --flush-l2 --parity --out "$out/scan_$1.json"
    }
    scan hscan_fwd bwd_opt || failed=1
    scan hscan_bwd bwd_opt,bwd_opt_ch4 || failed=1
    scan vscan_fwd batch4,batch4_w2,batch4_w2_cs || failed=1
    scan vscan_bwd strict_ro || failed=1
    exit "$failed"
' _ "$OUT"
stage reader timeout 210s python3 scripts/bench_reader_micro.py \
    --modes off proven_floor dinput_vec2 dinput_vec4 fwd_vec2_r4 fwd_vec4 dw_vec2_fullrank regslide_all \
    --flush-l2 --parity --out "$OUT/reader_wave48b.json"
stage readproj timeout 90s python3 h200/wave48_mega_screen/probe_readproj_graph.py --out "$OUT/readproj_graph.json"

python3 -c '
import json, os, platform, subprocess
from pathlib import Path
import torch
import triton

out = Path(os.environ["VA_SCREEN_OUT"])
props = torch.cuda.get_device_properties(0)
stages = {}
for line in (out / "stages").read_text().splitlines():
    name, rc, seconds = line.split()
    stages[name] = {"exit": int(rc), "seconds": int(seconds)}
meta = dict(sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    image="pytorch/pytorch:latest", python=platform.python_version(), pytorch=torch.__version__,
    cuda=torch.version.cuda, triton=triton.__version__, gpu=props.name, gpu_count=torch.cuda.device_count(),
    capability=torch.cuda.get_device_capability(), sm_count=props.multi_processor_count,
    vram_bytes=props.total_memory, bootstrap_seconds=0, stages=stages,
    nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True))
(out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
'

cd "$OUT"
python3 -c '
import base64, hashlib, lzma, textwrap
from pathlib import Path

for path in sorted(Path(".").glob("*.json")):
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    print(f"{sha}  {path.name}")
    print(f"VA_T167_BEGIN {path.name} {sha}")
    print(textwrap.fill(base64.b64encode(lzma.compress(data)).decode(), width=100))
    print(f"VA_T167_END {path.name}")
'
