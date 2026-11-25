#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT="$(mktemp -d /tmp/va-t164-results.XXXXXXXX)"
trap 'rm -rf "$OUT"' EXIT
export PYTHONUNBUFFERED=1
export VA_SCREEN_OUT="$OUT"

stage() {
    local name="$1" started="$SECONDS" rc
    shift
    set +e
    "$@" >"$OUT/$name.log" 2>&1
    rc=$?
    set -e
    printf '%s %s %s\n' "$name" "$rc" "$((SECONDS-started))" >> "$OUT/stages"
    if (( rc != 0 )); then
        tail -n 30 "$OUT/$name.log"
    fi
}

stage reader timeout 210s python3 scripts/bench_reader_micro.py --modes off proven_floor fwd_vec128 dinput_vec128 dw_smem_vec128 vec128_all --flush-l2 --parity --out "$OUT/reader_wave47.json"
stage readproj timeout 90s python3 h200/wave47_screen/probe_readproj_graph.py --out "$OUT/readproj_graph.json"
stage vscan timeout 90s python3 h200/wave47_screen/probe_vscan_decomp.py --out "$OUT/vscan_decomp.json"
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
    stages[name] = dict(exit=int(rc), seconds=int(seconds))
result = dict(sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    image="pytorch/pytorch:latest", python=platform.python_version(), pytorch=torch.__version__,
    cuda=torch.version.cuda, triton=triton.__version__, gpu=props.name, gpu_count=torch.cuda.device_count(),
    capability=torch.cuda.get_device_capability(), sm_count=props.multi_processor_count,
    vram_bytes=props.total_memory, bootstrap_seconds=0, stages=stages,
    nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True))
(out / "meta.json").write_text(json.dumps(result, indent=2) + "\n")
'
cd "$OUT"
for file in meta.json reader_wave47.json readproj_graph.json vscan_decomp.json; do
    if [[ -f "$file" ]]; then
        sha256sum "$file"
        checksum="$(sha256sum "$file" | cut -d ' ' -f 1)"
        printf 'VA_T164_BEGIN %s %s\n' "$file" "$checksum"
        gzip -c "$file" | base64
        printf 'VA_T164_END %s\n' "$file"
    fi
done
