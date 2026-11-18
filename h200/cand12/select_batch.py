"""Measure cand12 update latency and confirm the selected batch against the default."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

def measure(batch, effective, label):
    root = Path.cwd()
    output = root / "results" / f"{label}-{batch}.json"
    output.parent.mkdir(exist_ok=True)
    command = [
        sys.executable, "benchmark_dw_terminal_rmsnorm_training.py",
        "--checkpoint", "checkpoint.pt", "--contract", "contract.json",
        "--images", "images.pt", "--batch", str(batch),
        "--accumulation", str(effective // batch), "--output", str(output),
        "--warmup", "6", "--steps", "5", "--repeats", "3",
        "--spatial", "--vector", "--reset-each-step",
    ]
    env = dict(os.environ, PYTHONPATH="code-cand12/scripts:code-cand12/src:code-cand12",
               TORCHINDUCTOR_CACHE_DIR=str(root / "cache-cand12"))
    rc = subprocess.run(command, env=env).returncode
    if rc == 42:
        return None
    if rc:
        raise RuntimeError(f"Benchmark {label} B={batch} exited {rc}")
    data = json.loads(output.read_text())
    ms = data["median_ms"]
    if data["effective_batch"] != effective or not data["final_finite"] or not 0 < ms < math.inf:
        raise ValueError(f"Invalid benchmark result: {output}")
    return ms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effective-batch", type=int, required=True)
    parser.add_argument("--candidates", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if any(b <= 0 or args.effective_batch % b for b in args.candidates):
        parser.error("Each candidate must divide the effective batch")
    timings = {b: measure(b, args.effective_batch, "search") for b in args.candidates}
    valid = {b: ms for b, ms in timings.items() if ms is not None}
    winner = min(valid, key=valid.get)
    default = 256 if 256 in valid else 128
    paired = []
    for i in range(3):
        pair = {}
        order = [("default", default), ("selected", winner)]
        for role, batch in (order if i % 2 == 0 else reversed(order)):
            pair[role] = measure(batch, args.effective_batch, f"confirm-{i}-{role}")
        paired.append(pair)
    before, after = [statistics.median(row[k] for row in paired) for k in ("default", "selected")]
    result = dict(effective_batch=args.effective_batch, search_ms=timings, physical_batch=winner,
                  accumulation=args.effective_batch // winner, default_batch=default,
                  pairs_ms=paired, throughput_gain_pct=100*(before/after-1))
    Path("selection.json").write_text(json.dumps(result, indent=2))
    print("BATCH_SELECTION_RESULT=" + json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
