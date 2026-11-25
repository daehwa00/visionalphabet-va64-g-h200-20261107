"""Measure a single Wave 40 BF16 training arm."""

import argparse
import os
import traceback
from pathlib import Path
from time import perf_counter

from h200.liverms_v2_opt import bench_h200

MODES = ("off", "strict", "chunked", "nocsrc", "partial", "auto")
PRESET = "cuda_graphs,tma_gemm,max_autotune_pointwise"
VARIANT = "17-p1-nr4-1024d-v3"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-sgs", choices=MODES, required=True)
    parser.add_argument("--preset", choices=("off", PRESET), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(LNET_SCAN_SGS=args.scan_sgs, LNET_MODEL_W36B=args.preset)
    args.variant, args.batch, args.profile = VARIANT, 1024, False
    started = perf_counter()
    try:
        metrics = bench_h200.train_arm(args)
        replays = metrics.get("cudagraph_replays_by_phase", {})
        row = {
            "status": "ok",
            "compile_warmup_s": metrics["compile_warmup_s"],
            "timed_loop_s": metrics["timed_loop_s"],
            "step_ms": {key: metrics[f"{key}_ms"] for key in ("median", "mean", "p90")},
            "img_per_s": metrics["images_per_second"],
            "peak_vram_gib": metrics["peak_allocated_gib"],
            "graph_replays": {
                "fwd": replays.get("forward", 0), "bwd": replays.get("backward", 0),
                "upd": replays.get("update", 0),
            },
            "loss_finite": metrics["finite_losses"],
            "finite_grad_count": metrics["finite_grad_count"],
            "grad_tensor_count": metrics["grad_tensor_count"],
        }
    except Exception:  # noqa: BLE001 - report child errors in JSON before exiting
        row = {"status": "error", "error": traceback.format_exc()}
    bench_h200.write_json(args.out, {
        **row, "total_arm_wall_s": perf_counter() - started,
        "scan_sgs_mode": args.scan_sgs, "preset": args.preset,
    })
    return int(row["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
