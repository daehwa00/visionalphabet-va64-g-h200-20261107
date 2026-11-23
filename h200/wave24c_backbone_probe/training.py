"""Use the Wave 21C harness and retain complete timed training trajectories."""

# ruff: noqa: ANN001, ANN002, ANN003, ANN201
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liverms_v2_opt"))
import bench_h200 as bench

_setup = bench.setup
_measure = bench.measure
_clip = torch.nn.utils.clip_grad_norm_
_norms = []


def setup(batch):
    result = _setup(batch)
    torch.backends.cudnn.benchmark = os.environ.get("WAVE24C_CUDNN_BENCHMARK", "1") == "1"
    return result


def clip(*args, **kwargs):
    norm = _clip(*args, **kwargs)
    _norms.append(norm.detach())
    return norm


def measure(step, batch, *, warmup, count):
    _norms.clear()
    result, losses = _measure(step, batch, warmup=warmup, count=count)
    result.update(
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        trajectory_loss=torch.stack(losses).tolist(),
        trajectory_preclip_norm=torch.stack(_norms[warmup:]).tolist(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
    )
    return result, losses


bench.setup = setup
bench.measure = measure
torch.nn.utils.clip_grad_norm_ = clip

if __name__ == "__main__":
    bench.main()
