"""Collect CUDA event timings and effective payload bandwidth for path projection."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from lnet import general_cuda_scans as scans


@contextmanager
def projection_metrics() -> Iterator[Callable[[], dict]]:
    launches = []
    original = scans._launch_kernel  # noqa: SLF001

    def tracked(source: torch.Tensor, key: scans.Kernel, grid: int, smem: int, args: tuple) -> None:
        if not key.projection:
            original(source, key, grid, smem, args)
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        original(source, key, grid, smem, args)
        end.record()
        # Feature and adjoint planes plus the two outputs or their gradients.
        payload = source.numel() * source.element_size() // 8 * (18 if key.backward else 10)
        launches.append((key.backward, start, end, payload, key.warps, grid, smem, key.shared_acc))

    def summarize() -> dict:
        result = {}
        for direction, label in ((False, "forward"), (True, "backward")):
            selected = [launch for launch in launches if launch[0] == direction]
            if not selected:
                continue
            milliseconds = sum(a.elapsed_time(b) for _, a, b, *_ in selected)
            payload = sum(launch[3] for launch in selected)
            result[label] = {
                "event_ms": milliseconds,
                "payload_bytes": payload,
                "effective_gbps": payload / (milliseconds * 1e6),
                "launches": len(selected),
                "geometries": sorted({launch[4:] for launch in selected}),
            }
        return result

    scans._launch_kernel = tracked  # noqa: SLF001
    try:
        yield summarize
    finally:
        scans._launch_kernel = original  # noqa: SLF001


def instrument_training(profile: Callable[..., Any]) -> Callable[..., dict]:
    def wrapped(step: Callable) -> dict:
        with projection_metrics() as summarize:
            result = profile(step)
        result["path_mlp"] = summarize()
        return result
    return wrapped
