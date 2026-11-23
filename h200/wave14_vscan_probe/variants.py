"""Experimental launch plans for fused vertical scans."""

from dataclasses import dataclass, replace
from functools import lru_cache

import torch

from lnet import general_cuda_scans as scans

# This module deliberately drives private scan internals and is never imported by LNet.
# ruff: noqa: SLF001, FBT003


@dataclass(frozen=True)
class Trial:
    vector_weights: bool = True
    fast_forward: bool = False
    cache_line: bool | None = None
    cache_source: bool | None = None
    shared_acc: bool = False
    shared_hidden: int = 0
    chunk: int | None = None
    prefetch_rows: bool = False
    cache_positive: bool = False
    async_source: bool = False


FORWARD = {
    "weights": Trial(),
    "fast": Trial(fast_forward=True),
    "fast-packed": Trial(vector_weights=False, fast_forward=True),
    "line": Trial(vector_weights=False, cache_line=True),
    "line-source": Trial(vector_weights=False, cache_line=True, cache_source=True),
    "vector-line": Trial(fast_forward=True, cache_line=True),
    "vector-line-source": Trial(fast_forward=True, cache_line=True, cache_source=True),
    "prefetch": Trial(fast_forward=True, cache_line=True, cache_source=True, prefetch_rows=True),
    "async": Trial(fast_forward=True, cache_line=True, cache_source=True, async_source=True),
    "async-line": Trial(fast_forward=True, cache_line=False, cache_source=True, async_source=True),
}
BACKWARD = {
    "weights": Trial(),
    "line": Trial(cache_line=True),
    "shared": Trial(cache_line=True, shared_acc=True),
    "shared-packed": Trial(vector_weights=False, cache_line=True, shared_acc=True),
    "half-dwi": Trial(cache_line=True, shared_hidden=4),
    "dwi": Trial(cache_line=True, shared_hidden=8),
    "chunk14": Trial(cache_line=True, cache_source=True, chunk=14, shared_hidden=4),
    "chunk28": Trial(cache_line=True, cache_source=True, chunk=28, shared_hidden=4),
    "full": Trial(cache_line=True, cache_source=True, chunk=0),
    "full-half-dwi": Trial(cache_line=True, cache_source=True, chunk=0, shared_hidden=4),
    "full-shared": Trial(cache_line=True, cache_source=True, chunk=0, shared_acc=True),
    "full-dwi": Trial(cache_line=True, cache_source=True, chunk=0, shared_hidden=8),
    "cached-ds": Trial(cache_line=True, cache_positive=True, shared_acc=True),
    "async-line": Trial(cache_line=True, cache_source=True, async_source=True),
    "async-full": Trial(cache_line=True, cache_source=True, chunk=0, async_source=True),
    "async-shared": Trial(cache_line=True, cache_source=True, shared_acc=True, async_source=True),
    "full-cached-ds": Trial(
        cache_line=True, cache_source=True, cache_positive=True, chunk=0, shared_acc=True,
    ),
}

original_launch_for_shape = scans._launch_for_shape
original_scan_plans = scans._scan_plans
original_resident = scans._resident
type Arm = tuple[str, str]
type Shape = tuple[int, int, int, int]
type Flag = bool | int | tuple[bool, bool]


def geometry(
    base: scans.Launch, trial: Trial, height: int, hidden: int, warps: int, itemsize: int,
    *, backward: bool, coarse: bool,
) -> tuple[scans.ScanPlan, scans.Kernel]:
    """Mirror the dynamic shared storage layout of vertical()."""
    threads = 32 * warps
    cache_line = base.plan.cache_line if trial.cache_line is None else trial.cache_line
    cache_source = base.plan.cache_source if trial.cache_source is None else trial.cache_source
    chunk = base.key.chunk if trial.chunk is None else trial.chunk
    full = backward and cache_source and (
        chunk == 0 or (trial.chunk is not None and chunk >= height)
    )
    span = height if full else chunk
    weights = 11 * hidden * 32 * (4 if trial.vector_weights else itemsize)
    if not (trial.vector_weights or base.plan.cache_weights):
        weights = 0
    boundaries = (height - 1) // span * threads * 16 if backward else 0
    trajectory = 16 * span * threads if backward else 0
    source = (height if cache_source else span if backward else 0) * 4 * itemsize * threads
    fields = 11 * hidden + 6 if trial.shared_acc else 8 * trial.shared_hidden
    accumulator = fields * threads * 4
    line = (height // (1 + coarse)) * 4 * itemsize * threads if cache_line else 0
    reduction = 32 * threads * 4 if backward and warps > 1 else 0
    ds = height * 4 * itemsize * threads if trial.cache_positive else 0
    smem = max(weights + boundaries + trajectory + source + accumulator + line + ds, reduction)
    plan = scans.ScanPlan(
        warps, smem, span, bool(weights), False, cache_line, trial.shared_acc, full, cache_source,
    )
    key = replace(
        base.key,
        warps=warps,
        chunk=0 if full else span,
        cache_weights=plan.cache_weights,
        global_checkpoints=False,
        cache_line=cache_line,
        shared_acc=trial.shared_acc,
        precise=base.key.precise and not trial.fast_forward,
        cache_source=cache_source,
        vector_weights=trial.vector_weights,
        fast_forward=trial.fast_forward,
        shared_hidden=trial.shared_hidden,
        prefetch_rows=trial.prefetch_rows,
        cache_positive=trial.cache_positive,
        async_source=trial.async_source,
    )
    return plan, key


def trial_launch(
    base: scans.Launch, trial: Trial, device: int, dtype: torch.dtype, shape: Shape, *,
    backward: bool, coarse: bool, hidden: int, forced_warps: int | None,
) -> scans.Launch | None:
    b, h, w, m = shape
    sms, limit, _ = scans.device_limits(device)
    lines, tiles = b * (w // (1 + coarse)), scans._ceil(m, 32)
    fields = 11 * hidden + 6 if backward else 0
    extra = scans._extra_workspace_budget(device)
    best = None
    available_warps = scans._SCAN_WARPS if trial.async_source else range(base.plan.warps, 0, -1)
    for warps in (forced_warps,) if forced_warps else available_warps:
        plan, key = geometry(base, trial, h, hidden, warps, dtype.itemsize,
                             backward=backward, coarse=coarse)
        if plan.smem > limit:
            continue
        resident = scans._resident(device, key, plan.smem)
        if not resident:
            continue
        waves = scans._SHORT_VSCAN_WAVES if backward and h <= scans._SHORT_SCAN else 8
        workers = scans._workers(
            lines, m, fields, warps, sms * resident * waves, scans._MAX_PARTIAL_BYTES + extra,
        )
        if not backward:
            if not (plan.cache_line or coarse):
                waves = scans._VSCAN_UNCACHED_FORWARD_WAVES
            workers = min(lines, scans._ceil(sms * resident * waves, tiles) * warps)
        scratch = m * scans._scratch_per_worker(
            plan, h, vertical=True, coarse=coarse, backward=backward, itemsize=dtype.itemsize,
        )
        if scratch:
            workers = min(workers, (scans._MAX_SCRATCH_BYTES + extra) // scratch)
        if not workers:
            continue
        if backward and workers * m * fields >= 2**31 and key.index32:
            key = replace(key, index32=False)
        launch = scans.Launch(plan, key, workers, scans._ceil(workers, warps))
        score = min(resident * warps, scans._ceil(workers * tiles, sms)), workers, warps
        if best is None or score > best[0]:
            best = (score, launch)
    return None if best is None else best[1]


def forward_hbm_proxy(
    launch: scans.Launch, shape: Shape, itemsize: int, *, coarse: bool, l2: int,
    resident_tiles: int,
) -> float:
    """Estimate traffic with L2 reuse of the live source and scratch."""
    b, h, w, m = shape
    ratio = 1 + coarse
    lines = b * (w // ratio)
    state_bytes = lines * h * m * 4 * itemsize
    output_bytes = lines * (h // ratio) * m * 2 * itemsize
    scratch_per_line = m * scans._scratch_per_worker(
        launch.plan, h, vertical=True, coarse=coarse, backward=False, itemsize=itemsize,
    )
    scratch_bytes = scratch_per_line * launch.workers
    live_source_bytes = min(state_bytes, resident_tiles * 32 * h * 4 * itemsize)
    # Reserve cache for output and unrelated concurrent accesses.
    reuse_budget = 7 * l2 // 8
    scratch_reuse = min(1.0, max(0, reuse_budget - live_source_bytes) / max(1, scratch_bytes))
    scratch_traffic = 2 * lines * scratch_per_line * (1 - scratch_reuse)
    source_reuse = min(1.0, max(0, reuse_budget - scratch_bytes) / max(1, live_source_bytes))
    reload = 0.0 if launch.plan.cache_source else state_bytes * (1 - source_reuse)
    return state_bytes + output_bytes + scratch_traffic + reload


def traffic_launch(
    base: scans.Launch, device: int, dtype: torch.dtype, shape: Shape, *, coarse: bool,
    hidden: int, forced_warps: int | None,
) -> scans.Launch:
    sms, _, max_warps = scans.device_limits(device)
    l2 = scans._l2_size(device)
    tiles = scans._ceil(shape[-1], 32)

    def score(launch: scans.Launch) -> float:
        blocks = scans._resident(device, launch.key, launch.plan.smem)
        live = min(launch.workers * tiles, sms * blocks * launch.plan.warps)
        traffic = forward_hbm_proxy(
            launch, shape, dtype.itemsize, coarse=coarse, l2=l2, resident_tiles=live,
        )
        return min(live, sms * max_warps // 4) / traffic

    labels = ("fast", "fast-packed", "line", "line-source", "vector-line", "vector-line-source")
    options = [base]
    for label in labels:
        launch = trial_launch(
            base, FORWARD[label], device, dtype, shape, backward=False, coarse=coarse,
            hidden=hidden, forced_warps=forced_warps,
        )
        if launch is not None:
            options.append(launch)
    return max(options, key=score)


@lru_cache(maxsize=1024)
def _override(
    arm: Arm, forced_warps: int | None, device: int, dtype: torch.dtype,
    shape: Shape, index_bound: int, **kwargs: Flag,
) -> scans.Launch | None:
    base = original_launch_for_shape(device, dtype, shape, index_bound, **kwargs)
    if (
        base is None or not kwargs["vertical"] or kwargs.get("pure_vertical")
        or scans._scan_plans is not original_scan_plans
        or scans._resident is not original_resident
    ):
        return base
    label = arm[kwargs["backward"]]
    if label == "baseline" or dtype.itemsize != 2 or kwargs["hidden"] != scans._REGISTER_HIDDEN:
        return base
    if label == "traffic":
        return traffic_launch(
            base, device, dtype, shape, coarse=bool(kwargs["coarse"]), hidden=int(kwargs["hidden"]),
            forced_warps=forced_warps,
        )
    if label == "fast-packed":
        return base._replace(key=replace(base.key, precise=False, fast_forward=True))
    options = BACKWARD if kwargs["backward"] else FORWARD
    trial = options[label]
    if trial.async_source and (shape[-1] % 32 or torch.cuda.get_device_capability(device)[0] < 8):
        return base
    launch = trial_launch(
        base, trial, device, dtype, shape, forced_warps=forced_warps,
        **{k: kwargs[k] for k in ("backward", "coarse", "hidden")},
    )
    return launch or base


def select(arm: Arm, forced_warps: int | None = None) -> None:
    def wrapped(
        device: int, dtype: torch.dtype, shape: Shape, index_bound: int, **kwargs: Flag,
    ) -> scans.Launch | None:
        return _override(arm, forced_warps, device, dtype, shape, index_bound, **kwargs)

    def clear() -> None:
        original_launch_for_shape.cache_clear()
        _override.cache_clear()

    wrapped.cache_clear = clear
    scans._launch_for_shape = (
        original_launch_for_shape if arm == ("baseline", "baseline") else wrapped
    )
