"""Bounded, deterministic execution-plan search driven by whole-model measurements.

Only structural backend limits prune candidates. Device, model, batch, and width
never select a preferred implementation; that decision belongs to ``measure``.
"""

from __future__ import annotations

# Script entry points and their tests share the same top-level value classes.
# pyright: reportImplicitRelativeImport=false
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

from va_inference_plan import (
    POST_VARIANTS,
    PROJECTIONS,
    READERS,
    BlockPlan,
    ExecutionPlan,
    PostPlan,
    plan_id,
    plan_to_dict,
    validate_plan,
)


@dataclass(frozen=True)
class SearchResult:
    best_plan: ExecutionPlan
    best_measurement: dict[str, object]
    history: tuple[dict[str, object], ...]
    stopped_reason: str


def canonicalize_plan(plan: ExecutionPlan) -> ExecutionPlan:
    """Remove no-op overrides and geometry unused by the selected backend."""
    blocks = []
    for block in plan.blocks:
        defaults = BlockPlan(block.site)
        changes: dict[str, object] = {}
        if block.reader == plan.reader and block.scan != "scan2d":
            changes["reader"] = "inherit"
        if block.scan != "scan2d":
            changes.update(
                scan_block_modes=defaults.scan_block_modes, scan_num_warps=defaults.scan_num_warps
            )
        if block.scan != "vertical_projection":
            changes.update(
                vertical_pole_tile=defaults.vertical_pole_tile,
                vertical_output_tile=defaults.vertical_output_tile,
                vertical_num_warps=defaults.vertical_num_warps,
            )
        if block.projection == "none":
            changes.update(
                projection_block_rows=defaults.projection_block_rows,
                projection_output_tile=defaults.projection_output_tile,
                projection_num_warps=defaults.projection_num_warps,
            )
        normalized = replace(block, **changes)
        if normalized != defaults:
            blocks.append(normalized)
    return replace(
        plan,
        blocks=tuple(blocks),
        posts=tuple(post for post in plan.posts if post.variant != "none"),
    )


def _hint_records(hints: Mapping[str, object], axis: str) -> dict[str, dict[str, object]]:
    value = hints.get(axis, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{axis} hints must map sites to records")
    records = {}
    for site, record in value.items():
        if not isinstance(site, str) or not isinstance(record, Mapping):
            raise TypeError(f"{axis} hints must map string sites to records")
        records[site] = dict(record)
    return records


def _spatial_shape(node: Mapping[str, object]) -> tuple[int, int] | None:
    shape = node.get("input_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(type(size) is not int or size < 1 for size in shape)
    ):
        return None
    return cast("int", shape[1]), cast("int", shape[2])


def _next_power_two(value: int) -> int:
    return 1 << (max(4, value) - 1).bit_length()


def search_execution_plans(
    catalog: Sequence[dict[str, object]],
    measure: Callable[[ExecutionPlan], dict[str, object]],
    *,
    hints: Mapping[str, object] | None = None,
    beam_width: int = 2,
    max_trials: int = 40,
    shared_memory_limit: int | None = None,
) -> SearchResult:
    """Measure joint foundations, then expand post/projection/scan combinations.

    The budget counts unique callback invocations, including the baseline. A
    failed baseline raises explicitly. Failed later candidates remain evidence
    and cannot win. Completing this bounded beam search is not a claim of a
    globally optimal plan.
    """
    for name, value in (("beam_width", beam_width), ("max_trials", max_trials)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if shared_memory_limit is not None and (
        type(shared_memory_limit) is not int or shared_memory_limit < 1
    ):
        raise ValueError("shared_memory_limit must be a positive integer or None")
    validate_plan(ExecutionPlan(), catalog)
    nodes = {cast("str", node["site"]): node for node in catalog}
    post_sites = {
        cast("str", node["post_site"]) for node in catalog if node.get("post_site") is not None
    }
    hints = {} if hints is None else hints
    post_hints = _hint_records(hints, "post")
    projection_hints = _hint_records(hints, "projection")
    scan_hints = _hint_records(hints, "scan")
    if (
        post_hints.keys() - post_sites
        or (projection_hints.keys() | scan_hints.keys()) - nodes.keys()
    ):
        raise ValueError("hints contain unknown catalog sites")
    history: list[dict[str, object]] = []
    measurements: dict[str, dict[str, object]] = {}
    plans: dict[str, ExecutionPlan] = {}
    pruning: set[tuple[str, str, str]] = set()
    best: str | None = None
    budget_reached = False

    def prune(stage: str, site: str, reason: str) -> None:
        key = stage, site, reason
        if key not in pruning:
            pruning.add(key)
            history.append({"stage": stage, "site": site, "status": "pruned", "reason": reason})

    def evaluate(candidate: ExecutionPlan, stage: str) -> str | None:
        nonlocal best, budget_reached
        candidate = canonicalize_plan(candidate)
        identity = plan_id(candidate)
        if identity in measurements:
            return identity
        try:
            validate_plan(candidate, catalog)
        except ValueError as error:
            prune(stage, identity, str(error))
            return None
        if len(measurements) >= max_trials:
            budget_reached = True
            return None
        measurement: dict[str, object]
        try:
            measurement = dict(measure(candidate))
        except Exception as error:  # noqa: BLE001 - A failed trial is search evidence.
            measurement = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
        status = measurement.get("status")
        if status == "error" and measurement.get("fatal") is True:
            reason = measurement.get("error", measurement.get("reason", "unspecified fatal error"))
            raise RuntimeError(f"fatal measurement failure: {reason}")
        timing = measurement.get("median_ms")
        valid_timing = (
            type(timing) in (int, float)
            and math.isfinite(cast("float", timing))
            and cast("float", timing) > 0
        )
        if status != "ok" or not valid_timing:
            measurement = dict(measurement)
            if status not in ("invalid", "error"):
                measurement["status"] = "invalid"
                measurement["reason"] = (
                    "measurement requires status='ok' and finite positive median_ms"
                )
            if type(timing) is float and not math.isfinite(timing):
                measurement["median_ms"] = None
        measurements[identity] = measurement
        plans[identity] = candidate
        history.append(
            {
                "stage": stage,
                "plan_id": identity,
                "plan": plan_to_dict(candidate),
                "status": measurement["status"],
                "measurement": measurement,
            }
        )
        if measurement["status"] == "ok" and (
            best is None
            or cast("float", measurement["median_ms"])
            < cast("float", measurements[best]["median_ms"])
        ):
            best = identity
        return identity

    def retain(candidates: Sequence[ExecutionPlan], stage: str) -> list[ExecutionPlan]:
        identities: list[str] = []
        for candidate in candidates:
            identity = evaluate(candidate, stage)
            if identity is not None and measurements[identity]["status"] == "ok":
                identities.append(identity)
            if budget_reached:
                break
        ordered = sorted(
            set(identities), key=lambda item: (cast("float", measurements[item]["median_ms"]), item)
        )
        return [plans[item] for item in ordered[:beam_width]]

    baseline = evaluate(ExecutionPlan(), "foundation")
    if baseline is None or measurements[baseline]["status"] != "ok":
        raise RuntimeError(f"baseline measurement failed: {measurements.get(baseline or '', {})}")
    beam = retain(
        [
            ExecutionPlan(reader=reader, launch=launch)
            for launch in ("standard", "spatial", "scan", "wide")
            for reader in READERS
        ],
        "foundation",
    )

    def post_candidates(parent: ExecutionPlan) -> list[ExecutionPlan]:
        candidates = [
            replace(parent, posts=tuple(PostPlan(site, variant) for site in sorted(post_sites)))
            for variant in POST_VARIANTS
        ]
        if post_hints:
            candidates.append(
                replace(
                    parent,
                    posts=tuple(
                        PostPlan(site, cast("str", record["variant"]))
                        for site, record in sorted(post_hints.items())
                    ),
                )
            )
        return candidates

    projection_fields = (
        "projection",
        "projection_block_rows",
        "projection_output_tile",
        "projection_num_warps",
    )
    hinted_projection_geometries = set()
    for site, record in sorted(projection_hints.items()):
        block = replace(
            BlockPlan(site),
            **{key: value for key, value in record.items() if key in projection_fields},
        )
        hinted_projection_geometries.add(
            (
                block.projection,
                block.projection_block_rows,
                block.projection_output_tile,
                block.projection_num_warps,
            )
        )

    def projection_candidates(parent: ExecutionPlan) -> list[ExecutionPlan]:
        result = []
        for variant in (*PROJECTIONS, "hinted"):
            if variant == "hinted" and not projection_hints:
                continue
            blocks = {block.site: block for block in parent.blocks}
            for site, node in sorted(nodes.items()):
                if not node["memory_projection"] or node["kind"] == "terminal":
                    continue
                changes = (
                    {
                        key: value
                        for key, value in projection_hints.get(site, {}).items()
                        if key in projection_fields
                    }
                    if variant == "hinted"
                    else {"projection": variant}
                )
                blocks[site] = replace(blocks.get(site, BlockPlan(site)), **changes)
            result.append(replace(parent, blocks=tuple(blocks.values())))
        for geometry in sorted(hinted_projection_geometries):
            blocks = {block.site: block for block in parent.blocks}
            changes = dict(zip(projection_fields, geometry, strict=True))
            for site, node in sorted(nodes.items()):
                if node["memory_projection"] and node["kind"] in ("same_resolution", "transition"):
                    blocks[site] = replace(blocks.get(site, BlockPlan(site)), **changes)
            result.append(replace(parent, blocks=tuple(blocks.values())))
        return result

    scan_fields = (
        "scan",
        "scan_block_modes",
        "scan_num_warps",
        "vertical_pole_tile",
        "vertical_output_tile",
        "vertical_num_warps",
    )

    def scan_candidates(parent: ExecutionPlan) -> list[ExecutionPlan]:
        result = [parent]
        for variant in ("scan2d", "vertical_projection", "hinted"):
            if variant == "hinted" and not scan_hints:
                continue
            blocks = {block.site: block for block in parent.blocks}
            for site, node in sorted(nodes.items()):
                changes = (
                    {
                        key: value
                        for key, value in scan_hints.get(site, {}).items()
                        if key in scan_fields
                    }
                    if variant == "hinted"
                    else {"scan": variant}
                )
                scan = changes.get("scan", "split")
                if scan == "split":
                    continue
                if scan == "scan2d" and node["kind"] == "transition":
                    continue
                if scan == "vertical_projection" and (
                    node["kind"] != "same_resolution" or not node["memory_projection"]
                ):
                    continue
                shape = _spatial_shape(node)
                if shape is None:
                    prune("scan", site, f"{scan} requires observed NHWC input_shape")
                    continue
                height, width = shape
                if (scan == "scan2d" and max(height, width) > 56) or (
                    scan == "vertical_projection" and height > 56
                ):
                    prune("scan", site, f"{scan} exceeds the backend spatial limit of 56")
                    continue
                block = replace(blocks.get(site, BlockPlan(site)), **changes)
                if scan == "scan2d":
                    block = replace(block, reader="none")
                    lower_bound = (
                        _next_power_two(height)
                        * _next_power_two(width)
                        * 16
                        * 2
                        * block.scan_block_modes
                    )
                    if (
                        node["kind"] != "terminal"
                        and shared_memory_limit is not None
                        and lower_bound > shared_memory_limit
                    ):
                        prune(
                            "scan",
                            site,
                            f"scan2d shared-memory lower bound {lower_bound} exceeds device limit {shared_memory_limit}",
                        )
                        continue
                elif scan == "vertical_projection":
                    pole_tile = changes.get("vertical_pole_tile", 16)
                    block = replace(block, projection="none", vertical_pole_tile=pole_tile)
                blocks[site] = block
            result.append(replace(parent, blocks=tuple(blocks.values())))
        return result

    for stage, expand in (
        ("post", post_candidates),
        ("projection", projection_candidates),
        ("scan", scan_candidates),
    ):
        if budget_reached:
            break
        candidates = [candidate for parent in beam for candidate in expand(parent)]
        beam = retain(candidates, stage)
    if best is None:  # Baseline success establishes a measured fallback.
        raise RuntimeError("search lost its successful baseline measurement")
    return SearchResult(
        plans[best],
        measurements[best],
        tuple(history),
        "max_trials" if budget_reached else "completed_bounded_beam",
    )
