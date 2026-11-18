"""Pure execution-plan values and auditable JSON cache storage.

This module deliberately has no model, tensor, device, or benchmark dependency.
The caller builds a structural catalog and measures candidate plans separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, cast

CACHE_VERSION = 1
READERS = ("none", "rms_scan", "spatial_scan", "scaled_scan")
SCANS = ("split", "scan2d", "vertical_projection")
PROJECTIONS = ("none", "projection_residual", "projection_residual_norm")
POST_VARIANTS = ("none", "joint_gated", "gate_output", "full")


def _choice(value: object, choices: tuple[object, ...], name: str) -> None:
    if type(value) is not type(choices[0]) or value not in choices:
        raise ValueError(f"{name} must be one of {choices!r}, got {value!r}")


def _site(value: object) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("site must be a nonempty string without surrounding whitespace")


@dataclass(frozen=True)
class BlockPlan:
    site: str
    reader: str = "inherit"
    scan: str = "split"
    projection: str = "none"
    scan_block_modes: int = 2
    scan_num_warps: int = 4
    vertical_pole_tile: int = 32
    vertical_output_tile: int = 128
    vertical_num_warps: int = 8
    projection_block_rows: int = 16
    projection_output_tile: int = 64
    projection_num_warps: int = 4

    def __post_init__(self) -> None:
        _site(self.site)
        _choice(self.reader, ("inherit", *READERS), "reader")
        _choice(self.scan, SCANS, "scan")
        _choice(self.projection, PROJECTIONS, "projection")
        for name, choices in (
            ("scan_block_modes", (1, 2, 4, 8)),
            ("scan_num_warps", (4, 8)),
            ("vertical_pole_tile", (16, 32, 64)),
            ("vertical_output_tile", (32, 64, 128)),
            ("vertical_num_warps", (4, 8)),
            ("projection_block_rows", (16, 32)),
            ("projection_output_tile", (32, 64, 128)),
            ("projection_num_warps", (4, 8)),
        ):
            _choice(getattr(self, name), choices, name)


@dataclass(frozen=True)
class PostPlan:
    site: str
    variant: str = "none"

    def __post_init__(self) -> None:
        _site(self.site)
        _choice(self.variant, POST_VARIANTS, "post variant")


@dataclass(frozen=True)
class ExecutionPlan:
    reader: str = "none"
    launch: str = "standard"
    blocks: tuple[BlockPlan, ...] = ()
    posts: tuple[PostPlan, ...] = ()

    def __post_init__(self) -> None:
        _choice(self.reader, READERS, "reader")
        _choice(self.launch, ("standard", "spatial", "scan", "wide"), "launch")
        for name, expected in (("blocks", BlockPlan), ("posts", PostPlan)):
            entries: object = getattr(self, name)
            if not isinstance(entries, tuple) or any(
                type(item) is not expected for item in entries
            ):
                raise ValueError(f"{name} must be a tuple of {expected.__name__} values")
            typed_entries = cast("tuple[BlockPlan | PostPlan, ...]", entries)
            if len({item.site for item in typed_entries}) != len(typed_entries):
                raise ValueError(f"duplicate sites in {name}")
            object.__setattr__(self, name, tuple(sorted(typed_entries, key=lambda item: item.site)))


def plan_to_dict(plan: ExecutionPlan) -> dict[str, object]:
    if type(plan) is not ExecutionPlan:
        raise ValueError("plan must be an ExecutionPlan")
    result = cast("dict[str, object]", asdict(plan))
    result["blocks"] = list(cast("tuple[dict[str, object], ...]", result["blocks"]))
    result["posts"] = list(cast("tuple[dict[str, object], ...]", result["posts"]))
    return result


def _record(value: object, cls: type) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{cls.__name__} must be an object")
    unknown = value.keys() - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return dict(value)


def plan_from_dict(value: object) -> ExecutionPlan:
    data = _record(value, ExecutionPlan)
    for name, cls in (("blocks", BlockPlan), ("posts", PostPlan)):
        entries = data.get(name, [])
        if not isinstance(entries, list):
            raise ValueError(f"{name} must be a JSON array")  # noqa: TRY004
        try:
            # Field values are validated by each dataclass after dynamic JSON construction.
            constructor = cast("Callable[..., BlockPlan | PostPlan]", cls)
            data[name] = tuple(constructor(**_record(item, cls)) for item in entries)
        except TypeError as error:
            raise ValueError(f"invalid {name}: {error}") from error
    return cast("Callable[..., ExecutionPlan]", ExecutionPlan)(**data)


def _json_value(value: object, *, nonnegative: bool = False) -> None:
    """Accept only actual JSON values; avoid implicit tuple/key coercion."""
    if value is None or type(value) in (str, bool):
        return
    if type(value) in (int, float):
        number = cast("int | float", value)
        if (type(value) is float and not math.isfinite(number)) or (nonnegative and number < 0):
            raise ValueError(
                "numeric metadata must be finite" + (" and nonnegative" if nonnegative else "")
            )
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item, nonnegative=nonnegative)
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item, nonnegative=nonnegative)
        return
    raise ValueError("metadata must contain only JSON values and string object keys")


def _canonical(value: object) -> str:
    _json_value(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(value: str) -> object:
    result: object = json.loads(value, object_pairs_hook=_pairs_no_duplicates)
    _json_value(result)
    return result


def plan_to_json(plan: ExecutionPlan) -> str:
    return _canonical(plan_to_dict(plan))


def plan_from_json(value: str) -> ExecutionPlan:
    return plan_from_dict(_load_json(value))


def plan_id(plan: ExecutionPlan) -> str:
    return hashlib.sha256(plan_to_json(plan).encode("utf-8")).hexdigest()


def validate_plan(plan: ExecutionPlan, catalog: Sequence[object]) -> None:
    """Check fusion compatibility against structure, never model width or batch."""
    plan_to_dict(plan)
    nodes = {}
    post_sites = set()
    for node in catalog:
        if not isinstance(node, dict):
            raise ValueError("catalog entries must be objects")  # noqa: TRY004
        site = node.get("site")
        _site(site)
        if site in nodes:
            raise ValueError(f"duplicate catalog site: {site}")
        _choice(node.get("kind"), ("same_resolution", "transition", "terminal"), "catalog kind")
        if type(node.get("memory_projection")) is not bool:
            raise ValueError("catalog memory_projection must be a boolean")
        post_site = node.get("post_site")
        if post_site is not None:
            _site(post_site)
            if post_site in post_sites:
                raise ValueError(f"duplicate catalog post_site: {post_site}")
            post_sites.add(post_site)
        nodes[site] = node
    for block in plan.blocks:
        if block.site not in nodes:
            raise ValueError(f"unknown block site: {block.site}")
        node = nodes[block.site]
        if block.scan == "vertical_projection":
            if node["kind"] != "same_resolution" or not node["memory_projection"]:
                raise ValueError("vertical_projection requires a projected same_resolution block")
            if block.projection != "none":
                raise ValueError(
                    "vertical_projection cannot combine with a separate projection mode"
                )
        if block.scan == "scan2d":
            if node["kind"] not in ("same_resolution", "terminal"):
                raise ValueError("scan2d requires a same_resolution or terminal block")
            if block.reader != "none":
                raise ValueError("scan2d requires an explicit reader='none' override")
        if block.projection != "none" and (
            not node["memory_projection"] or node["kind"] not in ("same_resolution", "transition")
        ):
            raise ValueError(
                "projection fusion requires a projected same_resolution or transition block"
            )
    for post in plan.posts:
        if post.site not in post_sites:
            raise ValueError(f"unknown post site: {post.site}")


@dataclass(frozen=True)
class CacheReadResult:
    status: Literal["hit", "miss", "stale", "invalid"]
    reason: str
    plan: ExecutionPlan | None = None
    measurements: object = None
    search_evidence: object = None


def signature_sha(signature: object) -> str:
    if not isinstance(signature, dict):
        raise ValueError("signature must be a JSON object")  # noqa: TRY004
    return hashlib.sha256(_canonical(signature).encode("utf-8")).hexdigest()


def cache_path(root: str | Path, signature: object) -> Path:
    return Path(root) / f"{signature_sha(signature)}.json"


def read_plan_cache(  # noqa: PLR0911 - Each rejection returns an explicit cache status.
    root: str | Path,
    signature: object,
    *,
    catalog: Sequence[object] | None = None,
) -> CacheReadResult:
    path = cache_path(root, signature)
    try:
        payload = _load_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return CacheReadResult("miss", "cache file does not exist")
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        return CacheReadResult("invalid", f"cache could not be read: {error}")
    try:
        required = {
            "version",
            "signature",
            "signature_sha",
            "plan",
            "plan_id",
            "measurements",
            "search_evidence",
        }
        if not isinstance(payload, dict) or payload.keys() != required:
            return CacheReadResult("invalid", "cache fields do not match the schema")
        if type(payload["version"]) is not int:
            return CacheReadResult("invalid", "cache version must be an integer")
        if payload["version"] != CACHE_VERSION:
            return CacheReadResult("stale", "cache version has changed")
        stored_sha = signature_sha(payload["signature"])
        if payload["signature_sha"] != stored_sha:
            return CacheReadResult("invalid", "stored signature digest does not match")
        if _canonical(payload["signature"]) != _canonical(signature):
            return CacheReadResult(
                "stale", "cache signature does not match the requested environment"
            )
        plan = plan_from_dict(payload["plan"])
        if payload["plan_id"] != plan_id(plan):
            return CacheReadResult("invalid", "stored plan digest does not match")
        if catalog is not None:
            validate_plan(plan, catalog)
        _json_value(payload["measurements"], nonnegative=True)
        return CacheReadResult(
            "hit",
            "signature and plan validated",
            plan,
            payload["measurements"],
            payload["search_evidence"],
        )
    except (ValueError, TypeError) as error:
        return CacheReadResult("invalid", f"cache validation failed: {error}")


def write_plan_cache(
    root: str | Path,
    signature: object,
    plan: ExecutionPlan,
    measurements: object,
    search_evidence: object,
) -> Path:
    """Atomically publish a JSON record; failed writes preserve an existing record."""
    path = cache_path(root, signature)
    _json_value(measurements, nonnegative=True)
    payload = {
        "version": CACHE_VERSION,
        "signature": signature,
        "signature_sha": signature_sha(signature),
        "plan": plan_to_dict(plan),
        "plan_id": plan_id(plan),
        "measurements": measurements,
        "search_evidence": search_evidence,
    }
    serialized = _canonical(payload) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path
