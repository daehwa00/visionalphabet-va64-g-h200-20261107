"""Geometry-only trial proposals from previously measured complete updates.

Catalog entries are indivisible candidates, never dispatch authority. Every source
setting must map onto actual compiled target scopes, and parent specializers must
be explicit and consistent. Device keys and kernel revisions are never imported.
There is no arbitrary donor cap: the catalog itself is the finite portfolio.
Catalog contents are intentionally absent from reusable workload policy hashes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from lnet.output_dot_dispatch import DOT_NAME
from lnet.path_backward_layout import LAYOUT_NAME, candidate_geometry
from lnet.retained_path_dispatch import PROFILE_NAME, step_geometry
from lnet.segmented_path_scan import SEGMENT_NAME

VJP_NAME = "effective_output_vjp"
STRUCT_NAME = "effective_structural_scan"
CATALOG = Path(__file__).resolve().parents[1] / "src/lnet/effective_update_proposals.json"
FAMILIES = (LAYOUT_NAME, PROFILE_NAME, SEGMENT_NAME, DOT_NAME, VJP_NAME, STRUCT_NAME)
OFF = {"blocks": {}, "num_warps": 4, "num_stages": 1}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _scope(key):
    return json.loads(key)


def _identity(fields):
    # These are the current target's identity, not anything supplied by a donor.
    return {k: v for k, v in fields.items() if k != "kernel_revision"}


def same_launch_scope(first, second):
    return _identity(_scope(first)) == _identity(_scope(second))


def related_geometry(plan, family, key):
    return next((v for k, v in plan.get(family, {}).items() if same_launch_scope(k, key)), None)


def _active(geometry):
    return bool(geometry and geometry.get("blocks"))


def _geometry(family, geometry, fields):
    if not isinstance(geometry, dict) or set(geometry) != {"blocks", "num_warps", "num_stages"}:
        raise ValueError("geometry requires exact blocks/num_warps/num_stages")
    blocks, shape = geometry["blocks"], fields["shape"]
    if (
        not isinstance(blocks, dict)
        or any(type(v) is not int for v in blocks.values())
        or type(geometry["num_warps"]) is not int
        or type(geometry["num_stages"]) is not int
    ):
        raise ValueError("geometry dimensions must be integers")
    if not blocks:
        if family == LAYOUT_NAME or geometry != OFF:
            raise ValueError("invalid disabled geometry sentinel")
        return
    if family == VJP_NAME:
        if (
            geometry
            != {"blocks": {"VARIANT": blocks.get("VARIANT")}, "num_warps": 4, "num_stages": 1}
            or blocks.get("VARIANT") not in (1, 2)
            or (shape["height"], shape["width"], shape["operation"], shape["coarse"])
            != (56, 56, 1, 0)
            or fields["dtype"] != "bfloat16"
        ):
            raise ValueError("unsupported VJP marker")
        return
    if family == STRUCT_NAME:
        variant = blocks.get("VARIANT")
        op, coarse = shape.get("operation"), shape.get("coarse")
        valid_op = (
            (op == 1 and coarse == 0 and variant in (1, 2, 3, 4, 5, 6))
            or (op == 1 and coarse == 1 and variant in (1, 2, 3, 4, 5))
            or (op == 0 and coarse == 0 and variant in (1, 2))
        )
        if (
            geometry != {"blocks": {"VARIANT": variant}, "num_warps": 4, "num_stages": 1}
            or not valid_op
            or (shape["height"], shape["width"]) != (56, 56)
            or fields["dtype"] != "bfloat16"
        ):
            raise ValueError("unsupported structural scan marker")
        return
    if family == LAYOUT_NAME:
        checked = candidate_geometry(
            blocks["LINES_PER_GROUP"],
            blocks["BLOCK_MODES"],
            geometry["num_warps"],
            blocks["COMPACT_CHANNELS"],
        )
        tile, group, width = blocks["BLOCK_MODES"], blocks["LINES_PER_GROUP"], shape["width"]
    else:
        if shape["height"] != 56 or shape["width"] != 56 or fields["dtype"] != "bfloat16":
            raise ValueError("retained scopes require compiled BF16 56x56")
        checked = step_geometry(
            shape["operation"],
            blocks["GROUP"],
            blocks["BM"],
            geometry["num_warps"],
            blocks.get("DOUBLING", 0),
        )
        tile, group = blocks["BM"], blocks["GROUP"]
        width = (
            shape["width"] // 2 if shape["operation"] != 2 and shape["coarse"] else shape["width"]
        )
    if (
        checked.blocks != blocks
        or checked.num_stages != geometry["num_stages"]
        or shape["modes"] % tile
        or (group and shape["batch"] * width % group)
    ):
        raise ValueError("unsupported/misaligned target geometry")


def _parents(plan):
    for family in (SEGMENT_NAME, DOT_NAME, VJP_NAME):
        for key, geometry in plan.get(family, {}).items():
            if not _active(geometry):
                continue
            shape = _scope(key)["shape"]
            retained = related_geometry(plan, PROFILE_NAME, key)
            if (
                not _active(retained)
                or retained["blocks"].get("DOUBLING") != 1
                or shape.get("operation") != 1
                or shape.get("height") != 56
            ):
                raise ValueError("active specializer lacks explicit four-row retained parent")
            if family in (SEGMENT_NAME, DOT_NAME) and geometry != retained:
                raise ValueError("specializer/retained parent geometry mismatch")
            if family in (DOT_NAME, VJP_NAME) and related_geometry(plan, SEGMENT_NAME, key) != retained:
                raise ValueError("dot/VJP lacks identical explicit segment parent")
            if family == VJP_NAME and related_geometry(plan, DOT_NAME, key) != retained:
                raise ValueError("VJP lacks identical explicit dot parent")
    for key, geometry in plan.get(STRUCT_NAME, {}).items():
        if not _active(geometry):
            continue
        shape = _scope(key)["shape"]
        retained = related_geometry(plan, PROFILE_NAME, key)
        if not _active(retained) or shape.get("height") != 56:
            raise ValueError("active structural scan lacks explicit retained parent")
        if shape.get("operation") == 1:
            if (
                retained["blocks"].get("DOUBLING") != 1
                or related_geometry(plan, SEGMENT_NAME, key) != retained
                or related_geometry(plan, DOT_NAME, key) != retained
            ):
                raise ValueError("backward structural scan requires four-row retained/segment/dot parents")
            if _active(related_geometry(plan, VJP_NAME, key)):
                raise ValueError("a structural backward implements its own output VJP")


def _map_proposal(proposal, observed):
    if (
        not isinstance(proposal, dict)
        or not isinstance(proposal.get("name"), str)
        or not proposal["name"]
        or not proposal.get("origin")
        or proposal.get("effective_samples") != 1024
    ):
        raise ValueError("proposal requires name/origin/effective_samples=1024")
    dtype = proposal.get("dtype")
    if dtype not in ("bfloat16", "float32"):
        raise ValueError("proposal requires an explicit supported source dtype")
    settings = proposal.get("settings")
    if not isinstance(settings, list) or not settings:
        raise ValueError("proposal requires nonempty settings")
    if "settings_count" in proposal and proposal["settings_count"] != len(settings):
        raise ValueError("source settings count mismatch")
    if "settings_digest" in proposal and proposal["settings_digest"] != _digest(settings):
        raise ValueError("source settings digest mismatch")
    plan, seen = {}, set()
    for setting in settings:
        family, shape, geometry = setting["family"], setting["shape"], setting["geometry"]
        allowed = {"height", "width", "modes"} | (
            set() if family == LAYOUT_NAME else {"operation", "coarse"}
        )
        if (
            family not in FAMILIES
            or not isinstance(shape, dict)
            or set(shape) != allowed
            or any(type(v) is not int for v in shape.values())
            or any(shape[k] <= 0 for k in ("height", "width", "modes"))
        ):
            raise ValueError("source shape must contain only exact portable shape fields")
        if family != LAYOUT_NAME and (
            shape["operation"] not in (0, 1, 2) or shape["coarse"] not in (0, 1)
        ):
            raise ValueError("unsupported source operation/coarse")
        if setting.get("dtype", dtype) != dtype:
            raise ValueError("conflicting source dtype")
        signature = family, _digest(shape)
        if signature in seen:
            raise ValueError("duplicate/ambiguous source setting")
        seen.add(signature)
        matches = [
            k
            for k in observed.get(family, [])
            if _scope(k).get("execution_regime") == "compiled"
            and _scope(k).get("dtype") == dtype
            and all(_scope(k)["shape"].get(f) == v for f, v in shape.items())
        ]
        if not matches:
            raise ValueError(f"no actual target scope for {family} {shape}")
        for key in matches:
            _geometry(family, geometry, _scope(key))
            plan.setdefault(family, {})[key] = copy.deepcopy(geometry)
    _parents(plan)
    # An omitted specializer must not silently inherit an unrelated target winner.
    # Add off markers only where the proposal explicitly chooses the parent.
    for parent_family, child_family in (
        (PROFILE_NAME, SEGMENT_NAME),
        (SEGMENT_NAME, DOT_NAME),
        (DOT_NAME, VJP_NAME),
    ):
        for parent_key in list(plan.get(parent_family, {})):
            for key in observed.get(child_family, []):
                if same_launch_scope(parent_key, key) and key not in plan.get(child_family, {}):
                    plan.setdefault(child_family, {})[key] = copy.deepcopy(OFF)
    for parent_key in plan.get(PROFILE_NAME, {}):
        for key in observed.get(STRUCT_NAME, []):
            if same_launch_scope(parent_key, key):
                plan.setdefault(STRUCT_NAME, {}).setdefault(key, copy.deepcopy(OFF))
    _parents(plan)
    return plan


def proposal_candidates(observed, path=None, *, diagnostics=None, provenance=None):
    """Return atomic trial plans and donor-named direct VJP alternatives.

    A bad donor is rejected in full. Other independent valid donors remain usable.
    Optional count/digest provenance guards detect catalog truncation without ever
    treating an origin's source records digest as a destination plan digest.
    """
    diagnostics = diagnostics if diagnostics is not None else []
    provenance = provenance if provenance is not None else {}
    path = CATALOG if path is None else Path(path)
    if not path.exists():
        return {}
    try:
        catalog = json.loads(path.read_text())
        if (
            not isinstance(catalog, dict)
            or catalog.get("schema_version") != 1
            or not isinstance(catalog.get("proposals"), list)
        ):
            raise ValueError("unsupported proposal catalog schema")
    except (OSError, ValueError) as exc:
        diagnostics.append({"proposal": "catalog", "error": str(exc)})
        return {}
    result, names = {}, set()
    for proposal in catalog["proposals"]:
        name = proposal.get("name", "unnamed") if isinstance(proposal, dict) else "unnamed"
        try:
            if name in names:
                raise ValueError("duplicate donor name")
            names.add(name)
            plan = _map_proposal(proposal, observed)
            label = f"proposal-{name}"
            donor_plans = {label: plan}
            direct = [
                key
                for key in plan.get(VJP_NAME, {})
                if (
                    _scope(key)["shape"].get("height"),
                    _scope(key)["shape"].get("operation"),
                    _scope(key)["shape"].get("coarse"),
                )
                == (56, 1, 0)
                and _active(related_geometry(plan, DOT_NAME, key))
                and not _active(related_geometry(plan, STRUCT_NAME, key))
            ]
            for variant in (1, 2):
                if not direct:
                    break
                alternative = copy.deepcopy(plan)
                for key in direct:
                    alternative[VJP_NAME][key] = {
                        "blocks": {"VARIANT": variant},
                        "num_warps": 4,
                        "num_stages": 1,
                    }
                    _geometry(VJP_NAME, alternative[VJP_NAME][key], _scope(key))
                _parents(alternative)
                if alternative != plan:
                    donor_plans[f"{label}-direct-vjp{variant}"] = alternative
            result.update(donor_plans)
            for candidate_name, candidate in donor_plans.items():
                provenance[candidate_name] = dict(
                    source_name=proposal["name"],
                    origin=copy.deepcopy(proposal["origin"]),
                    source_dtype=proposal["dtype"],
                    source_settings_count=len(proposal["settings"]),
                    source_settings_digest=_digest(proposal["settings"]),
                    destination_records_digest=_digest(candidate),
                )
        except (KeyError, TypeError, ValueError) as exc:
            diagnostics.append({"proposal": name, "error": str(exc)})
    return result
