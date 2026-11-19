"""Step 4: One-by-one structural candidate evaluation against the H200 r3 winner control
and 3-pair order-alternating promotion on fixed 1,024-image input.
"""
from __future__ import annotations

import copy
import json
import os
import statistics
from pathlib import Path

import fixed_vjp as f
from automatic_training_layout import DOT_NAME, LAYOUT_NAME, PROFILE_NAME, SEGMENT_NAME, digest
from effective_batch_proposals import _map_proposal
from effective_batch_search import STRUCT_NAME, VJP_NAME, _structural_contract, _vjp_contract

DISABLED = dict(blocks={}, num_warps=4, num_stages=1)


def canonical(m: int, records: dict) -> dict:
    defaults = f.disc[m]["defaults"]
    merged = f.complete(m, records)
    for name in list(merged):
        merged[name] = {
            k: g
            for k, g in merged[name].items()
            if g.get("blocks") or defaults.get(name, {}).get(k, {}).get("blocks")
        }
        if not merged[name]:
            del merged[name]
    return merged


def register(m: int, records: dict, label: str) -> str:
    return f.put(m, canonical(m, records), label)


def keys(m: int, n: str, h: int, op: int | None = None, coarse: int | None = None):
    for k in f.disc[m]["observed"].get(n, []):
        s = json.loads(k)["shape"]
        if (
            s["height"] == h
            and (op is None or s.get("operation") == op)
            and (coarse is None or s.get("coarse") == coarse)
        ):
            yield k


def write_scope(plan: dict, m: int, n: str, h: int, op: int | None, coarse: int | None, geometry: dict) -> int:
    found = list(keys(m, n, h, op, coarse))
    if found:
        for k in found:
            plan.setdefault(n, {})[k] = copy.deepcopy(geometry)
    return len(found)


def clear_struct(plan: dict) -> None:
    plan.pop(STRUCT_NAME, None)


def set_struct(plan: dict, m: int, op: int, coarse: int, variant: int) -> None:
    if variant == 0:
        for k in keys(m, STRUCT_NAME, 56, op, coarse):
            plan.get(STRUCT_NAME, {}).pop(k, None)
    else:
        write_scope(
            plan,
            m,
            STRUCT_NAME,
            56,
            op,
            coarse,
            dict(blocks=dict(VARIANT=variant), num_warps=4, num_stages=1),
        )


def set_vback_tile(plan: dict, m: int, coarse: int, bm: int, warps: int, group: int) -> None:
    geom = dict(blocks=dict(BM=bm, GROUP=group, DOUBLING=1), num_warps=warps, num_stages=1)
    for fam in (PROFILE_NAME, SEGMENT_NAME, DOT_NAME):
        write_scope(plan, m, fam, 56, 1, coarse, geom)


def measured_confirm_single(tag: str, label: str) -> dict | None:
    # Run with confirm=True (2 warmups, 3 groups x 2 updates) and profile=True
    ans = f.measure(tag, label, confirm=True, profile=True)
    if ans is None:
        return None
    row = ans[0]
    return dict(
        tag=tag,
        label=f.PLANS[tag][2],
        ms=row["ms"],
        groups_ms=row["groups_ms"],
        img_s=row["img_s"],
        peak_mib=row["peak_mib"],
        numeric=row["reference"]["passed"],
        reference=row["reference"],
        applied=row.get("applied", []),
        trace=row.get("trace", {}),
    )


def run() -> None:
    f.snapshot()
    m = 1024
    assert f.observed(m)
    assert f.observed(128)

    # Reconstruct the exact H200 r3 winner (h200-e1024-718) from effective_update_proposals.json
    catalog_path = f.source / "src/lnet/effective_update_proposals.json"
    catalog = json.loads(catalog_path.read_text())
    h200_prop = next(p for p in catalog["proposals"] if p["name"] == "h200-e1024-718")
    incumbent_records = _map_proposal(h200_prop, f.disc[m]["observed"])

    control_tag = register(m, incumbent_records, "control-h200-r3-winner")
    f.champions.add(control_tag)

    print("EVALUATING_CONTROL", control_tag, flush=True)
    control_eval = measured_confirm_single(control_tag, "onebyone-00-control")
    assert control_eval is not None and control_eval["numeric"], "Control evaluation failed"
    f.audit["initial"] = dict(
        tag=control_tag,
        source="h200-e1024-718",
        ms=control_eval["ms"],
        trace=control_eval["trace"],
    )

    one_by_one_results = [control_eval]
    f.audit["one_by_one"] = one_by_one_results
    f.save()

    # Step 3/4: Test each structural change one by one on 56x56 Direct (c0) against control
    c0_variants = [
        (1, "v1-recompute_prev-c0"),
        (2, "v2-fast_mlp-c0"),
        (3, "v3-half_hidden-c0"),
        (4, "v4-unpack_prev-c0"),
        (5, "v5-unpack_rows-c0"),
        (6, "v6-col_loop2-c0"),
    ]
    c0_tags = {}
    for vid, vname in c0_variants:
        plan = copy.deepcopy(incumbent_records)
        clear_struct(plan)
        set_struct(plan, m, 1, 0, vid)
        _vjp_contract(plan)
        _structural_contract(plan)
        tag = register(m, plan, vname)
        c0_tags[vid] = tag
        res = measured_confirm_single(tag, f"onebyone-c0-v{vid}")
        if res is not None:
            res["delta_vs_control_ms"] = res["ms"] - control_eval["ms"]
            one_by_one_results.append(res)
            f.save()

    # Also test c0 warp=4 (REG=248 < 255) on v4 (unpack_prev) and v3 (half_hidden)
    for vid, vname in ((4, "v4-unpack_prev-c0-bm4w4"), (3, "v3-half_hidden-c0-bm4w4")):
        plan = copy.deepcopy(incumbent_records)
        clear_struct(plan)
        set_vback_tile(plan, m, 0, 4, 4, 4)
        set_struct(plan, m, 1, 0, vid)
        _vjp_contract(plan)
        _structural_contract(plan)
        tag = register(m, plan, vname)
        res = measured_confirm_single(tag, f"onebyone-c0-v{vid}-w4")
        if res is not None:
            res["delta_vs_control_ms"] = res["ms"] - control_eval["ms"]
            one_by_one_results.append(res)
            f.save()

    # Test 56->28 Coarse (c1) structural variants (v4 unpack_prev and v3 half_hidden) isolated
    for vid, vname in ((4, "v4-unpack_prev-c1-only"), (3, "v3-half_hidden-c1-only")):
        plan = copy.deepcopy(incumbent_records)
        clear_struct(plan)
        set_struct(plan, m, 1, 1, vid)
        _vjp_contract(plan)
        _structural_contract(plan)
        tag = register(m, plan, vname)
        res = measured_confirm_single(tag, f"onebyone-c1-v{vid}")
        if res is not None:
            res["delta_vs_control_ms"] = res["ms"] - control_eval["ms"]
            one_by_one_results.append(res)
            f.save()

    # Test Forward 56x56 Direct (op=0, c0) fast_forward (VARIANT=1) isolated
    plan_fwd = copy.deepcopy(incumbent_records)
    clear_struct(plan_fwd)
    set_struct(plan_fwd, m, 0, 0, 1)
    _vjp_contract(plan_fwd)
    _structural_contract(plan_fwd)
    tag_fwd = register(m, plan_fwd, "vfwd-v1-fast_forward-only")
    res_fwd = measured_confirm_single(tag_fwd, "onebyone-vfwd-v1")
    if res_fwd is not None:
        res_fwd["delta_vs_control_ms"] = res_fwd["ms"] - control_eval["ms"]
        one_by_one_results.append(res_fwd)
        f.save()

    # Now build combined structural candidates from the top c0 variants + c1 + vfwd
    valid_c0 = [
        r
        for r in one_by_one_results[1:]
        if r["numeric"] and r["label"].startswith(("v1-", "v2-", "v3-", "v4-", "v5-", "v6-")) and "c1-only" not in r["label"]
    ]
    valid_c0.sort(key=lambda r: r["ms"])
    top_c0_tags = [r["tag"] for r in valid_c0[:3]]

    combo_tags = []
    for base_t in top_c0_tags:
        _, base_rec, base_lbl = f.PLANS[base_t]
        # Add c1 VARIANT=4 (unpack_prev) and VARIANT=3 (half_hidden)
        for c1_v in (4, 3):
            plan = copy.deepcopy(base_rec)
            set_struct(plan, m, 1, 1, c1_v)
            _structural_contract(plan)
            t_c1 = register(m, plan, f"{base_lbl}+c1_v{c1_v}")
            combo_tags.append(t_c1)
            res = measured_confirm_single(t_c1, f"combo-{base_lbl}-c1v{c1_v}")
            if res is not None:
                res["delta_vs_control_ms"] = res["ms"] - control_eval["ms"]
                one_by_one_results.append(res)
                f.save()
            # Also add forward VARIANT=1
            plan_f = copy.deepcopy(plan)
            set_struct(plan_f, m, 0, 0, 1)
            _structural_contract(plan_f)
            t_f = register(m, plan_f, f"{base_lbl}+c1_v{c1_v}+vfwd1")
            combo_tags.append(t_f)
            res_f = measured_confirm_single(t_f, f"combo-{base_lbl}-c1v{c1_v}-vfwd1")
            if res_f is not None:
                res_f["delta_vs_control_ms"] = res_f["ms"] - control_eval["ms"]
                one_by_one_results.append(res_f)
                f.save()

    # Rank all tested candidates by median ms and run 3-pair order-alternating confirmation!
    all_candidates = [
        r for r in one_by_one_results[1:] if r["numeric"] and r["ms"] < control_eval["ms"]
    ]
    all_candidates.sort(key=lambda r: r["ms"])

    best = control_tag
    # Also ensure we run 3-pair confirmation for the best single c0 change against control_tag,
    # and then the overall fastest combined candidate against control_tag / new best.
    single_c0_candidates = [
        r
        for r in all_candidates
        if "+c1" not in r["label"] and "only" not in r["label"]
    ]
    if single_c0_candidates:
        best_single_tag = single_c0_candidates[0]["tag"]
        print("CONFIRMING_BEST_SINGLE", best_single_tag, single_c0_candidates[0]["label"], flush=True)
        if f.confirm(best, best_single_tag):
            best = best_single_tag

    for cand in all_candidates[:4]:
        ctag = cand["tag"]
        if ctag == best or ctag in f.champions:
            continue
        print("CONFIRMING_CANDIDATE", ctag, cand["label"], cand["ms"], flush=True)
        if f.confirm(best, ctag):
            best = ctag
            break
        # Also test directly against initial control_tag if best had already advanced
        if best != control_tag and (control_tag, ctag) not in f.seen:
            if f.confirm(control_tag, ctag, advance=False):
                pass

    assert f.profile_winner(best)
    wm, records, label = f.PLANS[best]
    f.audit["observed_shapes"] = {
        str(k): {n: [json.loads(x)["shape"] for x in ks] for n, ks in v["observed"].items()}
        for k, v in f.disc.items()
    }
    (f.root / "final-discovery.json").write_text(
        json.dumps(dict(observed=f.disc[wm]["observed"], defaults=f.disc[wm]["defaults"]))
    )
    f.audit["alternatives_exhausted"] = not f.audit["errors"]
    f.audit["fixed_best"] = dict(micro=wm, records=records, name=label, tag=best)
    # In matched_table.py:
    # 1. 'native': compare winner vs control_h200_r3_winner (and vs bundle defaults)
    # We pass incumbent_records as defaults for 'native_vs_r3_winner' or run both native (vs bundle default) and vs r3_winner!
    f.audit["comparison_plans"] = [
        dict(
            role="native",
            micro=wm,
            winner=records,
            control_override=incumbent_records,
            defaults=canonical(wm, incumbent_records),
        )
    ]
    if wm != 128 and f.observed(128):
        p128 = f.transplant(records, 128)
        inc128 = f.transplant(incumbent_records, 128)
        f.audit["comparison_plans"].append(
            dict(
                role="matched_4090_micro_trial",
                micro=128,
                winner=canonical(128, p128),
                control_override=canonical(128, inc128),
                defaults=canonical(128, inc128),
                mapped_scopes=sum(map(len, p128.values())),
                parent_scopes=sum(map(len, records.values())),
            )
        )
    f.save()


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        f.audit["errors"].append(repr(exc)[-400:])
        raise
    finally:
        f.save()
