"""Validate on the production comparator and persist raw per-tensor metrics."""

import argparse
import json
import math
from pathlib import Path

import torch

from effective_batch_search import compare_tensors

COUNTS = {"outputs": 5, "grads": 193, "updates": 193}
POLE = "backbone.terminal.pole_input_projection.point_weight"


def compare(reference, candidate):
    ref, cand = reference["checks"], candidate["checks"]
    if any(len(group) != COUNTS[name] for groups in (ref, cand) for name, group in groups.items()):
        raise ValueError("Unexpected output, gradient, or AdamW tensor count")
    official = compare_tensors(ref, cand)
    numbers = {}
    thresholds = {}
    for name in COUNTS:
        if ref[name].keys() != cand[name].keys():
            raise ValueError(f"Tensor keys differ in {name}")
        rows = numbers[name] = {}
        for key, a in ref[name].items():
            b = cand[name][key]
            finite = a.shape == b.shape and bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            if finite:
                a, b = a.float(), b.float()
                delta = b - a
                absolute, relative = delta.abs().max().item(), (delta.norm() / a.norm().clamp_min(1e-12)).item()
                finite = math.isfinite(absolute) and math.isfinite(relative)
            rows[key] = dict(
                max_abs=absolute if finite else None, rel_l2=relative if finite else None, finite=finite,
                requested_pass=finite and relative <= .02 and (name != "updates" or absolute <= 3e-5),
            )
        failures = [key for key, v in rows.items() if not v["requested_pass"]]
        thresholds[name] = dict(count=len(rows), failed=failures)
    if any(reference[k] != candidate[k] for k in ("samples", "microbatches", "optimizer_steps")):
        raise ValueError("Capture sample, microbatch, or optimizer step counts differ")
    scalars = {}
    for key in ("loss", "clip_norm"):
        a, b = float(reference[key]), float(candidate[key])
        finite = math.isfinite(a) and math.isfinite(b)
        delta = abs(a - b) if finite else None
        scalars[key] = dict(reference=a if math.isfinite(a) else None,
                            candidate=b if math.isfinite(b) else None,
                            max_abs=delta, rel_l2=delta / max(abs(a), 1e-12) if finite else None,
                            passed=finite and delta <= .02 * abs(a) + 1e-6)
    scalar_pass = all(v["passed"] for v in scalars.values())
    return dict(passed=official["passed"] and scalar_pass, official=official, raw=numbers,
                requested_thresholds=dict(passed=scalar_pass and not any(v["failed"] for v in thresholds.values()),
                                          groups=thresholds),
                scalars=scalars)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    for key in ("microbatch", "checkpoint_sha256", "contract_sha256", "fixture_sha256", "primitive_sha256",
                "workload_sha256"):
        if json.loads(args.reference.read_text())[key] != json.loads(args.candidate.read_text())[key]:
            raise ValueError(f"Cannot compare work with different {key}")
    ref, cand = [torch.load(path.with_suffix(".capture.pt"), map_location="cpu", weights_only=True)
                 for path in (args.reference, args.candidate)]
    result = compare(ref, cand)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, allow_nan=False))
    tmp.replace(args.output)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
