"""CPU forward arithmetic audit and accuracy-blind VA64 convolution sizing.

Uses the frozen VA real-MAC=2/scalar-function=1 arithmetic convention. Candidate
hidden widths are fixed hardware-aligned multiples of eight before profiling;
the closest complete forward FLOP total wins, without reading training results.
Orthogonal analysis weights are materialized once before counting, as in the
frozen inference-operator census; their logits must remain exactly unchanged.
"""
from __future__ import annotations

# Dynamic JSON event descriptors follow the existing frozen auditor schema.
# pyright: reportExplicitAny=false

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from functools import partial
import sys
from typing import Any, cast

import torch
from torch import nn
from torch.nn.utils import parametrize

from profile_rmt_common_flops import DEFAULT_AUDITOR, load_module, profile, sha256

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (8, 16, 24, 32)
TOLERANCE = 0.05
BASELINE_COMMIT = "af875d129dc6dc7a711bce4a8ca217ffe4b22518"


class ConvArithmeticRules:
    """Keep frozen rules, adding the explicitly equivalent scalar log1p call."""

    def __init__(self, auditor: Any) -> None:
        self.auditor = auditor

    def tensor_numel(self, value: Any) -> int:
        return self.auditor.tensor_numel(value)

    def aten_flops(self, event: dict[str, Any]) -> tuple[str, int]:
        if event["op"].split(".")[1] == "log1p":
            return "scalar_aten", self.tensor_numel(event["output"])
        return self.auditor.aten_flops(event)


def fold_parametrizations(model: nn.Module, sample: torch.Tensor) -> list[str]:
    """Exclude fixed inference weight construction, checking exact logits."""
    folded = []
    with torch.inference_mode():
        before = model(sample)
        for name, module in list(model.named_modules()):
            if parametrize.is_parametrized(module):
                for parameter_name in list(cast(nn.ModuleDict, module.parametrizations)):
                    parametrize.remove_parametrizations(
                        module, parameter_name, leave_parametrized=True,
                    )
                    folded.append(f"{name}.{parameter_name}")
        after = model(sample)
    if not torch.equal(before, after):
        raise ValueError("Materializing inference weights changed logits")
    return folded


def choose_candidate(rows: list[dict[str, Any]], reference_flops: int) -> dict[str, Any]:
    if not rows or reference_flops <= 0:
        raise ValueError("Missing candidate or reference arithmetic")
    if any(not row["valid"] or row["unsupported_operators"] for row in rows):
        raise ValueError("Incomplete candidate arithmetic census")
    def distance(row: dict[str, Any]) -> tuple[int, int]:
        return abs(row["flops_per_image"] - reference_flops), row["hidden_channels"]

    chosen = min(rows, key=distance)
    if abs(chosen["flops_per_image"] / reference_flops - 1) > TOLERANCE:
        raise ValueError("No candidate satisfies the predeclared 5% FLOP budget")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, default=DEFAULT_AUDITOR)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--baseline-source-root", type=Path,
                        default=ROOT / "artifacts/h200-va64-g-20261107/repository/src")
    parser.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    args = parser.parse_args()
    source_root = args.baseline_source_root.resolve()
    if not (source_root / "lnet/__init__.py").is_file():
        raise ValueError("Missing frozen baseline source root")
    sys.path.insert(0, str(source_root))
    import lnet
    if Path(lnet.__file__).resolve().parent != source_root / "lnet":
        raise ValueError("Already-imported lnet differs from requested frozen baseline")
    import va64_conv_control as control

    torch.set_num_threads(args.threads)
    auditor = load_module(args.auditor, "frozen_va_conv_forward_auditor")
    reference_path = args.auditor.parent / "rebuttal-full-flops-audit.json"
    reference_document = json.loads(reference_path.read_text())
    if reference_document["auditor_sha256"] != sha256(args.auditor):
        raise ValueError("Frozen reference and active arithmetic rules differ")
    reference = next(row for row in reference_document["models"] if row["model"] == "va_k64")
    if reference["flops_per_image"] != control.BASELINE_FLOPS:
        raise ValueError("Control and frozen baseline FLOP budgets differ")

    rows = []
    scan_inventory = []
    for hidden in CANDIDATES:
        torch.manual_seed(0)
        model = control.build_model(hidden, prepared=False).cpu().float().eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        sample = torch.randn(1, 3, 224, 224)
        handles = []
        if hidden == CANDIDATES[0]:
            def record_scan(name: str, module: control.FourPathConvolution,
                            inputs: tuple[torch.Tensor, ...]) -> None:
                scan_inventory.append({"name": name, "input_shape": list(inputs[0].shape),
                                       "modes": module.modes})

            for name, module in model.named_modules():
                if isinstance(module, control.FourPathConvolution):
                    handles.append(module.register_forward_pre_hook(
                        partial(record_scan, name),
                    ))
        folded = fold_parametrizations(model, sample)
        for handle in handles:
            handle.remove()
        row = profile(model, ConvArithmeticRules(auditor))
        if row["params"] != parameter_count:
            raise ValueError("Inference materialization changed parameter count")
        row.update(hidden_channels=hidden, folded_parameters=folded,
                   materialized_logits_exactly_equal=True)
        rows.append(row)
    chosen = choose_candidate(rows, reference["flops_per_image"])
    ratio = chosen["flops_per_image"] / reference["flops_per_image"]
    source_paths = [Path(__file__), Path(control.__file__), ROOT / "scripts/profile_rmt_common_flops.py",
                    args.auditor, reference_path]
    import run_va64_retrain_imagenet1k as baseline
    source_paths.append(Path(baseline.__file__))
    sources = {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path)
               for path in source_paths}
    frozen_source = Path(lnet.__file__).resolve().parent
    frozen_sources = {str(path.relative_to(frozen_source)): sha256(path)
                      for path in sorted(frozen_source.rglob("*.py"))}
    frozen_source_sha = hashlib.sha256(json.dumps(frozen_sources, sort_keys=True).encode()).hexdigest()
    checkout_commit = subprocess.check_output(
        ["git", "-C", str(frozen_source.parents[1]), "rev-parse", "HEAD"], text=True,
    ).strip()
    # The checkout may receive unrelated runner commits while this job runs.
    # Bind the actual profiled backbone bytes to the intended archive commit.
    for relative_path, digest in frozen_sources.items():
        pinned_contents = subprocess.check_output(
            ["git", "-C", str(frozen_source.parents[1]), "show",
             f"{args.baseline_commit}:src/lnet/{relative_path}"],
        )
        if hashlib.sha256(pinned_contents).hexdigest() != digest:
            raise ValueError(f"Frozen backbone differs from pinned commit: {relative_path}")
    # Two parity forwards observe the same sites; each entry is one scan site.
    scan_inventory = list({entry["name"]: entry for entry in scan_inventory}.values())
    if len(scan_inventory) != 12:
        raise ValueError("Expected twelve executed convolution replacement sites")
    expected_shapes = sorted([(56, 56, 80)] * 2 + [(28, 28, 80)] * 2
                             + [(14, 14, 80)] * 6 + [(7, 7, 80)] * 2)
    if sorted(tuple(entry["input_shape"][1:]) for entry in scan_inventory) != expected_shapes:
        raise ValueError("Executed scan geometry differs from frozen VA64")
    result = {
        "schema": "visionalphabet.va64_conv_budget.v1", "valid": True,
        "hidden_channels": chosen["hidden_channels"], "flops_per_image": chosen["flops_per_image"],
        "params": chosen["params"], "unsupported_operators": chosen["unsupported_operators"],
        "definition": control.definition(chosen["hidden_channels"]),
        "stage_pole_widths": [80, 80, 80, 80], "stage_depths": [2, 2, 6, 2],
        "stage_scan_resolutions": [56, 28, 14, 7], "scan_site_inventory": scan_inventory,
        "convention": auditor.__doc__, "torch_version": torch.__version__,
        "additional_scalar_rule": "log1p counts one real scalar special-function equivalent per output",
        "inference_weight_materialization": "Exact-logit checked; excludes fixed orthogonal weight construction",
        "comparison": {"reference_model": "va_k64", "reference_params": reference["params"],
                       "reference_flops_per_image": reference["flops_per_image"],
                       "conv_to_baseline_flops_ratio": ratio, "tolerance_fraction": TOLERANCE,
                       "within_tolerance": True, "parameter_matched": False},
        "selection": {"candidate_hidden_channels": list(CANDIDATES),
                      "rule": "minimum absolute full-forward FLOP difference; ties prefer fewer hidden channels",
                      "validation_results_used": False,
                      "candidate_constraint": "positive multiples of 8 for accelerator channel alignment"},
        "source_sha256": sources, "frozen_deployment_commit": args.baseline_commit,
        "observed_source_checkout_commit": checkout_commit,
        "profiled_backbone_matches_pinned_commit": True,
        "frozen_lnet_source_sha256": frozen_source_sha,
        "frozen_lnet_source_files_sha256": frozen_sources, "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("valid", "hidden_channels", "params", "flops_per_image", "comparison")}, indent=2))


if __name__ == "__main__":
    main()
