"""Build cost tables and update a private rebuttal draft after all nine runs finish."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path


def finalize(root: Path, template: Path) -> None:
    summary = json.loads((root / "summary.json").read_text())
    if summary["completed"] != 9 or any(r["status"] != "completed" for r in summary["records"]):
        raise ValueError("all nine verified endpoints are required for the final report")
    rows = []
    for result in summary["records"]:
        v, s = result["variant"], result["seed"]
        run = root / "runs" / f"va_k128_retrain_{v}" / f"seed_{s}"
        inventory = json.loads((run / "parameter-inventory.json").read_text())
        cost = json.loads((run / "cost.json").read_text())
        raw = json.loads((run / "result.json").read_text())
        bench = json.loads(
            (
                root
                / "preflight"
                / f"{v}-b{json.loads((root / 'campaign.json').read_text())['batch_size']}.json"
            ).read_text()
        )
        rows.append(
            {
                **result,
                "validation_loss": raw["final_validation"]["cross_entropy"],
                "validation_examples": raw["final_validation"]["examples"],
                "stored_parameters": inventory["total"]["stored_elements"],
                "trainable_parameters": inventory["total"]["trainable_elements"],
                "head_parameters": inventory["head"]["stored_elements"],
                "phase_trainable": inventory["phase"]["trainable_elements"],
                "gpu_hours": cost["gpu_hours"],
                "peak_allocated_bytes": cost["peak_allocated_bytes"],
                "peak_reserved_bytes": cost["peak_reserved_bytes"],
                "synthetic_train_images_s": bench["train_images_per_second"],
                "synthetic_b1_ms": bench["inference"]["1"]["median_ms"],
                "synthetic_b16_images_s": bench["inference"]["16"]["images_per_second"],
            }
        )
    report = root / "rebuttal"
    report.mkdir(exist_ok=True)
    with (report / "seed-results-and-cost.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def group(v: str) -> str:
        stats = summary["groups"][v]
        return f"{stats['mean']:.3f} ± {stats['sample_sd']:.3f}%"

    def difference(key: str) -> str:
        stats = summary["paired"][key]
        low, high = stats["t95_ci"]
        return f"{stats['mean_pp']:+.3f} pp (sample SD {stats['sample_sd_pp']:.3f}; exploratory 95% t interval [{low:+.3f}, {high:+.3f}] pp)"

    phase = (
        f"**T1. Retrained component ablations.** We trained the original K128 (B), "
        f"a model with all effective scan phases fixed to zero (P), and a model with "
        f"terminal direction-averaged energies (A), each from scratch for 100 epochs "
        f"on ImageNet-1K with seeds 0, 1, and 2 and an otherwise shared recipe. "
        f"Final-epoch Top-1 was {group('B')} for B and {group('P')} for P "
        f"(mean ± sample SD). The paired B-P difference was {difference('B-P')}. "
        "These results measure the effect of scan-phase freedom after retraining in this setting. "
        "P retains complex representations and is not a true-real baseline. "
        "A real-valued recurrence and a conventional pooled-feature classifier remain untested. "
        "The three-seed confidence interval is unstable and does not establish equivalence or universal necessity."
    )
    direction = (
        f"**Retrained directional-readout comparison.** A achieved {group('A')}; "
        f"the paired B-A difference was {difference('B-A')}. "
        "A averages normalized terminal energies before log1p and retains all four scans. "
        "Its head has 128 inputs instead of 512, reducing total parameters from 5,083,176 to "
        "4,699,176. Therefore the comparison includes readout/head-capacity and optimization effects; "
        "it does not isolate direction information alone. A fixed single-direction readout remains untested."
    )
    source = template.read_text()
    begin = source.index("**T1.")
    end = source.index("**T2.", begin)
    source = source[:begin] + phase + "\n\n" + source[end:]
    target = "A comparison between concatenated directional energies, direction-averaged energies, and a fixed single-direction readout is needed."
    if target not in source:
        raise ValueError("rebuttal template changed; refusing an ambiguous replacement")
    source = source.replace(
        target,
        "The comparison with direction-averaged energies is now available below; a fixed single-direction readout remains an open comparison.",
        1,
    )
    marker = "**T3. Computational trade-off.**"
    source = source.replace(marker, direction + "\n\n" + marker, 1)
    source = source.replace(
        "No new training results are included.",
        "This version incorporates all nine completed B/P/A retraining runs.",
    )
    note = "Internal note: This conservative draft addresses every review item but cannot substitute for the missing experiments."
    source = source.replace(
        note,
        "Internal note: This draft incorporates B/P/A retraining; other missing experiments remain explicit.",
    )
    (report / "response-draft-with-retraining.en.md").write_text(source)
    md = [
        "# K128 retraining: final endpoints",
        "",
        "| Variant | Top-1 mean ± sample SD |",
        "|---|---|",
    ]
    md += [f"| {v} | {group(v)} |" for v in "BPA"]
    md += [
        "",
        f"B-P: {difference('B-P')}",
        f"B-A: {difference('B-A')}",
        "",
        "Practical threshold fixed before training: 0.3 pp. No extra seeds selected.",
        "Full seed/cost table: seed-results-and-cost.csv.",
        "Synthetic benchmark throughput excludes image loading and is not end-to-end epoch throughput.",
        "P is not a true-real model. B-A includes head-capacity differences.",
        "Updated private response draft: response-draft-with-retraining.en.md.",
    ]
    (report / "final-report.md").write_text("\n".join(md) + "\n")
    provenance = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
        "completed_runs": 9,
    }
    (report / "report-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        "All nine runs complete; final cost table and private rebuttal draft generated.", flush=True
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--wait", action="store_true")
    a = p.parse_args()
    while a.wait:
        status = a.root / "status.json"
        if status.exists():
            state = json.loads(status.read_text())["state"]
            if state == "completed":
                break
            if state == "failed" or state.endswith("_failed"):
                raise RuntimeError(
                    f"campaign needs attention: {state}; no completed-results draft produced"
                )
        time.sleep(60)
    finalize(a.root, a.template)


if __name__ == "__main__":
    main()
