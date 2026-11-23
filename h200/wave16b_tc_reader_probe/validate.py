"""Validate the model checks emitted as JSON by the Wave 16B runner."""

import argparse
import json
from pathlib import Path


def passes(kind: str, report: dict) -> bool:
    if kind == "dynamo":
        graphs = [result for modes in report["graphs"].values() for result in modes.values()]
        return graphs == [[1, 0]] * 18
    if kind == "fold":
        return (
            report["strict_reload_passed"]
            and report["folded_train_guard"]
            and report["fp32_head_max_abs_logit_diff"] < 1e-5
        )
    return all(
        report[key]
        for key in (
            "finite_losses",
            "finite_gradients",
            "running_ms_finite",
        )
    )


def validate(out: Path) -> None:
    failures = []
    for policy in ("off", "force"):
        for kind in ("dynamo", "fold", "train"):
            path = out / f"{kind}-{policy}.json"
            if not path.is_file():
                failures.append(f"{path.name} is missing")
                continue
            report = json.loads(path.read_text())
            if report["status"] != "ok":
                failures.append(f"{path.name} has status {report['status']}")
            elif not passes(kind, report):
                failures.append(f"{path.name} failed the {kind} checks")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    validate(parser.parse_args().out)
