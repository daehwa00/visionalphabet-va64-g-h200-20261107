"""Print CUDA Graph and training metrics for the Wave 25C report."""

# ruff: noqa: T201
import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median


def isolated(root: Path) -> None:
    path = root / "isolated.log"
    if not path.exists():
        return
    print("| dtype | HxW | CI→CO | off F/B/total ms | on F/B/total ms | total 변화 |")
    print("|---|---|---|---:|---:|---:|")
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        off, on = [
            list(map(median, zip(*row["timings"][arm], strict=True))) for arm in ("off", "on")
        ]
        off_ms, on_ms = ("/".join(f"{value:.3f}" for value in values) for values in (off, on))
        h, w = row["hw"]
        change = 100 * (on[2] / off[2] - 1)
        print(
            f"| {row['dtype']} | {h}x{w} | {row['ci']}→{row['co']} | "
            f"{off_ms} | {on_ms} | {change:+.2f}% |"
        )


def accuracy(root: Path) -> None:
    path = root / "correctness.log"
    if not path.exists():
        return
    pattern = re.compile(
        r"(torch\.\w+) \(\d+, \d+\) \d+ \d+ (True|False) (True|False) "
        r"(?:True|False) (off|fp32) (\[.*?\])"
    )
    errors = defaultdict(list)
    for dtype, bias, input_grad, reference, values in pattern.findall(path.read_text()):
        fields = ["output", "dWeight"]
        if bias == "True":
            fields.append("dConvBias")
        fields.extend(("dLN_w", "dLN_b"))
        if input_grad == "True":
            fields.append("dInput")
        for field, value in zip(fields, ast.literal_eval(values), strict=True):
            errors[dtype, input_grad, reference, field].append(value)
    print("| dtype | dInput 요청 | 기준 | 항목 | 조합 수 | max abs | max rel L2 |")
    print("|---|---|---|---|---:|---:|---:|")
    for key, values in sorted(errors.items()):
        absolute, relative = map(max, zip(*values, strict=True))
        print("|", " | ".join(key), f"| {len(values)} | {absolute:.6g} | {relative:.6g} |")


def train(root: Path) -> None:
    print(
        "| 순서 | arm | wall ms | cuDNN ms | stem ms | 합계 ms | 최대 allocated B | SM clock / °C |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for pair in ("forward", "reverse", "controlled1", "controlled2"):
        modes = ("on", "off") if pair == "reverse" else ("off", "on")
        for mode in modes:
            path = root / "results" / f"train-{pair}-{mode}.json"
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            if "profiler" not in row:
                print(f"| {pair} | {mode} | {row['status']} | | | | | |")
                continue
            buckets = row["profiler"]["self_cuda_ms_by_bucket"]
            cudnn, stem = buckets["cudnn_conv"], buckets["fused_stem"]
            clock = " → ".join(
                line.strip()
                for line in path.with_suffix(".log").read_text().splitlines()
                if " MHz," in line
            )
            print(
                f"| {pair} | {mode} | {row['median_ms']:.3f} | {cudnn:.3f} | {stem:.3f} | "
                f"{cudnn + stem:.3f} | {row['peak_allocated_bytes']:,} | {clock} |"
            )


def trajectories(root: Path) -> None:
    print(
        "| 순서 | 최대 loss 차이 | 최대 pre-clip norm 차이 | wall 변화 | stem 합계 변화 |"
        " peak allocated 차이 |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for pair in ("forward", "reverse", "controlled1", "controlled2"):
        paths = [root / "results" / f"train-{pair}-{mode}.json" for mode in ("off", "on")]
        if not all(path.exists() for path in paths):
            continue
        off, on = [json.loads(path.read_text()) for path in paths]
        if not all(row["status"] == "ok" for row in (off, on)):
            continue
        loss, norm = [
            max(abs(a - b) for a, b in zip(off[key], on[key], strict=True))
            for key in ("trajectory_loss", "trajectory_preclip_norm")
        ]
        wall_pct = 100 * (on["median_ms"] / off["median_ms"] - 1)
        memory = on["peak_allocated_bytes"] - off["peak_allocated_bytes"]
        off_bucket, on_bucket = [
            sum(
                row["profiler"]["self_cuda_ms_by_bucket"][bucket]
                for bucket in ("cudnn_conv", "fused_stem")
            )
            for row in (off, on)
        ]
        print(
            f"| {pair} | {loss:.6g} | {norm:.6g} | {wall_pct:+.2f}% | "
            f"{on_bucket - off_bucket:+.3f} ms | {memory:+,} B |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    isolated(args.root)
    print()
    train(args.root)
    print()
    trajectories(args.root)
    print()
    accuracy(args.root)


if __name__ == "__main__":
    main()
