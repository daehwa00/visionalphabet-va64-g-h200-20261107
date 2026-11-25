# ruff: noqa: PLC0415, T201
"""Package verified H200 output and render the screening report."""

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import platform
import re
import textwrap
from collections.abc import Iterable, Iterator
from pathlib import Path

FILES = ("meta.json", "scan_vscan_bwd.json", "reader_wave45b.json", "stem_stage0.json")
DTYPES = ("bf16", "fp16")
Summary = tuple[list[str], tuple[str, float] | None]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def emit() -> None:
    import torch
    import triton

    out = Path(os.environ["OUT"])
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    meta = {
        "sha": os.environ["VA_SHA"],
        "image": "pytorch/pytorch:latest",
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": gpu.name if gpu else None,
        "vram_bytes": gpu.total_memory if gpu else None,
        "sm_count": gpu.multi_processor_count if gpu else None,
        "nvidia_smi": os.environ["VA_NVIDIA_SMI"],
        "bootstrap_seconds": 0,
        "stages": {},
    }
    for line in (out / "status").read_text().splitlines():
        stage, rc, seconds = line.split()
        meta["stages"][stage] = {"exit": int(rc), "seconds": int(seconds)}
    (out / FILES[0]).write_text(json.dumps(meta, indent=2) + "\n")
    values = {}
    for name in FILES:
        path = out / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        expected = digest(raw)
        values[name] = json.loads(raw)
        print(f"{expected}  {name}")
        print(f"VA_T159_BEGIN {name} {expected}")
        payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
        for line in textwrap.wrap(payload, 480):
            print(f"VA_T159_GZ {line}")
        print(f"VA_T159_END {name}")
    for _, title, lines, _ in summarize(values):
        print(title)
        print("\n".join(lines))


def decode(log: str, out: Path) -> tuple[dict, dict[str, str]]:
    out.mkdir(parents=True, exist_ok=True)
    values, hashes = {}, {}
    for name in FILES:
        quoted = re.escape(name)
        begin = rf"^VA_T159_BEGIN {quoted} ([0-9a-f]{{64}})\s*\n"
        end = rf"^VA_T159_END {quoted}\s*$"
        block = re.search(begin + "(.*?)" + end, log, re.MULTILINE | re.DOTALL)
        if block is None:
            if re.search(begin, log, re.MULTILINE):
                raise ValueError(f"Truncated artifact {name}")
            continue
        expected, body = block.groups()
        raw = gzip.decompress(
            base64.b64decode(
                "".join(re.findall(r"^VA_T159_GZ ([A-Za-z0-9+/=]+)$", body, re.MULTILINE)),
                validate=True,
            )
        )
        if digest(raw) != expected or not re.search(rf"^{expected}  {quoted}$", log, re.MULTILINE):
            raise ValueError(f"SHA-256 mismatch for {name}")
        values[name] = json.loads(raw)
        hashes[name] = expected
        (out / name).write_bytes(raw)
    return values, hashes


def table(headers: list[str], rows: Iterable[list]) -> list[str]:
    def row(fields: Iterable[object]) -> str:
        return "| " + " | ".join(str(field) for field in fields) + " |"

    return [row(headers), row(["---"] * len(headers)), *(row(fields) for fields in rows), ""]


def verdict(value: bool) -> str:  # noqa: FBT001
    return "PASS" if value else "FAIL"


def worst(rows: list[dict], key: str) -> str:
    vals = [row[key] for row in rows]
    if any(value is None or not math.isfinite(value) for value in vals):
        return "nonfinite"
    return f"{max(vals):.9g}"


def scan(data: dict) -> Summary:
    result = data["vscan_bwd"]
    shapes, totals = result["shapes"], result["weighted_ms"]
    modes = list(shapes)
    lines = table(
        ["Shape (calls)", *modes],
        (
            [
                f"{label} ({case['calls']})",
                *(f"{shapes[mode][label]['median_ms']:.6f}" for mode in modes),
            ]
            for label, case in shapes["off"].items()
        ),
    )
    lines += table(
        ["Mode", "Weighted ms", "Δ off", "Δ strict_ro", "Timing kernel coverage"],
        (
            [
                mode,
                f"{totals[mode]:.6f}",
                f"{totals[mode] - totals['off']:+.6f}",
                f"{totals[mode] - totals['strict_ro']:+.6f}",
                verdict(result["timing_kernel_covered"][mode]),
            ]
            for mode in modes
        ),
    )
    rows = []
    for mode in modes:
        for dtype in DTYPES:
            samples = [
                stats
                for label, case in result["parity"].items()
                if label.startswith(dtype + "/")
                for stats in case["modes"][mode]["tensors"].values()
            ]
            rows.append(
                [
                    mode,
                    dtype,
                    len(samples),
                    worst(samples, "max_abs"),
                    worst(samples, "max_rel"),
                    verdict(all(s["exact_match"] for s in samples)),
                    verdict(all(s["tol_pass"] for s in samples)),
                ]
            )
    lines += table(["Mode", "Dtype", "Tensors", "Max abs", "Max rel", "Exact", "Tolerance"], rows)
    winner = result["best_tolerated_mode"]
    delta = totals[winner] - totals["strict_ro"]
    return lines, (winner, delta) if delta < 0 else None


def logical_bytes(row: dict) -> int:
    batch, h, w, modes, rank = row["shape"]
    projected = 2 * batch * h * w * modes * rank
    grad = 4 * batch * h * w * modes
    filters = 2 * modes * rank * 9 * 2
    if row["op"] == "dweight":
        plan = row["plan"]
        partial = 4 * plan["grid"][1] * 2 * rank * 9 * modes
        return projected + grad + 2 * partial + 2 * filters
    return projected + grad + filters


def reader(data: dict) -> Summary:
    totals = data["weighted_ms"]
    modes = list(totals)
    grouped, traffic = {}, dict.fromkeys(modes, 0)
    for row in data["timing_rows"]:
        _, h, w, m, r = row["shape"]
        nbytes = logical_bytes(row)
        traffic[row["mode"]] += row["calls"] * nbytes
        grouped.setdefault((f"{h}x{w}x{m}x{r}", row["op"]), {})[row["mode"]] = (
            f"{row['ms']:.6f} / {nbytes / row['ms'] / 1e9:.3f}"
        )
    lines = [
        "각 셀은 ms / 유효 TB/s입니다. 유효 TB/s는 입출력과 dweight 임시 버퍼의 "
        "논리적 바이트 기준이며 HBM 하드웨어 카운터를 사용하지 않았습니다.",
        "",
    ]
    lines += table(
        ["Shape", "Operation", *modes],
        (
            [shape, op, *(samples[mode] for mode in modes)]
            for (shape, op), samples in grouped.items()
        ),
    )
    lines += table(
        [
            "Mode",
            "Weighted fwd",
            "Weighted dinput",
            "Weighted dweight",
            "Weighted total ms",
            "Δ off",
            "Effective TB/s",
        ],
        (
            [
                mode,
                *(f"{totals[mode][op]:.6f}" for op in ("fwd", "dinput", "dweight", "total")),
                f"{totals[mode]['delta_vs_off_ms']['total']:+.6f}",
                f"{traffic[mode] / totals[mode]['total'] / 1e9:.3f}",
            ]
            for mode in modes
        ),
    )
    jit = data["compilation"]
    lines += [f"NVRTC JIT {jit['count']}회, {jit['seconds']:.6f}초 (타이밍 및 parity 합계).", ""]
    rows, eligible = [], []
    for mode in modes:
        mode_pass = True
        for dtype in DTYPES:
            for op in ("fwd", "dinput", "dweight"):
                samples = [
                    s for s in data["rows"] if (s["mode"], s["dtype"], s["op"]) == (mode, dtype, op)
                ]
                passed = bool(samples) and all(s["tol_pass"] for s in samples)
                mode_pass = mode_pass and passed
                mismatches = sum(s["timing_plan_match"] is False for s in samples)
                rows.append(
                    [
                        mode,
                        dtype,
                        op,
                        len(samples),
                        worst(samples, "max_abs"),
                        worst(samples, "max_rel"),
                        verdict(all(s["exact_match"] for s in samples)),
                        verdict(passed),
                        mismatches,
                    ]
                )
        delta = totals[mode]["delta_vs_off_ms"]["total"]
        if mode_pass and delta < 0:
            eligible.append((mode, delta))
    lines += table(
        [
            "Mode",
            "Dtype",
            "Op",
            "Cases",
            "Max abs",
            "Max rel",
            "Exact",
            "Tolerance",
            "Timing plan differences",
        ],
        rows,
    )
    return lines, min(eligible, key=lambda item: item[1], default=None)


def stem(data: dict) -> Summary:
    timings = data["median_ms"]
    lines = table(
        ["Mode", "Fwd ms", "Bwd ms", "Fwd+Bwd ms", "Δ off"],
        (
            [
                mode,
                *(f"{times[key]:.6f}" for key in ("fwd", "bwd", "fwd_bwd")),
                f"{times['fwd_bwd'] - timings['off']['fwd_bwd']:+.6f}",
            ]
            for mode, times in timings.items()
        ),
    )
    lines += [
        "기존 Stem CUDA 검사의 rel_l2 기준인 BF16 < 0.035, FP16 < 0.006을 "
        "off와 FP32 양쪽 참조에 적용했습니다.",
        "",
    ]
    eligible = dict.fromkeys(timings, True)
    rows = []
    for case in data["parity"]:
        dtype = case["dtype"]
        limit = 0.035 if dtype == "bf16" else 0.006
        for mode, refs in case["modes"].items():
            for ref, tensors in refs.items():
                samples = list(tensors.values())
                passed = all(s["finite"] and s["rel_l2"] < limit for s in samples)
                eligible[mode] = eligible[mode] and passed
                rows.append(
                    [
                        mode,
                        dtype,
                        case["input_grad"],
                        ref,
                        worst(samples, "max_abs"),
                        worst(samples, "max_rel"),
                        worst(samples, "rel_l2"),
                        verdict(all(s["max_abs"] == 0 for s in samples)),
                        verdict(passed),
                    ]
                )
    lines += table(
        [
            "Mode",
            "Dtype",
            "dX",
            "Reference",
            "Max abs",
            "Max rel",
            "Max rel_l2",
            "Exact",
            "Tolerance",
        ],
        rows,
    )
    deltas = {
        mode: times["fwd_bwd"] - timings["off"]["fwd_bwd"]
        for mode, times in timings.items()
        if eligible[mode] and data["parity"]
    }
    winner = min(deltas, key=deltas.get, default=None)
    return lines, (winner, deltas[winner]) if winner and deltas[winner] < 0 else None


def summarize(values: dict) -> Iterator[tuple[str, str, list[str], tuple[str, float] | None]]:
    stages = (
        ("Scan", "Wave 45A · vscan_bwd", scan),
        ("Reader", "Wave 45B · Factorized Reader", reader),
        ("Stem", "Wave 46A · Stage-0 Stem", stem),
    )
    for filename, (name, title, summarize_stage) in zip(FILES[1:], stages, strict=True):
        data = values.get(filename)
        lines, winner = summarize_stage(data) if data else (["결과 JSON 미생성.", ""], None)
        yield name, title, lines, winner


def report(args: argparse.Namespace) -> None:
    record = json.loads(args.comment.read_text())
    returned = "자원이 반납되었습니다" in record["body"]
    if not returned or not record["body"].startswith("### ✅ 학습 결과 보고서"):
        raise ValueError("GPU return is unconfirmed")
    values, hashes = decode(record["body"], args.output / "decoded")
    meta = values.get("meta.json", {})
    if meta and meta["sha"] != args.sha:
        raise ValueError("Executed SHA differs from submitted SHA")
    lines = [
        "# Task 159 · Wave 45A + 45B + 46A H200 L2-cold eager screening",
        "",
        f"- Whole H200 (MIG=7) 단일 요청 [#{args.issue}]"
        f"(https://github.com/Aerodrone-H200/gpu-request/issues/{args.issue}).",
        f"- 브랜치 `wave45-h200-mega-screen`, 제출 SHA `{args.sha}`.",
        f"- GPU 반환 확인 댓글 UTC `{record['createdAt']}`.",
    ]
    if meta:
        lines.extend(
            [
                f"- 실행 SHA `{meta['sha']}`, `{meta['image']}`, {meta['gpu']}, "
                f"{meta['vram_bytes'] / 2**30:.2f} GiB, {meta['sm_count']} SM.",
                f"- Python {meta['python']}, PyTorch {meta['pytorch']}, CUDA {meta['cuda']}, "
                f"Triton {meta['triton']}. 패키지 부트스트랩 {meta['bootstrap_seconds']}초.",
                "- 단계별 실행 시간 및 종료 코드 "
                + ", ".join(
                    f"{stage} {status['seconds']}초 (코드 {status['exit']})"
                    for stage, status in meta["stages"].items()
                )
                + ".",
            ]
        )
    lines += ["", "## JSON SHA-256 검증", ""]
    lines += table(
        ["파일", "SHA-256", "상태"],
        (
            [
                f"[{name}](decoded/{name})" if name in hashes else name,
                f"`{hashes[name]}`" if name in hashes else "—",
                "일치" if name in hashes else "미생성",
            ]
            for name in FILES
        ),
    )
    winners = []
    for name, title, summary, winner in summarize(values):
        lines.extend([f"## {title}", "", *summary])
        if winner:
            winners.append((name, *winner))
    anchor, v44a, v44b = 181.1238, -0.2304, -0.5517
    projected = anchor + v44a + v44b + sum(delta for _, _, delta in winners)
    lines += [
        "## L2-cold eager 전체 모델 전망",
        "",
        f"Wave 43 #841 {anchor:.4f} ms에 요청된 Wave 44A 델타 "
        f"`batch4_w2` {v44a:+.4f} ms와 `bwd_opt_ch4` {v44b:+.4f} ms를 적용했습니다. "
        "Task 154의 CUDA 결과에서는 `bwd_opt_ch4`가 허용오차를 통과하지 못했습니다.",
        "",
    ]
    lines += [f"- {stage} `{mode}` {delta:+.6f} ms." for stage, mode, delta in winners]
    if winners:
        lines.append("")
    lines += [
        f"**{projected:.4f} ms**. Scan은 Wave 43의 strict_ro, Reader와 Stem은 off 대비 델타를 "
        "적용했습니다. 신규 단계는 자체 수치 허용오차를 통과한 개선만 반영했습니다. "
        "Type 2 전체 모델 `torch.compile` 실측이 남아 있습니다.",
        "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("emit")
    report_args = commands.add_parser("report")
    report_args.add_argument("comment", type=Path)
    report_args.add_argument("output", type=Path)
    report_args.add_argument("issue", type=int)
    report_args.add_argument("sha")
    args = parser.parse_args()
    if args.command == "emit":
        emit()
    else:
        report(args)
