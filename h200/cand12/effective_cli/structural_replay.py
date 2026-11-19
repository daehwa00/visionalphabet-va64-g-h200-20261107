"""Step 3: Replay the 56x56 Direct _path_backward_kernel bottleneck launch on captured operands
and compare Control (4-row output dot) against all 6 structural variants and warp/tile configs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys

import torch

OUTS = (19, 20, 21, 22, 24, 25, 26, 27, 28)
ATOMIC = OUTS[:-1]
RETAINED_ORDER = (*range(10), 15, 16, 17, 13, 14, 19, 20, 21, 22, 24, 25, 26, 27, 28)


def sha(x: torch.Tensor) -> str:
    return hashlib.sha256(x.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()


def restore(meta: dict, bundle: dict) -> list[torch.Tensor]:
    slots = {}
    for v in meta["views"]:
        g = v["group"]
        n = (
            v["offset"] + 1 + sum((s - 1) * t for s, t in zip(v["shape"], v["stride"]))
            if all(v["shape"])
            else 0
        )
        slots[g] = max(slots.get(g, 0), n)
    buffers = {
        g: torch.empty(
            n,
            dtype=getattr(torch, next(v["dtype"] for v in meta["views"] if v["group"] == g)),
            device="cuda",
        )
        for g, n in slots.items()
    }
    args = []
    for i, v in enumerate(meta["views"]):
        t = buffers[v["group"]].as_strided(v["shape"], v["stride"], v["offset"])
        assert t.data_ptr() % 512 == v["ptr_mod_512"]
        if v.get("value") == i:
            val = bundle["values"][i]
            assert sha(val) == meta["sha256"][str(i)]
            t.copy_(val)
            assert sha(t.cpu()) == meta["sha256"][str(i)]
        args.append(t)
    return args


def error(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.flatten()
    bf = b.flatten()
    sq = torch.zeros(3, device="cuda", dtype=torch.float64)
    dmax = 0.0
    refmax = 0.0
    for i in range(0, af.numel(), 4 * 1024 * 1024):
        x = af[i : i + 4194304].float()
        y = bf[i : i + 4194304].float()
        diff = x - y
        dmax = max(dmax, float(diff.abs().max()))
        refmax = max(refmax, float(y.abs().max()))
        sq += torch.stack(
            (diff.square().sum(), y.square().sum(), (~torch.isfinite(diff)).float().sum())
        ).double()
    dl, rl, nonfinite = sq.tolist()
    rel = (dl / max(rl, 1e-24)) ** 0.5
    return dict(
        max_abs=dmax,
        relative_l2=rel,
        finite=nonfinite == 0,
        passed=nonfinite == 0 and rel <= 0.02 and dmax <= 0.03 * refmax + 1e-6,
    )


def compiled_stats(compiled) -> dict:
    out = dict(
        regs=getattr(compiled, "n_regs", None),
        spill_slots=getattr(compiled, "n_spills", None),
        shared=getattr(getattr(compiled, "metadata", None), "shared", None),
    )
    asm_map = getattr(compiled, "asm", {})
    cubin_bytes = asm_map.get("cubin", b"")
    if cubin_bytes:
        out["binary_sha256"] = hashlib.sha256(cubin_bytes).hexdigest()
        cu = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
        if Path(cu).exists():
            tmp = Path(os.environ["RUN_ROOT"]) / "tmp_replay.cubin"
            tmp.write_bytes(cubin_bytes)
            res = subprocess.run([cu, "--dump-resource-usage", str(tmp)], capture_output=True, text=True)
            asm = subprocess.run([cu, "--dump-sass", str(tmp)], capture_output=True, text=True)
            tmp.unlink(missing_ok=True)
            if not asm.returncode:
                ops = Counter(re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z0-9]+)", asm.stdout))
                out.update(opcodes=dict(ops), static_resource=res.stdout.strip()[-400:])
                m_reg = re.search(r"REG:(\d+)", res.stdout)
                m_stk = re.search(r"STACK:(\d+)", res.stdout)
                if m_reg:
                    out["regs"] = int(m_reg.group(1))
                if m_stk:
                    out["stack_bytes"] = int(m_stk.group(1))
    return out


def main() -> None:
    root = Path(os.environ["RUN_ROOT"])
    fixture = Path(os.environ["COLD_INPUTS"])
    out_dir = root / "replay-structural"
    out_dir.mkdir(parents=True, exist_ok=True)

    from lnet.output_dot_kernel import _path_backward_kernel as control_kernel
    from lnet.effective_output_vjp import _rows_kernel, _transpose_kernel
    from lnet.effective_structural_scan import _BACKWARD_VARIANTS

    meta = json.loads((fixture / "capture.json").read_text())
    bundle = torch.load(fixture / "operands.pt", weights_only=True, mmap=True, map_location="cpu")
    args = restore(meta, bundle)
    del bundle["values"]

    # Retained kernel arguments before height constexpr
    tensor_args = [args[j] for j in RETAINED_ORDER]
    # scalars in capture.json: [56, 7168, 160, 8, 160, 1792, 1] -> width=56, line_count=7168, modes=160, packed_path_hidden=8
    width, line_count, modes, packed_path_hidden = 56, 7168, 160, 8

    # Specs: (name, kernel_fn, bm, group, warps, doubling, steps)
    specs = [
        ("control_dot_bm4_w2", control_kernel, 4, 4, 2, 1, None),
        ("vjp1_transpose_bm4_w2", _transpose_kernel, 4, 4, 2, 1, None),
        ("vjp2_rows_bm4_w2", _rows_kernel, 4, 4, 2, 1, None),
        ("v1_recompute_prev_bm4_w2", _BACKWARD_VARIANTS[1], 4, 4, 2, 1, None),
        ("v2_fast_mlp_bm4_w2", _BACKWARD_VARIANTS[2], 4, 4, 2, 1, None),
        ("v3_half_hidden_bm4_w2", _BACKWARD_VARIANTS[3], 4, 4, 2, 1, None),
        ("v4_unpack_prev_bm4_w2", _BACKWARD_VARIANTS[4], 4, 4, 2, 1, None),
        ("v5_unpack_rows_bm4_w2", _BACKWARD_VARIANTS[5], 4, 4, 2, 1, None),
        ("v6_col_loop2_bm4_w2", _BACKWARD_VARIANTS[6], 4, 4, 2, 1, 2),
        ("v4_unpack_prev_bm4_w4", _BACKWARD_VARIANTS[4], 4, 4, 4, 1, None),
        ("v3_half_hidden_bm4_w4", _BACKWARD_VARIANTS[3], 4, 4, 4, 1, None),
        ("v4_unpack_prev_bm8_w4", _BACKWARD_VARIANTS[4], 8, 4, 4, 1, None),
    ]

    def reset() -> None:
        for i in ATOMIC:
            args[i].zero_()

    def run_spec(spec) -> object:
        name, fn, bm, group, warps, doubling, steps = spec
        if steps is None:
            grid = ((modes // bm) * group, line_count // group, 1)
            return fn[grid](
                *tensor_args,
                56,
                width,
                line_count,
                modes,
                packed_path_hidden,
                1e-8,
                BM=bm,
                GROUP=group,
                COARSE=False,
                DOUBLING=doubling,
                num_warps=warps,
                num_stages=1,
            )
        else:
            eff_lines = 128 * (width // steps)
            grid = ((modes // bm) * group, eff_lines // group, 1)
            return fn[grid](
                *tensor_args,
                56,
                width,
                line_count,
                modes,
                packed_path_hidden,
                1e-8,
                BM=bm,
                GROUP=group,
                COARSE=False,
                DOUBLING=doubling,
                STEPS=steps,
                num_warps=warps,
                num_stages=1,
            )

    ref = {}
    checks = {}
    resources = {}
    active_specs = []

    for spec in specs:
        name = spec[0]
        try:
            reset()
            args[28].fill_(float("nan"))
            compiled = run_spec(spec)
            torch.cuda.synchronize()
            resources[name] = compiled_stats(compiled)
            if name == "control_dot_bm4_w2":
                for i in OUTS:
                    ref[i] = args[i].clone()
            else:
                checks[name] = {str(i): error(args[i], ref[i]) for i in OUTS}
            active_specs.append(spec)
            passed_all = (
                True
                if name == "control_dot_bm4_w2"
                else all(v["passed"] for v in checks[name].values())
            )
            print(
                "STRUCT_REPLAY_COMPILED",
                name,
                "regs=",
                resources[name].get("regs"),
                "stack=",
                resources[name].get("stack_bytes", 0),
                "passed=",
                passed_all,
                flush=True,
            )
        except Exception as exc:
            if name == "control_dot_bm4_w2":
                raise
            print("STRUCT_REPLAY_FAILED", name, repr(exc)[-400:], flush=True)

    for _ in range(4):
        for spec in active_specs:
            reset()
            run_spec(spec)
    torch.cuda.synchronize()

    samples = {spec[0]: [] for spec in active_specs}
    telemetry = []
    for i in range(9):
        order = active_specs.copy()
        random.Random(149 + i).shuffle(order)
        for spec in order:
            name = spec[0]
            pairs = []
            for _ in range(5):
                reset()
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                run_spec(spec)
                end.record()
                pairs.append((begin, end))
            torch.cuda.synchronize()
            samples[name].append(statistics.median(x.elapsed_time(y) for x, y in pairs))
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
        )
        telemetry.append(q.stdout.strip())

    zap = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.float32)
    cold = {}
    for spec in active_specs:
        name = spec[0]
        times = []
        for _ in range(7):
            reset()
            zap.fill_(1)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            run_spec(spec)
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end))
        cold[name] = times

    summary_rows = {}
    for spec in active_specs:
        name = spec[0]
        warm_med = statistics.median(samples[name])
        cold_med = statistics.median(cold[name])
        chk = checks.get(name, {})
        max_rel_l2 = max((v["relative_l2"] for v in chk.values()), default=0.0)
        max_abs = max((v["max_abs"] for v in chk.values()), default=0.0)
        passed = True if name == "control_dot_bm4_w2" else all(v["passed"] for v in chk.values())
        summary_rows[name] = dict(
            warm_median_ms=warm_med,
            warm_8calls_ms=8.0 * warm_med,
            cold_median_ms=cold_med,
            cold_8calls_ms=8.0 * cold_med,
            warm_samples_ms=samples[name],
            cold_samples_ms=cold[name],
            max_relative_l2=max_rel_l2,
            max_abs=max_abs,
            numeric_passed=passed,
            resources=resources[name],
            checks=chk,
        )
        print(
            "STRUCT_REPLAY_ROW",
            name,
            f"warm={warm_med:.3f}ms (x8={8*warm_med:.2f}ms)",
            f"cold={cold_med:.3f}ms (x8={8*cold_med:.2f}ms)",
            f"max_rel_l2={max_rel_l2:.6f}",
            f"passed={passed}",
            flush=True,
        )

    result = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        rows=summary_rows,
        telemetry=telemetry,
    )
    (root / "structural-replay.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
