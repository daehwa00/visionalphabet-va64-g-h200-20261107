"""CPU SM90a resource and row-pair issue audit, with optional Hopper replay."""

import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import bench_wave66_split_bf16 as bench
import bench_wave76_vb_innerloop as screen
import pytest
import torch
from test_w66_split_bf16 import compile_source, offline_plans

from lnet import w76_vb_innerloop_scan as vb
from lnet.w76_vb_innerloop_scan import fp


def loop_counts(sass, label, rows) -> tuple[int, int, int, int]:
    """Count issued PCs through the paired path only (odd scalar tail excluded)."""
    code = {
        int(pc, 16): ins.strip() for pc, ins in re.findall(r"/\*([\da-f]+)\*/\s+([^/\n]+);", sass)
    }
    loops = []
    for pc, ins in code.items():
        branch = re.search(r"BRA (0x[\da-f]+)", ins)
        if (
            branch
            and (start := int(branch[1], 16)) < pc
            and sum("HFMA2" in op for addr, op in code.items() if start <= addr <= pc) >= 64
        ):
            loops.append((pc - start, start, pc))
    _, start, stop = min(loops)
    height = int(label[:-1])
    step = 2 if label.endswith("c") else 1
    pairs = height // (2 * step)
    counts = Counter()
    for iteration in range(pairs):
        pc = start
        tile_row = height - 1 - iteration * 2 * step
        while pc <= stop:
            ins = code[pc]
            opcode = ins.split()[int(ins.startswith("@"))].split(".")[0]
            counts[opcode] += 1
            if pc < stop and opcode == "BRA":
                dest = int(ins.split()[-1], 16)
                if ins.startswith("BRA"):
                    pc = dest
                    continue
                block = [code[i] for i in range(pc + 16, dest, 16)]
                if any(op.startswith("STG") for op in block):
                    use_tile = tile_row < rows
                    tile_row -= 1
                    if use_tile:
                        pc = dest
                        continue
                elif iteration == pairs - 1:
                    pc = dest
                    continue
            pc += 16
    return sum(counts.values()), counts["LDS"], counts["PRMT"], pairs


def test_cubins(tmp_path) -> None:
    jobs = []
    for label, op, h, m, base in offline_plans(bench.SHAPES, plane_gradients=True):
        if op != "VB":
            continue
        launch = base
        if h == 28:
            launch = fp.scans._plan_w57(0, launch, (1024, h, h, m), "v57_bwd_seg16_2chunk")
        launch = fp.configure(launch, h)
        jobs.extend((label, h, m, launch, arm) for arm in ("w74", *vb.VB_ARMS[label]))

    def audit(job):
        label, h, m, launch, arm = job
        key = launch.key
        folder = tmp_path / (label + arm)
        folder.mkdir()
        symbol = fp.profile_name(key, h)
        source = fp.kernel_source if arm == "w74" else vb.source_for(arm)
        row = compile_source(
            source(key, h, h, m), folder, [*fp.compile_flags(key, h), "-lineinfo"], symbol
        )
        smem = fp.shared_bytes(launch, h) if arm == "w74" else vb.shared_bytes(launch, h, arm)
        aw = 4 * min(
            16,
            65536 // (128 * ((row["regs"] + 7) // 8) * 8),
            (228 << 10) // (((smem + row["static"] + 255) // 256) * 256),
        )
        assert row["stack"] == row["LDL"] == row["STL"] == 0
        assert row["regs"] <= 255
        assert aw == 8
        sass = subprocess.check_output(
            ["cuobjdump", "-fun", symbol, "-sass", next(folder.glob("*.cubin"))],  # noqa: S607
            text=True,
        )
        k = (49 if arm == vb.K49 else 46) if label == "56d" else h if h >= 28 else 0
        counts = loop_counts(sass, label, k)
        print("AUDIT", label, arm, k, row["regs"], smem, aw, *counts, flush=True)  # noqa: T201
        return label, arm, counts[0]

    with ThreadPoolExecutor(6) as pool:
        rows = {(label, arm): n for label, arm, n in pool.map(audit, jobs)}
    # Total issued instructions across full paired iterations (CUDA 12.8 sm_90a).
    expected = {
        "56d": (26896, 25440),
        "56c": (16232, 15448),
        "28d": (13748, 13076),
        "28c": (8407, 8036),
        "14d": (6621, 6285),
        "14c": (3420, 3264),
        "7d": (2976, 2676),
    }
    for label, (control, candidate) in expected.items():
        assert rows[label, "w74"] == control
        assert rows[label, vb.REG] == candidate
    assert rows["56d", vb.K49] == 25416
    for (label, arm), n in rows.items():
        if arm != "w74":
            assert n < rows[label, "w74"]


@pytest.mark.parametrize("shape", bench.SHAPES)
def test_buffers(monkeypatch, shape) -> None:
    label, h, w, m, coarse = shape
    plan = next(
        launch for _, op, _, _, launch in offline_plans([shape], plane_gradients=True) if op == "VB"
    )
    monkeypatch.setattr(fp.scans, "device_limits", lambda _: (132, 227 << 10, 64))
    monkeypatch.setattr(fp.scans, "_plan_launch", lambda *_, **__: plan)
    monkeypatch.setattr(fp, "_check_source", lambda *_: None)
    r = torch.empty(1, h, w, m, dtype=torch.bfloat16)
    states = fp._words(r, w)
    pole = [torch.empty(m)] * 4
    aux = [torch.empty(2, h, m)] * 2
    path = [torch.empty(s) for s in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    grad = r[:, : h // (1 + coarse), : w // (1 + coarse)].contiguous()
    seen = []

    def dispatch(_, launch, _width, args, scale_args, **_options: object):
        assert scale_args
        assert args[17].value is None
        assert args[18].value is None
        seen.append(fp.scans._ceil(launch.workers, launch.key.warps))

    def reduce_partial(tensor, params, _hidden):
        assert tensor.shape == (seen.pop(), 94, m)
        return params

    monkeypatch.setattr(fp, "_dispatch", dispatch)
    monkeypatch.setattr(fp.scans, "_reduce_partials", reduce_partial)
    for arm in vb.VB_ARMS[label]:
        vb.vertical(pole, r, states, aux, path, coarse, (grad, grad), arm=arm)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires Hopper",
)
@pytest.mark.parametrize("shape", bench.SHAPES)
def test_graph_parity(monkeypatch, shape) -> None:
    for k, v in bench.lockin_arm_env("w74_fp8_lockin").items():
        monkeypatch.setenv(k, v)
    result = screen.screen(shape)
    for arm in vb.VB_ARMS[shape[0]]:
        row = result["arms"][arm]
        assert row["graph_pass"]
        assert row["finite"]
        assert min(row["normal_min_cos"], row["tiny_min_cos"]) >= 0.999999
