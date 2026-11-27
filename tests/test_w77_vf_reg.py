# ruff: noqa: ANN201, S607, T201
"""Compile actual forward entry points on CPU and check Hopper resources."""

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from test_w66_split_bf16 import SHAPES, compile_source, offline_plans

from lnet import w77_vf_reg_scan as vf
from lnet.w77_vf_reg_scan import fp


def occupancy(regs, smem, warps):
    rounded = ((regs + 7) // 8) * 8
    subpart = 16384 // (32 * rounded)
    allocation = ((smem + 1024 + 255) // 256) * 256
    return warps * min(32, 64 // warps, 4 * subpart // warps, 233472 // allocation)


def pair_counts(sass, label):
    code = {
        int(pc, 16): ins.strip() for pc, ins in re.findall(r"/\*([\da-f]+)\*/\s+([^/\n]+);", sass)
    }
    loops = []
    for pc, ins in code.items():
        match = re.search(r"BRA (0x[\da-f]+)", ins)
        if match and (begin := int(match[1], 16)) < pc:
            body = [code[i] for i in range(begin, pc + 16, 16)]
            if any("HFMA2" in op for op in body) and any("STG." in op for op in body):
                loops.append((begin, pc, body))
    # The batch loop precedes the optional scalar tail and sits inside the line loop.
    _, _, body = min(
        (x for x in loops if not any(x[0] <= y[0] and y[1] < x[1] for y in loops)),
        key=lambda x: x[0],
    )
    assert not any("BRA " in op for op in body[:-1]), label
    pairs = 1 if label in ("14c", "7d") else 2
    return len(body) / pairs, sum("LDS." in op or "LDS " in op for op in body) / pairs


def test_cubins(tmp_path):
    jobs = []
    for label, op, h, m, base in offline_plans(SHAPES, plane_outputs=True):
        if op == "VF" or (label == "56d" and op == "HB"):
            launch = fp.configure(base, h) if op == "VF" else base
            jobs.extend(
                (label, op, h, m, launch, arm)
                for arm in (("w74", *vf.arms(label)) if op == "VF" else ("w74",))
            )

    def audit(job):
        label, op, h, m, launch, arm = job
        folder = tmp_path / (label + op + arm)
        folder.mkdir()
        key = launch.key
        symbol = fp.profile_name(key, h)
        source = fp.kernel_source if arm == "w74" else vf.source_for(arm)
        row = compile_source(source(key, h, h, m), folder, fp.compile_flags(key, h), symbol)
        sass = subprocess.check_output(
            ["cuobjdump", "-fun", symbol, "-sass", next(folder.glob("*.cubin"))], text=True
        )
        (folder / "disasm").write_text(sass)
        smem = fp.shared_bytes(launch, h) if arm == "w74" else vf.shared_bytes(launch, h, arm)
        spills = (row["stack"], row["LDL"], row["STL"])
        aw = occupancy(row["regs"], smem + row["static"], key.warps)
        counts = pair_counts(sass, label) if op == "VF" else ()
        print("AUDIT", label, op, arm, row["regs"], smem, aw, spills, *counts, flush=True)
        if (label, op, arm) == ("14d", "VF", "w74"):
            assert spills in ((0, 0, 0), (8, 2, 1))
        else:
            assert spills == (0, 0, 0)
        return (label, op, arm), (aw, counts)

    with ThreadPoolExecutor(6) as pool:
        results = dict(pool.map(audit, jobs))
    control_aw, control_counts = results["56d", "VF", "w74"]
    assert results["56d", "VF", vf.CACHED][0] > control_aw
    assert results["56d", "VF", vf.ALL][1][0] < control_counts[0]


@pytest.mark.parametrize("shape", SHAPES)
def test_dispatch(monkeypatch, shape):
    label, h, w, m, coarse = shape
    base = next(p for _, op, _, _, p in offline_plans([shape], plane_outputs=True) if op == "VF")
    monkeypatch.setattr(fp.scans, "_plan_launch", lambda *_a, **_kw: base)
    monkeypatch.setattr(fp, "_check_source", lambda *_: None)
    r = torch.empty(1, h, w, m, dtype=torch.bfloat16)
    states = fp._words(r, w)
    pole = [torch.empty(m)] * 4
    aux = [torch.empty(2, h, m)] * 2
    path = [r.new_empty(s) for s in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    calls = []

    def record(_, launch, width, args, scales, **kwargs: object):
        assert not scales
        assert args[13].value
        assert args[14].value
        calls.append((launch, width, kwargs))

    monkeypatch.setattr(fp, "_dispatch", record)
    for arm in ("w74", *vf.arms(label)):
        out = vf.vertical(pole, r, states, aux, path, coarse, arm=arm, separate=h != 7)
        launch, width, options = calls.pop()
        assert width == w
        assert options == (
            {}
            if arm == "w74"
            else {"source_fn": vf.source_for(arm), "smem": vf.shared_bytes(launch, h, arm)}
        )
        oh = h // (1 + coarse)
        if h == 7:
            assert out.shape == (1, oh, oh, 2 * m)
        else:
            assert all(t.shape == (1, oh, oh, m) for t in out)
