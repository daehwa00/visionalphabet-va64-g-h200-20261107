"""Compile real-layout SM90a VB kernels without a CUDA device."""

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import bench_wave66_split_bf16 as bench
import bench_wave75_vb_3warp as screen
import pytest
import torch
from test_w66_split_bf16 import compile_source, offline_plans

from lnet import w72_split_fp8_scan as fp

EXPECTED = {
    ("14c", "w74"): (219, 37888, 8),
    ("14c", "w75_3w"): (219, 29824, 9),
    ("14d", "w74"): (204, 41472, 8),
    ("14d", "w75_2w"): (204, 23552, 8),
    ("14d", "w75_3w"): (204, 32512, 9),
    ("28c", "w74"): (225, 70144, 8),
    ("28c", "w75_3w_full"): (223, 54016, 9),
    ("28d", "w74"): (214, 77312, 8),
    ("28d", "w75_3w_full"): (213, 59392, 9),
    ("56c", "w74"): (212, 105984, 8),
    ("56c", "w75_3w_k17_cachew"): (219, 65920, 9),
    ("56c", "w75_3w_kmax"): (219, 76672, 9),
    ("56c", "w75_3w_nocachew_full"): (224, 75264, 9),
    ("56d", "w74"): (208, 115200, 8),
    ("56d", "w75_3w_kmax"): (208, 76672, 9),
    ("56d", "w75_3w_nocachew_kmax"): (216, 76800, 9),
    ("7d", "w74"): (223, 29184, 8),
    ("7d", "w75_3w"): (223, 24704, 9),
}


def test_cubins(tmp_path) -> None:
    jobs = []
    for label, op, h, m, base in offline_plans(bench.SHAPES, plane_gradients=True):
        if op != "VB":
            continue
        launch = base
        if h == 28:
            launch = fp.scans._plan_w57(0, launch, (1024, h, h, m), "v57_bwd_seg16_2chunk")
        control = fp.configure(launch, h)
        for arm in ("w74", *fp.VB_ARMS[label]):
            candidate = control if arm == "w74" else fp.vb_candidate(control, h, arm)
            jobs.append((label, arm, h, m, candidate))

    def audit(job):
        label, arm, h, m, launch = job
        k = launch.key
        folder = tmp_path / (label + arm)
        folder.mkdir()
        symbol = fp.profile_name(k, h)
        source = fp.kernel_source(k, h, h, m)
        assert "for (int k = 1; k < WARPS; ++k)" in source
        assert "worker = group * WARPS + warp" in source
        assert "m < M && worker < workers" in source
        assert "row += I(reduce ? group : worker) * fields * M + m" in source
        assert k.backward_reduce_warps == 0
        assert launch.partials == fp.scans._ceil(launch.workers, k.warps)
        row = compile_source(source, folder, fp.compile_flags(k, h), symbol)
        (cubin,) = folder.glob("*.cubin")
        sass = subprocess.check_output(["cuobjdump", "-fun", symbol, "-sass", cubin], text=True)  # noqa: S607
        count = len(re.findall(r"^\s*/\*[\da-f]+\*/\s+[^/\s]", sass, re.MULTILINE))
        smem = fp.shared_bytes(launch, h)
        assert smem >= 32 * k.warps * 32 * 4  # FP32 reduction staging
        regs = row["regs"]
        # SM90a cubins include the 1 KiB system reservation in SHARED.
        assert row["static"] == 1024
        aw = k.warps * min(
            64 // k.warps,
            65536 // (k.warps * 256 * ((regs + 7) // 8)),
            (228 << 10) // (((smem + row["static"] + 255) // 256) * 256),
        )
        result = (
            label,
            arm,
            k.warps,
            fp.tile_rows(k, h) if h >= 28 else 0,
            int(k.cache_weights),
            regs,
            smem,
            aw,
            row["stack"],
            row["LDL"],
            row["STL"],
            count,
        )
        print("AUDIT", *result, flush=True)  # noqa: T201
        return result

    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(audit, jobs))
    for row in rows:
        assert row[8:11] == (0, 0, 0), row
        assert row[5:8] == EXPECTED[row[:2]]
        if row[2] == 3:
            assert row[7] == 9, row


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

    def dispatch(_, launch, _width, args, scale_args):
        assert scale_args
        assert args[17].value is None
        assert args[18].value is None
        seen.append(fp.scans._ceil(launch.workers, launch.key.warps))

    def reduce_partial(tensor, params, _hidden):
        assert tensor.shape == (seen.pop(), 94, m)
        return params

    monkeypatch.setattr(fp, "_dispatch", dispatch)
    monkeypatch.setattr(fp.scans, "_reduce_partials", reduce_partial)
    for arm in fp.VB_ARMS[label]:
        fp.vertical(pole, r, states, aux, path, coarse, (grad, grad), vb_arm=arm)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires Hopper",
)
@pytest.mark.parametrize("shape", bench.SHAPES)
def test_graph_parity(monkeypatch, shape) -> None:
    for k, v in bench.lockin_arm_env("w74_fp8_lockin").items():
        monkeypatch.setenv(k, v)
    screen.screen(shape)
