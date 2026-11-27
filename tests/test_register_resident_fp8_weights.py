"""Audit production planar-gradient SM90a code and the SMALL_SPATIAL_FP8 fallback."""

from concurrent.futures import ThreadPoolExecutor

import bench_split_bf16_scan as bench
from test_split_bf16_model import compile_source, cubin_resources, offline_plans

from alphabet import split_fp8_scan as fp
from h200.bench_h200 import benchmark_arm_env

SMEM = {
    "56d": 115200, "56c": 104448, "28d": 77312, "28c": 70144,
    "14d": 39936, "14c": 36352, "7d": 22016,
}


def test_cubins(monkeypatch, tmp_path) -> None:
    for key, value in benchmark_arm_env("hopper_fp8_resident").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(fp.scans, "device_limits", lambda _: (132, 227 << 10, 64))
    jobs = list(offline_plans(bench.SHAPES, plane_gradients=True))

    def audit(job):
        label, op, h, modes, launch = job
        if op == "VB" and h == 28:
            launch = fp.scans._plan_vertical_segmented_backward(
                0, launch, (1024, h, h, modes), "backward_segment16_two_chunks"
            )
        if launch.key.vertical:
            launch = fp.configure(launch, h)
        elif op == "HF":
            launch = fp._cached_horizontal(launch, h)
        key = launch.key
        folder = tmp_path / (label + op)
        folder.mkdir()
        src = fp.kernel_source(key, h, h, modes)
        row = compile_source(src, folder, fp.compile_flags(key, h), fp.profile_name(key, h))
        smem = fp.shared_bytes(launch, h)
        regs = row["regs"]
        # cuobjdump SHARED already includes the SM90 reserved 1 KiB.
        aw = key.warps * min(
            64 // key.warps,
            65536 // (key.warps * 256 * ((regs + 7) // 8)),
            (228 << 10) // (((smem + row["static"] + 255) // 256) * 256),
        )
        assert row["stack"] == row["LDL"] == row["STL"] == 0
        if op == "VB":
            assert smem == SMEM[label]
            assert aw == 8
            if label == "56d":
                assert "if (y < 49)" in src
            assert regs <= 246
            scale = cubin_resources(
                next(folder.glob("*.cubin")), "alphabet_profile_scale__split_fp8"
            )
            assert scale["stack"] == scale["LDL"] == scale["STL"] == 0
        return label, op, regs, smem, aw

    assert len(jobs) == 28
    with ThreadPoolExecutor(6) as pool:
        for row in pool.map(audit, jobs):
            print("REGISTER_RESIDENT_FP8", *row)  # noqa: T201


def test_fallback(monkeypatch) -> None:
    monkeypatch.setenv("ALPHABET_SCAN_SPLIT_FP8", "1")
    monkeypatch.setenv("ALPHABET_SCAN_RESIDENT_BACKWARD_WEIGHTS", "0")
    for label, op, h, modes, base in offline_plans(bench.SHAPES, plane_gradients=True):
        if op != "VB":
            continue
        launch = base
        if h == 28:
            launch = fp.scans._plan_vertical_segmented_backward(
                0, launch, (1024, h, h, modes), "backward_segment16_two_chunks"
            )
        launch = fp.configure(launch, h)
        assert (
            fp.shared_bytes(launch, h)
            == {
                "56d": 115200,
                "56c": 105984,
                "28d": 77312,
                "28c": 70144,
                "14d": 41472,
                "14c": 37888,
                "7d": 29184,
            }[label]
        )
        source = fp.kernel_source(launch.key, h, h, modes)
        assert "result.resident" not in source
        if label == "56d":
            assert "if (y < 46)" in source
