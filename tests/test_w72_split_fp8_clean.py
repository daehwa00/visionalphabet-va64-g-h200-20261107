"""Wave 72 full-model routing, offline Hopper audit and training checks."""

import concurrent.futures

import bench_wave66_split_bf16 as bench
import pytest
import torch
from test_w66_split_bf16 import compile_source, offline_plans
from test_w66_split_bf16_scan import model_scan_graph
from torch._functorch.aot_autograd import aot_export_module
from torch._subclasses.fake_tensor import FakeTensorMode

from lnet import product_scan_pipeline as pipeline
from lnet import w72_split_fp8_scan as fp


def test_model_graph(monkeypatch) -> None:
    monkeypatch.setenv("VA_W74_SMALL_FP8", "0")
    ops = model_scan_graph(monkeypatch, fp8=True)
    assert [(shape, ep) for name, shape, ep in ops if name.endswith("_w72_fp8.default")] == [
        ((56, 56), "full16"),
        ((56, 56), "coarse_path"),
        ((28, 28), "full16"),
        ((28, 28), "coarse_path"),
    ]
    assert len(ops) == 11
    assert (
        sum(name.endswith(("planes_v1.default", "storage_v2.default")) for name, _, _ in ops) == 7
    )


def test_56d_cubins(tmp_path) -> None:
    def audit(job):
        _, op, w, m, launch = job
        if launch.key.vertical:
            launch = fp.configure(launch, w)
        key = launch.key
        source = fp.kernel_source(key, w, w, m)
        assert fp.profile_name(key) in source
        folder = tmp_path / op
        folder.mkdir()
        flags = [*fp.compile_flags(key, w), "--keep", "--keep-dir", str(folder)]
        row = compile_source(source, folder, flags, fp.profile_name(key))
        assert (row["stack"], row["LDL"], row["STL"]) == (0, 0, 0), (op, row)
        (ptx_path,) = folder.glob("*.ptx")
        ptx = ptx_path.read_text()
        assert "e4m3x2" in ptx, op
        assert ("e5m2x2" in ptx) == key.backward, op
        if op in ("VF", "HB"):
            assert launch.plan.cache_line
        if op == "VF":
            smem = fp.shared_bytes(launch, w)
            regs = ((row["regs"] + 7) // 8) * 8
            blocks = min(
                64 // key.warps, (228 << 10) // (1024 + smem), 65536 // (key.warps * 32 * regs)
            )
            assert smem == 25600
            assert blocks * key.warps == 16
        if op == "VB":
            assert key.chunk == 0
            assert not launch.plan.global_checkpoints
            assert fp.shared_bytes(launch, w) == 115200
        return op

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        assert set(pool.map(audit, [job for job in offline_plans() if job[0] == "56d"])) == {
            "HF",
            "VF",
            "VB",
            "HB",
        }


def test_28_vertical_winners() -> None:
    for label, op, w, m, base in offline_plans(bench.SHAPES[2:4]):
        launch = base
        if not launch.key.vertical:
            continue
        if op == "VB":
            launch = fp.scans._plan_w57(0, launch, (1024, w, w, m), "v57_bwd_seg16_2chunk")
        launch = fp.configure(launch, w)
        if op == "VB":
            assert launch.key.chunk == 0
            assert fp.shared_bytes(launch, w) == (77312 if label == "28d" else 70144)
        else:
            assert launch.plan.cache_line
            src = fp.kernel_source(launch.key, w, w, m)
            assert ("reinterpret_cast<Fp8Quad*>(line_buf)" in src) == (label == "28d")


def test_saved_storage() -> None:
    with FakeTensorMode():
        r = torch.empty(2, 56, 56, 160, dtype=torch.bfloat16)
        states = fp.hforward([], r, r)
        assert states.shape == (2, 5, 2, 56, 56, 32)
        assert states.dtype == fp._words(r, 28).dtype == torch.int16


@pytest.mark.parametrize("planes", [False, True])
def test_aot_retained_state(planes) -> None:
    check_joint_state(planes, 56, 160, "full16")


def check_joint_state(planes, h, m, epilogue) -> None:
    class Loss(torch.nn.Module):
        def forward(self, *args: torch.Tensor):
            op = (
                pipeline._packed_pipeline_planes_op_w72_fp8
                if planes
                else pipeline._packed_pipeline_op_w72_fp8
            )
            outputs = op(
                list(args[:9]),
                list(args[9:]),
                epilogue,
                1e-6,
                0,
                recompute=False,
                path_swiglu=False,
                **({} if planes else {"planar_output": False}),
            )
            return (outputs[0].sum(),)

    with FakeTensorMode():
        args = [torch.empty(1, 1, 1, m, requires_grad=True) for _ in range(8)]
        args += [
            torch.empty(shape, dtype=torch.bfloat16, requires_grad=True)
            for shape in (
                (1, h, h, 2 * m),
                (m, 8, 8),
                (m, 8),
                (m, 2, 8),
                (m, 2),
            )
        ]
        graph, _ = aot_export_module(Loss(), tuple(args), trace_joint=True, output_loss_index=0)
    saved = next(
        n for n in graph.graph.nodes if n.target == torch.ops.lnet.w72_fp8_horizontal.default
    )
    backward = next(
        n for n in graph.graph.nodes if n.target == torch.ops.lnet.w72_fp8_scan_backward.default
    )
    assert saved.meta["val"].dtype == torch.int16
    assert saved.meta["val"].shape == (1, m // 32, 2, h, h, 32)
    assert backward.args[4] is saved


def test_null_workspace(monkeypatch) -> None:
    check_workspace(monkeypatch, bench.SHAPES[0])


def check_workspace(monkeypatch, shape) -> None:
    monkeypatch.setattr(fp.scans, "device_limits", lambda _: (132, 227 << 10, 64))
    _, h, w, m, coarse = shape
    plans = {op: plan for _, op, _, _, plan in offline_plans([shape])}
    monkeypatch.setattr(fp, "_check_source", lambda *_: None)
    monkeypatch.setattr(fp.scans, "_reduce_partials", lambda _, params, *_args: params)
    r = torch.empty(1, h, w, m, dtype=torch.bfloat16)
    pole = [torch.empty(m)] * 4
    aux = [torch.empty(2, h, m)] * 2
    path = [torch.empty(s) for s in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    seen = []

    def capture(_, launch, _width, args, scale_args=()):
        scratch = (args[17], args[18]) if launch.key.vertical else (args[10],)
        assert all(ptr.value is None for ptr in scratch)
        seen.append(fp.profile_name(launch.key, h))
        assert bool(scale_args) == (launch.key.vertical and launch.key.backward)

    def plan(*_: object, vertical, backward, **__: object):
        return plans[("V" if vertical else "H") + ("B" if backward else "F")]

    monkeypatch.setattr(fp, "_dispatch", capture)
    monkeypatch.setattr(fp.scans, "_plan_launch", plan)
    states = fp.horizontal(pole, r, r)
    fp.vertical(pole, r, states, aux, path, coarse=coarse)
    out = r[:, :h // (1 + coarse), :w // (1 + coarse)].contiguous()
    grad = fp.vertical(pole, r, states, aux, path, coarse=coarse, grad=(out, out))
    assert grad[4].dtype == torch.int16
    fp.horizontal(pole, r, r, states=states, adjoints=grad[4], scales=grad[-1], compact=coarse)
    assert len(set(seen)) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize(("height", "coarse"), [(56, False), (56, True), (28, False), (28, True)])
def test_compiled_training(monkeypatch, request, height, coarse) -> None:
    check_compiled_training(monkeypatch, request, height, coarse)


def check_compiled_training(monkeypatch, request, height, coarse, *, planes=False) -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires Hopper")
    torch._dynamo.reset()
    request.addfinalizer(torch._dynamo.reset)
    for k, v in bench.lockin_arm_env("w57_lockin").items():
        monkeypatch.setenv(k, v)
    x, a, g, _, path = bench.reference_inputs(height, height, 128 if height == 7 else 160)
    drive = torch.cat((x.real, x.imag), -1).detach().bfloat16().requires_grad_()
    pole = [
        p.detach().reshape(1, 1, 1, -1).contiguous().requires_grad_()
        for p in (a.real, a.imag, g.real, g.imag)
    ]
    params = [p.detach().bfloat16().requires_grad_() for p in path]
    operands = [*pole, drive, *params]
    fn = (
        pipeline.run_product_scan_coarse_path_collapse_packed
        if coarse
        else pipeline.run_product_scan_path_collapse_pipeline_packed
    )

    if planes:
        fn = (pipeline.run_product_scan_coarse_path_collapse_planes_packed if coarse
              else pipeline.run_product_scan_path_collapse_planes_packed)

    def run(*args: torch.Tensor):
        output = fn(
            tuple(args[:4]),
            tuple(args[:4]),
            args[4],
            tuple(args[5:]),
            gain_normalization="pointwise",
            memory_policy="retain",
        )

        return torch.cat(output, -1) if planes else output

    monkeypatch.setenv("VA_W72_SPLIT_FP8", "0")
    ref = run(*operands)
    compiled = torch.compile(run, fullgraph=True)
    monkeypatch.setenv("VA_W72_SPLIT_FP8", "1")
    factors = (1.0, 1e-5) if height in (14, 56) and not coarse else (1.0,)
    gradients = [torch.randn_like(ref) * factor for factor in factors]
    stream = torch.cuda.Stream()
    for dy in gradients:
        expected = (ref, *torch.autograd.grad(ref, operands, dy, retain_graph=True))

        def step(dy=dy):
            out = compiled(*operands)
            return (out, *torch.autograd.grad(out, operands, dy))

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            observed = step()
        graph.replay()
        torch.cuda.synchronize()
        for lhs, rhs in zip(expected, observed, strict=True):
            assert torch.isfinite(rhs).all()
            assert bench.metrics(lhs, rhs)["rel_l2"] < 0.12
