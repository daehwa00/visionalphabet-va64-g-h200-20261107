"""Parity, layout handling, and capture for the Wave 29C kernels."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton
from torch import Tensor
from torch._higher_order_ops.triton_kernel_wrap import generate_ttir
from torch.nn import functional
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend

from lnet import model_w29c
from lnet.memory_backbone import ChannelRMSNorm
from lnet.model_w29c import _row_pitch, model_w29c_enabled, packed_swiglu_vec2d

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(("mode", "features"), [
    ("off", ()),
    ("swiglu_vec2d", ("swiglu_vec2d",)),
    ("rmsnorm_bwd", ("rmsnorm_bwd",)),
    ("rmsnorm_full", ("rmsnorm_bwd", "rmsnorm_full")),
    ("combined_bwd", ("swiglu_vec2d", "rmsnorm_bwd", "nr4")),
    ("combined_full", ("swiglu_vec2d", "rmsnorm_bwd", "rmsnorm_full", "nr4")),
])
def test_presets(mode: str, features: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LNET_MODEL_OPT", raising=False)
    monkeypatch.setenv("LNET_MODEL_W29C", mode)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 6))
    keys = ("swiglu_vec2d", "rmsnorm_bwd", "rmsnorm_full", "nr4")
    assert tuple(key for key in keys if model_w29c_enabled(torch.device("cuda"), key)) == features


@pytest.mark.parametrize("capability", [(7, 5), (8, 6), (8, 9), (9, 0), (10, 0)])
def test_auto_dispatch(capability: tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LNET_MODEL_OPT", raising=False)
    monkeypatch.delenv("LNET_MODEL_W28C", raising=False)
    monkeypatch.delenv("LNET_MODEL_W29C", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    device = torch.device("cuda")
    winners = ("swiglu_vec2d", "rmsnorm_bwd", "nr4")
    for feature in winners:
        assert model_w29c_enabled(device, feature) == (capability >= (8, 0))
        assert not model_w29c_enabled(torch.device("cpu"), feature)
    assert not model_w29c_enabled(device, "rmsnorm_full")
    monkeypatch.setenv("LNET_MODEL_W28C", "swiglu")
    assert not model_w29c_enabled(device, "swiglu_vec2d")
    monkeypatch.setenv("LNET_MODEL_W29C", "combined_bwd")
    monkeypatch.setenv("LNET_MODEL_OPT", "off")
    assert not any(model_w29c_enabled(device, feature) for feature in winners)


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_auto_dispatch_fullgraph(device_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_OPT", "auto")
    monkeypatch.setenv("LNET_MODEL_W28C", "off")
    monkeypatch.setenv("LNET_MODEL_W29C", "auto")
    if device_type == "cpu":
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda _: SimpleNamespace(major=10, minor=0),
        )
    sample = torch.zeros(4, device=device_type)
    supported = torch.cuda.get_device_capability(
        sample.device if device_type == "cuda" else torch.device("cuda"),
    ) >= (8, 0)

    def run(x: Tensor) -> Tensor:
        device = x.device if x.is_cuda else torch.device("cuda")
        return x + 1 if model_w29c_enabled(device, "rmsnorm_bwd") else x - 1

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(sample), sample + (1 if supported else -1))
    monkeypatch.setenv("LNET_MODEL_W29C", "off")
    torch.testing.assert_close(compiled(sample), sample - 1)


def _expression_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _spread_size(  # noqa: PLR0911
    node: ast.expr, bindings: dict[str, set[int | None]],
) -> int | None:
    match node:
        case ast.Tuple(elts=items) | ast.List(elts=items):
            sizes = [
                _spread_size(item.value, bindings) if isinstance(item, ast.Starred) else 1
                for item in items
            ]
            return sum(size for size in sizes if size is not None) if None not in sizes else None
        case ast.Name(id=name):
            sizes = bindings.get(name, set())
            return next(iter(sizes)) if len(sizes) == 1 else None
        case ast.IfExp(body=body, orelse=otherwise):
            size = _spread_size(body, bindings)
            return size if size == _spread_size(otherwise, bindings) else None
        case ast.GeneratorExp(generators=[generator]):
            return None if generator.ifs else _spread_size(generator.iter, bindings)
        case ast.Call(func=ast.Name(id="tuple"), args=[value]):
            return _spread_size(value, bindings)
        case ast.Call(func=ast.Name(id="cast"), args=[_, value]):
            return _spread_size(value, bindings)
        case ast.Call(func=ast.Name(id="range"), args=[ast.Constant(value=int(count))]):
            return count
        case ast.BinOp(left=items, op=ast.Mult(), right=ast.Constant(value=int(count))):
            size = _spread_size(items, bindings)
            return None if size is None else size * count
        case _:
            return None


def _tuple_annotation_size(node: ast.expr | None, aliases: dict[str, int]) -> int | None:
    match node:
        case ast.Subscript(value=ast.Name(id="tuple"), slice=ast.Tuple(elts=items)):
            if not any(isinstance(item, ast.Constant) and item.value is Ellipsis for item in items):
                return len(items)
        case ast.BinOp(left=items, op=ast.BitOr(), right=ast.Constant(value=None)):
            return _tuple_annotation_size(items, aliases)
        case ast.Name(id=name):
            return aliases.get(name)
    return None


def _spread_bindings(tree: ast.Module, lineno: int) -> dict[str, set[int | None]]:
    # Resolve fixed-size tensor packs, including the optional packs selected by the scan paths.
    scope = max(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
         and node.lineno < lineno <= node.end_lineno),
        key=lambda node: node.lineno,
    )
    aliases = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and (
            size := _tuple_annotation_size(statement.value, {})
        ):
            aliases.update({t.id: size for t in statement.targets if isinstance(t, ast.Name)})
    bindings: dict[str, set[int | None]] = {}
    for arg in (*scope.args.args, *scope.args.kwonlyargs):
        if (size := _tuple_annotation_size(arg.annotation, aliases)) is not None:
            bindings[arg.arg] = {size}
    leading_dims = {}
    assignments = sorted(
        (node for node in ast.walk(scope)
         if isinstance(node, ast.Assign) and node.lineno < lineno),
        key=lambda node: node.lineno,
    )
    for assignment in assignments:
        value = assignment.value
        size = _spread_size(value, bindings)
        for target in assignment.targets:
            if not isinstance(target, ast.Name):
                continue
            match value:
                case ast.Call(func=ast.Attribute(attr="new_empty"),
                              args=[ast.Tuple(elts=[ast.Constant(value=int(dim)), *_])]):
                    leading_dims[target.id] = dim
                case ast.Call(func=ast.Name(id="tuple"), args=[
                    ast.Call(func=ast.Attribute(value=ast.Name(id=name), attr="unbind"),
                             args=[ast.Constant(value=0)])
                ]):
                    size = leading_dims.get(name)
            bindings.setdefault(target.id, set()).add(size)
    return bindings


def _launch_positional_count(call: ast.Call, tree: ast.Module, location: str) -> int:
    positional = len(call.args)
    bindings = None
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            if bindings is None:
                bindings = _spread_bindings(tree, call.lineno)
            size = _spread_size(arg.value, bindings)
            assert size is not None, f"{location}: unresolved spread {ast.unparse(arg)}"
            positional += size - 1
    return positional


def test_triton_launch_signatures() -> None:
    source = Path(__file__).resolve().parents[1]
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for folder in ("src", "tests", "scripts", "h200")
        for path in (source / folder).rglob("*.py")
    }
    signatures = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and any(
                _expression_name(decorator.func if isinstance(decorator, ast.Call) else decorator)
                == "jit" for decorator in node.decorator_list
            ):
                assert node.name not in signatures, f"Ambiguous Triton kernel name: {node.name}"
                args = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                signatures[node.name] = [arg.arg for arg in args]

    target = GPUTarget("cuda", 90, 32)
    backend_options = vars(make_backend(target).parse_options({}))
    launches = 0
    errors = []
    for path, tree in trees.items():
        aliases = {
            name.asname: name.name
            for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            for name in node.names if name.asname and name.name in signatures
        }
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Subscript):
                continue
            kernel = call.func.value
            wrapped = (
                isinstance(kernel, ast.Call) and _expression_name(kernel.func) == "wrap_triton"
            )
            name = _expression_name(kernel.args[0] if wrapped else kernel)
            name = aliases.get(name, name)
            if not wrapped and name not in signatures:
                continue
            launches += 1
            location = f"{path.relative_to(source)}:{call.lineno} ({name})"
            assert name in signatures, f"{location}: unresolved Triton kernel"
            positional = _launch_positional_count(call, tree, location)
            arg_names = signatures[name]
            passed = arg_names[:positional] + [
                arg.arg for arg in call.keywords
                if arg.arg in arg_names or arg.arg not in backend_options
            ]
            if positional > len(arg_names) or sorted(map(str, passed)) != sorted(arg_names):
                errors.append(f"{location}: passed {passed}, expected {arg_names}")
    assert launches, "No Triton launches found"
    assert not errors, "\n".join(errors)


def test_representative_triton_launches_generate_ttir(monkeypatch: pytest.MonkeyPatch) -> None:
    # The target is the only driver information required to generate TTIR.
    target = GPUTarget("cuda", 90, 32)
    monkeypatch.setattr(
        triton.runtime.driver, "_active", SimpleNamespace(get_current_target=lambda: target),
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=132),
    )
    modules = []

    def wrap(kernel):
        def launch(*args: object, **kwargs: object) -> None:
            named = dict(zip(kernel.arg_names[:len(args)], args, strict=True))
            module, _ = generate_ttir(kernel, {**named, **kwargs}, {})
            modules.append(str(module))

        return {(1,): launch, (1, 1): launch}

    monkeypatch.setattr(model_w29c, "wrap_triton", wrap)
    x = torch.empty(4, 64, dtype=torch.bfloat16, device="cpu")
    model_w29c._channel_rmsnorm_backward(
        x, x[0], torch.ones(4, device="cpu"), x, need_dx=True, need_dw=True,
    )
    model_w29c._launch_swiglu(x, x, x[:, :32], backward=False)
    assert len(modules) == 2
    assert all("tt.func" in module for module in modules)


def test_row_pitch() -> None:
    tensor = torch.empty(3, 5, 17)
    assert _row_pitch(tensor) == 17
    assert _row_pitch(tensor[..., :9]) == 17
    assert _row_pitch(tensor.transpose(0, 1)) == 0
    assert _row_pitch(tensor[:, ::2, :]) == 0
    assert _row_pitch(tensor[..., :1].expand_as(tensor)) == 0


def _output_and_grads(
    output: Tensor, grad: Tensor, params: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    return output, *torch.autograd.grad(output, params, grad)


def _assert_close(
    actual: tuple[Tensor, ...], expected: tuple[Tensor, ...], dtype: torch.dtype,
) -> None:
    for got, want in zip(actual, expected, strict=True):
        error = (got.float() - want.float()).norm()
        assert error <= 2 * torch.finfo(dtype).eps * want.float().norm() + 1e-6


def _compare(output: Tensor, ref: Tensor, grad: Tensor, params: tuple[Tensor, ...]) -> None:
    _assert_close(
        _output_and_grads(output, grad, params), _output_and_grads(ref, grad, params), ref.dtype,
    )


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [65, 96, 144, 192, 384, 520])
def test_swiglu_vec2d(dtype: torch.dtype, width: int) -> None:
    packed = torch.randn(3, 11, 2 * width, device="cuda", dtype=dtype, requires_grad=True)
    value, gate = packed.chunk(2, -1)
    ref = value * functional.silu(gate)
    _compare(packed_swiglu_vec2d(packed), ref, torch.randn_like(ref), (packed,))


@CUDA
@pytest.mark.parametrize("mode", ["rmsnorm_bwd", "rmsnorm_full"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [64, 65, 96, 128, 256])
@pytest.mark.parametrize("layout", ["dense", "pitched", "transposed", "broadcast_grad"])
def test_channel_rmsnorm(
    mode: str, dtype: torch.dtype, width: int, layout: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LNET_MODEL_W29C", mode)
    torch.manual_seed(502)
    norm = ChannelRMSNorm(width).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        norm.weight.normal_()
    pitch = width + 16 if layout == "pitched" else width
    x = torch.randn(3, 179, pitch, device="cuda", dtype=dtype)[..., :width]
    grad = torch.randn(3, 179, pitch, device="cuda", dtype=dtype)[..., :width]
    if layout == "transposed":
        x = x.transpose(0, 1)
    if layout == "broadcast_grad":
        grad = grad[..., :1].expand_as(x)
    if layout == "transposed":
        grad = grad.transpose(0, 1)
    x.requires_grad_()
    ref = functional.rms_norm(x, (width,), norm.weight, norm.epsilon)
    _compare(norm(x), ref, grad, (x, norm.weight))


@CUDA
@pytest.mark.parametrize("mode", ["combined_bwd", "combined_full"])
def test_compile_fullgraph(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W29C", mode)
    torch.manual_seed(503)
    norm = ChannelRMSNorm(96).cuda()
    packed = torch.randn(29, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def run(x: Tensor) -> Tensor:
        return norm(packed_swiglu_vec2d(x))

    eager = run(packed)
    compiled = torch.compile(run, fullgraph=True)(packed)
    upstream = torch.randn_like(eager)
    params = packed, norm.weight
    actual = _output_and_grads(compiled, upstream, params)
    expected = _output_and_grads(eager, upstream, params)
    # The scale gradient is stored as FP32 after low-precision activation math.
    # Check at activation precision and independently against FP32 arithmetic.
    _assert_close(actual, expected, packed.dtype)

    packed_fp32 = packed.detach().float().requires_grad_()
    weight_fp32 = norm.weight.detach().float().requires_grad_()
    value, gate = packed_fp32.chunk(2, -1)
    reference = functional.rms_norm(value * functional.silu(gate), (96,), weight_fp32, norm.epsilon)
    fp32 = _output_and_grads(reference, upstream.float(), (packed_fp32, weight_fp32))
    _assert_close(expected, fp32, packed.dtype)
    _assert_close(actual, fp32, packed.dtype)
