"""Triton source transformations for bounded launch experiments."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import triton
import triton.language as tl


@triton.jit
def wide_batched_dot(a, b):
    """Supply independent 2-D, 64-row MLP tiles to the Hopper dot lowering."""
    modes: tl.constexpr = a.shape[0]
    rows: tl.constexpr = a.shape[1]
    inner: tl.constexpr = a.shape[2]
    cols: tl.constexpr = b.shape[2]
    tile: tl.constexpr = 64 if rows < 64 else rows
    row = tl.arange(0, tile)[:, None]
    result = tl.full((modes, rows, cols), 0, tl.float32)
    for mode in tl.static_range(modes):
        left = tl.reshape(tl.gather(a, tl.full((1, rows, inner), mode, tl.int32), 0), (rows, inner))
        right = tl.reshape(tl.gather(b, tl.full((1, inner, cols), mode, tl.int32), 0), (inner, cols))
        if tile != rows:
            index = tl.broadcast_to(row % rows, (tile, inner))
            left = tl.where(row < rows, tl.gather(left, index, 0), 0).to(a.dtype)
        product = tl.dot(left, right)
        if tile != rows:
            index = tl.broadcast_to(tl.arange(0, rows)[:, None], (rows, cols))
            product = tl.gather(product, index, 0)
        lane = tl.arange(0, modes)[:, None, None] == mode
        result = tl.where(lane, product[None, :, :], result)
    return result


@triton.jit
def load_vertical_tma(base, batch, x, mode, h, w, m, B: tl.constexpr, BM: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        base, shape=[B, h, w, m], strides=[h * w * m, w * m, m, 1],
        block_shape=[1, 64, 1, BM],
    )
    return tl.reshape(desc.load([batch, 0, x, mode]), (64, 1, BM))


@triton.jit
def store_vertical_tma(base, data, batch, x, mode, h, w, m, B: tl.constexpr, BM: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        base, shape=[B, h, w, m], strides=[h * w * m, w * m, m, 1],
        block_shape=[1, 64, 1, BM],
    )
    desc.store([batch, 0, x, mode], tl.reshape(tl.trans(data), (1, 64, 1, BM)))


def load_module(name, path, namespace=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if namespace:
        module.__dict__.update({k: v for k, v in namespace.items() if not k.startswith("__")})
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_for(fn, folder, *, scheduler="grid", line_axis=1, dot=None, tma=False, forward=False):
    if dot == "wgmma" and forward:
        helper = fn.__globals__["_first_layer"]
        helper = source_for(helper, folder, dot="wgmma")
    else:
        helper = None

    class Rewrite(ast.NodeTransformer):
        def __init__(self):
            self.tma_loads = 0

        def visit_Call(self, node):
            self.generic_visit(node)
            name = ast.unparse(node.func)
            if name == "tl.program_id" and scheduler != "grid":
                return ast.copy_location(ast.Name(f"logical_{node.args[0].value}", ast.Load()), node)
            if name == "tl.dot" and dot == "ieee":
                a, b = [ast.unparse(arg) for arg in node.args[:2]]
                return ast.parse(f'tl.dot(({a}).to(tl.float32), ({b}).to(tl.float32), input_precision="ieee")',
                                 mode="eval").body
            if name == "tl.dot" and dot == "wgmma":
                if ast.unparse(node.args[0]) in {"packed_path", "packed", "hidden"} and not forward:
                    node.func = ast.Name("wide_batched_dot", ast.Load())
            if name == "tl.store" and tma and forward:
                ptr = node.args[0]
                if isinstance(ptr, ast.BinOp) and ast.unparse(ptr.left) in ("OR", "OI"):
                    mode = "tl.program_id(0)" if scheduler == "grid" else "logical_0"
                    expr = (f"store_vertical_tma({ast.unparse(ptr.left)}, {ast.unparse(node.args[1])}, "
                            f"batch, x, {mode} // GROUP * BM, H, W, M, TMA_BATCH, BM)")
                    return ast.parse(expr, mode="eval").body
            if name == "tl.load" and tma:
                ptr = node.args[0]
                if isinstance(ptr, ast.BinOp) and isinstance(ptr.left, ast.Name):
                    names = {"SA": "x", "IA": "x", "SB": "x", "IB": "x"} if forward else {
                        "source_real_a": "source_x_a", "source_imag_a": "source_x_a",
                        "source_real_b": "source_x_b", "source_imag_b": "source_x_b",
                    }
                    if ptr.left.id in names and ast.unparse(ptr.right) in {"off", "offset", "offset_b"}:
                        self.tma_loads += 1
                        if forward:
                            expr = f"load_vertical_tma({ptr.left.id}, batch, x, logical_0 // GROUP * BM, H, W, M, TMA_BATCH, BM)"
                        else:
                            coord = f"{names[ptr.left.id]}_scalar"
                            expr = f"load_vertical_tma({ptr.left.id}, batch, {coord}, mode_group * BM, height, width, modes, TMA_BATCH, BM)"
                        if scheduler == "grid":
                            expr = expr.replace("logical_0", "tl.program_id(0)")
                        return ast.parse(expr, mode="eval").body
            return node

    tree = ast.parse(fn.src)
    func = tree.body[-1]
    func.decorator_list = [ast.parse("triton.jit", mode="eval").body]
    rewriter = Rewrite()
    rewriter.visit(func)
    if tma:
        if rewriter.tma_loads < 4:
            raise ValueError("TMA source mapping did not find all four source planes")
        if not forward:
            insert = next(i for i, stmt in enumerate(func.body)
                          if isinstance(stmt, ast.Assign) and ast.unparse(stmt.targets[0]) == "x_group") + 1
            func.body[insert:insert] = ast.parse(
                "source_x_a_scalar = 2 * x_group + 1 if COARSE else x_group\n"
                "source_x_b_scalar = 2 * x_group if COARSE else x_group"
            ).body
    if scheduler != "grid":
        original = func.body
        original_args = func.args.args[:]
        func.args.args[:0] = [
            ast.arg(name, ast.parse("tl.constexpr", mode="eval").body)
            for name in ("GRID0", "GRID1", "SERIAL_LINES")
        ]
        logical = (
            "logical_0 = work_id // GRID1\nlogical_1 = work_id % GRID1"
            if line_axis == 1 else
            "logical_1 = work_id // GRID0\nlogical_0 = work_id % GRID0"
        )
        if scheduler == "serial":
            loop = ast.parse(
                "start = tl.program_id(0) * SERIAL_LINES\n"
                "for work_id in tl.range(start, tl.minimum(start + SERIAL_LINES, GRID0 * GRID1)):\n    pass"
            ).body
        else:
            specialize = ", warp_specialize=True" if scheduler == "warp_specialize" else ""
            loop = ast.parse(
                f"for work_id in tl.range(tl.program_id(0), GRID0 * GRID1, tl.num_programs(0){specialize}):\n"
                "    pass"
            ).body
        args = [a.arg for a in original_args]
        # Each tile starts from the same base pointers, including mutated gradient offsets.
        tile = ast.FunctionDef(
            name="run_tile",
            args=ast.arguments(posonlyargs=[], args=[*original_args, ast.arg("logical_0"), ast.arg("logical_1")],
                               vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=original, decorator_list=func.decorator_list, returns=None,
        )
        tile_call = "run_tile(" + ", ".join([*args, "logical_0", "logical_1"]) + ")"
        loop[-1].body = ast.parse(logical + "\n" + tile_call).body
        func.body = loop
        tree.body.insert(0, tile)
    if tma:
        func.args.args.insert(0, ast.arg("TMA_BATCH", ast.parse("tl.constexpr", mode="eval").body))
        if scheduler != "grid":
            tile.args.args.append(ast.arg("TMA_BATCH", ast.parse("tl.constexpr", mode="eval").body))
            loop[-1].body[-1].value.args.append(ast.Name("TMA_BATCH", ast.Load()))
    ast.fix_missing_locations(tree)
    code = "import triton\n\n" + ast.unparse(tree) + "\n"
    key = hashlib.sha256((fn.cache_key + code).encode()).hexdigest()[:20]
    path = Path(folder) / f"r8_{key}.py"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(code)
    namespace = dict(fn.__globals__, wide_batched_dot=wide_batched_dot,
                     load_vertical_tma=load_vertical_tma, store_vertical_tma=store_vertical_tma)
    if helper is not None:
        namespace["_first_layer"] = helper
    module = load_module(f"r8_{key}", path, namespace)
    return getattr(module, func.name)



def hback_mode_tail(fn, folder):
    """Mask the final BM64 tile when 160 modes leave 32 live lanes."""
    code = fn.src.replace("active = x < W", "active = (x < W) & valid_mode")
    code = code.replace("    active =", "    valid_mode = mode < M\n    active =", 1)
    for pole in ("AR", "AI", "GR", "GI"):
        code = code.replace(f"tl.load({pole} + mode)", f"tl.load({pole} + mode, valid_mode, 0)")
    code = code.replace("x < W - 1, 0", "(x < W - 1) & valid_mode, 0")
    code = code.replace('sem="relaxed",', 'mask=valid_mode, sem="relaxed",')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "hback_mode_tail.py"
    path.write_text("import triton\n@triton.jit\n" + code + "\n")
    return load_module("hback_mode_tail", path, fn.__globals__).hback56
