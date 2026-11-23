"""Join compiled ATen matrix calls to readable FX, with signature validation."""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

from audit import MATRIX_OPS

if TYPE_CHECKING:
    from pathlib import Path

TENSOR = re.compile(r'\b(\w+): "(\w+)\[([\d, ]*)\]"')
OP = re.compile(r"torch\.ops\.aten\.(mm|addmm|bmm|baddbmm)\.default\(([^)]*)\)")


def fx_calls(path: Path, phase: str) -> list[dict]:
    """Read concrete tensor signatures and source annotations from Torch 2.9 dumps."""
    tensors = {}
    sources = {}
    calls = []
    source = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "# File: " in line:
            source = line.split("# File: ", 1)[1]
        elif "# No stacktrace found" in line:
            source = None
        for name, dtype, dims in TENSOR.findall(line):
            tensors[name] = dtype, ast.literal_eval(f"({dims},)") if dims else ()
            sources[name] = source
        match = OP.search(line)
        if not match:
            continue
        name = "aten::" + match[1]
        operands = [part.strip() for part in match[2].split(",")]
        start = MATRIX_OPS[name]
        left, right = operands[start : start + 2]
        left_dtype, left_dims = tensors[left]
        right_dtype, right_dims = tensors[right]
        calls.append(
            {
                "name": name,
                "phase": phase,
                "source": source,
                "operand_sources": [sources[left], sources[right]],
                "dims": [left_dims, right_dims],
                "types": [left_dtype, right_dtype],
            }
        )
    return calls


def generated_layouts(path: Path) -> list[tuple]:  # noqa: C901
    """Read explicit Inductor strides without importing generated GPU code."""
    variables = {}
    layouts = []

    def signature(node: ast.expr) -> tuple:
        if isinstance(node, ast.Name):
            return variables.get(node.id, ())
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return ()
        name = node.func.id
        offset = 1 if name in {"reinterpret_tensor", "as_strided"} else 0
        if name not in {"reinterpret_tensor", "as_strided", "empty_strided_cuda"}:
            return ()
        return tuple(ast.literal_eval(arg) for arg in node.args[offset : offset + 2])

    code = ast.parse(path.read_text(encoding="utf-8"))
    for node in sorted(ast.walk(code), key=lambda node: getattr(node, "lineno", 0)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            layout = signature(node.value)
            if layout:
                variables[node.targets[0].id] = layout
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "extern_kernels":
            continue
        start = MATRIX_OPS.get("aten::" + node.func.attr)
        if start is not None:
            layouts.append(tuple(signature(arg) for arg in node.args[start : start + 2]))
    return layouts


def compiled_calls(events: list[dict], paths: tuple[Path, Path]) -> dict[int, dict]:  # noqa: C901
    """Require equal operator sequences in compiled forward and backward."""
    compiled = {"forward": [], "backward": []}
    for event in events:
        if event.get("cat") != "cpu_op":
            continue
        if event["name"].startswith("CompiledFunctionBackward"):
            compiled["backward"].append(event)
        elif event["name"] == "CompiledFunction":
            compiled["forward"].append(event)
    matched = {}
    for phase, path in zip(compiled, paths, strict=True):
        scopes = compiled[phase]
        calls = fx_calls(path, phase)
        generated = path.with_name("output_code.py")
        if generated.is_file():
            layouts = generated_layouts(generated)
            if len(layouts) == len(calls):
                for call, pair in zip(calls, layouts, strict=True):
                    call["strides"] = [
                        layout[1] if layout and layout[0] == shape else None
                        for layout, shape in zip(pair, call["dims"], strict=True)
                    ]
        names = {call["name"] for call in calls}
        ops = [
            event
            for event in sorted(events, key=lambda event: event.get("ts", 0))
            if event.get("cat") == "cpu_op"
            and event["name"] in names
            and any(
                event["tid"] == scope["tid"]
                and scope["ts"] <= event["ts"] <= scope["ts"] + scope["dur"]
                for scope in scopes
            )
        ]
        if not ops:
            continue
        if [event["name"] for event in ops] != [call["name"] for call in calls]:
            continue
        for event, call in zip(ops, calls, strict=True):
            matched[event["args"]["External id"]] = call
    return matched
