"""Join profiled Triton names to AOT/Inductor FX source locations."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from h200.liverms_v2_opt.bench_h200 import write_json

SOURCE = re.compile(r"^\s*# File: (.+?):(\d+) in ([^,]+), code: (.*)")
ASSIGN = re.compile(r"^ {8}(\w+)(?:: .*?)? = ")


def fx_sources(path: Path) -> dict[str, dict]:
    sources = {}
    site = {}
    for line in path.read_text().splitlines():
        match = SOURCE.match(line)
        if match:
            filename, lineno, function, code = match.groups()
            site = {"file": filename, "line": int(lineno), "function": function, "code": code}
        assignment = ASSIGN.match(line)
        if assignment and site:
            sources[assignment[1]] = site
    return sources


def kernel_sources(debug_dir: Path) -> dict[str, list[dict]]:
    kernels = defaultdict(list)
    for path in sorted(debug_dir.rglob("output_code.py")):
        sources = fx_sources(path.with_name("fx_graph_readable.py"))
        nodes = []
        for line in path.read_text().splitlines():
            if line.startswith("# kernel path:"):
                nodes = []
            match = re.match(r"#   %(\w+) : .* = call_function", line)
            if match:
                nodes.append(match[1])
            kernel = re.match(r"(triton_\w+) = async_compile.triton", line)
            if kernel:
                sites = [{"node": node, **sources[node]} for node in nodes if node in sources]
                kernels[kernel[1]].append({
                    "graph": path.parent.name,
                    "nodes": nodes,
                    "sites": sites,
                })
    return dict(kernels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.profile.read_text())
    graph_kernels = kernel_sources(args.debug_dir)
    kernels = [
        {**row, "origins": graph_kernels.get(row["name"], [])}
        for row in result["profiler"]["kernels"]
        if row["bucket"] == "triton"
    ]
    write_json(args.out, {"profiled_triton": kernels, "generated_kernels": graph_kernels})


if __name__ == "__main__":
    main()
