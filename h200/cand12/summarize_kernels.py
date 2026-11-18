"""Report CUDA kernel durations and separate scan launch geometries."""
from collections import defaultdict
import json


def summarize(path, batch, accumulation, updates):
    groups = defaultdict(lambda: [0, 0.0])
    for event in json.loads(path.read_text())["traceEvents"]:
        if event.get("cat") != "kernel":
            continue
        name = event["name"]
        metadata = event.get("args", {})
        scan = "scan" in name.lower() or "_bidirectional_" in name.lower()
        if not scan:
            name = name.split("(")[0][:140]
        grid = tuple(metadata.get("grid", [])) if scan else ()
        block = tuple(metadata.get("block", [])) if scan else ()
        key = (name, grid, block, metadata.get("registers per thread") if scan else None)
        row = groups[key]
        row[0] += 1
        row[1] += event["dur"]
    rows = [
        dict(kernel=name, grid=grid, block=block, registers=regs,
             launches=calls, ms_per_update=round(us / updates / 1000, 4))
        for (name, grid, block, regs), (calls, us) in
        sorted(groups.items(), key=lambda item: item[1][1], reverse=True)
    ]
    if not rows:
        raise RuntimeError("CUDA trace contained no kernels")
    print("KERNEL_TOTAL=" + json.dumps(dict(batch=batch, accumulation=accumulation,
          ms_per_update=sum(row["ms_per_update"] for row in rows))), flush=True)
    for row in rows:
        if row in rows[:15] or "scan" in row["kernel"].lower() or "_bidirectional_" in row["kernel"]:
            print("KERNEL_ROW=" + json.dumps(row), flush=True)
