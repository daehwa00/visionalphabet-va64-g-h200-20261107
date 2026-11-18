"""Use the exact 4090 scopes with a device-specific H200 tuning store."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy
import sys

from lnet.kernel_launch_config import capture_launch_scopes, store_geometry, stored_geometry
from lnet.path_backward_layout import LAYOUT_NAME, candidate_geometry
from tune_path_backward import launch_contracts, load_scope


def main():
    arm = os.environ["ARM"]
    root = Path(os.environ["RUN_ROOT"])
    shapes = (14, 28, 56)
    scope_file = root / "matched-scopes.json"
    selected = {14: (4, 8, 4, 3), 28: (4, 8, 4, 3), 56: (4, 4, 2, 3)}
    if arm == "h8w4":
        selected[56] = (4, 8, 4, 3)
    elif arm == "h4w1":
        selected[56] = (4, 4, 1, 3)
    if arm == "discover":
        with capture_launch_scopes(LAYOUT_NAME) as found:
            runpy.run_path("benchmark_matched.py", run_name="__main__")
        observed = {}
        for size in shapes:
            matches = [s for s in found if s.execution_regime == "compiled" and
                       tuple(dict(s.shape_items)[k] for k in ("batch","height","width","modes")) == (128,size,size,160)]
            assert len(matches) == 1, found
            observed[str(size)] = asdict(matches[0])
        scope_file.write_text(json.dumps(observed))
        return
    scopes = json.loads(scope_file.read_text())
    for size, spec in selected.items():
        scope = load_scope(scopes[str(size)])
        geometry = candidate_geometry(*spec)
        if stored_geometry(LAYOUT_NAME,scope=scope) is None:
            store_geometry(LAYOUT_NAME, geometry, scope=scope)
        assert stored_geometry(LAYOUT_NAME,scope=scope) == geometry
    runpy.run_path(os.environ.get("MATCHED_ENTRY","benchmark_matched.py"), run_name="__main__")
    if "--profile" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--output")+1])
        runs = {}
        for size in shapes:
            launches = launch_contracts(out.with_suffix(".trace.json"), (128,size,size,160))
            assert all((v["group"],v["modes"],v["warps"],v["compact"]) == selected[size] and v["lines"] == 1 for v in launches)
            runs[str(size)] = launches
        out.with_suffix(".launches.json").write_text(json.dumps(runs))


if __name__ == "__main__":
    main()
