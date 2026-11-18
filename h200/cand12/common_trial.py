"""Discover the compiled H200 scope, stage trial records and check the resulting launch."""
import json
import os
from dataclasses import asdict
from pathlib import Path
import runpy
import sys


def main():
    from lnet.kernel_launch_config import capture_launch_scopes, store_geometry, stored_geometry
    from lnet.path_backward_layout import LAYOUT_NAME, candidate_geometry
    from tune_path_backward import launch_contracts, load_scope
    root=Path(os.environ["RUN_ROOT"])
    arm=os.environ["ARM"]
    shape=(1024,56,56,160)
    scope_file=root/"observed.json"
    specs={"common8":(4,8,4,3),"common2w":(4,4,2,3)}
    if arm=="discovery":
        with capture_launch_scopes(LAYOUT_NAME) as found:
            # Discovery uses a fresh graph cache. Read the scope produced by an actual backward.
            try:
                runpy.run_path("benchmark_imagenet1k.py",run_name="__main__")
            except SystemExit as exc:
                if exc.code!=43:
                    raise
        matched=[s for s in found if s.execution_regime=="compiled" and tuple(dict(s.shape_items)[k] for k in ("batch","height","width","modes"))==shape]
        assert len(matched)==1, found
        scope_file.write_text(json.dumps(asdict(matched[0])))
        print("OBSERVED_SCOPE", scope_file.read_text())
        return
    scope=load_scope(json.loads(scope_file.read_text()))
    spec=specs[arm]
    geometry=candidate_geometry(*spec)
    if "--layout-check" in sys.argv:
        store_geometry(LAYOUT_NAME,geometry,scope=scope)
    assert stored_geometry(LAYOUT_NAME,scope=scope)==geometry
    runpy.run_path("benchmark_imagenet1k.py",run_name="__main__")
    if "--kernel-profile" in sys.argv:
        launches=launch_contracts(Path("kernel-trace.json"),shape)
        assert len(launches)==3 and all((v["group"],v["modes"],v["warps"],v["compact"])==spec and v["lines"]==1 for v in launches), launches
        print("VERIFIED_LAUNCH",arm,json.dumps(launches))


if __name__=="__main__":
    main()
