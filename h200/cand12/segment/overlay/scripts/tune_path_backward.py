"""Validate and persist a device-specific direct-path backward layout on matched training steps."""

from __future__ import annotations

import argparse
import ast
import json
import os
import runpy
import shutil
import statistics
import subprocess
from contextlib import nullcontext
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lnet.kernel_launch_config import (
    LaunchScope,
    capture_launch_scopes,
    device_key,
    store_geometry,
    stored_geometry,
)
from lnet.path_backward_layout import (
    BASELINE,
    BUNDLED_LAYOUTS,
    LAYOUT_NAME,
    candidate_geometry,
)


from lnet.retained_path_dispatch import (
    BUNDLE as STEP_BUNDLE,
    PROFILE_NAME,
    HORIZONTAL_BACKWARD,
    VERTICAL_BACKWARD,
    VERTICAL_FORWARD,
    legacy_layout_trial,
    step_geometry,
)

PHASES = {
    "forward": (VERTICAL_FORWARD, 0),
    "backward": (VERTICAL_BACKWARD, 0),
    "coarse-backward": (VERTICAL_BACKWARD, 1),
    "horizontal-backward": (HORIZONTAL_BACKWARD, 0),
    "coarse-horizontal-backward": (HORIZONTAL_BACKWARD, 1),
}


def load_scope(fields: dict) -> LaunchScope:
    return LaunchScope(**{**fields, "shape_items": tuple(map(tuple, fields["shape_items"]))})


def trial(job: dict) -> None:
    phase = job.get("phase", "legacy")
    name = LAYOUT_NAME if phase == "legacy" else PROFILE_NAME
    if job["scope"] is not None:
        scope = load_scope(job["scope"])
        spec = job["candidate"]
        geometry = (
            (
                candidate_geometry(*spec)
                if phase == "legacy"
                else step_geometry(PHASES[phase][0], *spec)
            )
            if spec
            else BASELINE
        )
        store_geometry(name, geometry, scope=scope)
        if stored_geometry(name, scope=scope) != geometry:
            raise RuntimeError("Could not stage the trial geometry")
    sys.argv = [job["benchmark"], *job["arguments"]]
    # The model runner lives in this source tree. The evaluator can live elsewhere.
    with (
        legacy_layout_trial() if phase == "legacy" else nullcontext(),
        capture_launch_scopes(name) as scopes,
    ):
        runpy.run_path(job["benchmark"], run_name="__main__")
    Path(job["observed"]).write_text(json.dumps([asdict(s) for s in scopes], indent=2))


def numerical_gate(result: dict) -> bool:
    checks = result.get("gradient_checks", {})
    return (
        result.get("status") == "passed"
        and result.get("validation_finite")
        and result.get("final_finite")
        and result.get("missing_gradients") == []
        and result.get("output_check", {}).get("passed")
        and all(
            checks.get(k) and all(v["passed"] for v in checks[k].values())
            for k in ("grads", "updates")
        )
        and checks["grads"].keys() == checks["updates"].keys()
    )


def launch_contracts(trace: Path, shape: tuple[int, ...]) -> list[dict]:
    """Read what Inductor actually compiled, including the two path layout flags."""
    events = json.loads(trace.read_text())["traceEvents"]
    dispatches = {
        e["args"]["External id"]: e["args"]
        for e in events
        if e.get("cat") == "cpu_op" and "kernel_file" in e.get("args", {})
    }
    launches = []
    for e in events:
        if (
            e.get("cat") != "kernel"
            or "_product_scan_coarse4_associative_backward_kernel" not in e["name"]
        ):
            continue
        launch = e["args"]
        call = dispatches[launch["External id"]]
        if tuple(call["Input Dims"][4]) != shape or tuple(call["Input Dims"][13]) != shape:
            continue
        tree = ast.parse(Path(call["kernel_file"]).read_text())
        meta = next(
            n.value for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == "triton_meta"
        )
        constants = next(
            ast.literal_eval(v)
            for k, v in zip(meta.keys, meta.values, strict=True)
            if isinstance(k, ast.Constant) and k.value == "constants"
        )
        tile = dict(x.split("=") for x in call["kernel_kwargs"].split(","))
        launches.append(
            dict(
                group=constants["LINES_PER_GROUP"],
                compact=constants["COMPACT_CHANNELS"],
                lines=int(tile["BLOCK_LINES"]),
                modes=int(tile["BLOCK_MODES"]),
                warps=call["num_warps"],
                grid=launch["grid"],
                ms=e["dur"] / 1000,
            )
        )
    if not launches:
        raise RuntimeError(f"Could not identify measured path backward for {shape}")
    return launches


def retained_launches(
    trace: Path, target: tuple[int, ...], phase: str, candidate: bool
) -> list[dict]:
    operation, coarse = PHASES[phase]
    symbol = (
        ("_path_forward_kernel", "_path_backward_kernel", "_horizontal_path_backward_kernel")
        if candidate
        else (
            "_product_scan_coarse4_associative_forward_kernel",
            "_product_scan_coarse4_associative_backward_kernel",
            "_bidirectional_backward_kernel",
        )
    )[operation]
    events = json.loads(trace.read_text())["traceEvents"]
    calls = {
        e["args"]["External id"]: e["args"]
        for e in events
        if e.get("cat") == "cpu_op" and "kernel_file" in e.get("args", {})
    }
    launches = []
    for e in events:
        if e.get("cat") != "kernel" or e["name"] != symbol:
            continue
        launch = e["args"]
        call = calls[launch["External id"]]
        if tuple(call["Input Dims"][4]) != target:
            continue
        tree = ast.parse(Path(call["kernel_file"]).read_text())
        meta = next(
            n.value for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == "triton_meta"
        )
        constants = next(
            ast.literal_eval(v)
            for k, v in zip(meta.keys, meta.values, strict=True)
            if isinstance(k, ast.Constant) and k.value == "constants"
        )
        if (
            int(
                constants.get(
                    "COARSE", constants.get("COARSE_EVEN_ODD", constants.get("COARSE_PATHS", False))
                )
            )
            != coarse
        ):
            continue
        tile = dict(x.split("=") for x in call.get("kernel_kwargs", "").split(",") if "=" in x)
        launches.append(
            dict(
                segment_rows=4
                if any(
                    isinstance(n, ast.FunctionDef) and n.name == "_segment_four"
                    for n in ast.walk(tree)
                )
                else 1,
                group=int(tile.get("GROUP", 0)),
                modes=int(tile.get("BM", tile.get("BLOCK_MODES", 0))),
                warps=call["num_warps"],
                compact=int(tile.get("DOUBLING", 0)),
                lines=1,
                grid=launch["grid"],
                ms=e["dur"] / 1000,
            )
        )
    if not launches:
        raise RuntimeError(f"No {symbol} launch matched {target} for {phase}")
    return launches


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmark",
        required=True,
        type=Path,
        help="Matched cand12 training benchmark script",
    )
    for key in ("checkpoint", "contract", "images", "output-dir"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument(
        "--phase",
        choices=("legacy", *PHASES),
        default="legacy",
        help="Use legacy for the original path layout, or select a retained-state kernel.",
    )
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--shape", type=int, nargs=3, default=(56, 56, 160), metavar=("H", "W", "M"))
    p.add_argument(
        "--candidate",
        type=int,
        nargs=4,
        action="append",
        metavar=("GROUP", "MODES", "WARPS", "OPTION"),
        help="OPTION is the legacy compact-channel mask, or the retained kernel doubling flag.",
    )
    p.add_argument(
        "--seed-inductor-cache",
        type=Path,
        help="Reuse saved kernel autotune choices from the same benchmark",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--mode", default="default")
    args = p.parse_args()
    if min(args.batch, *args.shape, args.warmup, args.steps, args.repeats) < 1:
        p.error("Dimensions and measurement counts must be positive")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.seed_inductor_cache:
        shutil.copytree(
            args.seed_inductor_cache,
            root / "graphs",
            ignore=shutil.ignore_patterns("aotautograd", "fxgraph", "locks"),
        )
    scope = None
    legacy = args.phase == "legacy"
    name = LAYOUT_NAME if legacy else PROFILE_NAME
    bundle = BUNDLED_LAYOUTS if legacy else STEP_BUNDLE

    def run(arm: str, label: str, spec=None, reference=None, profile=False) -> dict:
        output = root / f"{label}.json"
        cmd = [
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--contract",
            str(args.contract.resolve()),
            "--images",
            str(args.images.resolve()),
            "--output",
            str(output),
            "--batch",
            str(args.batch),
            "--mode",
            args.mode,
            "--spatial",
            "--vector",
            "--reset-each-step",
            "--warmup",
            str(args.warmup),
            "--steps",
            str(args.steps),
            "--repeats",
            str(args.repeats),
        ]
        if reference:
            cmd += ["--reference", str(reference)]
        if profile:
            cmd += ["--profile"]
        job = dict(
            benchmark=str(args.benchmark.resolve()),
            arguments=cmd,
            scope=asdict(scope) if scope else None,
            candidate=spec,
            phase=args.phase,
            observed=str(output.with_suffix(".scopes.json")),
        )
        job_file = output.with_suffix(".job.json")
        job_file.write_text(json.dumps(job))
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                (
                    str(Path(__file__).resolve().parent),
                    str(Path(__file__).resolve().parents[1] / "src"),
                    os.environ.get("PYTHONPATH", ""),
                )
            ),
            LNET_LAUNCH_CACHE=str(root / "trial-stores" / arm),
            TORCHINDUCTOR_CACHE_DIR=str(root / "graphs"),
            TRITON_CACHE_DIR=str(root / "triton"),
            PYTHONHASHSEED="0",
        )
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(
                [sys.executable, __file__, "--trial", str(job_file)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return json.loads(output.read_text())

    run("discovery", "discovery")
    observed = [load_scope(d) for d in json.loads((root / "discovery.scopes.json").read_text())]
    target = (args.batch, *args.shape)
    matches = [
        s
        for s in observed
        if tuple(dict(s.shape_items)[k] for k in ("batch", "height", "width", "modes")) == target
        and s.execution_regime == ("eager" if args.mode == "eager" else "compiled")
        and (
            legacy
            or (dict(s.shape_items)["operation"], dict(s.shape_items)["coarse"])
            == PHASES[args.phase]
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one observed scope for {target}, found {len(matches)}")
    scope = matches[0]
    baseline = run("control", "reference")
    reference = root / "reference.json"
    candidates = args.candidate or [
        [4, 4, 2, 3],
        [1, 8, 4, 3],
        [4, 8, 4, 0],
        [4, 8, 8, 3],
    ]
    if not legacy and args.candidate is None:
        op = PHASES[args.phase][0]
        candidates = [[1, 32, 4, 0]] if op == HORIZONTAL_BACKWARD else [[4, 4, 2, 1], [4, 8, 4, 0]]
    previous = stored_geometry(name, scope=scope) or stored_geometry(name, scope=scope, path=bundle)
    if previous and previous != BASELINE:
        blocks = previous.blocks
        candidates = [
            *candidates,
            [
                blocks["LINES_PER_GROUP"] if legacy else blocks["GROUP"],
                blocks["BLOCK_MODES"] if legacy else blocks["BM"],
                previous.num_warps,
                blocks["COMPACT_CHANNELS"] if legacy else blocks.get("DOUBLING", 0),
            ],
        ]
    audit = dict(
        device=device_key(),
        scope=asdict(scope),
        baseline_ms=baseline["median_ms"],
        trials=[],
    )
    best_score, winner = 1.0, None
    for index, spec in enumerate(dict.fromkeys(tuple(c) for c in candidates)):
        geometry = (
            candidate_geometry(*spec) if legacy else step_geometry(PHASES[args.phase][0], *spec)
        )
        if spec[0] and args.batch * args.shape[1] % spec[0]:
            continue
        arm = f"candidate-{index}"
        record = dict(candidate=spec, pairs=[], passed=False)
        audit["trials"].append(record)
        for i in range(3):
            order = (arm, "control") if i == 1 else ("control", arm)
            pair = {"order": order}
            for side in order:
                label = f"{arm}-pair-{i + 1}-{side}"
                try:
                    result = run(
                        side,
                        label,
                        spec if side == arm else None,
                        reference,
                        profile=i == 0,
                    )
                except subprocess.CalledProcessError:
                    if side == "control":
                        raise
                    record["failed_run"] = label
                    break
                if not numerical_gate(result):
                    raise RuntimeError(f"Incomplete numerical checks in {label}")
                if i == 0:
                    trace = root / f"{label}.trace.json"
                    launches = (
                        launch_contracts(trace, target)
                        if legacy
                        else retained_launches(trace, target, args.phase, side == arm)
                    )
                    record[f"launches_{side}"] = launches
                    if side == arm:
                        for launch in launches:
                            actual = (
                                launch["group"],
                                launch["modes"],
                                launch["warps"],
                                launch["compact"],
                            )
                            if launch["lines"] != 1 or actual != spec:
                                raise RuntimeError(f"Compiled launch mismatch {actual} != {spec}")
                    elif legacy and any(v["group"] or v["compact"] for v in launches):
                        raise RuntimeError("Control did not run the cand12 layout")
                pair[side] = {k: result[k] for k in ("median_ms", "images_s", "peak_allocated_mib")}
            if "failed_run" in record:
                break
            record["pairs"].append(pair)
        ratios = [x["control"]["median_ms"] / x[arm]["median_ms"] for x in record["pairs"]]
        controls = [x["control"]["median_ms"] for x in record["pairs"]]
        candidates_ms = [x[arm]["median_ms"] for x in record["pairs"]]
        if len(ratios) == 3 and max(candidates_ms) < min(controls):
            record["passed"] = True
            score = statistics.median(ratios)
            if score > best_score:
                best_score, winner = score, geometry
        (root / "calibration.json").write_text(json.dumps(audit, indent=2) + "\n")
    if winner is not None:
        store_geometry(name, winner, scope=scope)
        if stored_geometry(name, scope=scope) != winner:
            raise RuntimeError("Validated winner could not be persisted")
    audit["winner"] = asdict(winner) if winner else None
    (root / "calibration.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--trial"]:
        trial(json.loads(Path(sys.argv[2]).read_text()))
    else:
        main()
